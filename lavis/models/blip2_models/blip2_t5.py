import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from positional_encodings.torch_encodings import PositionalEncoding1D

from lavis.common.utils import _TORCH_GREATER_EQUAL_1_12

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.wrap import enable_wrap, wrap
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from lavis.common.utils import apply_with_stopping_condition


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, cache_dir="./cache")

        # ================ Warning: do not change the order of the **below** tokens >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # ==== Add special tokens ====
        # Location
        self.special_token_ids = []
        location_tokens = []
        for i in range(256):  # pc -> x,y,z tokens
            location_tokens.append("<loc%d>" % i)
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": location_tokens})
        self.special_token_ids += self.t5_tokenizer.convert_tokens_to_ids(location_tokens)
        self.start_loc_id = self.t5_tokenizer.convert_tokens_to_ids(["<loc0>"])[0]
        # 3D feature
        tokens3d = ["<scene>", "</scene>"]
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": tokens3d})
        self.special_token_ids += self.t5_tokenizer.convert_tokens_to_ids(tokens3d)
        self.tokens3d_id = self.t5_tokenizer.convert_tokens_to_ids(["<scene>"])[0]
        # object
        object_tokens = ["<obj>", "</obj>"]
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": object_tokens})
        self.special_token_ids += self.t5_tokenizer.convert_tokens_to_ids(object_tokens)
        self.start_obj_id = self.t5_tokenizer.convert_tokens_to_ids(["<obj>"])[0]
        # Goal image
        image_tokens = ["<image>", "</image>"]
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": image_tokens})
        self.special_token_ids += self.t5_tokenizer.convert_tokens_to_ids(image_tokens)
        self.start_image_id = self.t5_tokenizer.convert_tokens_to_ids(["<image>"])[0]
        self.end_image_id = self.t5_tokenizer.convert_tokens_to_ids(["</image>"])[0]
        # Goal PointCloud
        pcd_tokens = ["<pcd>", "</pcd>"]
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": pcd_tokens})
        self.special_token_ids += self.t5_tokenizer.convert_tokens_to_ids(pcd_tokens)
        self.start_pc_id = self.t5_tokenizer.convert_tokens_to_ids(["<pcd>"])[0]
        self.end_pc_id = self.t5_tokenizer.convert_tokens_to_ids(["</pcd>"])[0]
        # Action 7D
        self.num_act_tokens = 256
        action7d_tokens = []
        for i in range(self.num_act_tokens):
            action7d_tokens.append("<aloc%d>" % i)
            action7d_tokens.append("<arot%d>" % i)
        action7d_tokens.append("<gripper0>")
        action7d_tokens.append("<gripper1>")
        action7d_tokens.append("<ACT_SEP>")
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": action7d_tokens})
        self.special_token_ids += self.t5_tokenizer.convert_tokens_to_ids(action7d_tokens)

        # ==== Load T5 model ====
        t5_config = T5Config.from_pretrained(t5_model, cache_dir="./cache")
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config, cache_dir="./cache")
        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        # ==== Freeze T5 model ====
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            # param.data = param.data.bfloat16()  # V100 doesn't support bfloat16
        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.input_embeddings = self.t5_model.get_input_embeddings()

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        # ==== Positional encoding ====
        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.pos_embedding = pos_model(x).squeeze().cuda()

        # ==== Other ====
        self.max_txt_len = max_txt_len
        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def prepare_inputs(self, pc_embeds, pc, num_feat):
        B, T, N, _ = pc.shape
        with torch.cuda.amp.autocast(dtype=torch.float32):
            all_pcs = torch.zeros((pc_embeds.shape))  # B, T, N, 1408
            clamped_pc = torch.clamp(pc, 0, self.pos_embedding.shape[0] - 1).long()
            all_pcs_emb = self.pos_embedding[clamped_pc]  # (B, T, N, 3, 469)
            all_pcs[..., :1407] = all_pcs_emb.reshape(B, T, N, -1)  # (B, T, N, 1407)
            all_pcs = all_pcs.cuda()
            pc_embeds = pc_embeds + all_pcs * 0.1  # B, T, N, 1408

        image_atts = torch.ones(pc_embeds.size()[:-1]).long().to(pc_embeds.device)  # (B, T, N)
        query_tokens = self.query_tokens.expand(B * T, -1, -1)  # (B * T, 32, 768)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds.reshape(B * T, N, -1),
            encoder_attention_mask=image_atts.reshape(B * T, N),
            return_dict=True,
        )
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)  # (B * T, 32, 2048)
        inputs_t5 = inputs_t5.reshape(B, T, -1, inputs_t5.shape[-1])  # (B, T, 32, 2048)
        atts_t5 = torch.ones(inputs_t5.size()[:-1]).long().to(pc_embeds.device)  # (B, T, 32)

        return inputs_t5, atts_t5

    def insert_3d_feats(
        self, batch_input_tokens_input_ids, batch_input_tokens_atts, batch_atts_t5, batch_inputs_t5, num_feat
    ):
        B = batch_input_tokens_input_ids.shape[0]
        inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)  # (B, L_input, 1024)
        inputs_list = []
        atts_list = []
        for b in range(B):
            current_embeds = inputs_embeds[b]  # (L_input, 1024)
            current_att = batch_input_tokens_atts[b]  # (L_input)
            current_3d_feats = batch_inputs_t5[b]  # (T, 32, 2048)
            current_3d_att = batch_atts_t5[b]  # (T, 32)
            feat3d_indices = (batch_input_tokens_input_ids[b] == self.tokens3d_id).nonzero(as_tuple=True)[
                0
            ]  # (num_3d_feats)
            assert len(feat3d_indices) == num_feat[b], f"{len(feat3d_indices)} != {num_feat[b]}"
            for i, idx in enumerate(reversed(feat3d_indices)):
                left = current_embeds[: idx + 1]
                right = current_embeds[idx + 1 :]
                current_embeds = torch.cat([left, current_3d_feats[i], right], dim=0)
                left = current_att[: idx + 1]
                right = current_att[idx + 1 :]
                current_att = torch.cat([left, current_3d_att[i], right], dim=0)
            inputs_list.append(current_embeds)
            atts_list.append(current_att)
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            inputs_list, batch_first=True, padding_value=0
        )  # (B, L_input + T * 32, 1024)
        encoder_atts = torch.nn.utils.rnn.pad_sequence(
            atts_list, batch_first=True, padding_value=0
        )  # (B, L_input + T * 32)
        return inputs_embeds, encoder_atts

    def forward(self, samples):
        pc_embeds = samples["pc_feat"]  # B, T, N, 1408
        pc = samples["pc"].long()  # B, T, N, 3
        num_feat = samples["num_features"]  # B, each element <= T
        B, T, N, _ = pc.shape

        inputs_t5, atts_t5 = self.prepare_inputs(pc_embeds, pc, num_feat)

        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(
                pc_embeds.device
            )  # (B, L_input)
            output_tokens = self.t5_tokenizer(
                samples["answer"],
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(
                pc_embeds.device
            )  # (B, L_output)

            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []
            batch_num_feat = []
            for b, n in enumerate(samples["n_answers"]):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                batch_atts_t5 += [atts_t5[b]] * n
                batch_inputs_t5 += [inputs_t5[b]] * n
                batch_num_feat += [num_feat[b]] * n

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)  # (B, L_input)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)  # (B, L_input)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)  # (B, T, 32)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)  # (B, T, 32, 2048)

            inputs_embeds, encoder_atts = self.insert_3d_feats(
                batch_input_tokens_input_ids, batch_input_tokens_atts, batch_atts_t5, batch_inputs_t5, batch_num_feat
            )

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )

            loss = outputs.loss
            return {"loss": loss}

    def predict_answers(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        pc_embeds = samples["pc_feat"]
        pc = samples["pc"].long()
        num_feat = samples["num_features"]
        B, T, N, _ = pc.shape

        inputs_t5, atts_t5 = self.prepare_inputs(pc_embeds, pc, num_feat)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        prompt = self.prompt

        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(text_input, padding="longest", return_tensors="pt").to(pc_embeds.device)

        inputs_embeds, encoder_atts = self.insert_3d_feats(
            input_tokens.input_ids, input_tokens.attention_mask, atts_t5, inputs_t5, num_feat
        )

        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=1,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=False)

        if self._apply_lemmatizer:
            output_text_new = self._lemmatize(output_text)
            output_text = output_text_new

        return output_text

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.

        The rough wrapping structure is:
        - BLIP2-T5
            - Qformer.bert
                - FSDP(FSDP(embeddings))
                - FSDP(FSDP(encoder.layer))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FSDP(FSDP(encoder.block))
                - FSDP(FSDP(output_embeddings))
                - FSDP(FSDP(output_embeddings))
                - other parameters
            - FSDP(FSDP(T5_Proj))

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the t5_model.encoder.block and t5_model.decoder.block?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        Great thanks to https://github.com/mlfoundations/open_flamingo for the FSDP wrapping strategy.
        """
        self.t5_model.encoder.block.requires_grad_(True)
        self.t5_model.decoder.block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.Qformer.bert.embeddings = wrap(wrap(self.Qformer.bert.embeddings))
            self.Qformer.bert.encoder.layer = nn.ModuleList(
                wrap(wrap(layer)) for layer in self.Qformer.bert.encoder.layer
            )
            self.t5_model.encoder.block = nn.ModuleList(wrap(wrap(block)) for block in self.t5_model.encoder.block)
            self.t5_model.decoder.block = nn.ModuleList(wrap(wrap(block)) for block in self.t5_model.decoder.block)
            self.t5_model.set_input_embeddings(wrap(wrap(self.t5_model.get_input_embeddings())))
            self.t5_model.set_output_embeddings(wrap(wrap(self.t5_model.get_output_embeddings())))
            self.t5_proj = wrap(wrap(self.t5_proj))

        # manually move non-FSDP managed parameters to device_id
        # these are all in t5_model
        apply_with_stopping_condition(
            module=self.t5_model,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # exclude the original decoder layers from the optimizer
        for block in self.t5_model.encoder.block:
            for p in block.parameters():
                p.exclude_from_optimizer = True  # type: ignore
        for block in self.t5_model.decoder.block:
            for p in block.parameters():
                p.exclude_from_optimizer = True  # type: ignore
        self.to_delete = ["t5_model.encoder.block", "t5_model.decoder.block"]

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
