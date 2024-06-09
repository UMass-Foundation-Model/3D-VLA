import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
import logging

from lavis.datasets.datasets.goal_pcd_rlbench_datasets import RLBench
from lavis.models.pointe.configs import MODEL_CONFIGS, model_from_config
from lavis.models.pointe.download import load_checkpoint
from lavis.common.logger import setup_logger


@dataclass
class TrainingConfig:
    train_batch_size = 1
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lavis/output/PE/rlbench-point-as-token/runs"  # the model name locally and on the HF Hub

    model_config = {
        "cond_drop_prob": 0.1,
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 2048,
        "name": "CLIPImageGoalPointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "token_cond": True,
        "width": 512,
        "cache_dir": "cache/point_e_model",
    }


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, resume_path=None, epoch=0):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if resume_path != None:
        logging.info(f"resuming checkpoint from {resume_path}") if accelerator.is_local_main_process else None
        accelerator.load_state(resume_path)
    else:
        logging.info("finetuning from the beginning") if accelerator.is_local_main_process else None

    if hasattr(model, "module"):
        model.module.load_clip_acc(accelerator.device)
    else:
        model.load_clip_acc(accelerator.device)
    global_step = 0

    # Now you train the model
    for epoch in range(epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        for step, batch in enumerate(train_dataloader):
            pointcloud = batch["start_pc"].type(torch.float32).permute(0, 2, 1)  # [-1,1]
            target = batch["end_pc"].type(torch.float32).permute(0, 2, 1)  # [-1,1]

            pointcloud[:, 2] = pointcloud[:, 2] - 1.0
            target[:, 2] = target[:, 2] - 1.0

            # Sample noise to add to the images
            noise = torch.randn(pointcloud.shape, device=pointcloud.device)
            bs = pointcloud.shape[0]

            text = batch["instruction"]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=pointcloud.device, dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_pointcloud = noise_scheduler.add_noise(target, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_pointcloud, timesteps, pointcloud, texts=text)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            save_path = os.path.join(config.output_dir, f"ckpt-{epoch}")
            accelerator.save_state(save_path)
            save_model = accelerator.unwrap_model(model)
            os.makedirs(os.path.join(config.output_dir, "raw_model"), exist_ok=True)
            torch.save(save_model, os.path.join(config.output_dir, "raw_model", f"ckpt_ep{epoch}.pth"))


if __name__ == "__main__":
    setup_logger()
    device = "cuda"
    config = TrainingConfig()
    model_config = config.model_config
    dataset = RLBench(split="train", sample_size=model_config["n_ctx"])
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1024, beta_schedule="squaredcos_cap_v2")

    # ==== Load model ====
    epoch = 0
    base_name = "base40M_textvec"  # use base300M or base1B for better results
    model = model_from_config(model_config, device)
    resume_path = None
    if os.path.exists(config.output_dir):
        ckpts = [int(name[5:].split(".")[0]) for name in os.listdir(config.output_dir) if "ckpt" in name]
        if ckpts:
            max_ckpt = max(ckpts)
            resume_path = os.path.join(config.output_dir, f"ckpt-{max_ckpt}")
            epoch = max_ckpt + 1
        else:
            model.load_state_dict(load_checkpoint(base_name, device, cache_dir=model_config["cache_dir"]), strict=False)
    else:  # init from point-e
        model.load_state_dict(load_checkpoint(base_name, device, cache_dir=model_config["cache_dir"]), strict=False)

    # ==== change input and output layer ====
    with torch.no_grad():
        new_linear_out = nn.Linear(model.output_proj.in_features, 6, bias=True)
        new_linear_out.weight.zero_()
        new_linear_out.weight.copy_(model.output_proj.weight[:6])
        model.output_proj = new_linear_out

    # ==== Optimizer and Scheduler ====
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, resume_path, epoch)
