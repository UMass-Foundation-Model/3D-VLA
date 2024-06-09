import os
import sys
import PIL
import torch
import json
import cv2
import numpy as np
import logging
import textwrap
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing as mp
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

from lavis.common.logger import setup_logger
from lavis.common.utils import TQDM_ARGS


def load_pipeline(ckpt_folder, model_id="stabilityai/stable-diffusion-2", device_id=0):
    device = f"cuda:{device_id}"

    # ==== Model Configuration ====
    # checkpoint-num
    latest_checkpoint = sorted(
        [os.path.join(ckpt_folder, f) for f in os.listdir(ckpt_folder) if f.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[-1]),
    )[-1]
    run_id = latest_checkpoint.split("/")[-3]
    run_id = os.path.join(run_id, "results", os.path.basename(latest_checkpoint))
    logging.info(f"Loading checkpoint from {latest_checkpoint}")

    # ==== Load model ====
    unet = UNet2DConditionModel.from_pretrained(latest_checkpoint, subfolder="unet")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, unet=unet, torch_dtype=torch.float32, use_safetensors=True
    ).to(device=device)
    generator = torch.Generator(device=device)

    # ==== Custom pipeline functions ====
    def my_prepare_image_latents(
        image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == 6:
            rgb, d = torch.split(image, 3, dim=1)
            image_embeds_rgb = pipe.vae.encode(rgb).latent_dist.mode()
            image_embeds_d = pipe.vae.encode(d).latent_dist.mode()
            image_latents = torch.cat([image_embeds_rgb, image_embeds_d], dim=1)
        elif image.shape[1] == 3:
            image_embeds = pipe.vae.encode(image).latent_dist.mode()
            image_latents = torch.cat([image_embeds], dim=1)
        else:
            raise ValueError("Invalid input shape")

        image_latents = torch.cat([image_latents], dim=0)
        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
        return image_latents

    pipe.safety_checker = lambda images, **kwargs: (images, [False])
    pipe.set_progress_bar_config(leave=False, desc="Inference", **TQDM_ARGS)
    pipe.prepare_image_latents = my_prepare_image_latents

    return pipe, generator, run_id


def decode_one_latent(pipe, latent, output_type="np"):
    pred_image = pipe.vae.decode(
        latent.unsqueeze(0) / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    pred_image = pred_image / 2 + 0.5
    pred_image = (
        pipe.image_processor.postprocess(
            pred_image.detach(),
            output_type=output_type,
            do_denormalize=[False],
        )
        * 255
    )
    return pred_image


def main():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--ckpt_folder", default="lavis/output/LDM/pix2pix/runs", help="path to checkpoint folder")
    parser.add_argument("--num_samples", type=int, default=100, help="number of samples to run inference on")
    parser.add_argument("--all", action="store_true", help="run inference on all samples")
    parser.add_argument("--include_depth", action="store_true", help="include depth in input")
    args = parser.parse_args()
    global INCLUDE_DEPTH
    INCLUDE_DEPTH = args.include_depth
    setup_logger()

    # ==== Load data ====
    data_root = "./data"
    test_samples = json.load(open("dataset/1.1/goal_image/ldm_goal_image_test.json"))
    np.random.seed(42)
    np.random.shuffle(test_samples)
    if not args.all and args.num_samples > 0:
        test_samples = test_samples[: args.num_samples]
    logging.info(f"Loaded {len(test_samples)} test samples")

    # ==== Inference Configuration ====
    num_inference_steps = 50
    image_guidance_scale = 2.5
    guidance_scale = 2.5
    logging.info(f"steps: {num_inference_steps}, img_guidance: {image_guidance_scale}, guidance: {guidance_scale}")
    H, W = 256, 256

    # ==== Run Inference ====
    num_workers = torch.cuda.device_count()
    logging.info(f"Running inference on {num_workers} GPUs")
    sub_test_samples = np.array_split(test_samples, num_workers)

    # spawn for cuda devices
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(args, sub_test_samples[i], (H, W), num_inference_steps, image_guidance_scale, guidance_scale, i),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logging.info("Inference done!")


def worker(
    args,
    sub_test_samples,
    resolution,
    num_inference_steps,
    image_guidance_scale,
    guidance_scale,
    device_id=0,
):
    H, W = resolution
    main_process = device_id == 0
    if not main_process:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        logging.getLogger().setLevel(logging.CRITICAL)

    # ==== Load model ====
    pipe, generator, run_id = load_pipeline(args.ckpt_folder, device_id=device_id)

    # ==== Save Folder ====
    result_path = f"lavis/output/LDM/{run_id}/test_g{guidance_scale}_s{num_inference_steps}"
    os.makedirs(result_path, exist_ok=True)
    logging.info(f"Saving results to {result_path}")

    bar = tqdm(sub_test_samples, **TQDM_ARGS) if main_process else sub_test_samples
    for idx, d in enumerate(bar):
        # === Prepare input ===
        image = PIL.Image.open(d["base_image_path"])
        image = np.array(image.resize((H, W)))[..., :3]
        if args.include_depth:
            path_to_load = d["base_image_path"].replace(".png", "_depth.png")
            if os.path.exists(path_to_load):
                depth = PIL.Image.open(d["base_image_path"].replace(".png", "_depth.png"))
            else:
                depth = PIL.Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
            depth = np.array(depth.resize((H, W)))[..., :3]
            input_image = np.concatenate([image, depth], axis=-1)  # (H, W, 6)
        else:
            input_image = image

        prompt = d["instruction"]

        pipe.vae.config.latent_channels = 8 if args.include_depth else 4
        edited_image = pipe(
            prompt,
            image=input_image / 255.0,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
        ).images[0]

        # ==== Decode edited image ====
        pipe.vae.config.latent_channels = 4
        latent_rgb = edited_image[:4]
        pred_image = decode_one_latent(pipe, latent_rgb)
        pred_image = pred_image.reshape(H, W, 3)
        if args.include_depth:
            latent_d = edited_image[4:]
            pred_depth = decode_one_latent(pipe, latent_d)
            pred_depth = pred_depth.reshape(H, W, 3)

        # === Load GT ===
        target = PIL.Image.open(d["final_image_path"])
        target = np.array(target.resize((H, W)))[..., :3]
        if args.include_depth:
            path_to_load = d["final_image_path"].replace(".png", "_depth.png")
            if os.path.exists(path_to_load):
                target_depth = PIL.Image.open(d["final_image_path"].replace(".png", "_depth.png"))
            else:
                target_depth = PIL.Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
            target_depth = np.array(target_depth.resize((H, W)))[..., :3]

        # === Save results ===
        sample_id = os.path.join(result_path, f"{d['dataset']}_{d['scene_id']}")

        comb_img = np.concatenate([image, pred_image, target], axis=1)
        if args.include_depth:
            pred_depth = pred_depth.reshape(H, W, 3)
            comb_depth = np.concatenate([depth, pred_depth, target_depth], axis=1)
            comb_img = np.concatenate([comb_img, comb_depth], axis=0)

        _, ax = plt.subplots(1, 1)
        ax.imshow(comb_img.clip(0, 255).astype(np.uint8))
        ax.axis("off")
        ax.set_title("\n".join(textwrap.wrap(prompt, width=50)))
        plt.tight_layout()
        plt.savefig(f"{sample_id}_comb.png")
        plt.close()

        cv2.imwrite(f"{sample_id}_image.png", np.flip(pred_image.reshape(H, W, 3), axis=-1))
        cv2.imwrite(f"{sample_id}_base_image.png", np.flip(image, axis=-1))
        cv2.imwrite(f"{sample_id}_target_image.png", np.flip(target, axis=-1))
        if args.include_depth:
            cv2.imwrite(f"{sample_id}_depth.png", np.flip(pred_depth.reshape(H, W, 3), axis=-1))
            cv2.imwrite(f"{sample_id}_base_depth.png", np.flip(depth, axis=-1))
            cv2.imwrite(f"{sample_id}_target_depth.png", np.flip(target_depth, axis=-1))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
