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


def center_crop_resize(image, H, W):
    W_img, H_img = image.size
    target_aspect = W / H
    input_aspect = W_img / H_img

    if input_aspect > target_aspect:
        new_width = int(target_aspect * H_img)
        new_height = H_img
    else:
        new_width = W_img
        new_height = int(W_img / target_aspect)

    left = (W_img - new_width) / 2
    top = (H_img - new_height) / 2
    right = (W_img + new_width) / 2
    bottom = (H_img + new_height) / 2

    image = image.crop((left, top, right, bottom))
    image = image.resize((W, H))

    return image


def load_image(image_path, H, W):
    image = PIL.Image.open(image_path)
    return np.array(center_crop_resize(image, H, W))[..., :3]


def load_depth_image(image_path, H, W):
    depth_image_path = image_path.replace(".png", "_depth.png")
    if os.path.exists(depth_image_path):
        depth_image = PIL.Image.open(depth_image_path)
    else:
        depth_image = PIL.Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    return np.array(center_crop_resize(depth_image, H, W))[..., :3]


def save_combined_image(image, pred_image, target, pred_depth, target_depth, sample_id, prompt, include_depth):
    comb_img = np.concatenate([image, pred_image, target], axis=1)
    if include_depth:
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


def save_images(image, pred_image, target, depth, pred_depth, target_depth, sample_id, has_target, include_depth):
    cv2.imwrite(f"{sample_id}_image.png", np.flip(pred_image, axis=-1))
    cv2.imwrite(f"{sample_id}_base_image.png", np.flip(image, axis=-1))
    if has_target:
        cv2.imwrite(f"{sample_id}_target_image.png", np.flip(target, axis=-1))
    if include_depth:
        cv2.imwrite(f"{sample_id}_depth.png", np.flip(pred_depth, axis=-1))
        cv2.imwrite(f"{sample_id}_base_depth.png", np.flip(depth, axis=-1))
        if has_target:
            cv2.imwrite(f"{sample_id}_target_depth.png", np.flip(target_depth, axis=-1))


def load_pipeline(ckpt_folder, model_id="stabilityai/stable-diffusion-2", device_id=0, seed=42):
    ckpt_folder = ckpt_folder.rstrip("/")
    device = f"cuda:{device_id}"

    # ==== Model Configuration ====
    if os.path.exists(ckpt_folder):
        folder_list = os.listdir(ckpt_folder)
        if "unet" in folder_list:
            latest_checkpoint = ckpt_folder
        else:
            checkpoint_files = [os.path.join(ckpt_folder, f) for f in folder_list if f.startswith("checkpoint")]
            latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split("-")[-1]))[-1]
        run_id = latest_checkpoint.split("/")[-3]
        run_id = os.path.join(run_id, "results", os.path.basename(latest_checkpoint))
    else:
        latest_checkpoint = ckpt_folder
        run_id = latest_checkpoint.split("/")[-1]
        run_id = os.path.join(run_id, "results")
    logging.info(f"Loading checkpoint from {latest_checkpoint}")

    # ==== Load model ====
    unet = UNet2DConditionModel.from_pretrained(latest_checkpoint, subfolder="unet")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, unet=unet, torch_dtype=torch.float32, use_safetensors=True
    ).to(device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

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
    parser.add_argument("--text", default=None, help="text input")
    parser.add_argument("--image", default=None, help="path to the image input")
    parser.add_argument("--save_path", default="tmp/result", help="path where inference results will be saved")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    global INCLUDE_DEPTH
    INCLUDE_DEPTH = args.include_depth
    setup_logger()

    # ==== Load data ====
    if args.image is None:
        test_samples = json.load(open("dataset/1.1/goal_image/ldm_goal_image_test.json"))
        np.random.seed(args.seed)
        np.random.shuffle(test_samples)
        if not args.all and args.num_samples > 0:
            test_samples = test_samples[: args.num_samples]
    else:
        assert args.image is not None and args.text is not None
        test_samples = [{"base_image_path": args.image, "instruction": args.text}]
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

    # ==== spawn for cuda devices ====
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
    setup_logger()
    has_target = args.image is None
    main_process = device_id == 0
    if not main_process:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        logging.getLogger().setLevel(logging.CRITICAL)

    # ==== Load model ====
    pipe, generator, run_id = load_pipeline(args.ckpt_folder, device_id=device_id, seed=args.seed)

    # ==== Save Folder ====
    if has_target:
        result_path = f"lavis/output/LDM/{run_id}/test_g{guidance_scale}_s{num_inference_steps}"
        os.makedirs(result_path, exist_ok=True)
        logging.info(f"Saving results to {result_path}")

    bar = tqdm(sub_test_samples, **TQDM_ARGS) if main_process else sub_test_samples
    for idx, d in enumerate(bar):
        # === Prepare input ===
        image = load_image(d["base_image_path"], H, W)
        if args.include_depth:
            depth = load_depth_image(d["base_image_path"], H, W)
            input_image = np.concatenate([image, depth], axis=-1)  # (H, W, 6)
        else:
            depth = None
            input_image = image

        prompt = d["instruction"]

        # ==== Inference ====
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
        else:
            pred_depth = None

        # === Load GT ===
        if has_target:
            final_image_path = d["final_image_path"]
            target = load_image(final_image_path, H, W)
            target_depth = load_depth_image(final_image_path, H, W) if args.include_depth else None
            sample_id = os.path.join(result_path, f"{d['dataset']}_{d['scene_id']}")
        else:
            target = np.zeros((H, W, 3), dtype=np.uint8)
            target_depth = target
            sample_id = args.save_path

        # ==== Save images ====
        save_combined_image(image, pred_image, target, pred_depth, target_depth, sample_id, prompt, args.include_depth)
        save_images(
            image, pred_image, target, depth, pred_depth, target_depth, sample_id, has_target, args.include_depth
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
