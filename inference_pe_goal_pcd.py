import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse

from lavis.datasets.datasets.goal_pcd_rlbench_datasets import RLBench
from lavis.models.pointe.transformer import GoalPointDiffusionTransformer
from lavis.models.pointe.download import load_checkpoint
from lavis.common.logger import setup_logger
from lavis.common.utils import TQDM_ARGS


@dataclass
class TrainingConfig:
    eval_batch_size = 1  # how many images to sample during evaluation
    ckpt_dir = "anyeZHY/3dvla-diffusion-pointcloud"  # the model name locally and on the HF Hub
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    classifier_free_guidance = 2.0  # the strength of classifier-free guidance

    model_config = {
        "cond_drop_prob": 0.1,
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 8192,
        "name": "GoalPointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "token_cond": True,
        "width": 512,
        "pointe_cache_dir": "cache/point_e_model",
        "device": "cuda",
        "dtype": torch.float32,
    }


def plot_point_cloud(
    pc,
    name="debug.jpg",
    color: bool = True,
    grid_size: int = 1,
    fixed_bounds=(
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
    ),
):
    """
    Render a point cloud as a plot to the given image path.

    :param pc: the PointCloud to plot.
    :param image_path: the path to save the image, with a file extension.
    :param color: if True, show the RGB colors from the point cloud.
    :param grid_size: the number of random rotations to render.
    """
    fig = plt.figure(figsize=(8, 8))
    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(grid_size, grid_size, 1 + j + i * grid_size, projection="3d")
            color_args = {}
            if color:
                color_args["c"] = np.clip((pc[:, 3:] + 1) / 2, 0.0, 1.0)
            c = pc[:, :3]

            if grid_size > 1:
                theta = np.pi * 2 * (i * grid_size + j) / (grid_size**2)
                rotation = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                c = c @ rotation

            ax.scatter(c[:, 0], c[:, 1], c[:, 2], **color_args)

            if fixed_bounds is None:
                min_point = c.min(0)
                max_point = c.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                ax.set_xlim3d(center[0] - size, center[0] + size)
                ax.set_ylim3d(center[1] - size, center[1] + size)
                ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])
    fig.savefig(name)
    plt.close(fig)
    return fig


def evaluate(
    config,
    accelerator,
    model,
    scheduler,
    train_dataloader,
    save_path,
    classifier_free_guidance=1.0,
):
    if hasattr(model, "module"):
        model.module.load_clip_acc(accelerator.device)
    else:
        model.load_clip_acc(accelerator.device)

    model.eval()
    for _, batch in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process, **TQDM_ARGS)):
        input_pointcloud = batch["start_pc"].type(torch.float32).permute(0, 2, 1)  # [-1,1]
        if "end_pc" in batch:
            gt_pointcloud = batch["end_pc"].type(torch.float32).permute(0, 2, 1)  # [-1,1]
        else:
            gt_pointcloud = None
        text = batch["instruction"][0]

        pointcloud = torch.randn((config.eval_batch_size, 6, config.model_config["n_ctx"])).to(accelerator.device)
        pointcloud = torch.cat([pointcloud, input_pointcloud], dim=1)

        # set step values
        scheduler.set_timesteps(64)

        model.eval()
        for t in tqdm(scheduler.timesteps, leave=False, disable=not accelerator.is_main_process, **TQDM_ARGS):
            # 1. predict noise model_output
            if classifier_free_guidance > 1.0:
                texts = [text, None]
                timestep = torch.ones(2 * config.eval_batch_size) * t
                pointcloud = torch.cat([pointcloud, pointcloud], dim=0)
            else:
                texts = [text]
                timestep = torch.ones(config.eval_batch_size) * t

            model_output = model(pointcloud, timestep.to(accelerator.device), input_pointcloud, texts=texts)

            if classifier_free_guidance > 1.0:
                cond_x_0, uncond_x_0 = torch.split(model_output, len(pointcloud) // 2, dim=0)
                mean = uncond_x_0 + classifier_free_guidance * (cond_x_0 - uncond_x_0)
                model_output = mean
                pointcloud = pointcloud[:1]

            # 2. compute previous image: x_t -> t_t-1
            pointcloud[:, :6] = scheduler.step(model_output, t, pointcloud[:, :6]).prev_sample

        # ==== Save results ====
        path_to_save = os.path.join(save_path, f"scene_{batch['scene_id'][0]}")
        os.makedirs(path_to_save, exist_ok=True)
        # gpu -> cpu
        pointcloud = pointcloud.cpu().numpy().transpose(0, 2, 1)[0, :, :6]
        input_pointcloud = input_pointcloud.detach().cpu().numpy().transpose(0, 2, 1)[0]
        if gt_pointcloud is not None:
            gt_pointcloud = gt_pointcloud.detach().cpu().numpy().transpose(0, 2, 1)[0]
        # save images
        pointclouds_to_concat = [pointcloud, input_pointcloud]
        if gt_pointcloud is not None:
            pointclouds_to_concat.append(gt_pointcloud)
        all_points = np.concatenate(pointclouds_to_concat, axis=0)
        bounds = np.array([all_points.min(0)[:3], all_points.max(0)[:3]])
        plot_point_cloud(pointcloud, os.path.join(path_to_save, "pred_pc.jpg"), fixed_bounds=bounds)
        plot_point_cloud(input_pointcloud, os.path.join(path_to_save, "input_pc.jpg"), fixed_bounds=bounds)
        if gt_pointcloud is not None:
            plot_point_cloud(gt_pointcloud, os.path.join(path_to_save, "target_pc.jpg"), fixed_bounds=bounds)
        np.save(os.path.join(path_to_save, "inference.npy"), pointcloud)
        np.save(os.path.join(path_to_save, "input.npy"), input_pointcloud)
        if gt_pointcloud is not None:
            np.save(os.path.join(path_to_save, "target.npy"), gt_pointcloud)
        # write text
        open(os.path.join(path_to_save, "instruction.txt"), "w").write(text)


# Define a custom dataset for single sample inference
class SingleSampleDataset(torch.utils.data.Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.batch


if __name__ == "__main__":
    setup_logger()
    config = TrainingConfig()
    torch.set_grad_enabled(False)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npy", type=str, default=None, help="Path to the input point cloud npy/npz file")
    parser.add_argument("--text", type=str, default=None, help="Instruction text")
    parser.add_argument("--output_dir", type=str, default="lavis/output/PE/single_sample", help="Output directory")
    args = parser.parse_args()

    # ==== Initialize accelerator ====
    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    # ==== Load model ====
    name = config.model_config.pop("name")
    ckpt_dir = config.ckpt_dir
    logging.info(f"Loading model {name} from {ckpt_dir}")

    if os.path.exists(ckpt_dir) and not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        # load from local
        model = GoalPointDiffusionTransformer(**config.model_config)
        if "ckpt" in ckpt_dir:
            path_to_load = ckpt_dir
            save_path = os.path.join(ckpt_dir.replace("runs", "results"))
        else:
            path_to_load = None
            ckpts = [int(name[5:].split(".")[0]) for name in os.listdir(ckpt_dir) if "ckpt" in name]
            assert len(ckpts) > 0, f"No checkpoints found in {ckpt_dir}"
            max_ckpt = max(ckpts)
            path_to_load = os.path.join(ckpt_dir, f"ckpt-{max_ckpt}")
            save_path = os.path.join(ckpt_dir.replace("runs", "results"), f"ckpt-{max_ckpt}")
        model = accelerator.prepare(model)
        accelerator.load_state(path_to_load)
    else:
        # load from Hugging Face
        model = GoalPointDiffusionTransformer.from_pretrained(ckpt_dir, strict=True)
        model = accelerator.prepare(model)
        name = ckpt_dir.split("/")[-1]
        save_path = os.path.join("lavis/output/PE", name)

    if args.input_npy is not None and args.text is not None:
        save_path = args.output_dir
        input_pointcloud = np.load(args.input_npy)
        if args.input_npy.endswith(".npz"):
            input_pointcloud = input_pointcloud["arr_0"]
        if input_pointcloud.shape[0] > config.model_config["n_ctx"]:
            selected_idx = np.random.choice(input_pointcloud.shape[0], config.model_config["n_ctx"])
            input_pointcloud = input_pointcloud[selected_idx]
        mean = np.mean(input_pointcloud[:, :3], axis=0)
        radius = np.max(np.linalg.norm(input_pointcloud[:, :3] - mean[:3], axis=1))
        input_pointcloud[:, :3] = (input_pointcloud[:, :3] - mean[:3]) / radius
        start_pc = torch.tensor(input_pointcloud, dtype=torch.float32)
        batch = {
            "start_pc": start_pc,
            "instruction": args.text,
            "scene_id": [0],
        }
        dataset = SingleSampleDataset(batch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    else:
        # ==== Initialize dataset and dataloader ====
        dataset = RLBench(split="test", sample_size=config.model_config["n_ctx"])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    dataloader = accelerator.prepare(dataloader)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1024, beta_schedule="squaredcos_cap_v2")

    with torch.no_grad():
        evaluate(config, accelerator, model, noise_scheduler, dataloader, save_path, config.classifier_free_guidance)
