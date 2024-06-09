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

from lavis.datasets.datasets.goal_pcd_rlbench_datasets import RLBench
from lavis.models.pointe.configs import MODEL_CONFIGS, model_from_config
from lavis.models.pointe.download import load_checkpoint
from lavis.common.logger import setup_logger


@dataclass
class TrainingConfig:
    eval_batch_size = 1  # how many images to sample during evaluation
    output_dir = "lavis/output/PE/rlbench-point-as-token/runs"  # the model name locally and on the HF Hub
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    gradient_accumulation_steps = 1

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
    model,
    scheduler,
    train_dataloader,
    max_ckpt,
    resume_path=None,
    classifier_free=False,
):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, train_dataloader = accelerator.prepare(model, train_dataloader)

    if resume_path != None:
        logging.info(f"resuming checkpoint from {resume_path}") if accelerator.is_main_process else None
        accelerator.load_state(resume_path)
    else:
        logging.info("finetuning from the beginning") if accelerator.is_main_process else None

    if hasattr(model, "module"):
        model.module.load_clip_acc(accelerator.device)
    else:
        model.load_clip_acc(accelerator.device)

    model.eval()
    for _, batch in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
        input_pointcloud = batch["start_pc"].type(torch.float32).permute(0, 2, 1)  # [-1,1]
        gt_pointcloud = batch["end_pc"].type(torch.float32).permute(0, 2, 1)  # [-1,1]

        input_pointcloud[:, 2] = input_pointcloud[:, 2] - 1.0
        gt_pointcloud[:, 2] = gt_pointcloud[:, 2] - 1.0

        text = batch["instruction"][0]

        pointcloud = torch.randn((config.eval_batch_size, 6, 2048))

        pointcloud = pointcloud.to(accelerator.device)

        if classifier_free:
            pointcloud = torch.randn((config.eval_batch_size, 6, 2048))
            scale = 2.0
            pointcloud = pointcloud.to(accelerator.device)

        # set step values
        scheduler.set_timesteps(130)

        model.eval()
        for t in tqdm(scheduler.timesteps, leave=False, disable=not accelerator.is_main_process):
            # 1. predict noise model_output
            if classifier_free:
                texts = [text, None]
                timestep = torch.ones(2 * config.eval_batch_size) * t
                pointcloud = torch.cat([pointcloud, pointcloud], dim=0)
            else:
                texts = [text]
                timestep = torch.ones(config.eval_batch_size) * t

            model_output = model(pointcloud, timestep.to(accelerator.device), input_pointcloud, texts=texts)

            if classifier_free:
                cond_x_0, uncond_x_0 = torch.split(model_output, len(pointcloud) // 2, dim=0)
                mean = uncond_x_0 + scale * (cond_x_0 - uncond_x_0)
                model_output = mean
                pointcloud = pointcloud[:1]

            # 2. compute previous image: x_t -> t_t-1
            pointcloud = scheduler.step(model_output, t, pointcloud[:, :6]).prev_sample

        # ==== Save results ====
        result_path = os.path.join(config.output_dir.replace("runs", "results"), f"ckpt_{max_ckpt}")
        path_to_save = os.path.join(result_path, f"scene_{batch['scene_id'][0]}")
        os.makedirs(path_to_save, exist_ok=True)
        # gpu -> cpu
        pointcloud = pointcloud.cpu().numpy().transpose(0, 2, 1)[0]
        input_pointcloud = input_pointcloud.detach().cpu().numpy().transpose(0, 2, 1)[0]
        gt_pointcloud = gt_pointcloud.detach().cpu().numpy().transpose(0, 2, 1)[0]
        # save images
        plot_point_cloud(pointcloud, os.path.join(path_to_save, "pred_pc.jpg"))
        plot_point_cloud(input_pointcloud, os.path.join(path_to_save, "input_pc.jpg"))
        plot_point_cloud(gt_pointcloud, os.path.join(path_to_save, "target_pc.jpg"))
        np.save(os.path.join(path_to_save, "inference.npy"), pointcloud)
        np.save(os.path.join(path_to_save, "input.npy"), input_pointcloud)
        np.save(os.path.join(path_to_save, "target.npy"), gt_pointcloud)
        # write text
        open(os.path.join(path_to_save, "instruction.txt"), "w").write(text)


if __name__ == "__main__":
    setup_logger()
    device = "cuda"
    config = TrainingConfig()
    torch.set_grad_enabled(False)
    dataset = RLBench(split="test", sample_size=config.model_config["n_ctx"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1024, beta_schedule="squaredcos_cap_v2")

    # ==== Load model ====
    model = model_from_config(config.model_config, device)
    assert os.path.exists(config.output_dir), f"Model not found at {config.output_dir}"
    path_to_load = None
    ckpts = [int(name[5:].split(".")[0]) for name in os.listdir(config.output_dir) if "ckpt" in name]
    assert len(ckpts) > 0, f"No checkpoints found in {config.output_dir}"
    max_ckpt = max(ckpts)
    path_to_load = os.path.join(config.output_dir, f"ckpt-{max_ckpt}")
    # change input and output layer
    with torch.no_grad():
        new_linear_out = nn.Linear(model.output_proj.in_features, 6, bias=True)
        new_linear_out.weight.zero_()
        new_linear_out.weight.copy_(model.output_proj.weight[:6])
        model.output_proj = new_linear_out

        evaluate(config, model, noise_scheduler, dataloader, max_ckpt, path_to_load)
