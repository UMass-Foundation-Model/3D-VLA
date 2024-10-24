import os
import json
import numpy as np
from torch.utils.data import Dataset
import random


class RLBench(Dataset):
    def __init__(self, split=None, sample_size=2048):
        self.rlbench_dir = "data/rlbench"
        task_definition = os.path.join(self.rlbench_dir, "taskvar_instructions.jsonl")
        self.sample_size = sample_size

        data = {}
        for line in open(task_definition, "r", encoding="utf-8"):
            task_js = json.loads(line)
            data[task_js["task"]] = task_js["variations"]
        self.task_definition = data

        episodes = json.load(open("dataset/rlbench_data.json", "r"))
        len_episodes = len(episodes)
        random.Random(42).shuffle(episodes)

        if isinstance(split, tuple):
            job, all_jobs = split
            ids = len_episodes // all_jobs
            self.episodes = episodes[job * ids : min(len_episodes, (job + 1) * ids)]
        elif split == "all":
            self.episodes = episodes
        elif split == "train":
            self.episodes = episodes[1000:]
        elif split == "test":
            self.episodes = episodes[:1000]
        else:
            raise ValueError("Invalid split")

    def __len__(self):
        return len(self.episodes)

    def getitem(self, index):
        ep = self.episodes[index]
        task, var, eps = ep.split("#")

        # ==== Load the point clouds ====
        episode_path = os.path.join(self.rlbench_dir, task, var, "episodes", eps)
        pcd_path = os.path.join(episode_path, "pcd")
        pcds = sorted(os.listdir(pcd_path), key=lambda x: int(x.split(".")[0]))
        input_pc_path = os.path.join(pcd_path, pcds[0])
        output_pc_path = os.path.join(pcd_path, pcds[-1])
        start_pc = self.laod_pcd(input_pc_path)
        end_pc = self.laod_pcd(output_pc_path)
        # sample the point clouds
        if start_pc.shape[0] > self.sample_size:
            selected_idx = np.random.choice(start_pc.shape[0], self.sample_size)
            start_pc = start_pc[selected_idx]
            end_pc = end_pc[selected_idx]
        else:  # duplicate the point clouds
            selected_idx = np.random.choice(self.sample_size - pcds.shape[0], pcds.shape[0])
            start_pc = np.concatenate([start_pc, start_pc[selected_idx]])
            end_pc = np.concatenate([end_pc, end_pc[selected_idx]])

        all_pc = np.concatenate([start_pc, end_pc], axis=0)[..., :3]
        mean = np.mean(all_pc, axis=0)
        radius = np.max(np.linalg.norm(all_pc - mean, axis=1))
        start_pc[..., :3] = (start_pc[..., :3] - mean) / radius
        end_pc[..., :3] = (end_pc[..., :3] - mean) / radius

        # ==== Load the instruction ====
        instruction = self.task_definition[task][var.replace("variation", "")][0]

        sample_dict = {
            "start_pc": start_pc,
            "instruction": instruction,
            "end_pc": end_pc,
            "scene_id": index,
        }

        return sample_dict

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self.getitem(index)
            except:
                index = np.random.randint(0, len(self.episodes))
        assert False, "Too many failures"

    def laod_pcd(self, path):
        if path.endswith(".npy") and os.path.exists(path):
            return np.load(path)
        else:
            return np.load(path.replace(".npy", ".npz"))["arr_0"]


if __name__ == "__main__":
    from tqdm import tqdm
    from multiprocessing import Pool

    def load_data(job):
        dataset = RLBench(split=(job, 32))
        for i in tqdm(range(len(dataset))):
            dataset[i]

    with Pool(32) as p:
        p.map(load_data, range(32))
