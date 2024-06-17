import os
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class GoalDataset(Dataset):
    def __init__(self, ann_path, include_depth=False, enable_sampling=True):
        self.enable_sampling = enable_sampling

        self.ann_path = ann_path
        self.include_depth = include_depth

        if isinstance(self.ann_path, str):
            ann_path_list = [self.ann_path]
        else:
            ann_path_list = self.ann_path

        self.annotations = defaultdict(list)
        for json_path in ann_path_list:
            ann = json.load(open(json_path))
            for sample in ann:
                self.annotations[sample["dataset"]].append(sample)
        self.ann_length = sum(len(samples) for samples in self.annotations.values())

        if self.enable_sampling:
            self.sample_weights = {}
            for dataset, samples in self.annotations.items():
                self.sample_weights[dataset] = np.sqrt(len(samples))
        else:
            self.annotations = [s for samples in self.annotations.values() for s in samples]
            random.shuffle(self.annotations)

    @property
    def dataset_to_robo_arm(self):
        return {
            "bc_z": "Google Robot",
            "berkeley_rpt_converted_externally_to_rlds": "Franka",
            "cmu_play_fusion": "Franka",
            "cmu_playing_with_food": "Franka",
            "rt1": "Google Robot",
            "fractal20220817_data": "Google Robot",  # actually rt1 is fractal20220817_data
            "furniture_bench_dataset_converted_externally_to_rlds": "Franka",
            "jaco_play": "Jaco 2",
            "language_table": "xArm",
            "maniskill_dataset_converted_externally_to_rlds": "Franka",
            "robot_vqa": "Google Robot",
            "roboturk": "Franka",
            "stanford_robocook_converted_externally_to_rlds": "Franka",
            "taco_play": "Franka",
            "ucsd_pick_and_place_dataset_converted_externally_to_rlds": "xArm",
            "utaustin_mutex": "Franka",
            "bridge": "WidowX",
            "epic_kitchen": "Hand",
            "droid": "Franka",
        }

    def with_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.ann_length

    def get_base_image(self, sample):
        return Image.open(sample["base_image_path"])

    def get_base_depth(self, sample):
        path_to_load = sample["base_image_path"].replace(".png", "_depth.png")
        if not os.path.exists(path_to_load):
            return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        return Image.open(path_to_load)

    def get_prompt(self, sample):
        return sample["instruction"]

    def get_goal_image(self, sample):
        return Image.open(sample["final_image_path"])

    def get_goal_depth(self, sample):
        path_to_load = sample["final_image_path"].replace(".png", "_depth.png")
        if not os.path.exists(path_to_load):
            return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        return Image.open(path_to_load)

    def get_sample(self, index):
        if self.enable_sampling:
            dataset = random.choices(list(self.sample_weights.keys()), weights=list(self.sample_weights.values()))[0]
            sample = np.random.choice(self.annotations[dataset])
        else:
            sample = self.annotations[index]
        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)

        sample_dict = {
            "input_image": self.get_base_image(sample),
            "edit_prompt": self.get_prompt(sample),
            "edited_image": self.get_goal_image(sample),
            "robo_arm": self.dataset_to_robo_arm[sample["dataset"]],
        }
        if self.include_depth:
            sample_dict["input_depth"] = self.get_base_depth(sample)
            sample_dict["edited_depth"] = self.get_goal_depth(sample)

        return self.transform(sample_dict)
