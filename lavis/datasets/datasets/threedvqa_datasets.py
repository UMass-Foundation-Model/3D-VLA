"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import random
import logging
import numpy as np
from collections import OrderedDict

from lavis.common.utils import parse_loc_answer
from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answer": "; ".join(ann["answers"]),
                "pc_feat": sample["pc_feat"],
                "pc": sample["pc"],
                "num_features": sample["num_features"],
            }
        )


def feature_path_to_point_path(feature_path):
    point_path = (
        feature_path.replace("voxelized_features_sam_nonzero", "voxelized_voxels_sam_nonzero")
        .replace("nps_blip/features", "nps_blip/points")
        .replace("nps_blip_filtered/features", "nps_blip_filtered/points")
        .replace("front_image_pred/features.npz", "front_image_pred/points.npz")
        .replace(".pt", ".npy")
    )
    return point_path


def load(to_load):
    if to_load.endswith(".pt"):
        return torch.load(to_load, map_location="cpu")
    elif to_load.endswith(".npy"):
        return np.load(to_load)
    elif to_load.endswith(".npz"):
        return np.load(to_load)["arr_0"]
    else:
        print(f"Error file type: {to_load}")
        raise NotImplementedError


def load_feature(feature_path, point_path, sample_num=10000):
    try:
        pc_feat = load(feature_path)
        pc = load(point_path)
    except:
        print(f"Error loading {feature_path}")
        pc_feat = torch.zeros((1, 1408))
        pc = np.zeros((1, 3))

    if isinstance(pc, torch.Tensor):
        pc = pc.clone().detach().cpu()
    elif isinstance(pc, np.ndarray):
        pc = torch.tensor(pc).float().cpu()
    else:
        raise ValueError("pc is not a tensor or numpy array")

    if isinstance(pc_feat, torch.Tensor):
        pc_feat = pc_feat.clone().detach().cpu()
    elif isinstance(pc_feat, np.ndarray):
        pc_feat = torch.tensor(pc_feat).float().cpu()
    else:
        raise ValueError("pc_feat is not a tensor or numpy array")

    pc_feat, pc = sample_pc(pc_feat, pc, sample_num)

    return pc_feat, pc


def sample_pc(pc_feat, pc, sample_num):
    # sample sample_num points: [N, 1408] -> [sample_num, 1408]
    if pc_feat.shape[0] == 0:
        pc_feat = torch.zeros((sample_num, 1408))
        pc = torch.zeros((sample_num, 3))
    elif pc_feat.shape[0] > sample_num:
        idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:sample_num])[1]
        pc_feat = pc_feat[idxes]
        pc = pc[idxes]
    else:
        idxes = torch.sort(torch.randperm(sample_num - pc_feat.shape[0]))[1][: sample_num - pc_feat.shape[0]]
        idxes = idxes % pc_feat.shape[0]
        pc_feat = torch.cat([pc_feat, pc_feat[idxes].clone().detach()], dim=0)
        pc = torch.cat([pc, pc[idxes].clone().detach()], dim=0)
    return pc_feat, pc


class ThreeDVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, **kwargs)
        random.shuffle(self.annotation)
        self.annotation = [ann for ann in self.annotation if "answers" in ann and "path" in ann]
        logging.info(f"Total number of samples: {len(self.annotation)}")
        self.sample_num = getattr(kwargs, "sample_num", 8000)
        self.max_T = 2

    def __getitem__(self, index):
        ann = self.annotation[index]

        caption = self.text_processor(ann["question"]).replace("3d-feat>", "scene>")

        # ==== Set paths ====
        pc_feat_root = ann["path"]
        if "pos_path" in ann:
            voxel_root = ann["pos_path"]
        else:
            voxel_root = [feature_path_to_point_path(path) for path in ann["path"]]

        # ==== Load voxel and point cloud features ====
        pc_feat = torch.zeros((self.max_T, self.sample_num, 1408))
        pc = torch.zeros((self.max_T, self.sample_num, 3))
        for feat_path, point_path, t in zip(pc_feat_root, voxel_root, range(len(pc_feat_root))):
            pc_feat[t], pc[t] = load_feature(feat_path, point_path, self.sample_num)

        # ==== Compute answer weights ====
        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        # find nan
        assert not torch.isnan(pc_feat).any()
        assert not torch.isnan(pc).any()
        return {
            "pc_feat": pc_feat,
            "pc": pc,
            "text_input": caption,
            "answer": answers,
            "weight": weights,
            "image_id": 0,
            "question_id": index,
            "num_features": torch.tensor(len(ann["path"])),
        }

    def __len__(self):
        return len(self.annotation)


class ThreeDVQAEvalDataset(VQAEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, **kwargs)
        self.annotation = [ann for ann in self.annotation if "answers" in ann and "path" in ann]
        logging.info(f"Total number of EVAL samples: {len(self.annotation)}")
        self.sample_num = getattr(kwargs, "sample_num", 8000)
        self.max_T = 2

    def __getitem__(self, index):
        ann = self.annotation[index]
        caption = self.text_processor(ann["question"]).replace("3d-feat>", "scene>")
        dataname = ann["dataset"]
        scene_id = ann["scene_id"]

        # ==== Set paths ====
        pc_feat_root = ann["path"]
        if "pos_path" in ann:
            voxel_root = ann["pos_path"]
        else:
            voxel_root = [feature_path_to_point_path(path) for path in ann["path"]]

        # ==== Load voxel and point cloud features ====
        pc_feat = torch.zeros((self.max_T, self.sample_num, 1408))
        pc = torch.zeros((self.max_T, self.sample_num, 3))
        for feat_path, point_path, t in zip(pc_feat_root, voxel_root, range(len(pc_feat_root))):
            pc_feat[t], pc[t] = load_feature(feat_path, point_path, self.sample_num)

        # ==== Compute answer weights ====
        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        return {
            "pc_feat": pc_feat,
            "pc": pc,
            "text_input": caption,
            "answer": answers,
            "num_features": torch.tensor(len(ann["path"])),
            # below are info for evaluation
            "dataname": dataname,
            "task": ann["task"],
            "scene_id": ann["scene_id"],
            "question_id": ann["question_id"],
            "index": index,
        }

    def __len__(self):
        return len(self.annotation)
