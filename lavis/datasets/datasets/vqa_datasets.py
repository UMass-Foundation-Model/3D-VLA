import torch
from collections import defaultdict

from lavis.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, **kwargs):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, **kwargs)

    def collater(self, samples):
        """
        Collate the samples into a batch
        """
        # create a dictionary to hold the batch
        batch_dict = defaultdict(list)
        for sample in samples:
            for key, value in sample.items():
                if key == "answer":
                    batch_dict[key].extend(value)
                    batch_dict["n_answers"].append(len(value))
                else:
                    batch_dict[key].append(value)

        # convert the list to tensor
        for key, value in batch_dict.items():
            if isinstance(value[0], torch.Tensor):
                batch_dict[key] = torch.stack(value, dim=0)
            elif isinstance(value[0], int):
                batch_dict[key] = torch.LongTensor(value)
            else:
                batch_dict[key] = value

        return batch_dict


class VQAEvalDataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, **kwargs):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, **kwargs)
