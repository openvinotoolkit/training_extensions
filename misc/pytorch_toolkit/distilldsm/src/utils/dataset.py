from torch.utils.data import Dataset
import torch
import numpy as np
from .sequences import WholeVolumeSegmentationSequence

class WholeVolumeSegmentationDataset(WholeVolumeSegmentationSequence, Dataset):
    def __init__(self, *args, batch_size=1, shuffle=False, metric_names=None, **kwargs):
        super().__init__(*args, batch_size=batch_size, shuffle=shuffle, metric_names=metric_names, **kwargs)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        item = self.epoch_filenames[idx]
        x, y = self.resample_input(item)
        return (torch.from_numpy(np.moveaxis(np.copy(x), [-1, -2], [0, 1])).float(),
                torch.from_numpy(np.moveaxis(np.copy(y), [-1, -2], [0, 1])).byte())
