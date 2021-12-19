from torch.utils.data import Dataset
import numpy as np
import cv2


class CustomDataset(Dataset):
    def __init__(self, data_array, transform=None):
        self.data = data_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.resize(self.data[idx]['img'], (320, 640), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(self.data[idx]['mask'], (320, 640), interpolation=cv2.INTER_CUBIC)
        if self.transform:
            image, mask = self.transform(image, mask)

        image = image[None, :, :].astype(np.float32)
        image = image/255.0

        mask = mask[None, :, :]
        mask = mask.astype(np.float32)
        mask = mask/255.0

        return {'image': image, 'mask': mask}
