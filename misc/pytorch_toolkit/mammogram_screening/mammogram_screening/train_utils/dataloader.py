from torch.utils.data import Dataset
import numpy as np
import cv2


class Stage1Dataset(Dataset):
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


class Stage2aDataset(Dataset):
    def __init__(self, data_array, transform=None):
        self.data = data_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.resize(self.data[idx]['img'], (320, 640), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(self.data[idx]['mask'], (320, 640), interpolation=cv2.INTER_CUBIC)
        cls=self.data[idx]['cls']
        file_name=self.data[idx]['file_name']

        if self.transform:
            image, mask = self.transform(image, mask)

        image = image[None, :, :].astype(np.float32)
        image = image/255.0
        mask = mask[None, :, :]
        mask = mask.astype(np.float32)
        mask = mask/255.0

        return {'image': image, 'mask': mask, 'cls': cls, 'file_name': file_name}

def to_one_hot(num):
    density = np.zeros((4,), dtype=np.float32)
    density[num] = 1
    return density

class Stage2bDataset(Dataset):
    def __init__(self, data_array, transform=None):
        self.data = data_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag = self.data[idx]['patches']
        cls = self.data[idx]['cls']
        is_random = self.data[idx]['random']
        input = []
        for roi in bag:
            for img in roi:
                if self.transform:
                    img, _ = self.transform(img)
                else:
                    pass
                input.append(img)

        input = np.array(input).astype(np.float32)
        input = input/255.0

        if is_random:
                indices = np.arange(input.shape[0])
                np.random.shuffle(indices)
                input = input[indices]
                input = input[:5]

        return {'bag': input, 'cls': cls, 'len': len(bag)}
