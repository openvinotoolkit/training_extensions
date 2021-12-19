from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import cv2


def to_one_hot(num):
    density = np.zeros((4,), dtype=np.float32)
    density[num] = 1
    return density



class CustomDataset(Dataset):
    def __init__(self, d, transform=None):
        self.d = d
        self.transform = transform

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        bag = self.d[idx]['patches'] # [[,,,], [...]]
        cls = self.d[idx]['cls']
        is_random = self.d[idx]['random']

        input = []


        # bag = np.array(bag).astype(np.float32)
        # bag = bag/255.0


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





if __name__ == '__main__':
    from transforms import augment_color
    import torch
    from torch.utils.data import DataLoader

    x_train = np.load('data/train_bags_pred.npy')
    x_test = np.load('data/test_bags_pred.npy')
    x_val = np.load('data/val_bags_pred.npy')


    from transforms import augment_color
    train_data = CustomDataset(x_train, transform=None)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2)

    test_data = CustomDataset(x_test, transform=None)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)


    val_data = CustomDataset(x_val, transform=None)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)

    for i, data in enumerate(test_loader):
        img = data['bag'][0].numpy()
        # import pdb; pdb.set_trace()
        for j, im in enumerate(img):
            im = im*255
            im = im.astype(np.uint8)
            cv2.imwrite('delete/%d_%d.png'%(i,j), im)





        # arr = np.zeros((img.shape[0], img.shape[1]*2), dtype=np.uint8)
        # arr[:, :img.shape[1]] = img[:, :]
        # arr[:, img.shape[1]:] = mask
        #
        #
        # arr[:, img.shape[1]-2:img.shape[1]+2] = 255
        #
        #
        # cv2.imwrite('delete/%d_%d.png'%(density, i), arr)
        print(i)
