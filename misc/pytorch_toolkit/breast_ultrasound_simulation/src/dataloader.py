import cv2
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from torch.utils import data
import torch


class IVUS_Dataset(data.Dataset):
    def __init__(
            self,
            file_names,
            Stage0_DIR,
            Images_DIR,
            resize=True,
            test=0):
        # Initiliaztion
        self.file_names = file_names
        self.Stage0_DIR = Stage0_DIR
        self.Images_DIR = Images_DIR
        self.ten_trans = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()])
        self.resize = resize
        self.test = test

    def __len__(self):
        # Return total number of samples in data set
        return len(self.file_names)

    def __getitem__(self, index):

        file_name = self.file_names[index]

        # stage 0 is input,x
        X = Image.open(os.path.join(self.Stage0_DIR, file_name))
        X = self.ten_trans(X)

        # images are the output, y
        y = Image.open(os.path.join(self.Images_DIR, file_name))
        if self.resize:
            y = y.resize((128, 128))

        y = self.ten_trans(y)

        return X, y, file_name


class BUS_dataset(data.Dataset):
    def __init__(
            self,
            file_names,
            bmode_dir,
            label_dir=None,
            resize=True,
            test=0):
        # Initiliaztion
        self.file_names = file_names
        self.bmode_dir = bmode_dir
        self.label_dir = label_dir
        self.ten_trans = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()])
        self.resize = resize
        self.test = test

    def __len__(self):
        # Return total number of samples in data set
        return len(self.file_names)

    def __getitem__(self, index):

        file_name = self.file_names[index]

        # stage 0 is input,x
        X = Image.open(os.path.join(self.bmode_dir, file_name))
        y = 0
        if self.label_dir is not None:
            y = Image.open(os.path.join(
                self.label_dir, file_name[:-10] + ".png"))
            y = y.convert('1')
            y = y.resize((128, 128))
            y = self.ten_trans(y)
        if self.test:
            X = self.ten_trans(X)
        return X, y, file_name


class IVUS3D_Dataset(data.Dataset):
    def __init__(self, folder_names, Stage0_DIR, Images_DIR):
        # Initiliaztion
        self.folder_names = folder_names
        self.Images_DIR = Images_DIR
        self.Stage0_DIR = Stage0_DIR
        self.ten_trans = transforms.ToTensor()

    def __len__(self):
        # Return total number of samples in data set
        return len(self.folder_names)

    def __getitem__(self, index):

        folder_name = self.folder_names[index]
        file_names = os.listdir(os.path.join(self.Images_DIR, folder_name))
        file_names = np.sort(np.array(file_names))
        X_l = []
        y_l = []

        # Concatenating images to form a volume,
        # out of 5 pullbacks given in IVUS, using only middle 3.
        # To create a volume of size 128x128x128
        c1 = 1
        c2 = 0
        for file_name in file_names:
            if c1 == 5:
                c1 = 1
                continue

            if 1 < c1 < 5:
                # stage 0 is input,x
                X = cv2.imread(
                    os.path.join(
                        self.Stage0_DIR,
                        folder_name,
                        file_name),
                    0)
                # X = cv2.resize(X, (192,192))
                X_l.append(X)
                # images are the output, y
                y = cv2.imread(
                    os.path.join(
                        self.Images_DIR,
                        folder_name,
                        file_name),
                    0)
                # y = cv2.resize(y, (192,192))
                y_l.append(y)

                c2 = c2 + 1

            c1 = c1 + 1
            if c2 == 128:
                break

        X = np.stack(X_l, axis=0)
        y = np.stack(y_l, axis=0)

        X = X.transpose(1, 2, 0)
        y = y.transpose(1, 2, 0)
        # print(X.shape, y.shape)

        X = cv2.resize(X, (128, 128))
        y = cv2.resize(y, (128, 128))
        # print(X.shape, y.shape)

        X = X.transpose(2, 0, 1)
        y = y.transpose(2, 0, 1)
        # print(X.shape, y.shape)

        X = torch.from_numpy(X[np.newaxis, :, :, :] / 256).type(torch.float32)
        y = torch.from_numpy(y[np.newaxis, :, :, :] / 256).type(torch.float32)

        return X, y
