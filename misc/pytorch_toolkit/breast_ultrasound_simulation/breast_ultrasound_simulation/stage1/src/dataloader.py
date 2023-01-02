from PIL import Image
import os
from torchvision import transforms
from torch.utils import data
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BUS_Dataset(data.Dataset):
    def __init__(
            self,
            file_names,
            Stage0_DIR,
            Images_DIR,
            resize=True,
            test=0):
        "Initiliaztion"
        self.file_names = file_names
        self.Stage0_DIR = Stage0_DIR
        self.Images_DIR = Images_DIR
        self.ten_trans = transforms.Compose(
            [transforms.ToTensor()])
        self.resize = resize
        self.test = test

    def __len__(self):
        "Return total number of samples in data set"
        return len(self.file_names)

    def __getitem__(self, index):

        file_name = self.file_names[index]

        # stage 0 is input,x
        X = Image.open(os.path.join(self.Stage0_DIR, file_name))
        X = self.ten_trans(X)

        # images are the output, y
        y = Image.open(os.path.join(self.Images_DIR, file_name))
        #if self.resize:
        #    y = y.resize((128, 128))

        y = self.ten_trans(y)

        return X, y, file_name
