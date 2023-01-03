import torch
from torch.utils import data
import os
from torchvision import transforms
from PIL import Image
import numpy as np


class LungDataLoader(data.Dataset):
    """Class represents the dataloader for Lung segmentation task

    Atributes
    ---------
    datapath: str
        Folder location where img is stored
    lung_path: str
        Folder location where lung seg mask is stored
    json_file: str
        Folder location where json files are stored
    split: str
        String to determine train/val and test set
    is_transform: Boolean
        True if transformation is to be applied
    img_size: int
        Size of input image

    """
    def __init__(self,datapath,lung_path,json_file,split="train_set",is_transform= True,img_size= 512):

        self.split=split
        self.path= datapath
        self.lung_path=lung_path
        self.json = json_file
        self.files = self.json[self.split]
        self.img_size= img_size
        self.is_transform= is_transform
        self.image_tf= transforms.Compose(
            [transforms.Resize(self.img_size),
             transforms.ToTensor()
              ])

        self.lung_tf = transforms.Compose(
            [transforms.Resize(self.img_size),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):

        filename = self.files[index]
        img = Image.fromarray(np.load(self.path+'image/'+filename).astype(float))
        lung_mask = Image.fromarray(np.load(self.lung_path+filename).astype(float))

        if self.is_transform:
            img, lung_mask = self.transform(img,lung_mask)
            labels = torch.cat((1.-lung_mask,lung_mask)) #

        return img, labels

    def transform(self,img,lung_mask):
        img = self.image_tf(img)
        img = img.type(torch.FloatTensor)
        lung_mask = self.lung_tf(lung_mask)
        lung_mask = lung_mask.type(torch.FloatTensor)

        return img,lung_mask


class LungPatchDataLoader(data.Dataset):

    def __init__(self,imgpath,split="train_set",is_transform= True):

        self.split = split
        self.imgpath = imgpath+self.split+'/img/'
        self.is_transform = is_transform
        self.files = os.listdir(self.imgpath)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):

        filename = self.files[index]
        l1 = int(filename.split('_')[1])
        if l1 == 1: # Complement  operator ~ gave negative labels eg: for label 0 o/p was 1
            l2 = 0
        else:
            l2 = 1
        label = torch.tensor([l1,l2])
        img = np.load(self.imgpath+filename)

        if self.is_transform:
            img= self.transform(img)

        return img,label

    def transform(self,img):
        img = torch.Tensor(img).unsqueeze(0)
        img = img.type(torch.FloatTensor)


        return img
