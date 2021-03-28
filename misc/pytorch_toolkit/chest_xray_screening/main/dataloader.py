import os
import os.path
import numpy as np
import time
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from PIL import Image
import torch.nn.functional as func
from generate import *
from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random
from pathlib import Path
from math import sqrt
import torchvision.models.densenet as modelzoo

class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file,image_directory, transform_type):
        image_directory = Path(image_directory)
        image_names = []
        labels = []
        f = open('rsna_annotation.json') 
        data = json.load(f)
        image_names = data['names']
        labels = data['labels']
        
        image_names = [Path.joinpath(image_directory , x) for x in image_names]
        self.image_names = image_names
        self.labels = labels
        
        if transform_type=='train':
            transform=transforms.Compose([
                                            #transforms.Resize(350),
                                            #transforms.RandomResizedCrop((320,320), scale=(0.8, 1.0)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([124.978,124.978,124.978], [10.868,10.868,10.868])
                                        ])
            
        elif transform_type=='test':
            transform=transforms.Compose([
                
                                            #transforms.Resize((320,320)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([124.925,124.925,124.925], [10.865,10.865,10.865])
                                        ])
        
        
        
        self.transform = transform


    def __getitem__(self, index):
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        label = self.labels[index]
        #label = transforms.ToTensor(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor([label])

    def __len__(self):
        return len(self.image_names)