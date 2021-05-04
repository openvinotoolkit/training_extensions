import os
import os.path
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path



class RSNADataSet(Dataset):
    def __init__(self, image_list,label_list,image_directory, transform_type):
        image_directory = Path(image_directory)
        image_names = []
        labels = []
        
        image_names = image_list
        labels = label_list
        
        
        image_names = [Path.joinpath(image_directory , x) for x in image_names]
        self.image_names = image_names
        self.labels = labels
        
        if transform_type=='train':
            transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([124.978,124.978,124.978], [10.868,10.868,10.868])
                                        ])
            
        elif transform_type=='test':
            transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([124.925,124.925,124.925], [10.865,10.865,10.865])
                                        ])
        
        
        
        self.transform = transform


    def __getitem__(self, index):
        
     
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor([label])

    def __len__(self):
        return len(self.image_names)
