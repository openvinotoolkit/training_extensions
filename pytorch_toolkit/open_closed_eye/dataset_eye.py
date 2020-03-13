from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

class EyeDataset(Dataset):
    def __init__(self, root_folder, mode = 'train', transformations = None):
        self.transformations = transformations
        self.data = []
        for subdir, dirs, files in os.walk(osp.join(root_folder, mode)):
            for i, file in enumerate(files):                
                full_path = os.path.join(subdir, file)
                state = 1 if file[0] == 'o' else 0
                if mode == 'train':
                    self.data.append({'filename' : full_path, 'label' : state})
                else:
                    self.data.append({'filename' : full_path, 'label' : state})    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['filename']).convert('RGB')
        if self.transformations is not None:
            img = self.transformations(img)
        return (img, item['label'], item['filename'])



if __name__ == '__main__':

    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    dataset = EyeDataset('../../data/open_closed_eye/train', transformations)
    print (dataset[10])