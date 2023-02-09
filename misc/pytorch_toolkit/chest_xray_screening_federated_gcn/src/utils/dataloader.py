from torch.utils import data
from PIL import Image
import torch
import numpy as np

################ Dataloader #########################

class construct_dataset(data.Dataset):
    def __init__(self, data_pth, split_npz, site, transforms, tn_vl_idx):
        # site [0,4] or -999 which means global
        # tn_vl_idx 0=> Train  1=> Val  2=> Test
        # load the npz file
        a=np.load(split_npz, allow_pickle=True)
        img_names=a['img_names']
        gt=a['gt']
        clstr_assgn=a['clstr_assgn']
        trn_val_tst=a['trn_val_tst']
        del a
        if site==-999:
            # This means that we are performing the global baseline experiments where data of all sites is combined.
            idx=np.where(trn_val_tst==tn_vl_idx)[0]
        else:
            # Select only those images which belong to the site and train/val/test
            idx=np.where((clstr_assgn==site) & (trn_val_tst==tn_vl_idx))[0]
        img_names=img_names[idx]
        gt=gt[idx]
        del idx
        self.img_names=img_names
        self.gt=gt
        self.transforms=transforms
        self.data_pth=data_pth
        
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        img_nm=self.img_names[index]
        # Read the image
        image = Image.open(self.data_pth+img_nm)
        # Apply the transforms
        image=self.transforms(image)
        # Convert datatype/Tensor
        gt=self.gt[index]
        gt=torch.FloatTensor(gt)
        if image.shape[0]==4:
            image = image[:1,:,:]
        else:
            pass
        sample={'img': image, 'gt': gt, 'img_nm': img_nm}
        return sample 

    def __len__(self):
        #print(self.img_pths.shape[0])
        return self.img_names.shape[0]

