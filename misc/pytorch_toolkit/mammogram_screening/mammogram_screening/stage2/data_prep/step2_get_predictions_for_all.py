import numpy as np
import os
import torch
import sys
from dataloader import CustomDataset
pth=os.getcwd()
sys.path.append('../../stage1_mass_segmentation/')
from network.models import UNet
os.chdir(pth)
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm as tq

def predict_mass_seg(val_loader, model, nm):
    save_list = []
    model.eval()
    with torch.no_grad():
        for data in tq(val_loader):
            img = data['image'].float()
            mask = data['mask'].float()
            cls = data['cls'].item()
            file_name = data['file_name'][0]
            mask_pred = model(img)
            mask_pred = torch.sigmoid(mask_pred)
            img = img.data.cpu().numpy()[0][0]*255
            img = img.astype(np.uint8)
            mask_pred_norm = mask_pred.data.cpu().numpy()[0][0]
            mask_pred = mask_pred.data.cpu().numpy()[0][0]*255
            mask_pred = mask_pred.astype(np.uint8)
            mask = mask.data.cpu().numpy()[0][0]*255
            mask = mask.astype(np.uint8)

            data_dict = {'img': img, 'mask_pred': mask_pred, 'mask_pred_norm': mask_pred_norm, 'mask': mask, 'cls': cls, 'file_name': file_name}
            save_list.append(data_dict)

    np.save(nm, save_list)
    model.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_model', type=str, default='', required=True, help='path to seg model')
    parser.add_argument('--data_path', type=str, default='', required=True, help='directory at which data is stored')
    parser.add_argument('--out_path', type=str, default='', required=True, help='directory at which output data is to be stored')
    args = parser.parse_args()
    
    data_path = args.data_path
    out_path = args.out_path
    mass_seg_wt_path = args.seg_model
    model = UNet(num_filters=32)

    saved_model = torch.load(mass_seg_wt_path, map_location='cuda')
    model.load_state_dict(saved_model['state_dict'])

    # Prepare val dataset
    x_train = np.load(os.path.join(data_path,'segmenter_test.npy'), allow_pickle=True)
    val_data = CustomDataset(x_train, transform=None)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

    # Inference of the mass segmentation network on the validation set, save in a npy file
    print('Starting Validation Set Inference .....')
    predict_mass_seg(val_loader, model, out_path+'val_all_pred.npy')
    del x_train, val_data, val_loader

    # Prepare Train set
    x_train = np.load(os.path.join(data_path,'segmenter_train.npy'), allow_pickle=True)
    train_data = CustomDataset(x_train, transform=None)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)

    # Inference on the Train set
    print('Starting Training Set Inference .....')
    predict_mass_seg(train_loader, model, out_path+'train_all_pred.npy')
    del x_train, train_data, train_loader
