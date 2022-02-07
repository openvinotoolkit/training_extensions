import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from ..train_utils.dataloader import Stage2aDataset
from ..train_utils.models import UNet
from ..train_utils.get_config import get_config
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

    configs = get_config(action='pred_all', stage='stage2')
    data_path = configs['data_path']
    out_path = configs['out_path']
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    mass_seg_wt_path = configs['seg_model']
    model = UNet(num_filters=32)

    saved_model = torch.load(mass_seg_wt_path, map_location='cuda')
    model.load_state_dict(saved_model['state_dict'])

    # Prepare val dataset
    x_train = np.load(os.path.join(data_path,'segmenter_test.npy'), allow_pickle=True)
    val_data = Stage2aDataset(x_train, transform=None)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

    # Inference of the mass segmentation network on the validation set, save in a npy file
    print('Starting Validation Set Inference .....')
    predict_mass_seg(val_loader, model, out_path+'val_all_pred.npy')
    del x_train, val_data, val_loader

    # Prepare Train set
    x_train = np.load(os.path.join(data_path,'segmenter_train.npy'), allow_pickle=True)
    train_data = Stage2aDataset(x_train, transform=None)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)

    # Inference on the Train set
    print('Starting Training Set Inference .....')
    predict_mass_seg(train_loader, model, out_path+'train_all_pred.npy')
    del x_train, train_data, train_loader
