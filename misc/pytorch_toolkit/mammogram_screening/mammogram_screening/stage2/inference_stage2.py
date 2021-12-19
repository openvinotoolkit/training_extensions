import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from train.dataloader import CustomDataset
import numpy as np
import os
import time
import argparse
from network.models import Model2 as Model

def inference(model, test_loader, out_nm):
    model.eval()
    n = 0
    test_acc = 0.0
    tic = time.time()

    arr_true = []
    arr_pred = []

    for i, data in enumerate(test_loader):
        X = data['bag'].float().to(device)[0].unsqueeze(1)
        y = data['cls'].float().to(device).unsqueeze(0)

        out = model(X)
        out = torch.sigmoid(out)

        y_pred = out.data > 0.5
        y_pred = y_pred.float()


        arr_true.append(y.item())
        arr_pred.append(out.item())
        
        test_acc += torch.sum(y_pred == y.data).item()

        n += 1 # y.size(0)



    arr_true = np.array(arr_true).flatten()
    arr_pred = np.array(arr_pred).flatten()
    auc = roc_auc_score(arr_true, arr_pred)
    
    print('Test accuracy: '+str(test_acc/n)+'  test_auc: '+str(auc))
    np.savez(out_nm, label=arr_true, pred=arr_pred, test_acc=(test_acc/n), test_auc=auc)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2, required=False, help='Number of workers..')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--model_wt', type=str, default='', required=False, help='file path to save trained model weights')
    parser.add_argument('--test_bags_pth', type=str, default='', required=False, help='.npy file containing test bags')
    parser.add_argument('--out_nm', type=str, default='result.npz', required=False, help='.npy output file path to save predictions')


    args = parser.parse_args()
    num_workers = args.num_workers
    gpu = args.gpu
    saved_model_wt=args.model_wt
    tst_pth=args.test_bags_pth
    out_nm=args.out_nm

    device = 'cuda' if gpu else 'cpu'
    x_tst = np.load(tst_pth, allow_pickle=True)
    

    tst_data = CustomDataset(x_tst, transform=None)
    tst_loader = DataLoader(tst_data, batch_size=1, shuffle=False, num_workers=num_workers)

    
    model = Model()
    # load model weights
    checkpoint = torch.load(saved_model_wt, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint 
    
    model.to(device)
    model.eval()
    
    inference(model, tst_loader, out_nm)
