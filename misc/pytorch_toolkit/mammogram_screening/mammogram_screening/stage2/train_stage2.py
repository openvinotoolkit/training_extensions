import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from train.dataloader import CustomDataset
from train.train_functions import train
from train.val_functions import val
import numpy as np
import time
import argparse
from network.models import Model2 as Model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, required=False, help='No. of epochs..')
    parser.add_argument('--num_workers', type=int, default=2, required=False, help='Number of workers..')
    parser.add_argument('--lr', type=float, default=0.0001, required=False, help='learning rate..')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--save_model', type=str, default='', required=False, help='file path to save trained model weights')
    parser.add_argument('--train_bags_path', type=str, default='', required=False, help='.npy file containing training bags')
    parser.add_argument('--val_bags_path', type=str, default='', required=False, help='.npy file containing validation bags')


    args = parser.parse_args()
    epochs = args.epochs
    num_workers = args.num_workers
    lr = args.lr
    gpu = args.gpu
    chk_pt_pth=args.save_model
    trn_pth=args.train_bags_path
    val_pth=args.val_bags_path

    device = 'cuda' if gpu else 'cpu'
    x_train = np.load(trn_pth, allow_pickle=True)
    x_val = np.load(val_pth, allow_pickle=True)

    x_train_pos = []
    x_train_neg = []
    count = 0
    for i, d_elem in enumerate(x_train):
        # print(d_elem['cls'])
        if d_elem['cls'] == 1:
            x_train_pos.append(d_elem)
            count += 1
        else:
            x_train_neg.append(d_elem)

    print(count)

    ratio = len(x_train_neg) // len(x_train_pos)
    x_train_pos = x_train_pos * ratio
    x_train = x_train_pos + x_train_neg

    train_data = CustomDataset(x_train, transform=None)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=num_workers)

    val_data = CustomDataset(x_val, transform=None)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)

    model = Model()
    if gpu:
        model.float().cuda()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    auc_save = {'train': [], 'val': []}
    # train and test
    best_acc = 0
    train_acc_save = []
    val_acc_save = []
    for epoch in range(epochs):
        tic = time.time()
        train_loss, train_acc, train_auc= train(model, train_loader, criterion, optimizer, epoch, epochs, device, verbose=True)
        val_loss, val_acc, val_auc= val(model, val_loader, criterion, epoch, epochs, device, verbose=True)

        print('  '*40)
        print('Epoch [%d/ %d]'%(epoch+1, epochs))
        print('Train Loss: ', train_loss)
        print('Val Loss: ', val_loss)
        print('Train Acc: %.4f %f'%(train_acc, train_auc))
        print('Val Acc: %.4f %f'%(val_acc, val_auc))
        print('Time: %d'%(time.time() - tic))

        auc_save['train'].append(train_auc)
        auc_save['val'].append(val_auc)

        np.save('auc_plot.npy', auc_save)
        if epoch % 10 == 0:
            torch.save({'state_dict': model.state_dict()}, chk_pt_pth+'epoch_'+str(epoch)+'.pth.tar')
