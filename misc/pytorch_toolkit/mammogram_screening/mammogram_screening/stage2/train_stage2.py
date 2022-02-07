import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from ..train_utils.dataloader import Stage2bDataset
from ..train_utils.train_function import train_stage2,train_pos_neg_split
from ..train_utils.val_function import val_stage2
from ..train_utils.get_config import get_config 
from ..train_utils.models import Model2 as Model


if __name__ == '__main__':

    config = get_config(action='train', stage='stage2')
    epochs = config['epochs']
    num_workers = config['num_workers']
    lr = config['lr']
    gpu = config['gpu']
    model_save_path = config['model_save_path']
    train_bags_path = config['train_bags_path']
    val_bags_path = config['val_bags_path']

    device = 'cuda' if gpu else 'cpu'
    x_train = np.load(train_bags_path, allow_pickle=True)
    x_val = np.load(val_bags_path, allow_pickle=True)
    x_train_pos, x_train_neg = train_pos_neg_split(x_train)

    ratio = len(x_train_neg) // len(x_train_pos)
    if ratio == 0:
        ratio = 1
    x_train_pos = x_train_pos * ratio
    x_train = x_train_pos + x_train_neg

    train_data = Stage2bDataset(x_train, transform=None)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=num_workers)

    val_data = Stage2bDataset(x_val, transform=None)
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

        train_loss, train_acc, train_auc= train_stage2(model, train_loader, criterion, optimizer, epoch, epochs, device, verbose=True)
        val_loss, val_acc, val_auc= val_stage2(model, val_loader, criterion, epoch, epochs, device, verbose=True)

        print('Epoch [%d/ %d]'%(epoch+1, epochs))
        print('Train Loss: ', train_loss)
        print('Val Loss: ', val_loss)
        print('Train Acc: %.4f %f'%(train_acc, train_auc))
        print('Val Acc: %.4f %f'%(val_acc, val_auc))

        auc_save['train'].append(train_auc)
        auc_save['val'].append(val_auc)

        np.save('auc_plot.npy', auc_save)
        if epoch % 10 == 0:
            torch.save({'state_dict': model.state_dict()}, model_save_path+'checkpoint_stage2.pth')
