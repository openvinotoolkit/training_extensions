import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import time
import argparse
import os
from os.path import dirname, abspath
from network.models import UNet
from train_utils.transforms import augment_color
from train_utils.dataloader import CustomDataset
from train_utils.train_function import train
from train_utils.val_function import val

if __name__ == '__main__':
    par_dir=dirname(dirname(abspath(__file__)))  # parent of current directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, required=False, help='No. of epochs..')
    parser.add_argument('--batch_size', type=int, default=8, required=False, help='Batch size..')
    parser.add_argument('--num_workers', type=int, default=4, required=False, help='Number of workers..')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='learning rate..')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--chkpt', type=str, default='', required=False, help='path to save checkpoint model weights')
    parser.add_argument('--trn_data', type=str, default=os.path.join(par_dir, 'Data_preparation', 'segmenter_train_0.npy'), required=False, help='location of npy file of training dataset')
    parser.add_argument('--val_data', type=str, default=os.path.join(par_dir, 'Data_preparation', 'segmenter_val_0.npy'), required=False, help='location of npy file of validation dataset')
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    gpu = args.gpu
    device = 'cuda' if gpu else 'cpu'
    
    checkpoint_pth=args.chkpt
    train_data_pth=args.trn_data
    val_data_pth=args.val_data

    x_train = np.load(train_data_pth, allow_pickle=True)
    x_train = np.repeat(x_train, 4, axis=0)
    x_val = np.load(val_data_pth, allow_pickle=True)

    
    train_data = CustomDataset(x_train, transform=augment_color)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    val_data = CustomDataset(x_val, transform=None)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)

    model = UNet(num_filters=32)
    model.to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[50, 200, 250], gamma=0.1)

    train_plot = {'bce_loss': [], 'dice_loss': [], 'dice_coeff': []}
    val_plot = {'bce_loss': [], 'dice_loss': [], 'dice_coeff': []}


    best_loss = float('inf')
    for epoch in range(epochs):
        tic = time.time()
        train_loss_bce, train_loss_dice, train_dice = train(model, train_loader, optimizer, epoch, epochs, device, verbose=True)
        
        # here if you want to save the qualitative images of the validation set, then make save_qual=True, 
        # and provide the path for saving the images in save_pth
        val_loss_bce, val_loss_dice, val_dice = val(model, val_loader, epoch, epochs, device, verbose=True, save_qual=False, save_pth='')
        scheduler.step()

        train_plot['bce_loss'].append(train_loss_bce)
        train_plot['dice_loss'].append(train_loss_dice)
        train_plot['dice_coeff'].append(train_dice)

        val_plot['bce_loss'].append(val_loss_bce)
        val_plot['dice_loss'].append(val_loss_dice)
        val_plot['dice_coeff'].append(val_dice)


        print('######'*40)
        print('Epoch [%d/ %d]'%(epoch+1, epochs))
        print('Train loss: BCE [%f] log Dice [%f] Weighted Sum [%f]'%(train_loss_bce, train_loss_dice, train_loss_bce+train_loss_dice))
        print('Train metrics: Dice [%f]'%(train_dice))
        print('Val: BCE [%f] log Dice loss, Dice Metric[%f %f]'%(val_loss_bce, val_loss_dice, val_dice))
        print('Time to train the last epoch: '+str(time.time() - tic)+' seconds')
        print()


        np.save('train_plot.npy', train_plot)
        np.save('val_plot.npy', val_plot)

        # if epoch > 200:
    torch.save({"epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epochs": epochs,
                    }, 
                os.path.join(checkpoint_pth, 'checkpoint_'+str(epoch)+'.pth.tar')
                )
