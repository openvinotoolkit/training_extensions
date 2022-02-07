import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import os
from ..train_utils.models import UNet
from ..train_utils.transforms import augment_color
from ..train_utils.dataloader import Stage1Dataset
from ..train_utils.train_function import train_stage1
from ..train_utils.val_function import val_stage1
from ..train_utils.get_config import get_config
from tqdm import tqdm as tq


if __name__ == '__main__':

    config = get_config(action='train')
    epochs = config['epochs']
    batch_sz = config['batch_size']
    num_workers = config['num_workers']
    l_rate = config['lr']
    model_path = config['model_save_path']
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    gpu = config['gpu']
    device = 'cuda' if gpu else 'cpu'
    valid_data_pth = config['val_data_path']
    train_data_pth = config['tr_data_path']

    x_train = np.load(train_data_pth, allow_pickle=True)
    x_train = np.repeat(x_train, 4, axis=0)
    x_val = np.load(valid_data_pth, allow_pickle=True)

    train_data = Stage1Dataset(x_train, transform=augment_color)
    train_loader = DataLoader(train_data, batch_size=batch_sz, shuffle=True, num_workers=num_workers)

    val_data = Stage1Dataset(x_val, transform=None)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)

    model = UNet(num_filters=32)
    model.to(device)

    optimizer = optim.SGD(filter(
                            lambda p: p.requires_grad,
                            model.parameters()),
                            lr=l_rate, momentum=0.9,
                            weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[50, 200, 250], gamma=0.1)

    train_plot = {'bce_loss': [], 'dice_loss': [], 'dice_coeff': []}
    val_plot = {'bce_loss': [], 'dice_loss': [], 'dice_coeff': []}

    best_loss = float('inf')
    for epoch in tq(range(epochs)):
        train_loss_bce, train_loss_dice, train_dice = train_stage1(
                                                                model, train_loader,
                                                                optimizer, epoch,
                                                                epochs, device,
                                                                verbose=True)

        # here if you want to save the qualitative images of the validation set, then make save_qual=True,
        # and provide the path for saving the images in save_pth
        val_loss_bce, val_loss_dice, val_dice = val_stage1(
                                                        model, val_loader,
                                                        epoch, epochs, device,
                                                        verbose=True, save_qual=False,
                                                        save_pth='')
        scheduler.step()

        train_plot['bce_loss'].append(train_loss_bce)
        train_plot['dice_loss'].append(train_loss_dice)
        train_plot['dice_coeff'].append(train_dice)

        val_plot['bce_loss'].append(val_loss_bce)
        val_plot['dice_loss'].append(val_loss_dice)
        val_plot['dice_coeff'].append(val_dice)

        print(f'Epoch [{epoch+1}/ {epochs}]')
        print(f'Trainloss: BCE:{train_loss_bce} logDice:{train_loss_dice}WeightedSum:{train_loss_bce+train_loss_dice}')
        print(f'Train metrics: Dice {train_dice}')
        print(f'Val: BCE {val_loss_bce} log Dice loss, Dice Metric{val_loss_dice}{val_dice}')

        np.save(os.path.join(model_path,'train_plot.npy'), train_plot)
        np.save(os.path.join(model_path,'val_plot.npy'), val_plot)

        torch.save({"epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epochs": epochs,
                        },
                    os.path.join(model_path, 'checkpoint_stage1_sample.pth')
                    )
