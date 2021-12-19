from torch.autograd import Variable
from .loss_functions import bceLoss, diceLoss, diceCoeff
import torch
import torch.nn as nn
import numpy as np



def train(model, train_loader, optimizer, epoch, epochs, device, verbose=True):
    model.train()
    # D.train()

    n = 0
    train_loss_bce, train_loss_dice = 0.0, 0.0
    train_dice = 0


    for i, data in enumerate(train_loader):
        img = Variable(data['image'].float().to(device))
        mask = data['mask'].float().to(device)


        # optimizerD.zero_grad()
        optimizer.zero_grad()

        # Segmentation Network #####################
        mask_pred = model(img)
        mask_pred = torch.sigmoid(mask_pred)



        loss_bce = bceLoss(mask_pred, mask)
        loss_dice = diceLoss(mask_pred, mask)
        dice_coeff = diceCoeff(mask_pred, mask, reduce=True)


        # val = 0

        loss = 0.2*loss_bce + 0.8*loss_dice ##########################################

        loss.backward()


        optimizer.step()

        train_loss_bce += loss_bce.item() * img.size(0)
        train_loss_dice += loss_dice.item() * img.size(0)
        train_dice += dice_coeff.item() * img.size(0)


        n += img.size(0)

        if verbose == True:
            print('Train [%2d/%2d]: [%4d/%4d]\tLoss: BCE [%1.8f] Dice [%1.6f]'
                %(epoch+1, epochs, i+1, len(train_loader), train_loss_bce/n, train_loss_dice/n), end="\r")

    return train_loss_bce/n, train_loss_dice/n, train_dice/n
