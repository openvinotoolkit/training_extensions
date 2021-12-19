import torch
import numpy as np
import cv2
import torch.nn as nn


def bceLoss(mask_pred, mask, pos_weight=28):
    eps = 1e-15
    bce = -1.0*pos_weight*mask*torch.log(mask_pred+eps) - (1.0-mask)*torch.log(1.0-mask_pred+eps)

    return torch.mean(bce)

criterion = nn.CrossEntropyLoss()

def ceLoss(pred, target):
    return criterion(pred, target)



def diceCoeff(mask_pred, mask, reduce=False):
    mask_pred = mask_pred.view(mask.size(0), -1) # (N, C*H*W)
    mask = mask.view(mask.size(0), -1)

    smooth = 1.0
    intersection = torch.sum(mask_pred*mask, dim=1) # (N, 1)


    denominator = torch.sum(mask_pred, dim=1) + torch.sum(mask, dim=1) + smooth
    numerator = 2*intersection + smooth

    if reduce:
        dice = numerator/denominator
        return dice.mean()
    return numerator/denominator





def diceLoss(mask_pred, mask, gamma=0.5):
    dice_coeff = diceCoeff(mask_pred, mask, reduce=False)

    loss = -torch.log(dice_coeff)
    return loss.mean()




