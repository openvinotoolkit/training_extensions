from torch.autograd import Variable
from .loss_functions import bceLoss, diceLoss, diceCoeff, ceLoss
import torch
import os
import numpy as np
import cv2



######################### The validation function #####
# save_qual: whether to save the qualitative results
# save_pth: path where the qualitative results are saved if save_qual==True
# The images (image, GT_mask, pred_mask) are concatenated and saved inside save_pth/<epoch>/<image_index>

def val(model, test_loader, epoch, epochs, device, verbose=True, save_qual=False, save_pth=''):
    model.eval()

    n = 0
    val_loss_bce, val_loss_dice = 0.0, 0.0
    val_dice = 0.0

    if save_qual==True:
        pth=os.path.join(save_pth, str(epoch))
        os.makedirs(pth, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img = Variable(data['image'].float().to(device))
            mask = Variable(data['mask'].float().to(device))


            mask_pred = model(img)
            mask_pred = torch.sigmoid(mask_pred)


            loss_bce = bceLoss(mask_pred, mask)
            loss_dice = diceLoss(mask_pred, mask)
            dice_coeff = diceCoeff(mask_pred, mask, reduce=True)



            val_loss_bce += loss_bce.item() * img.size(0)
            val_loss_dice += loss_dice.item() * img.size(0)
            val_dice += dice_coeff.item() * img.size(0)


            n += img.size(0)
            
            if save_qual==True:
                
                img = img.data.cpu().numpy()[0][0]*255
                img = img.astype(np.uint8)
                mask_pred = mask_pred.data.cpu().numpy()[0][0]*255
                mask_pred = mask_pred.astype(np.uint8)
                mask = mask.data.cpu().numpy()[0][0]*255
                mask = mask.astype(np.uint8)


                SHAPE = img.shape
                arr = np.zeros((SHAPE[0], SHAPE[1]*3), dtype=np.uint8)
                arr[:, :SHAPE[1]] = img
                arr[:, SHAPE[1]:2*SHAPE[1]] = mask_pred
                arr[:, 2*SHAPE[1]:] = mask
                arr[:, SHAPE[1]-2:SHAPE[1]] = 250
                arr[:, SHAPE[1]:SHAPE[1]+2] = 250
                arr[:, 2*SHAPE[1]-2:2*SHAPE[1]] = 250
                arr[:, 2*SHAPE[1]:2*SHAPE[1]+2] = 250
                
                cv2.imwrite(os.path.join(pth,str(i)+'.png'), arr)

            if verbose == True:
                print('Val [%2d/%2d]: [%4d/%4d]\tLoss: [%1.8f] Dice [%1.6f]'
                    %(epoch+1, epochs, i+1, len(test_loader), val_loss_bce/n, val_loss_dice/n), end="\r")

    return val_loss_bce/n, val_loss_dice/n, val_dice/n
