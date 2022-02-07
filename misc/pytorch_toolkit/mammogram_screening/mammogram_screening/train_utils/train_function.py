from torch.autograd import Variable
from .loss_functions import bceLoss, diceLoss, diceCoeff
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def train_pos_neg_split(x_train):
    x_train_pos = []
    x_train_neg = []
    for d_elem in x_train:
        if d_elem['cls'] == 1:
            x_train_pos.append(d_elem)
        else:
            x_train_neg.append(d_elem)
    return x_train_pos, x_train_neg


def train_stage1(model, train_loader, optimizer, epoch, epochs, device, verbose=True):
    model.train()
    n = 0
    train_loss_bce, train_loss_dice = 0.0, 0.0
    train_dice = 0

    for i, data in enumerate(train_loader):
        img = Variable(data['image'].float().to(device))
        mask = data['mask'].float().to(device)
        optimizer.zero_grad()
        mask_pred = model(img)
        mask_pred = torch.sigmoid(mask_pred)
        loss_bce = bceLoss(mask_pred, mask)
        loss_dice = diceLoss(mask_pred, mask)
        dice_coeff = diceCoeff(mask_pred, mask, reduce=True)
        loss = 0.2*loss_bce + 0.8*loss_dice
        loss.backward()

        optimizer.step()

        train_loss_bce += loss_bce.item() * img.size(0)
        train_loss_dice += loss_dice.item() * img.size(0)
        train_dice += dice_coeff.item() * img.size(0)

        n += img.size(0)

        if verbose:
            print(f'''Train
                    [{(epoch+1)/epochs}]:[{(i+1)/len(train_loader)}]
                    |Loss: {train_loss_bce/n}
                    |Acc: {train_loss_dice/n}''')

    return train_loss_bce/n, train_loss_dice/n, train_dice/n

def train_stage2(model, train_loader, criterion, optimizer, epoch, epochs, device, verbose=True):
    model.train()
    train_loss, n = 0.0, 0
    train_acc = 0
    arr_true = []
    arr_pred = []

    for i, data in enumerate(train_loader):
        X = data['bag'].float().to(device)[0].unsqueeze(1)
        y = data['cls'].float().to(device).unsqueeze(0)
        optimizer.zero_grad()
        print(X.shape)

        outputs = model(X)
        outputs = torch.sigmoid(outputs)
        y_pred = outputs.data > 0.5
        y_pred = y_pred.float()
        arr_true.append(y.item())
        arr_pred.append(outputs.item())

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * y.size(0)
        train_acc += torch.sum(y_pred == y.data).item()
        n += 1

        if verbose:
            print(f'Train [{(epoch+1)/epochs}]:[{(i+1)/len(train_loader)}]|Loss: {train_loss/n}|Acc: {train_acc/n}')

    arr_true = np.array(arr_true).flatten()
    arr_pred = np.array(arr_pred).flatten()
    auc = roc_auc_score(arr_true, arr_pred)

    return train_loss/n, train_acc/n, auc
