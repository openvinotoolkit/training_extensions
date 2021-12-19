from torch.autograd import Variable
import torch
import time
from sklearn.metrics import roc_auc_score
import numpy as np



def train(model, train_loader, criterion, optimizer, epoch, epochs, device, verbose=True):
    model.train()
    train_loss, n = 0.0, 0
    train_acc = 0

    arr_true = []
    arr_pred = []

    accuracies = [0,0,0]

    tic = time.time()
    for i, data in enumerate(train_loader):
        X = data['bag'].float().to(device)[0].unsqueeze(1)
        y = data['cls'].float().to(device).unsqueeze(0)
        

        optimizer.zero_grad()

        outputs = model(X)
        outputs = torch.sigmoid(outputs)

        y_pred = outputs.data > 0.5
        y_pred = y_pred.float()
        


        arr_true.append(y.item())
        arr_pred.append(outputs.item()) # probability score
        



        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * y.size(0)
        train_acc += torch.sum(y_pred == y.data).item()

        n += 1 # X.size(0)

        if verbose == True:
            print('Train [%2d/%2d]: [%4d/%4d]\tLoss: %1.4f\tAcc: %1.8f'
                %(epoch+1, epochs, i+1, len(train_loader), train_loss/n, train_acc/n), end="\r")


    arr_true = np.array(arr_true).flatten()
    arr_pred = np.array(arr_pred).flatten()
    auc = roc_auc_score(arr_true, arr_pred)


    return train_loss/n, train_acc/n, auc
