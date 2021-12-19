from torch.autograd import Variable
import torch
import time
import numpy as np
from sklearn.metrics import roc_auc_score


def val(model, test_loader, criterion, epoch, epochs, device, verbose=True):
    model.eval()
    n = 0
    test_loss, test_acc = 0.0, 0.0
    accuracies = [0,0,0]
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

        loss = criterion(out, y)



        arr_true.append(y.item())
        arr_pred.append(out.item())
        
        test_loss += loss.item() * y.size(0)
        test_acc += torch.sum(y_pred == y.data).item()

        n += 1 # y.size(0)

        if verbose == True:
            print('Test [%2d/%2d]: [%4d/%4d]\tLoss: %1.4f\tAcc: %1.4f'
                %(epoch+1, epochs, i+1, len(test_loader), test_loss/n, test_acc/n), end="\r")


    arr_true = np.array(arr_true).flatten()
    arr_pred = np.array(arr_pred).flatten()
    auc = roc_auc_score(arr_true, arr_pred)


    return test_loss/n, test_acc/n, auc
