from torch import nn
import torch
import numpy as np
from .get_config import get_config
############# Define the Weighted Loss. The weights are different for each class ########
class Custom_Loss(nn.Module):
    def __init__(self, site, device=torch.device('cpu')):
        super().__init__()

        config = get_config(action='loss')
        wts_pos = np.array(config[str(site)]['wts_pos'])
        wts_neg = np.array(config[str(site)]['wts_neg'])
        wts_pos = torch.from_numpy(wts_pos)
        wts_pos = wts_pos.type(torch.Tensor)
        wts_pos=wts_pos.to(device) # size 1 by cls

        wts_neg = torch.from_numpy(wts_neg)
        wts_neg = wts_neg.type(torch.Tensor)
        wts_neg=wts_neg.to(device) # size 1 by cls

        self.wts_pos=wts_pos
        self.wts_neg=wts_neg
        self.bce=nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, ypred, ytrue):
        msk = ((1-ytrue)*self.wts_neg) + (ytrue*self.wts_pos) #1 if ytrue is 0
        loss=self.bce(ypred,ytrue) # bsz, cls
        loss=loss*msk
        loss=loss.view(-1) # flatten all batches and class losses
        loss=torch.mean(loss)
        return loss
