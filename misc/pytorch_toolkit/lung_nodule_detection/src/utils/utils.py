import numpy as np
import torch
from torch.utils import data

def dice_coefficient(pred1, target):
    smooth = 1e-15
    pred = torch.argmax(pred1,dim=1)
    num = pred.size()[0]
    pred_1_hot = torch.eye(3)[pred.squeeze(1)].cuda()
    pred_1_hot = pred_1_hot.permute(0, 3, 1, 2).float()

    target_1_hot = torch.eye(3)[target].cuda()
    target_1_hot = target_1_hot.permute(0,3, 1, 2).float()

    m1_1 = pred_1_hot[:,1,:,:].view(num, -1).float()
    m2_1 = target_1_hot[:,1,:,:].view(num, -1).float()
    m1_2 = pred_1_hot[:,2,:,:].view(num, -1).float()
    m2_2 = target_1_hot[:,2,:,:].view(num, -1).float()
    
    intersection_1 = (m1_1*m2_1).sum(1)
    intersection_2 = (m1_2*m2_2).sum(1)
    union_1 = (m1_1+m2_1).sum(1) + smooth - intersection_1
    union_2 = (m1_2+m2_2).sum(1) + smooth - intersection_2
    score_1 = intersection_1/union_1
    score_2 = intersection_2/union_2

    return [score_1.mean()]
    