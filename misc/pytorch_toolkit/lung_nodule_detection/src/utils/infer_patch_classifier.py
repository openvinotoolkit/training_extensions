#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import transforms
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from tqdm import tqdm_notebook as tq
from sklearn.metrics import confusion_matrix
from .data_loader import LungPatchDataLoader
from .lenet import LeNet

def lungpatch_classifier(modelpath,imgpath):

    testDset = LungPatchDataLoader(imgpath,is_transform=True,split="test")
    testDataLoader = data.DataLoader(testDset,batch_size=1,shuffle=True,num_workers=4,pin_memory=True)
    classification_model_loadPath = modelpath
    net = LeNet()

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        net = net.cuda()
    net.load_state_dict(torch.load(classification_model_loadPath+'lenet_best.pt'))

    optimizer = optim.Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    testRunningCorrects = 0
    testRunningLoss = 0
    testBatches = 0
    pred_arr = []
    label_arr = []
    for data1 in tq(testDataLoader):
        img, label = data1
        if use_gpu:
            inputs = img.cuda()
            label = label.float()
            label = label.cuda()

        net_out = net(Variable(inputs))

        net_loss = criterion(net_out,label)
        preds = torch.zeros(net_out.shape).cuda()
        preds[net_out > 0.5] = 1
        preds[net_out <= 0.5] = 0

        testRunningLoss += net_loss.item()
        testRunningCorrects += torch.sum(preds == label.data.float())

        for i,j in zip(preds.cpu().numpy(),label.cpu().numpy()):
            pred_arr.append(i)
            label_arr.append(j)

        testBatches += 1
        # if testBatches>0:
        #     break

    testepoch_loss = testRunningLoss/testBatches
    testepoch_acc = 100*(int(testRunningCorrects)/len(pred_arr))

    print(' Loss: {:.4f} | accuracy: {:.4f} '.format(
             testepoch_loss,testepoch_acc))


    tn, fp, fn, tp = confusion_matrix(np.array(label_arr).flatten(), np.array(pred_arr).flatten()).ravel()

    print('True Negative :',tn)
    print('false Negative :',fn)
    print('True positive :',tp)
    print('False positive :',fp)
    specificity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    print('Specificity :',specificity)
    print('Sensitivity :',sensitivity)
