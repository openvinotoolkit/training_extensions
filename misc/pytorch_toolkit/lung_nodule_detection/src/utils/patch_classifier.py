
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


def lungpatch_classifier(savepath,imgpath,lrate=1e-4,epochs=35):
    """Trains network to classify patches based on the presence of nodule

    Parameters
    ----------
    savepath: str
        Folder location to save the plots and model
    imgpath:
        Folder location where patch images are stored.
    lrate: int,Default = 1e-4
        Learning rate
    epochs: int, default = 35
        Total epochs

    Returns
    -------

    None
    """

    trainDset = LungPatchDataLoader(imgpath=imgpath,is_transform=True,split="train_set")
    valDset = LungPatchDataLoader(imgpath=imgpath,is_transform=True,split="valid_set")
    trainDataLoader = data.DataLoader(trainDset,batch_size=16,shuffle=True,num_workers=4,pin_memory=True)
    validDataLoader = data.DataLoader(valDset,batch_size=16,shuffle=True,num_workers=4,pin_memory=True)


    savePath = savepath
    if not os.path.isdir(savePath):
    	os.makedirs(savePath)
        
    trainDset = LungDataLoader(is_transform=True,split="train")
    valDset = LungDataLoader(is_transform=True,split="valid")
    trainDataLoader = data.DataLoader(trainDset,batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
    validDataLoader = data.DataLoader(valDset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)

    net = LeNet()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = lrate, weight_decay = 1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    epochs = epochs
    trainLoss = []
    validLoss = []
    trainAcc = []
    validAcc = []
    start = time.time()
    bestValidAcc = 0.0

    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainRunningCorrects = 0
        validRunningCorrects = 0    
        trainBatches = 0
        validBatches = 0
        
        net.train(True)
        
        for data1 in tq(trainDataLoader):
            img, label = data1
            if use_gpu:
                inputs = img.cuda()
                label = label.cuda()

            net_out = net(Variable(inputs))
            
            net_loss = criterion(net_out,label.float())
            preds = torch.zeros(net_out.shape).cuda()
            preds[net_out > 0.5] = 1
            preds[net_out <= 0.5] = 0

            optimizer.zero_grad()

            net_loss.backward() 
       
            optimizer.step()
                      
            trainRunningLoss += net_loss.item()
            for i in range(len(preds[:,0])):
                if preds[:,0][i] == label[:,0][i].float():
                    trainRunningCorrects += 1

            trainBatches += 1
    #         if trainBatches>1:
    #             break          

        trainepoch_loss = trainRunningLoss/trainBatches
        trainepoch_acc = 100*(int(trainRunningCorrects)/32594)
        trainLoss.append(trainepoch_loss)
        trainAcc.append(trainepoch_acc)
        
        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.5f} |Train running : {:.5f}| Train acc: {:.5f} ' 
              .format(epoch+1, epochs, trainepoch_loss,trainRunningCorrects,trainepoch_acc))

        with torch.no_grad():
            for data1 in tq(validDataLoader):
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

                validRunningLoss += net_loss.item()
                for i in range(len(preds[:,0])):
                    if preds[:,0][i] == label[:,0][i].float():
                        validRunningCorrects += 1

                validBatches += 1
    #             if validBatches>10:
    #                 break   

            validepoch_loss = validRunningLoss/validBatches
            validepoch_acc = 100*(int(validRunningCorrects)/3666)
            validLoss.append(validepoch_loss)
            validAcc.append(validepoch_acc)

            print('{:.0f} Loss: {:.4f} | accuracy: {:.4f} '.format(
                    epoch, validepoch_loss,validepoch_acc))
            
        if (validepoch_acc > bestValidAcc):
            bestValidAcc = validepoch_acc
            torch.save(net.state_dict(), savePath+'lenet_best.pt')
            
        scheduler.step(validepoch_loss)
        
        plot=plt.figure()
        plt.plot(range(len(trainLoss)),trainLoss,'-r',label='Train')
        plt.plot(range(len(validLoss)),validLoss,'-g',label='Valid')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if epoch==0:
            plt.legend()
        plt.savefig(savePath+'LossPlot.png')
        plt.close()
        epochEnd = time.time()-epochStart
        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.5f} | Valid Loss: {:.5f}' 
              .format(epoch+1, epochs, trainepoch_loss, validepoch_loss))
        print('Accuracy | Train_acc {:.5f} | Valid_acc  {:.5f} |'
              .format(trainepoch_acc,validepoch_acc))


        print('\nTime: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))
        trainLoss_np = np.array(trainLoss)
        validLoss_np = np.array(validLoss)
        trainAcc_np = np.array(trainAcc)
        validAcc_np = np.array(validAcc)

        print('Saving losses')

        torch.save(trainLoss_np, savePath+'trainLoss.pt')
        torch.save(validLoss_np, savePath+'validLoss.pt')
        torch.save(trainAcc_np, savePath+'train_acc.pt')
        torch.save(validAcc_np, savePath+'valid_acc.pt')

    #     if epoch>1:
    #         break

    end = time.time()-start
    print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
    plt.figure()
    plt.plot(range(len(trainLoss)),trainLoss,'-r',label='Train')
    plt.plot(range(len(validLoss)),validLoss,'-g',label='Valid')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.legend()
    plt.savefig(savePath+'trainLossFinal.png')
    plt.close()


    plt.figure()
    plt.plot(range(len(trainAcc)),trainAcc,'-r',label='Train')
    plt.plot(range(len(validAcc)),validAcc,'-g',label='Valid')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot')
    plt.savefig(savePath+'acc_plot.png')
    plt.close()