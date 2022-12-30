#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm as tq
import time
from torch.utils import data
import os
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .sumnet_bn_vgg import SUMNet
from .r2unet import U_Net
from .r2unet import R2U_Net
from torchvision import transforms
import json
from PIL import Image
from .utils import dice_coefficient
from .data_loader import LungDataLoader

class Discriminator(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_ch, 64, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
	    nn.MaxPool2d(3),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
	    nn.MaxPool2d(3),
            nn.Conv2d(64*2, 64 * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
	    nn.MaxPool2d(3),
            nn.Conv2d(64 * 4, 64 * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
	    nn.MaxPool2d(2),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, out_ch, 7, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 2) #.squeeze(1)

def ch_shuffle(x):
    shuffIdx1 = torch.from_numpy(np.random.randint(0,2,x.size(0)))
    shuffIdx2 = 1-shuffIdx1        
    d_in = torch.Tensor(x.size()).cuda()
    d_in[:,shuffIdx1] = x[:,0]
    d_in[:,shuffIdx2] = x[:,1]
    shuffLabel = torch.cat((shuffIdx1.unsqueeze(1),shuffIdx2.unsqueeze(1)),dim=1)
    return d_in, shuffLabel

def train_advnetwork(fold_no,savepath,jsonpath,datapath,lung_segpath,network,epochs=35,lrate=1e-4):
    """Training function for SUMNet,UNet,R2Unet

    Parameters
    ----------
    fold_no: str
        Fold number on which training is to be performed
    save_path: str
        Folder location to save the models and other plots
    json_path: str
        Folder location at which json files are stored
    datapath: str
        Folder location of data
    lung_segpath: str
        Folder location at which lung segmentation files are stored
    network: str
        Network to be trained
    epochs: int, Default: 35
        Number of epochs for training
    lrate: int, Default= 1e-4
        Learnig rate

    Returns
    -------

    None
    """


    fold = 'fold'+str(fold_no)
    savePath = savepath+'/'+network+'/'+fold+'/'
    if not os.path.isdir(savePath):
    	os.makedirs(savePath)

    with open(jsonpath+fold+'_pos_neg_eq.json') as f:
        json_file = json.load(f)
        train_set = json_file['train_set']
        val_set = json_file['valid_set']


    trainDset = LungDataLoader(datapath=datapath,lung_path=lung_segpath,is_transform=True,json_file=json_file,split="train_set",img_size=512)
    valDset = LungDataLoader(datapath=datapath,lung_path=lung_segpath,is_transform=True,json_file=json_file,split="valid_set",img_size=512)
    trainDataLoader = data.DataLoader(trainDset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    validDataLoader = data.DataLoader(valDset,batch_size=8,shuffle=False,num_workers=4,pin_memory=True,drop_last=True)

    if network == 'sumnet':
        net = SUMNet(in_ch=1,out_ch=2)
    if network == 'unet':
        net = U_Net(img_ch=1,output_ch=2)
    if network == 'r2unet':
        net = R2U_Net(img_ch=1,output_ch=2)


    netD2 = Discriminator(in_ch=2,out_ch=2) 

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        net = net.cuda()
        netD2 = netD2.cuda()


    optimizer = optim.Adam(net.parameters(), lr = lrate, weight_decay = 1e-5)
    optimizerD2 = optim.Adam(netD2.parameters(), lr = 1e-4, weight_decay = 1e-5)


    criterion = nn.BCEWithLogitsLoss()
    criterionD = nn.BCELoss()

    epochs = epochs
    trainLoss = []
    validLoss = []
    D2_losses = []
    trainDiceCoeff_lungs = []
    validDiceCoeff_lungs = []
    start = time.time()

    bestValidDice = torch.zeros(1)
    bestValidDice_lungs = 0.0

    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        trainDice_lungs = 0
        validDice_lungs = 0


        net.train(True)

        for data1 in tq(trainDataLoader):
            img, mask = data1
            labels = mask
            if use_gpu:
                inputs = img.cuda()
                labels = labels.cuda()

            net_out = net(Variable(inputs))
            net_out_sf = F.softmax(net_out,dim=1)
            
            
            ############################
            # DISCRIMINATOR 2 TRAINING #
            ############################

            optimizerD2.zero_grad()
            # Concatenate real (GT) and fake (segmented) samples along dim 1
            d_in = torch.cat((net_out[:,1].unsqueeze(1),labels[:,1].unsqueeze(1).float()),dim=1)
            # Shuffling aling dim 1: {real,fake} OR {fake,real}
            d_in,shuffLabel = ch_shuffle(d_in)
            # D2 prediction
            confr = netD2(Variable(d_in)).view(d_in.size(0),-1)  
            # Compute loss
            LD2 = criterionD(confr,shuffLabel.float().cuda())
            # Compute gradients
            LD2.backward()
            # Backpropagate   
            optimizerD2.step()
            # Appending loss for each batch into the list
            D2_losses.append(LD2.item())            
            optimizerD2.zero_grad()
            d2_in = torch.cat((net_out[:,1].unsqueeze(1),labels[:,1].unsqueeze(1).float()),dim=1)
            d2_in, d2_lb = ch_shuffle(d2_in)  
            conffs2 = netD2(d2_in).view(d2_in.size(0),-1)
            LGadv2 = criterionD(conffs2,d2_lb.float().cuda()) # Aversarial loss 2

            
            
            BCE_Loss = criterion(net_out[:,1],labels[:,1])

            net_loss = BCE_Loss  - 0.001*LGadv2 

            optimizer.zero_grad()

            net_loss.backward() 
       
            optimizer.step()
                      
            trainRunningLoss += net_loss.item()
     
            trainDice = dice_coefficient(net_out_sf,torch.argmax(labels,dim=1))
            trainDice_lungs += trainDice[0]  
             
            trainBatches += 1 
    #         if trainBatches>1:
    # #             break

        trainLoss.append(trainRunningLoss/trainBatches)
        trainDiceCoeff_lungs.append(trainDice_lungs/trainBatches)

        print("\n{}][{}]| Net_loss: {:.4f}  | BCE_Loss: {:.4f} |adv_loss: {:.4f}"
        .format(epoch,epochs,net_loss.item(),BCE_Loss,LGadv2) )    

        with torch.no_grad():
            for data1 in tq(validDataLoader):

                imgs, mask = data1
                labels = mask
                if use_gpu:
                    inputs = imgs.cuda()
                    labels = labels.cuda()

                net_out = net(Variable(inputs))
                net_out_sf = F.softmax(net_out.data,dim=1)


                BCE_Loss = criterion(net_out[:,1],labels[:,1])            

                net_loss = BCE_Loss 


                val_dice = dice_coefficient(net_out_sf,torch.argmax(labels,dim=1))
                validDice_lungs += val_dice[0]                        
                validRunningLoss += net_loss.item()
                validBatches += 1 
    #             if validBatches>1:
    #                 break   

            validLoss.append(validRunningLoss/validBatches)
            validDiceCoeff_lungs.append(validDice_lungs/validBatches)



        if (validDice_lungs.cpu() > bestValidDice_lungs):
            bestValidDice_lungs = validDice_lungs.cpu()
            torch.save(net.state_dict(), savePath+'sumnet_adv_best_lungs.pt')
       
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
              .format(epoch+1, epochs, trainRunningLoss/trainBatches, validRunningLoss/validBatches))
        print('Dice | Train  | Lung {:.3f}  | Valid | Lung {:.3f} | '
              .format(trainDice_lungs/trainBatches, validDice_lungs/validBatches))

        print('\nTime: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))
        trainLoss_np = np.array(trainLoss)
        validLoss_np = np.array(validLoss)
        trainDiceCoeff_lungs_np = np.array(trainDiceCoeff_lungs)
        validDiceCoeff_lungs_np = np.array(validDiceCoeff_lungs)


        print('Saving losses')

        torch.save(trainLoss_np, savePath+'trainLoss.pt')
        torch.save(validLoss_np, savePath+'validLoss.pt')
        torch.save(trainDiceCoeff_lungs_np, savePath+'trainDice_lungs.pt')
        torch.save(validDiceCoeff_lungs_np, savePath+'validDice_lungs.pt')

    #     if epoch>0:
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
    plt.plot(range(len(trainDiceCoeff_lungs)),trainDiceCoeff_lungs,'-r',label='Lungs')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.title('Dice coefficient: Train')
    plt.savefig(savePath+'trainDice.png')
    plt.close()

    plt.figure()
    plt.plot(range(len(validDiceCoeff_lungs)),validDiceCoeff_lungs,'-g',label='Lungs')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.title('Dice coefficient: Valid')
    plt.savefig(savePath+'validDice.png')
    plt.close()

    plt.figure()
    plt.plot(range(len(trainDiceCoeff_lungs)),trainDiceCoeff_lungs,'-r',label='Train')
    plt.plot(range(len(validDiceCoeff_lungs)),validDiceCoeff_lungs,'-g',label='Valid')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.savefig(savePath+'Dice_final.png')
    plt.close()

