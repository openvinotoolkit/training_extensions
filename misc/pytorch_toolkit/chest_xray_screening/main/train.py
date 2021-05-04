
import os
import os.path
import numpy as np
import time
import sys
import csv
import argparse
import torch
from torchvision import models
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from PIL import Image
import torch.nn.functional as func
import random
from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import json
from pathlib import Path
from dataloader import RSNADataSet


class RSNATrainer():

    def train (model, dataLoaderTrain, dataLoaderVal,dataLoaderTest, nnClassCount, trMaxEpoch, launchTimestamp, checkpoint,lr, device, class_names):

        ############Define Loss ########
        loss1 = torch.nn.BCELoss()
        aurocMax=0.0
        
        ######### LOAD CHECKPOINT  ##########
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        if checkpoint is not None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            for param in model.parameters():
                param.requires_grad = True
            print("Model loaded")

        ############## TRAIN THE NETWORK ########
        lossMIN = 100000
        lossVal = 100000
        for epochID in range(0, trMaxEpoch):
            # update_lr(optimizer,0.0001)
            print("Epoch "+ str(epochID+1)+"/"+str(trMaxEpoch))    
            timestampSTART = time.strftime("%H%M%S-%d%m%Y")
            global gepochID
            gepochID=epochID
            batchs, losst, losse, aurocMax = RSNATrainer.epochTrain(model, dataLoaderTrain, dataLoaderVal,dataLoaderTest, optimizer,
                                                              trMaxEpoch, nnClassCount,loss1,device,
                                                                        class_names, aurocMax)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 
                        'optimizer' : optimizer.state_dict()},'models/m-epoch'+str(epochID)+'-' + launchTimestamp + '.pth.tar')
            
            outGT1, outPRED1,accuracy = RSNATrainer.test(model, dataLoaderVal, nnClassCount, 
                                                             'models/m-epoch'+str(epochID)+'-' + launchTimestamp + '.pth.tar', 
                                                             class_names,device)
            
            print ('\nEpoch [' + str(epochID + 1) + '] [-----] [' + timestampEND + '] loss= ' + str(lossVal))
#             
        return batchs, losst, losse        
    #-------------------------------------------------------------------------------- 
       
    def epochTrain(model, dataLoaderTrain, dataLoaderVal,dataLoaderTest, optimizer, epochMax, classCount,loss1,device,
                   class_names, aurocMax):
        
        batch = []
        losstrain = []
        losseval = []
        
        model.train()
        scheduler = StepLR(optimizer, step_size=6, gamma=0.002)
        for batchID, (varInput, target) in enumerate(dataLoaderTrain):
            global gepochID
            epochID = gepochID
            
            varTarget = target.to(device)
            varInput = varInput.to(device)         
            varOutput=model(varInput)
                
            lossvalue = loss1(varOutput,tfunc.one_hot(varTarget.squeeze(1).long(),num_classes=3).float())
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            l = lossvalue.item()
            losstrain.append(l)
            if batchID%140==1:
                print(f"Batch::{batchID};Loss::{l}")
            if batchID%2500==0 and batchID!=0:
                print("\nbatchID:"+str(batchID))
                outGd,out,aurocMean=RSNATrainer.test(model, dataLoaderTest, classCount,None,class_names,device)
                print("\n")
                
                if aurocMean>aurocMax:
                    print('Better auroc obtained')
                    aurocMax = aurocMean
                    global model_val
                    model_val='m-epoch'+str(epochID)+'-batchId'+str(batchID)+'-aurocMean-'+str(aurocMean) + '.pth.tar'
                    torch.save({'batch': batchID + 1, 'state_dict': model.state_dict(), 'aucmean_loss': aurocMean, 'optimizer' : optimizer.state_dict()},
                               'models/m-epoch-'+str(epochID)+'-batchId-'+str(batchID) +'-aurocMean-'+str(aurocMean)+ '.pth.tar')
                scheduler.step()
                
        return batch, losstrain, losseval, aurocMax
    
    #-------------------------------------------------------------------------------- 
    
    def epochVal(model, dataLoader, optimizer, epochMax, classCount,loss1,device):
        
        model.eval()
        
        lossVal = 0
        lossValNorm = 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderVal):
                print(f"Batch {i} in Val")
                target = target.to(device)
                varOutput = model(varInput.to(device))
                
                losstensor = loss1(varOutput,tfunc.one_hot(target.squeeze(1).long(),num_classes=3).float())
                lossVal += losstensor
                lossValNorm += 1
                
        outLoss = lossVal / lossValNorm
        return outLoss
    
    
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC(dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        #print(datanpGT.shape)
        #print(datanpPRED.shape)
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                outAUROC.append(0)#when the number of counts is zero
        return outAUROC
        
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names,device):   
        cudnn.benchmark = True
        
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
        else:
            model.state_dict()
        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        
        model.eval()
        
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):

                target = target.to(device)
                outGT = torch.cat((outGT, target), 0).to(device)

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
            
                out = model(varInput.to(device))
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = RSNATrainer.computeAUROC(tfunc.one_hot(outGT.squeeze(1).long()).float(), outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('\nAUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED,aurocMean

def main(args):

    lr= args.lr
    checkpoint= args.checkpoint
    trBatchSize = args.bs
    trMaxEpoch = args.epochs

    nnClassCount = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available
    class_names = ['Lung Opacity','Normal','No Lung Opacity / Not Normal']

    ############# Data Loader ####################
    img_pth='/home/rakshith/RSNA_pneumonia/dataset/rsna-pneumonia-detection-challenge/train_data/'
    tr_list = np.load('/tools/train_list.npy').tolist()
    tr_labels = np.load('/tools/train_labels.npy').tolist()
    val_list = np.load('/tools/valid_list.npy').tolist()
    val_labels = np.load('/tools/val_labels.npy').tolist()
    test_list = np.load('/tools/test_list.npy').tolist()
    test_labels = np.load('/tools/test_labels.npy').tolist()

    datasetTrain = RSNADataSet(tr_list,tr_labels,img_pth,transform_type='train')
    datasetValid = RSNADataSet(val_list,val_labels,img_pth, transform_type='train')            
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=4, pin_memory=False)
    dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=False)

    datasetTest = RSNADataSet(test_list,test_labels,img_pth,transform_type='test')
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=True,  num_workers=4, pin_memory=False)

    ########### Construct Model ##############

    model=models.densenet121(pretrained=True)
    for param in model.parameters():
         param.requires_grad = False
    model.classifier=nn.Sequential(nn.Linear(1024, nnClassCount), nn.Sigmoid())
    model = model.to(device)

    ############## Train the  Model #################
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    batch, losst, losse = RSNATrainer.train(model, dataLoaderTrain, dataLoaderVal,dataLoaderTest, nnClassCount, trMaxEpoch, timestampLaunch, checkpoint,lr ,device, class_names)
    print("Model trained !")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",required=False, help="The learning rate of the model",default=1e-4,type = float)
    parser.add_argument("--checkpoint",required=False, help="start training from a checkpoint model weight",default= None ,type = str)
    parser.add_argument("--bs",required=False, help="Batchsize")
    parser.add_argument("--imgpath",required=True, help="Path containing train and test images", type =str)
    parser.add_argument("--epochs",required=False,default=15, help="Number of epochs", type=int)
    
    args = parser.parse_args()

    main(args)
