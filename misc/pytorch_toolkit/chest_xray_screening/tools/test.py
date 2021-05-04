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


class RSNAInference():

    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
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

    test_list = np.load('.test_list.npy').tolist()
    test_labels = np.load('.test_labels.npy').tolist()

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
    batch, losst, losse = RSNAInference.test(model, dataLoaderTest, nnClassCount, checkpoint, class_names, device)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",required=False, help="The learning rate of the model",default=1e-4,type = float)
    parser.add_argument("--checkpoint",required=False, help="start training from a checkpoint model weight",default= None ,type = str)
    parser.add_argument("--bs",required=False, help="Batchsize")
    parser.add_argument("--imgpath",required=True, help="Path containing train and test images", type =str)
    parser.add_argument("--epochs",required=False,default=15, help="Number of epochs", type=int)
    
    args = parser.parse_args()

    main(args)