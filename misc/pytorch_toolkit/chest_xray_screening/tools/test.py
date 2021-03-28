import os
import os.path
import numpy as np
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt
import torch
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
from generate import *
from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random
from pathlib import Path
import argparse
from math import sqrt
import torchvision.models.densenet as modelzoo



class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file,image_directory, transform_type):
        image_directory = Path(image_directory)
        image_names = []
        labels = []
        f = open('rsna_annotation.json') 
        data = json.load(f)
        image_names = data['names']
        labels = data['labels']
        
        image_names = [Path.joinpath(image_directory , x) for x in image_names]
        self.image_names = image_names
        self.labels = labels
        
        if transform_type=='train':
            transform=transforms.Compose([
                                            #transforms.Resize(350),
                                            #transforms.RandomResizedCrop((320,320), scale=(0.8, 1.0)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([124.978,124.978,124.978], [10.868,10.868,10.868])
                                        ])
            
        elif transform_type=='test':
            transform=transforms.Compose([
                
                                            #transforms.Resize((320,320)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([124.925,124.925,124.925], [10.865,10.865,10.865])
                                        ])
        
        
        
        self.transform = transform


    def __getitem__(self, index):
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        label = self.labels[index]
        #label = transforms.ToTensor(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor([label])

    def __len__(self):
        return len(self.image_names)

###################





class CheXpertTester():
    
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                outAUROC.append(0)#when the number of counts is zero
        return outAUROC
        
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names, device):   
        cudnn.benchmark = True
        
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
        
        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):

                target = target.to(device)
                outGT = torch.cat((outGT, target), 0).to(device)

                #bs, c, h, w = input.size()
                #varInput = input.view(-1, c, h, w)
                varInput = input
            
                out = model(varInput.to(device))
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTester.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('\nAUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED,aurocMean




def main(args):
    
    alpha = args.alpha
    phi = args.phi
    beta = args.beta
    checkpoint = args.checkpoint
    
    
    
    if beta == None:
        beta = round(sqrt(2/alpha),3)
        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use gpu if available
    

    class_names = ['Lung Opacity','Normal','No Lung Opacity / Not Normal']
    
    BatchSize = args.bs
    nnClassCount = 3
    
    
    
    ############# Data Loader ####################
    img_pth= args.imgpath
    pathFileTest = 'rsna_annotation.json'

    datasetTest = CheXpertDataSet(pathFileTest,img_pth)            
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=BatchSize, shuffle=False, num_workers=4, pin_memory=False)
   
    ########### Construct Model ##############
    alpha = alpha ** phi
    beta = beta ** phi
    
    model, total_macs = give_model(alpha,beta,nnClassCount)
    print(f"{total_macs} is the number of macs.")
    model = nn.Sequential(model, nn.Sigmoid())
    model = model.to(device)
    
    ############## Train the  Model #################

    batch, losst, losse = CheXpertTester.test(model, dataLoaderTest, nnClassCount,checkpoint,class_names,device)
    
    #test(model, dataLoaderTest, nnClassCount, checkpoint, class_names)





if __name__ == "__main__":

    parser.add_argument("--alpha",required=False, help="alpha for the model",default= (11/6) ,type = float)
    parser.add_argument("--phi",required=False, help="Phi for the model.",default= 1.0 ,type = float)
    parser.add_argument("--beta",required=False, help="Beta for the model.",default= None ,type = float)
    parser.add_argument("--checkpoint",required=False, help="start training from a checkpoint model weight",default= None ,type = str)
    parser.add_argument("--bs",required=False, help="Batchsize")
    parser.add_argument("--imgpath",required=True, help="Path containing train and test images", type =str)

    args = parser.parse_args()
    # execute only if run as a script
    main(args)







