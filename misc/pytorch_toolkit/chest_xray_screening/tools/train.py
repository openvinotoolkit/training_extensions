import numpy as np
import time
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils.dataloader import RSNADataSet
from utils.score import compute_auroc
from utils.model import DenseNet121,DenseNet121Eff
from utils.generate import *


class RSNATrainer():

    def __init__(
        self, model, 
        data_loader_train, data_loader_valid, data_loader_test,
        class_count, checkpoint, device, class_names):

        self.gepoch_id = 0
        self.model_val = ''
        self.model = model
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.data_loader_test = data_loader_test
        self.class_names = class_names
        self.class_count = class_count
        self.checkpoint = checkpoint
        self.device = device


    def train(
        self, max_epoch, timestamp_launch,lr):

        loss_fn = torch.nn.BCELoss()
        # Setting maximum AUROC value as zero
        auroc_max = 0.0                 
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        if self.checkpoint is not None:
            model_checkpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(model_checkpoint['state_dict'])
            optimizer.load_state_dict(model_checkpoint['optimizer'])
            for param in self.model.parameters():
                param.requires_grad = True
            print(f"Model loaded")

        train_loss_min = 100000  # A random very high number
        valid_loss_min = 100000

        for epoch_id in range(0, max_epoch):
            
            print(f"Epoch {epoch_id+1}/{max_epoch}")    
            self.gepoch_id = epoch_id


            train_loss, valid_loss, auroc_max = RSNATrainer.epoch_train(
                optimizer,loss_fn,auroc_max)

            timestamp_end = time.strftime("%H%M%S-%d%m%Y")

            
            if train_loss < train_loss_min:
                train_loss_min = train_loss

            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss

            torch.save({'epoch': epoch_id + 1, 'state_dict': model.state_dict(), 'best_loss': valid_loss_min, 
                        'optimizer' : optimizer.state_dict()},'models/m-epoch'+str(epoch_id)+'-' + timestamp_launch + '.pth.tar')

            test_auroc = RSNATrainer.test('models/m-epoch'+str(epoch_id)+'-' + timestamp_launch + '.pth.tar')
            
            print (f"Epoch:{epoch_id + 1}| EndTime:{timestamp_end}| TestAUROC: {test_auroc}| ValidAUROC: {auroc_max}")
   

    def valid(loss_fn):
        
        self.model.eval()
        loss_valid_r = 0
        valid_batches = 0      # Counter for valid batches

        out_gt = torch.FloatTensor().to(self.device)
        out_pred = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (var_input, var_target) in enumerate(self.data_loader_valid):
                print(f"Batch {i} in Val")

                var_target = var_target.to(self.device)
                out_gt = torch.cat((out_gt, var_target), 0).to(self.device)

                _, c, h, w = var_input.size()
                var_input = var_input.view(-1, c, h, w)

                var_output = model(var_input.to(self.device))
                out_pred = torch.cat((out_pred, var_output), 0)
                
                lossvalue = loss_fn(
                    var_output,tfunc.one_hot(var_target.squeeze(1).long(),num_classes =self.class_count).float())

                loss_valid_r += lossvalue.item()
                valid_batches += 1
                
            valid_loss = loss_valid_r / valid_batches

            auroc_individual = compute_auroc(tfunc.one_hot(out_gt.squeeze(1).long()).float(), out_pred, self.class_count)
            auroc_mean = np.array(auroc_individual).mean()

        return valid_loss,auroc_mean

       
    def epoch_train(optimizer, loss_fn, auroc_max):
        
        
        loss_train_list = []
        loss_valid_list = []
        
        self.model.train()
        scheduler = StepLR(optimizer, step_size=6, gamma=0.002)

        for batch_id, (var_input, var_target) in enumerate(self.data_loader_train):

            epoch_id = self.gepoch_id
            
            var_target = var_target.to(self.device)
            var_input = var_input.to(self.device)         
            var_output= self.model(var_input)
                
            trainloss_value = loss_fn(var_output,tfunc.one_hot(var_target.squeeze(1).long(),num_classes=self.class_count).float())
            
            optimizer.zero_grad()
            trainloss_value.backward()
            optimizer.step()
            
            train_loss_value = trainloss_value.item()
            loss_train_list.append(trainloss_value)

            # Evaluate the performance of the model on validation_set
            # every 2500th iteration. 2500 is a random choice, this could be changed.

            if batch_id%2500==0 and batch_id!=0:                   
                print(f"batch_id::{batch_id}")

                validloss_value,auroc_mean = RSNATrainer.valid(loss_valid_list,loss_fn)

                loss_valid_list.append(validloss_value)
                
                print("\n")
                
                if auroc_mean>auroc_max:
                    print(f'Better auroc obtained')
                    auroc_max = auroc_mean
                    
                    self.model_val='m-epoch'+str(epoch_id)+'-batch_id'+str(batch_id)+'-aurocMean-'+str(auroc_mean) + '.pth.tar'
                    torch.save({'batch': batch_id + 1, 'state_dict': model.state_dict(), 'aucmean_loss': auroc_mean, 'optimizer' : optimizer.state_dict()},
                               'models/m-epoch-'+str(epoch_id)+'-batch_id-'+str(batch_id) +'-aurocMean-'+str(auroc_mean)+ '.pth.tar')

                scheduler.step()

        train_loss_mean = np.mean(loss_train_list)
        valid_loss_mean = np.mean(loss_valid_list)
        
                
        return train_loss_mean, valid_loss_mean, auroc_max
    
    def test(): 

        cudnn.benchmark = True
        if self.checkpoint != None:
            model_checkpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(model_checkpoint['state_dict'])
        else:
            self.model.state_dict()

        out_gt = torch.FloatTensor().to(self.device)
        out_pred = torch.FloatTensor().to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (var_input, var_target) in enumerate(self.data_loader_test):

                var_target = var_target.to(self.device)
                out_gt = torch.cat((out_gt, var_target), 0).to(self.device)

                _, c, h, w = input.size()
                var_input = input.view(-1, c, h, w)
            
                out = model(var_input.to(self.device))
                out_pred = torch.cat((out_pred, out), 0)

        auroc_individual = compute_auroc(tfunc.one_hot(out_gt.squeeze(1).long()).float(), out_pred, self.class_count)
        auroc_mean = np.array(auroc_individual).mean()
        
        print(f'AUROC mean:{auroc_mean}')
        
        for i in range (0, len(auroc_individual)):
            print(f"{self.class_names[i]}:{auroc_individual[i]}")
        
        return auroc_mean

def main(args):

    lr = args.lr
    checkpoint = args.checkpoint
    batch_size = args.bs
    max_epoch = args.epochs


    class_count = args.clscount  #The objective is to classify the image into 3 classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available
    class_names = ['Lung Opacity','Normal','No Lung Opacity / Not Normal']

    # Data Loader 
    img_pth = args.imgpath
    numpy_path = args.npypath

    # Place numpy file containing train-valid-test split on tools folder

    tr_list = np.load(os.path.join(numpy_path,'train_list.npy')).tolist()
    tr_labels = np.load(os.path.join(numpy_path,'train_labels.npy')).tolist()
    val_list = np.load(os.path.join(numpy_path,'valid_list.npy')).tolist()
    val_labels = np.load(os.path.join(numpy_path,'valid_labels.npy')).tolist()
    test_list = np.load(os.path.join(numpy_path,'test_list.npy')).tolist()
    test_labels = np.load(os.path.join(numpy_path,'test_labels.npy')).tolist()

    dataset_train = RSNADataSet(tr_list, tr_labels, img_pth, transform=True)
    dataset_valid = RSNADataSet(val_list, val_labels, img_pth, transform=True) 

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=False)
    data_loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    dataset_test = RSNADataSet(test_list, test_labels, img_pth, transform=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False,  num_workers=4, pin_memory=False)

    # Construct Model 

    if args.optimised is not None:
        alpha = args.alpha
        phi = args.phi
        beta = args.beta
        

        if beta is None:
            beta = round(sqrt(2 / alpha), 3)

        alpha = alpha ** phi
        beta = beta ** phi

        model = DenseNet121Eff(alpha,beta,class_count)
        model = model.to(device)
    else: 
        model = DenseNet121(class_count)
        model = model.to(device)

    # Train the  Model 
    timestamp_launch = time.strftime("%d%m%Y - %H%M%S")

    rsna_trainer = RSNATrainer(
        model, data_loader_train, data_loader_valid, data_loader_test, 
        class_count,checkpoint, device, class_names)

    RSNATrainer.train(max_epoch, timestamp_launch, lr)

    print(f"Model trained !")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",required=False, help="Learning rate",default=1e-4,type = float)
    parser.add_argument("--checkpoint",required=False, help="Checkpoint model weight",default= None ,type = str)
    parser.add_argument("--bs",required=False, help="Batchsize", type=int)
    parser.add_argument("--imgpath",required=True, help="Path containing train and test images", type =str)
    parser.add_argument("--npypath",required=True, help="Path containing label list in npy format", type =str)
    parser.add_argument("--epochs",required=False,default=15, help="Number of epochs", type=int)
    parser.add_argument("--clscount",required=False,default=3, help="Number of classes", type=int)
    parser.add_argument("--alpha",
        required=False,
        help="alpha for the model",
        default=(11 / 6),
        type=float)
    parser.add_argument("--phi",
        required=False,
        help="Phi for the model.",
        default=1.0,
        type=float)
    parser.add_argument("--beta",
        required=False,
        help="Beta for the model.",
        default=None,
        type=float)
    
    args = parser.parse_args()

    main(args)

