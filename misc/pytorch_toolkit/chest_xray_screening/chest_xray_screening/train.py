import numpy as np
import time
import os
import argparse
import torch
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from .utils.dataloader import RSNADataSet
from .utils.score import compute_auroc
from .utils.model import DenseNet121, DenseNet121Eff
from math import sqrt
import json
from tqdm import tqdm as tq


class RSNATrainer():

    def __init__(self, model,
                      data_loader_train, data_loader_valid, data_loader_test,
                      class_count, checkpoint, device, class_names, lr):

        self.gepoch_id = 0
        self.device = device
        self.model = model.to(self.device)
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.data_loader_test = data_loader_test
        self.class_names = class_names
        self.class_count = class_count
        self.auroc_max = 0.0  # Setting maximum AUROC value as zero
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if checkpoint is not None:
            model_checkpoint = torch.load(checkpoint)
            self.optimizer.load_state_dict(model_checkpoint['optimizer'])
        else:
            model_checkpoint = None

        self.loss_fn = torch.nn.BCELoss()

    def train(self, max_epoch, savepath):
        train_loss_min = 1e+5  # A random very high number
        valid_loss_min = 1e+5

        for epoch_id in range(max_epoch):
            print(f"Epoch {epoch_id+1}/{max_epoch}")
            self.gepoch_id = epoch_id
            train_loss, valid_loss, auroc_max = self.epoch_train()
            self.current_train_loss = train_loss
            self.current_valid_loss = valid_loss
            timestamp_end = time.strftime("%H%M%S-%d%m%Y")

            if train_loss < train_loss_min:
                train_loss_min = train_loss

            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss

            torch.save({'epoch': epoch_id + 1,
                             'state_dict': self.model.state_dict(),
                             'best_loss': valid_loss_min,
                             'optimizer' : self.optimizer.state_dict()},
                             os.path.join(savepath, f'm-epoch-{epoch_id}.pth'))
            test_auroc = self.test()
            print(f"Epoch:{epoch_id + 1}| EndTime:{timestamp_end}| TestAUROC: {test_auroc}| ValidAUROC: {auroc_max}")

    def valid(self):
        self.model.eval()
        loss_valid_r = 0
        valid_batches = 0      # Counter for valid batches
        out_gt = torch.FloatTensor().to(self.device)
        out_pred = torch.FloatTensor().to(self.device)
        with torch.no_grad():
            for (var_input, var_target) in tq(self.data_loader_valid):
                var_target = var_target.to(self.device)
                out_gt = torch.cat((out_gt, var_target), 0).to(self.device)

                _, c, h, w = var_input.size()
                var_input = var_input.view(-1, c, h, w)

                var_output = self.model(var_input.to(self.device))
                out_pred = torch.cat((out_pred, var_output), 0)

                lossvalue = self.loss_fn(
                    var_output, tfunc.one_hot(var_target.squeeze(1).long(), num_classes=self.class_count).float())

                loss_valid_r += lossvalue.item()
                valid_batches += 1

            valid_loss = loss_valid_r / valid_batches

            auroc_individual = compute_auroc(
                tfunc.one_hot(out_gt.squeeze(1).long()).float(),
                out_pred, self.class_count)
            print(len(auroc_individual))
            auroc_mean = np.array(auroc_individual).mean()
        return valid_loss, auroc_mean


    def epoch_train(self):
        loss_train_list = []
        loss_valid_list = []
        self.model.train()
        scheduler = StepLR(self.optimizer, step_size=6, gamma=0.002)

        for batch_id, (var_input, var_target) in tq(enumerate(self.data_loader_train)):
            var_target = var_target.to(self.device)
            var_input = var_input.to(self.device)
            var_output= self.model(var_input)
            trainloss_value = self.loss_fn(
                var_output,
                tfunc.one_hot(var_target.squeeze(1).long(), num_classes=self.class_count).float())

            self.optimizer.zero_grad()
            trainloss_value.backward()
            self.optimizer.step()
            train_loss_value = trainloss_value.item()
            loss_train_list.append(train_loss_value)

            if batch_id % (len(self.data_loader_train)-1) == 0 and batch_id != 0:
                validloss_value, auroc_mean = self.valid()
                loss_valid_list.append(validloss_value)
                if auroc_mean > self.auroc_max:
                    print('Better auroc obtained')
                    self.auroc_max = auroc_mean

                scheduler.step()

        train_loss_mean = np.mean(loss_train_list)
        valid_loss_mean = np.mean(loss_valid_list)
        return train_loss_mean, valid_loss_mean, auroc_mean

    def test(self):
        cudnn.benchmark = True
        out_gt = torch.FloatTensor().to(self.device)
        out_pred = torch.FloatTensor().to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i, (var_input, var_target) in enumerate(self.data_loader_test):
                var_target = var_target.to(self.device)
                var_input = var_input.to(self.device)
                out_gt = torch.cat((out_gt, var_target), 0).to(self.device)
                _, c, h, w = var_input.size()
                var_input = var_input.view(-1, c, h, w)
                out = self.model(var_input)
                out_pred = torch.cat((out_pred, out), 0)

        auroc_individual = compute_auroc(tfunc.one_hot(out_gt.squeeze(1).long()).float(), out_pred, self.class_count)
        auroc_mean = np.array(auroc_individual).mean()
        print(f'AUROC mean:{auroc_mean}')

        for i, auroc_val in enumerate(auroc_individual):
            print(f"{self.class_names[i]}:{auroc_val}")

        return auroc_mean

def main(args):
    lr = args.lr
    checkpoint = args.checkpoint
    batch_size = args.bs
    max_epoch = args.epochs
    class_count = args.clscount  #The objective is to classify the image into 3 classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available
    class_names = ['Lung Opacity', 'Normal', 'No Lung Opacity / Not Normal']

    # Data Loader
    dpath = args.dpath
    img_pth = os.path.join(args.dpath, 'processed_data/')
    numpy_path = os.path.join(args.dpath, 'data_split/')
    with open(os.path.join(dpath, 'rsna_annotation.json')) as lab_file:
        labels = json.load(lab_file)

    # Place numpy file containing train-valid-test split on tools folder

    tr_list = np.load(os.path.join(numpy_path,'train_list.npy')).tolist()
    val_list = np.load(os.path.join(numpy_path,'valid_list.npy')).tolist()
    test_list = np.load(os.path.join(numpy_path,'test_list.npy')).tolist()

    dataset_train = RSNADataSet(tr_list, labels, img_pth, transform=True)
    dataset_valid = RSNADataSet(val_list, labels, img_pth, transform=True)

    data_loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False)
    data_loader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False)

    dataset_test = RSNADataSet(test_list, labels, img_pth, transform=True)
    data_loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False)

    # Construct Model

    if args.optimised:
        alpha = args.alpha
        phi = args.phi
        beta = args.beta
        if beta is None:
            beta = round(sqrt(2 / alpha), 3)

        alpha = alpha ** phi
        beta = beta ** phi

        model = DenseNet121Eff(alpha, beta, class_count)
    else:
        model = DenseNet121(class_count)

    # Train the  Model
    savepath = args.spath

    rsna_trainer = RSNATrainer(
        model, data_loader_train, data_loader_valid, data_loader_test,
        class_count,checkpoint, device, class_names, lr)
    rsna_trainer.train(max_epoch, savepath)
    print("Model trained !")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
        required=False,
        help="Learning rate",
        default=1e-4,
        type = float)
    parser.add_argument("--checkpoint",
        required=False,
        help="Checkpoint model weight",
        default= None,
        type = str)
    parser.add_argument("--bs",
        required=False,
        default=16,
        help="Batchsize",
        type=int)
    parser.add_argument("--dpath",
        required=True,
        help="Path to folder containing all data",
        type =str)
    parser.add_argument("--epochs",
        required=False,
        default=15,
        help="Number of epochs",
        type=int)
    parser.add_argument("--clscount",
        required=False,
        default=3,
        help="Number of classes",
        type=int)
    parser.add_argument("--spath",
        required=True,
        help="Path to folder in which models should be saved",
        type =str)
    parser.add_argument("--optimised",
        required=False, default=False,
        help="enable flag->eff model",
        action='store_true')
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
    custom_args = parser.parse_args()
    main(custom_args)
