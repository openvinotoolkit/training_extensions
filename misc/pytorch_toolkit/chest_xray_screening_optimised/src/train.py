import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics.ranking import roc_auc_score
import argparse
from math import sqrt
from chest_xray_screening_optimised.tools.dataloader import RSNADataSet
from chest_xray_screening_optimised.tools.generate import *


def compute_auroc(data_gt, data_pred, class_count):
    """ Computes the area under ROC Curve
    data_gt: ground truth data
    data_pred: predicted data
    class_count: Number of classes

    """

    out_auroc_list = []

    data_np_gt = data_gt.cpu().numpy()
    data_np_pred = data_pred.cpu().numpy()

    for i in range(class_count):
        try:
            out_auroc_list.append(roc_auc_score(
                data_np_gt[:, i], data_np_pred[:, i]))
        except ValueError:
            out_auroc_list.append(0)
    return out_auroc_list


class RSNATrainer():

    def __init__(self) -> None:
        self.gepoch_id = 0
        self.model_val = ''
        # pass

    def train(
            self, model, data_loader_train, data_loader_valid, data_loader_test,
            class_count, max_epoch, timestamp_launch, checkpoint, lr,
            device, class_names):

        loss_fn = torch.nn.BCELoss()
        # Setting maximum AUROC value as zero
        auroc_max = 0.0
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        if checkpoint is not None:
            model_checkpoint = torch.load(checkpoint)
            model.load_state_dict(model_checkpoint['state_dict'])
            optimizer.load_state_dict(model_checkpoint['optimizer'])
            for param in model.parameters():
                param.requires_grad = True
            print(f"Model loaded")

        train_loss_min = 100000  # A random very high number
        valid_loss_min = 100000

        for epoch_id in range(0, max_epoch):

            print(f"Epoch {epoch_id+1}/{max_epoch}")
            self.gepoch_id = epoch_id

            train_loss, valid_loss, auroc_max = RSNATrainer.epoch_train(
                model, data_loader_train, data_loader_valid,
                optimizer, class_count, loss_fn, device, auroc_max)

            timestamp_end = time.strftime("%H%M%S-%d%m%Y")

            if train_loss < train_loss_min:
                train_loss_min = train_loss

            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss

            torch.save(
                {'epoch': epoch_id + 1,
                 'state_dict': model.state_dict(),
                 'best_loss': valid_loss_min,
                 'optimizer': optimizer.state_dict()},
                'models/m-epoch' + str(epoch_id) + '-' +
                timestamp_launch + '.pth.tar')

            test_auroc = RSNATrainer.test(model, data_loader_test, class_count,
                                          'models/m-epoch' +
                                          str(epoch_id) + '-' +
                                          timestamp_launch + '.pth.tar',
                                          class_names, device)

            print(
                f" Epoch:{epoch_id + 1}| EndTime:{timestamp_end}| TestAUROC: {test_auroc}| ValidAUROC: {auroc_max}")

    def valid(model, data_loader_valid, loss_fn, class_count, device):

        model.eval()
        loss_valid_r = 0
        valid_batches = 0      # Counter for valid batches

        out_gt = torch.FloatTensor().to(device)
        out_pred = torch.FloatTensor().to(device)

        with torch.no_grad():
            for i, (var_input, var_target) in enumerate(data_loader_valid):
                print(f"Batch {i} in Val")

                var_target = var_target.to(device)
                out_gt = torch.cat((out_gt, var_target), 0).to(device)

                _, c, h, w = var_input.size()
                var_input = var_input.view(-1, c, h, w)

                var_output = model(var_input.to(device))
                out_pred = torch.cat((out_pred, var_output), 0)

                lossvalue = loss_fn(
                    var_output, tfunc.one_hot(var_target.squeeze(1).long(),
                                              num_classes=class_count).float())

                loss_valid_r += lossvalue.item()
                valid_batches += 1

            valid_loss = loss_valid_r / valid_batches

            auroc_individual = compute_auroc(
                tfunc.one_hot(
                    out_gt.squeeze(1).long()).float(),
                out_pred,
                class_count)
            auroc_mean = np.array(auroc_individual).mean()

        return valid_loss, auroc_mean

    def epoch_train(
            self, model, data_loader_train, data_loader_valid,
            optimizer, class_count, loss_fn, device, auroc_max):

        loss_train_list = []
        loss_valid_list = []

        model.train()
        scheduler = StepLR(optimizer, step_size=6, gamma=0.002)

        for batch_id, (var_input, var_target) in enumerate(data_loader_train):

            epoch_id = self.gepoch_id

            var_target = var_target.to(device)
            var_input = var_input.to(device)
            var_output = model(var_input)

            trainloss_value = loss_fn(
                var_output,
                tfunc.one_hot(
                    var_target.squeeze(1).long(),
                    num_classes=class_count).float())

            optimizer.zero_grad()
            trainloss_value.backward()
            optimizer.step()

            train_loss_value = trainloss_value.item()
            loss_train_list.append(trainloss_value)

            if batch_id % 140 == 1:
                print(f"Batch No::{batch_id};Loss::{train_loss_value}")

            # Evaluate the performance of the model on validation_set
            # every 2500th iteration

            if batch_id % 2500 == 0 and batch_id != 0:
                print(f"batch_id::{batch_id}")

                validloss_value, auroc_mean = RSNATrainer.valid(
                    model, data_loader_valid, loss_valid_list,
                    loss_fn, class_count, device)

                loss_valid_list.append(validloss_value)

                print("\n")

                if auroc_mean > auroc_max:
                    print(f'Better auroc obtained')
                    auroc_max = auroc_mean

                    self.model_val = 'm-epoch' + str(epoch_id) + '-batch_id' + str(
                        batch_id) + '-aurocMean-' + str(auroc_mean) + '.pth.tar'

                    torch.save(
                        {'batch': batch_id + 1,
                         'state_dict': model.state_dict(),
                         'aucmean_loss': auroc_mean,
                         'optimizer': optimizer.state_dict()},
                        'models/m-epoch-' + str(epoch_id) +
                        '-batch_id-' + str(batch_id) +
                        '-aurocMean-' + str(auroc_mean) +
                        '.pth.tar')

                scheduler.step()

        train_loss_mean = np.mean(loss_train_list)
        valid_loss_mean = np.mean(loss_valid_list)

        return train_loss_mean, valid_loss_mean, auroc_max

    def test(model, data_loader_test, class_count,
             checkpoint, class_names, device):

        cudnn.benchmark = True
        if checkpoint is not None:
            model_checkpoint = torch.load(checkpoint)
            model.load_state_dict(model_checkpoint['state_dict'])
        else:
            model.state_dict()

        out_gt = torch.FloatTensor().to(device)
        out_pred = torch.FloatTensor().to(device)

        model.eval()

        with torch.no_grad():
            for i, (var_input, var_target) in enumerate(data_loader_test):

                var_target = var_target.to(device)
                out_gt = torch.cat((out_gt, var_target), 0).to(device)

                _, c, h, w = input.size()
                var_input = input.view(-1, c, h, w)

                out = model(var_input.to(device))
                out_pred = torch.cat((out_pred, out), 0)

        auroc_individual = compute_auroc(
            tfunc.one_hot(
                out_gt.squeeze(1).long()).float(),
            out_pred,
            class_count)
        auroc_mean = np.array(auroc_individual).mean()

        print('\nAUROC mean ', auroc_mean)

        for i in range(0, len(auroc_individual)):
            print(f" {class_names[i]}:{auroc_individual[i]}")

        return auroc_mean


def main(args):

    lr = args.lr  # Learning rate

    # Specifying the variables for EfficientNet based optimisation

    alpha = args.alpha
    phi = args.phi
    beta = args.beta
    alpha = alpha ** phi
    beta = beta ** phi

    if beta is None:
        beta = round(sqrt(2 / alpha), 3)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # use gpu if available
    checkpoint = args.checkpoint

    class_names = ['Lung Opacity', 'Normal', 'No Lung Opacity / Not Normal']

    batch_size = args.bs
    max_epochs = args.epochs
    class_count = 3

    # Dataset
    img_pth = args.imgpath

    tr_list = np.load('../tools/train_list.npy').tolist()
    tr_labels = np.load('../tools/train_labels.npy').tolist()
    val_list = np.load('../tools/valid_list.npy').tolist()
    val_labels = np.load('../tools/val_labels.npy').tolist()
    test_list = np.load('../tools/test_list.npy').tolist()
    test_labels = np.load('../tools/test_labels.npy').tolist()

    dataset_train = RSNADataSet(
        tr_list,
        tr_labels,
        img_pth,
        transform=True)

    dataset_valid = RSNADataSet(
        val_list,
        val_labels,
        img_pth,
        transform=True)

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

    dataset_test = RSNADataSet(
        test_list,
        test_labels,
        img_pth,
        transform=True)

    data_loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False)

    model, total_macs = give_model(alpha, beta, class_count)
    print(f"{total_macs} is the number of macs.")
    model = nn.Sequential(model, nn.Sigmoid())
    model = model.to(device)

    # Train the  Model

    timestamp_launch = time.strftime("%H%M%S - %d%m%Y")

    RSNATrainer.train(
        model, data_loader_train, data_loader_valid, data_loader_test, class_count,
        max_epochs, timestamp_launch, checkpoint, lr, device, class_names)

    print("Model trained !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        required=False,
        help="alpha for the model",
        default=(
            11 / 6),
        type=float)
    parser.add_argument(
        "--lr",
        required=False,
        help="The learning rate of the model",
        default=1e-4,
        type=float)
    parser.add_argument(
        "--phi",
        required=False,
        help="Phi for the model.",
        default=1.0,
        type=float)
    parser.add_argument(
        "--beta",
        required=False,
        help="Beta for the model.",
        default=None,
        type=float)
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="start training from a checkpoint model weight",
        default=None,
        type=str)
    parser.add_argument("--bs", required=False, help="Batchsize")
    parser.add_argument(
        "--imgpath",
        required=True,
        help="Path containing train and test images",
        type=str)
    parser.add_argument(
        "--epochs",
        required=False,
        default=6,
        help="Number of epochs",
        type=int)

    args = parser.parse_args()
    main(args)
