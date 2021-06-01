import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from sklearn.metrics.ranking import roc_auc_score
from math import sqrt
import argparse
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


class RSNATester():

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

    alpha = args.alpha
    phi = args.phi
    beta = args.beta

    if beta is None:
        beta = round(sqrt(2 / alpha), 3)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # use gpu if available
    checkpoint = args.checkpoint

    class_names = ['Lung Opacity', 'Normal', 'No Lung Opacity / Not Normal']

    batch_size = args.bs
    max_epochs = args.epochs
    class_count = 3

    # Specify paths
    img_pth = args.imgpath
    tr_list = np.load('../tools/train_list.npy').tolist()
    tr_labels = np.load('../tools/train_labels.npy').tolist()
    val_list = np.load('../tools/valid_list.npy').tolist()
    val_labels = np.load('../tools/val_labels.npy').tolist()
    test_list = np.load('../tools/test_list.npy').tolist()
    test_labels = np.load('../tools/test_labels.npy').tolist()

    # Create dataloader

    dataset_train = RSNADataSet(tr_list, tr_labels, img_pth, transform=True)
    dataset_valid = RSNADataSet(val_list, val_labels, img_pth, transform=True)

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

    dataset_test = RSNADataSet(test_list, test_labels, img_pth, transform=True)
    data_loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False)

    # Generate Model

    model, total_macs = give_model(alpha, beta, class_count)
    print(f"{total_macs} is the number of macs.")
    model = nn.Sequential(model, nn.Sigmoid())
    model = model.to(device)

    # Test the  Model

    auroc_mean = RSNATester.test(
        model,
        data_loader_test,
        class_count,
        checkpoint,
        class_names,
        device)

    print(f"The average AUROC score is {auroc_mean}")


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

    args = parser.parse_args()

    main(args)
