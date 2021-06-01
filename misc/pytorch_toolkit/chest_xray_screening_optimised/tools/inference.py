import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from math import sqrt
import argparse
from utils.dataloader import RSNADataSet
from utils.generate import *
from utils.score import compute_auroc


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

    alpha = alpha ** phi
    beta = beta ** phi


    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # use gpu if available
    checkpoint = args.checkpoint

    class_names = ['Lung Opacity', 'Normal', 'No Lung Opacity / Not Normal']

    class_count = 3

    # Specify paths
    img_pth = args.imgpath
    np_path = args.npypath
    test_list = np.load(np_path+'/test_list.npy').tolist()
    test_labels = np.load(np_path+'/test_labels.npy').tolist()

    # Create dataloader

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
