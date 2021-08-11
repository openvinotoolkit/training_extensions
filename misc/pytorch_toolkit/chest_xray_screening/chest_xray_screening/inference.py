import numpy as np
import argparse
import torch
from torch.backends import cudnn
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from math import sqrt
import json
import os
from .utils.dataloader import RSNADataSet
from .utils.score import compute_auroc
from .utils.model import DenseNet121,DenseNet121Eff



class RSNAInference():
    def __init__(self,model, data_loader_test, class_count, checkpoint, class_names, device):
        self.device = device
        self.model = model.to(self.device)
        self.data_loader_test = data_loader_test
        self.class_count = class_count
        self.checkpoint = checkpoint
        self.class_names = class_names

    def test(self):
        cudnn.benchmark = True
        if self.checkpoint is not None:
            model_checkpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(model_checkpoint['state_dict'])
        else:
            self.model.state_dict()

        out_gt = torch.FloatTensor().to(self.device)
        out_pred = torch.FloatTensor().to(self.device)
        self.model.eval()
        with torch.no_grad():
            for var_input, var_target in self.data_loader_test:
                var_target = var_target.to(self.device)
                var_input = var_input.to(self.device)
                out_gt = torch.cat((out_gt, var_target), 0).to(self.device)

                _, c, h, w = var_input.size()
                var_input = var_input.view(-1, c, h, w)
                out = self.model(var_input.to(self.device))
                out_pred = torch.cat((out_pred, out), 0)
        auroc_individual = compute_auroc(tfunc.one_hot(out_gt.squeeze(1).long()).float(), out_pred, self.class_count)
        auroc_mean = np.array(auroc_individual).mean()
        print (f'AUROC mean: {auroc_mean}')
        for i, auroc_val in enumerate(auroc_individual):
            print(f"{self.class_names[i]}:{auroc_val}")
        return auroc_mean

def main(args):
    checkpoint= args.checkpoint
    class_count = 3

    class_names = ['Lung Opacity','Normal','No Lung Opacity / Not Normal']
    dpath = args.dpath
    img_pth = os.path.join(args.dpath, 'processed_data/')
    numpy_path = os.path.join(args.dpath, 'data_split/')
    with open(os.path.join(dpath, 'rsna_annotation.json')) as lab_file:
        labels = json.load(lab_file)
    test_list = np.load(os.path.join(numpy_path,'test_list.npy')).tolist()

    dataset_test = RSNADataSet(test_list,labels,img_pth,transform=True)
    data_loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available

    rsna_inference = RSNAInference(model, data_loader_test, class_count, checkpoint, class_names, device)

    test_auroc = rsna_inference.test()
    print(f"Test AUROC is {test_auroc}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
        required=False,
        help="start training from a checkpoint model weight",
        default= None,
        type = str)
    parser.add_argument("--dpath",
        required=True,
        help="Path to dataset",
        type =str)
    parser.add_argument("--optimised",
        required=False,
        default=False,
        help="enable flag for eff model",
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
