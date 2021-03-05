"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import os

import albumentations as A
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (Transform, build_model, load_checkpoint, make_dataset,
                   read_py_config)


def main():
    # parsing arguments
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--draw_graph', default=False, type=bool, required=False,
                        help='whether or not to draw graphics')
    parser.add_argument('--GPU', default=0, type=int, required=False,
                        help='specify which GPU to use')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='if you want to eval model on cpu, pass "cpu" param')
    args = parser.parse_args()

    # reading config and manage device
    path_to_config = args.config
    config = read_py_config(path_to_config)
    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'

    # building model
    model = build_model(config, device, strict=True, mode='eval')
    model.to(device)
    if config.data_parallel.use_parallel:
        model = nn.DataParallel(model, **config.data_parallel.parallel_params)

    # load snapshot
    path_to_experiment = os.path.join(config.checkpoint.experiment_path, config.checkpoint.snapshot_name)
    epoch_of_checkpoint = load_checkpoint(path_to_experiment, model, map_location=device, optimizer=None)

    # preprocessing, making dataset and loader
    normalize = A.Normalize(**config.img_norm_cfg)
    test_transform = A.Compose([
                                A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                                normalize
                               ])
    test_transform = Transform(val=test_transform)
    test_dataset = make_dataset(config, val_transform=test_transform, mode='eval')
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True, num_workers=2)

    # computing metrics
    auc_, eer, accur, apcer, bpcer, acer, fpr, tpr  = evaluate(model, test_loader,
                                                               config, device,
                                                               compute_accuracy=True)
    print((f'eer = {round(eer*100,2)}\n'
           + f'accuracy on test data = {round(np.mean(accur),3)}\n'
           + f'auc = {round(auc_,3)}\n'
           + f'apcer = {round(apcer*100,2)}\n'
           + f'bpcer = {round(bpcer*100,2)}\n'
           + f'acer = {round(acer*100,2)}\n'
           + f'checkpoint made on {epoch_of_checkpoint} epoch'))

    # draw graphics if needed
    if args.draw_graph:
        fnr = 1 - tpr
        plot_roc_curve(fpr, tpr, config)
        det_curve(fpr, fnr, eer, config)

def evaluate(model, loader, config, device, compute_accuracy=True):
    ''' evaluating AUC, EER, BPCER, APCER, ACER on given data loader and model '''
    model.eval()
    proba_accum = np.array([])
    target_accum = np.array([])
    accur=[]
    tp, tn, fp, fn = 0, 0, 0, 0
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (image, target) in loop:
        if config.test_steps == i:
            break
        image = image.to(device)
        if len(target.shape) > 1:
            target = target[:, 0].reshape(-1).to(device)
        with torch.no_grad():
            features = model(image)
            if config.data_parallel.use_parallel:
                model1 = model.module
            else:
                model1 = model
            output = model1.make_logits(features, all=False)
            if isinstance(output, tuple):
                output = output[0]

            y_true = target.detach().cpu().numpy()
            y_pred = output.argmax(dim=1).detach().cpu().numpy()
            tn_batch, fp_batch, fn_batch, tp_batch = metrics.confusion_matrix(y_true=y_true,
                                                                              y_pred=y_pred,
                                                                              ).ravel()
            tp += tp_batch
            tn += tn_batch
            fp += fp_batch
            fn += fn_batch

            if compute_accuracy:
                accur.append((y_pred == y_true).mean())
            if config.loss.amsoftmax.margin_type in ('cos', 'arcos'):
                output *= config.loss.amsoftmax.s
            if config.loss.loss_type == 'soft_triple':
                output *= config.loss.soft_triple.s
            positive_probabilities = F.softmax(output, dim=-1)[:,1].cpu().numpy()
        proba_accum = np.concatenate((proba_accum, positive_probabilities))
        target_accum = np.concatenate((target_accum, y_true))

    apcer = fp / (tn + fp) if (tn + fp) != 0 else 0
    bpcer = fn / (fn + tp) if (fn + tp) != 0 else 0
    acer = (apcer + bpcer) / 2

    fpr, tpr, _ = roc_curve(target_accum, proba_accum, pos_label=1)
    fnr = 1 - tpr
    fpr_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    fnr_eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = min(fpr_eer, fnr_eer)
    auc_ = auc(fpr, tpr)
    to_return = ((auc_, eer, accur, apcer, bpcer, acer, fpr, tpr)
                if compute_accuracy
                else (auc_, eer, apcer, bpcer, acer))
    return to_return

def plot_roc_curve(fpr, tpr, config):
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.00])
    plt.plot(fpr, tpr, lw=3, label="ROC curve (area= {:0.2f})".format(auc(fpr, tpr)))
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0,1],[0,1], lw=3, linestyle='--', color='navy')
    plt.savefig(config.curves.det_curve)

def det_curve(fps,fns, eer, config):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    fig,ax = plt.subplots(figsize=(8,8))
    plt.plot(fps,fns, label=f"DET curve, EER%={round(eer*100, 3)}")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('FAR', fontsize=16)
    plt.ylabel('FRR', fontsize=16)
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.xticks(rotation=45)
    plt.axis([0.001,1,0.001,1])
    plt.title('DET curve', fontsize=20)
    plt.legend(loc='upper right', fontsize=16)
    fig.savefig(config.curves.det_curve)

if __name__ == "__main__":
    main()
