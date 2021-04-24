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
import inspect
import os.path as osp
import sys

import cv2 as cv
import glog as log
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from demo_tools import TorchCNN, VectorCNN


def main():
    """Prepares data for the accuracy convertation checker"""
    parser = argparse.ArgumentParser(description='antispoofing recognition live demo script')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='Configuration file')
    parser.add_argument('--spf_model_openvino', type=str, default=None,
                        help='path to .xml IR OpenVINO model', required=True)
    parser.add_argument('--spf_model_torch', type=str, default=None,
                        help='path to .pth.tar checkpoint', required=True)
    parser.add_argument('--device', type=str, default='CPU')

    args = parser.parse_args()
    config = utils.read_py_config(args.config)
    assert args.spf_model_openvino.endswith('.xml') and (args.spf_model_torch.endswith('.pth.tar')
                                                            or args.spf_model_torch.endswith('.pth'))
    spoof_model_torch = utils.build_model(config, args.device.lower(), strict=True, mode='eval')
    spoof_model_torch = TorchCNN(spoof_model_torch, args.spf_model_torch, config, device=args.device.lower())
    spoof_model_openvino = VectorCNN(args.spf_model_openvino)
    # running checker
    avg_diff = run(spoof_model_torch, spoof_model_openvino)
    print((f'mean difference on the first predicted class : {avg_diff[0]}\n'
           + f'mean difference on the second predicted class : {avg_diff[1]}'))

def pred_spoof(batch, spoof_model_torch, spoof_model_openvino):
    """Get prediction for all detected faces on the frame"""
    output1 = spoof_model_torch.forward(batch)
    output1 = list(map(lambda x: x.reshape(-1), output1))
    output2 = spoof_model_openvino.forward(batch)
    output2 = list(map(lambda x: x.reshape(-1), output2))
    return output1, output2

def check_accuracy(torch_pred, openvino_pred):
    diff = np.abs(np.array(openvino_pred) - np.array(torch_pred))
    avg = diff.mean(axis=0)
    return avg

def run(spoof_model_torch, spoof_model_openvino):
    batch = np.float32(np.random.rand(100,128,128,3))
    torch_pred, openvino_pred = pred_spoof(batch, spoof_model_torch, spoof_model_openvino)
    avg_diff = check_accuracy(torch_pred, openvino_pred)
    return avg_diff

if __name__ == '__main__':
    main()
