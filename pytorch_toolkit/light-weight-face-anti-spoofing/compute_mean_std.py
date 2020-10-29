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

import albumentations as A
import cv2 as cv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CelebASpoofDataset
from utils import Transform


def main():
    parser = argparse.ArgumentParser(description='mean and std computing')
    parser.add_argument('--root', type=str, default=None, required=True,
                        help='path to root folder of the CelebA_Spoof')
    parser.add_argument('--img_size', type=tuple, default=(128,128), required=False,
                        help='height and width of the image to resize')
    args = parser.parse_args()
    # transform image
    transforms = A.Compose([
                                A.Resize(*args.img_size, interpolation=cv.INTER_CUBIC),
                                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
                                ])
    root_folder = args.root
    train_dataset = CelebASpoofDataset(root_folder, test_mode=False,
                                       transform=Transform(transforms),
                                       multi_learning=False)
    dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    mean, std = compute_mean_std(dataloader)
    print(f'mean:{mean}, std:{std}')

def compute_mean_std(loader):
    ''' based on next formulas: E[x] = sum(x*p) = sum(x)/N, D[X] = E[(X-E(X))**2] = E[X**2] - (E[x])**2'''
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader, leave=False):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean, std

if __name__=="__main__":
    main()
