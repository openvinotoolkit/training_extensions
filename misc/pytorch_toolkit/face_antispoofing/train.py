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
import torch.nn as nn

from trainer import Trainer
from utils import (Transform, build_criterion, build_model, make_dataset,
                   make_loader, make_weights, read_py_config)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether or not to save your model')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='Configuration file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='if you want to train model on cpu, pass "cpu" param')
    args = parser.parse_args()

    # manage device, arguments, reading config
    path_to_config = args.config
    config = read_py_config(path_to_config)
    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'
    if config.data_parallel.use_parallel:
        device = f'cuda:{config.data_parallel.parallel_params.output_device}'
    if config.multi_task_learning and config.dataset != 'celeba_spoof':
        raise NotImplementedError(
            'Note, that multi task learning is avaliable for celeba_spoof only. '
            'Please, switch it off in config file'
            )
    # launch training, validation, testing
    train(config, device, args.save_checkpoint)

def train(config, device='cuda:0', save_chkpt=True):
    ''' procedure launching all main functions of training,
        validation and testing pipelines'''
    # for pipeline testing purposes
    save_chkpt = False if config.test_steps else True
    # preprocessing data
    normalize = A.Normalize(**config.img_norm_cfg)
    train_transform_real = A.Compose([
                            A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                                                                intensity=(0.2, 0.5), p=0.2),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2,
                                                                                contrast_limit=0.2,
                                                                                brightness_by_max=True,
                                                                                always_apply=False, p=0.3),
                            A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.2),
                            normalize
                            ])
    train_transform_spoof = A.Compose([
                            A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                                                                intensity=(0.2, 0.5), p=0.2),
                            A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2,
                                                                                contrast_limit=0.2,
                                                                                brightness_by_max=True,
                                                                                always_apply=False, p=0.3),
                            A.augmentations.transforms.MotionBlur(blur_limit=5, p=0.2),
                            normalize
                            ])
    val_transform = A.Compose([
                A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                normalize
                ])

    # load data
    sampler = config.data.sampler
    if sampler:
        num_instances, weights = make_weights(config)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_instances, replacement=True)
    train_transform = Transform(train_spoof=train_transform_spoof,
                                train_real=train_transform_real, val=None)
    val_transform = Transform(train_spoof=None, train_real=None, val=val_transform)
    train_dataset, val_dataset, test_dataset = make_dataset(config, train_transform, val_transform)
    train_loader, val_loader, test_loader = make_loader(train_dataset, val_dataset,
                                                        test_dataset, config, sampler=sampler)

    # build model and put it to cuda and if it needed then wrap model to data parallel
    model = build_model(config, device=device, strict=False, mode='train')
    model.to(device)
    if config.data_parallel.use_parallel:
        model = torch.nn.DataParallel(model, **config.data_parallel.parallel_params)

    # build a criterion
    softmax = build_criterion(config, device, task='main').to(device)
    cross_entropy = build_criterion(config, device, task='rest').to(device)
    bce = nn.BCELoss().to(device)
    criterion = (softmax, cross_entropy, bce) if config.multi_task_learning else softmax

    # build optimizer and scheduler for it
    optimizer = torch.optim.SGD(model.parameters(), **config.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.scheduler)

    # create Trainer object and get experiment information
    trainer = Trainer(model, criterion, optimizer, device, config, train_loader, val_loader, test_loader)
    trainer.get_exp_info()

    # learning epochs
    for epoch in range(config.epochs.start_epoch, config.epochs.max_epoch):
        if epoch != config.epochs.start_epoch:
            scheduler.step()

        # train model for one epoch
        train_loss, train_accuracy = trainer.train(epoch)
        print(f'epoch: {epoch}  train loss: {train_loss}   train accuracy: {train_accuracy}')

        # validate your model
        accuracy = trainer.validate()

        # eval metrics such as AUC, APCER, BPCER, ACER on val and test dataset according to rule
        trainer.eval(epoch, accuracy, save_chkpt=save_chkpt)
        # for testing purposes
        if config.test_steps:
            exit()

    # evaluate in the end of training
    if config.evaluation:
        file_name = 'tests.txt'
        trainer.test(file_name=file_name)


if __name__=='__main__':
    main()
