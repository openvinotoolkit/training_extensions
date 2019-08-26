#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import argparse
import random
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data.dataloader import DataLoader
import numpy as np

from sr.dataset import DatasetFromSingleImages, DatasetTextImages
from sr.trainer import Trainer
from sr.metrics import PSNR, RMSE
from sr.models import make_model, MSE_loss
from sr.common import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Super Resolution PyTorch')
    parser.add_argument('--config', help='Path to config file')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print('Config:', config)

    print('===> Building model')
    scale = config['scale']
    model = make_model(config['model'], scale)

    if config['init_checkpoint']:
        s = torch.load(config['init_checkpoint'])
        model.load_state_dict(s['model'].state_dict())

    trainer = Trainer(model=model, name=config['exp_name'],
                      models_root=config['models_path'], resume=config['resume'])

    # Copy config in train directory agter create trainer object
    model_path = os.path.join(config['models_path'], config['exp_name'])
    shutil.copyfile(args.config, os.path.join(model_path, 'config.yaml'))

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    cudnn.benchmark = True

    print('===> Loading datasets')

    if config['model'] == 'TextTransposeModel':
        train_set = DatasetTextImages(path=config['train_path'],
                                      patch_size=config['patch_size'],
                                      aug_resize_factor_range=config['aug_resize_factor_range'],
                                      scale=scale,
                                      seed=config['seed'],
                                      dataset_size_factor=config['num_of_patches_per_image'],
                                      rotate=config['rotate'])

        val_set = DatasetTextImages(path=config['validation_path'],
                                    patch_size=None,
                                    aug_resize_factor_range=None,
                                    scale=scale,
                                    seed=config['seed'])

    else:
        train_set = DatasetFromSingleImages(path=config['train_path'],
                                            patch_size=config['patch_size'],
                                            aug_resize_factor_range=config['aug_resize_factor_range'],
                                            scale=scale,
                                            count=config['num_of_train_images'],
                                            cache_images=False,
                                            seed=config['seed'],
                                            dataset_size_factor=config['num_of_patches_per_image'])

        val_set = DatasetFromSingleImages(path=config['validation_path'],
                                          patch_size=None,
                                          aug_resize_factor_range=None,
                                          scale=scale,
                                          count=config['num_of_val_images'],
                                          cache_images=False,
                                          seed=config['seed'])

    training_data_loader = DataLoader(dataset=train_set, num_workers=config['num_of_data_loader_threads'],
                                      batch_size=config['batch_size'], shuffle=True, drop_last=True)

    batch_sampler = Data.BatchSampler(
        sampler=Data.SequentialSampler(val_set),
        batch_size=1,
        drop_last=True
    )

    evaluation_data_loader = DataLoader(dataset=val_set, num_workers=0,
                                        batch_sampler=batch_sampler)
    print('===> Building model')
    criterion = [MSE_loss(config['border']).cuda()]

    print('===> Training')

    trainer.train(criterion=criterion,
                  optimizer=optim.Adam,
                  optimizer_params={'lr': 1e-3},
                  scheduler=torch.optim.lr_scheduler.MultiStepLR,
                  scheduler_params={'milestones': config['milestones']},
                  training_data_loader=training_data_loader,
                  evaluation_data_loader=evaluation_data_loader,
                  pretrained_weights=None,
                  train_metrics=[PSNR(name='PSNR', border=config['border']),
                                 RMSE(name='RMSE', border=config['border'])],
                  val_metrics=[PSNR(name='PSNR', border=config['border']),
                               RMSE(name='RMSE', border=config['border'])],
                  track_metric='PSNR',
                  epoches=config['num_of_epochs']
                  )


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
