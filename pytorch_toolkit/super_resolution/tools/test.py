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
import time
import torch.utils.data as Data
from tqdm import tqdm
from sr.metrics import PSNR
from sr.dataset import DatasetFromSingleImages, DatasetTextImages
from sr.trainer import Trainer
from sr.common import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SR test')
    parser.add_argument('--test_data_path', default='', type=str, help='path to test data')
    parser.add_argument('--exp_name', default='test', type=str, help='experiment name')
    parser.add_argument('--models_path', default='./models', type=str, help='path to models folder')
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(os.path.join(args.models_path, args.exp_name))

    if config['model'] == 'TextTransposeModel':
        test_set = DatasetTextImages(path=args.test_data_path, patch_size=None,
                                     aug_resize_factor_range=None, scale=config['scale'])
    else:
        test_set = DatasetFromSingleImages(path=args.test_data_path, patch_size=None,
                                           aug_resize_factor_range=None, scale=config['scale'])

    batch_sampler = Data.BatchSampler(
        sampler=Data.SequentialSampler(test_set),
        batch_size=1,
        drop_last=True
    )

    evaluation_data_loader = Data.DataLoader(dataset=test_set, num_workers=0, batch_sampler=batch_sampler)

    trainer = Trainer(name=args.exp_name, models_root=args.models_path, resume=True)
    trainer.load_best()

    psnr = PSNR(name='PSNR', border=config['border'])

    tic = time.time()
    count = 0
    for batch in tqdm(evaluation_data_loader):
        output = trainer.predict(batch=batch)
        psnr.update(batch[1], output)
        count += 1

    toc = time.time()

    print('FPS: {}, SAMPLES: {}'.format(float(count) / (toc - tic), count))
    print('PSNR: {}'.format(psnr.get()))


if __name__ == '__main__':
    main()
