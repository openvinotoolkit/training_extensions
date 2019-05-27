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

import argparse
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data.dataloader import DataLoader

from dataset import DatasetFromSingleImages
import train
import metrics
from models import make_model, MSE_loss

# Training settings
parser = argparse.ArgumentParser(description="Super Resolution PyTorch")
parser.add_argument("--scale", type=int, default=4, help="Upsampling factor for SR")
parser.add_argument("--model", default="SmallModel", type=str, choices=['SRResNetLight', 'SmallModel'], help="SR model")
parser.add_argument("--patch_size", nargs='+', default=[196, 196], type=int, help="Patch size used for training (None - whole image)")
parser.add_argument("--border", type=int, default=4, help="Ignored border")
parser.add_argument("--aug_resize_factor_range", nargs='+', default=[0.75, 1.1], type=float,
                    help="Range of resize factor for training patch, used for augmentation")
parser.add_argument("--num_of_train_images", type=int, default=None,
                    help="Number of training images (None - use all images)")
parser.add_argument("--num_of_patches_per_image", type=int, default=10, help="Number of patches from one image")
parser.add_argument("--num_of_val_images", type=int, default=None,
                    help="Number of val images (None - use all images)")
parser.add_argument("--resume", action='store_true', help="Resume training from the latest state")

parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_epochs", type=int, default=500, help="Number of epochs to train for")
parser.add_argument("--num_of_data_loader_threads", type=int, default=0,
                    help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--train_path", default="", type=str, help="Path to train data")
parser.add_argument("--validation_path", default="", type=str, help="Path to folder with val images")
parser.add_argument("--exp_name", default="test", type=str, help="Experiment name")
parser.add_argument("--models_path", default="/models", type=str, help="Path to models folder")
parser.add_argument("--seed", default=1337, type=int, help="Seed for random generators")
parser.add_argument('--milestones', nargs='+', default=[8, 12, 16], type=int,
                    help='List of epoch indices, where learning rate decay is applied')



def main():
    opt = parser.parse_args()
    print('Parameters:', opt)

    print("===> Building model")
    scale = opt.scale
    model = make_model(opt.model, scale)

    trainer = train.Trainer(model=model, name=opt.exp_name, models_root=opt.models_path, resume=opt.resume)

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromSingleImages(path=opt.train_path, patch_size=opt.patch_size,
                                        aug_resize_factor_range=opt.aug_resize_factor_range,
                                        scale=scale, count=opt.num_of_train_images, cache_images=False,
                                        seed=opt.seed, dataset_size_factor=opt.num_of_patches_per_image)

    val_set = DatasetFromSingleImages(path=opt.validation_path, patch_size=None, aug_resize_factor_range=None,
                                      scale=scale, count=opt.num_of_val_images, cache_images=False, seed=opt.seed)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.num_of_data_loader_threads,
                                      batch_size=opt.batch_size, shuffle=True, drop_last=True)

    batch_sampler = Data.BatchSampler(
        sampler=Data.SequentialSampler(val_set),
        batch_size=1,
        drop_last=True
    )

    evaluation_data_loader = DataLoader(dataset=val_set, num_workers=0,
                                        batch_sampler=batch_sampler)
    print("===> Building model")
    criterion = [MSE_loss(opt.border).cuda()]

    print("===> Training")

    trainer.train(criterion=criterion,
                  optimizer=optim.Adam,
                  optimizer_params={"lr": 1e-3},
                  scheduler=torch.optim.lr_scheduler.MultiStepLR,
                  scheduler_params={"milestones": opt.milestones},
                  training_data_loader=training_data_loader,
                  evaluation_data_loader=evaluation_data_loader,
                  pretrained_weights=None,
                  train_metrics=[metrics.PSNR(name='PSNR', border=opt.border),
                                 metrics.RMSE(name='RMSE', border=opt.border)],
                  val_metrics=[metrics.PSNR(name='PSNR', border=opt.border),
                               metrics.RMSE(name='RMSE', border=opt.border)],
                  track_metric='PSNR',
                  epoches=opt.num_of_epochs
                  )


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
