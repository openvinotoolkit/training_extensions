"""
 Copyright (c) 2018 Intel Corporation
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
import datetime
import os.path as osp

import numpy as np
import glog as log
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets import VGGFace2, CelebA, NDG

from model.common import models_landmarks
from utils import landmarks_augmentation
from utils.utils import save_model_cpu, load_model_state
from losses.alignment import AlignmentLoss
from evaluate_landmarks import evaluate


def train(args):
    """Launches training of landmark regression model"""
    if args.dataset == 'vgg':
        drops_schedule = [1, 6, 9, 13]
        dataset = VGGFace2(args.train, args.t_list, args.t_land, landmarks_training=True)
    elif args.dataset == 'celeba':
        drops_schedule = [10, 20]
        dataset = CelebA(args.train, args.t_land)
    else:
        drops_schedule = [90, 140, 200]
        dataset = NDG(args.train, args.t_land)

    if dataset.have_landmarks:
        log.info('Use alignment for the train data')
        dataset.transform = transforms.Compose([landmarks_augmentation.Rescale((56, 56)),
                                                landmarks_augmentation.Blur(k=3, p=.2),
                                                landmarks_augmentation.HorizontalFlip(p=.5),
                                                landmarks_augmentation.RandomRotate(50),
                                                landmarks_augmentation.RandomScale(.8, .9, p=.4),
                                                landmarks_augmentation.RandomCrop(48),
                                                landmarks_augmentation.ToTensor(switch_rb=True)])
    else:
        log.info('Error: training dataset has no landmarks data')
        exit()

    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True)
    writer = SummaryWriter('./logs_landm/{:%Y_%m_%d_%H_%M}_'.format(datetime.datetime.now()) + args.snap_prefix)
    model = models_landmarks['landnet']

    if args.snap_to_resume is not None:
        log.info('Resuming snapshot ' + args.snap_to_resume + ' ...')
        model = load_model_state(model, args.snap_to_resume, args.device, eval_state=False)
        model = torch.nn.DataParallel(model, device_ids=[args.device])
    else:
        model = torch.nn.DataParallel(model, device_ids=[args.device])
        model.cuda()
        model.train()
        cudnn.enabled = True
        cudnn.benchmark = True

    log.info('Face landmarks model:')
    log.info(model)

    criterion = AlignmentLoss('wing')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, drops_schedule)
    for epoch_num in range(args.epoch_total_num):
        scheduler.step()
        if epoch_num > 5:
            model.module.set_dropout_ratio(0.)
        for i, data in enumerate(train_loader, 0):
            iteration = epoch_num * len(train_loader) + i

            data, gt_landmarks = data['img'].cuda(), data['landmarks'].cuda()
            predicted_landmarks = model(data)

            optimizer.zero_grad()
            loss = criterion(predicted_landmarks, gt_landmarks)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                log.info('Iteration %d, Loss: %.4f' % (iteration, loss))
                log.info('Learning rate: %f' % scheduler.get_lr()[0])
                writer.add_scalar('Loss/train_loss', loss.item(), iteration)
                writer.add_scalar('Learning_rate', scheduler.get_lr()[0], iteration)

            if iteration % args.val_step == 0:
                snapshot_name = osp.join(args.snap_folder, args.snap_prefix + '_{0}.pt'.format(iteration))
                log.info('Saving Snapshot: ' + snapshot_name)
                save_model_cpu(model, optimizer, snapshot_name, epoch_num)

                model.eval()
                log.info('Evaluating Snapshot: ' + snapshot_name)
                avg_err, per_point_avg_err, failures_rate = evaluate(train_loader, model)
                weights = per_point_avg_err / np.sum(per_point_avg_err)
                criterion.set_weights(weights)
                log.info(str(weights))
                log.info('Avg train error: {}'.format(avg_err))
                log.info('Train failure rate: {}'.format(failures_rate))
                writer.add_scalar('Quality/Avg_error', avg_err, iteration)
                writer.add_scalar('Quality/Failure_rate', failures_rate, iteration)
                model.train()

def main():
    """Creates a command line parser"""
    parser = argparse.ArgumentParser(description='Training Landmarks detector in PyTorch')
    parser.add_argument('--train_data_root', dest='train', required=True, type=str, help='Path to train data.')
    parser.add_argument('--train_list', dest='t_list', required=False, type=str, help='Path to train data image list.')
    parser.add_argument('--train_landmarks', default='', dest='t_land', required=False, type=str,
                        help='Path to landmarks for the train images.')
    parser.add_argument('--train_batch_size', type=int, default=170, help='Train batch size.')
    parser.add_argument('--epoch_total_num', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--val_step', type=int, default=2000, help='Evaluate model each val_step during each epoch.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--device', '-d', default=0, type=int)
    parser.add_argument('--snap_folder', type=str, default='./snapshots/', help='Folder to save snapshots.')
    parser.add_argument('--snap_prefix', type=str, default='LandmarksNet', help='Prefix for snapshots.')
    parser.add_argument('--snap_to_resume', type=str, default=None, help='Snapshot to resume.')
    parser.add_argument('--dataset', choices=['vgg', 'celeb', 'ngd'], type=str, default='vgg', help='Dataset.')
    arguments = parser.parse_args()

    with torch.cuda.device(arguments.device):
        train(arguments)

if __name__ == '__main__':
    main()
