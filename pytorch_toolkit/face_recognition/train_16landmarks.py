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

from datasets import IBUG

from model.common import models_landmarks
# from utils import landmarks_augmentation
from utils import landmarks_augmentation16
from utils.utils import save_model_cpu, load_model_state
from losses.alignment import AlignmentLoss
from evaluate_landmarks import evaluate


def train(args):
    """Launches training of landmark regression model"""

    drops_schedule = [2000]
    dataset = IBUG(args.train, args.t_land)
    val_dataset = IBUG(args.train, args.t_land, test=True)

    log.info('Use alignment for the train data')
    # dataset.transform = transforms.Compose([
    #                                         landmarks_augmentation16.Rescale((120, 120)),
    #                                         landmarks_augmentation16.Blur(k=3, p=.2),
    #                                         landmarks_augmentation16.HorizontalFlip(p=.5),
    #                                         landmarks_augmentation16.RandomRotate(30),
    #                                         landmarks_augmentation16.RandomScale(.8, .9, p=.4),
    #                                         landmarks_augmentation16.RandomCrop(112),
    #                                         landmarks_augmentation16.ToTensor(switch_rb=True)])
    dataset.transform = transforms.Compose([landmarks_augmentation16.Rescale((112, 112)),
                                            landmarks_augmentation16.HorizontalFlip(p=.5),
                                           landmarks_augmentation16.ToTensor(switch_rb=True)])
    val_dataset.transform = transforms.Compose([landmarks_augmentation16.Rescale((112, 112)),
                                                landmarks_augmentation16.ToTensor(switch_rb=True)])

    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=False)
    writer = SummaryWriter('./logs_landm/{:%Y_%m_%d_%H_%M}_'.format(datetime.datetime.now()) + args.snap_prefix)
    model = models_landmarks['mobilelandnet']()

    # print(model)

    if args.snap_to_resume is not None:
        log.info('Resuming snapshot ' + args.snap_to_resume + ' ...')
        model = load_model_state(model, args.snap_to_resume, args.device, eval_state=False)
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        cudnn.enabled = False
        cudnn.benchmark = False
    # else:
    #     model = torch.nn.DataParallel(model, device_ids=[0])
    #     model.cuda()
    #     model.train()
    #     cudnn.enabled = True
    #     cudnn.benchmark = True
    else:
        model.cuda(args.device)
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model.train()
        cudnn.enabled = False
        cudnn.benchmark = False

    log.info('Face landmarks model:')
    log.info(model)

    criterion = AlignmentLoss('l2')
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, drops_schedule)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, args.lr, 1.0, gamma=,mode='exp_range')
    for epoch_num in range(args.epoch_total_num):
        scheduler.step()
        # if epoch_num > 300:
        #     model.module.set_dropout_ratio(0.)
        for i, data in enumerate(train_loader, 0):
            iteration = epoch_num * len(train_loader) + i

            data, gt_landmarks = data['img'].cuda(), data['landmarks'].cuda()
            # print(gt_landmarks)
            predicted_landmarks = model(data)
            # print(predicted_landmarks)

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
    parser.add_argument('--train_batch_size', type=int, default=200, help='Train batch size.')
    parser.add_argument('--epoch_total_num', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--val_step', type=int, default=500, help='Evaluate model each val_step during each epoch.')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='Weight decay.')
    parser.add_argument('--device', '-d', default=0, type=int)
    parser.add_argument('--snap_folder', type=str, default='./snapshots/', help='Folder to save snapshots.')
    parser.add_argument('--snap_prefix', type=str, default='LandNet', help='Prefix for snapshots.')
    parser.add_argument('--snap_to_resume', type=str, default=None, help='Snapshot to resume.')
    parser.add_argument('--dataset', choices=['vgg', 'celeb', 'ngd', 'ibug'], type=str, default='ibug', help='Dataset.')
    arguments = parser.parse_args()

    with torch.cuda.device(arguments.device):
        train(arguments)

if __name__ == '__main__':
    main()
