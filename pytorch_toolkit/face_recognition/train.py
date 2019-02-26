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
import os
from pprint import pformat

import glog as log
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms as t
from tensorboardX import SummaryWriter

from datasets import LFW, VGGFace2, MSCeleb1M, IMDBFace, TrillionPairs

from losses.am_softmax import AMSoftmaxLoss
from losses.metric_losses import MetricLosses
from evaluate_lfw import evaluate, compute_embeddings_lfw

from utils.utils import load_model_state, save_model_cpu
import utils.augmentation as augm
from utils.parser_yaml import ArgumentParserWithYaml
from model.common import models_backbones

def train(args):
    """Performs training of a face recognition network"""
    input_size = models_backbones[args.model].get_input_res()
    if args.train_dataset == 'vgg':
        assert args.t_list
        dataset = VGGFace2(args.train, args.t_list, args.t_land)
    elif args.train_dataset == 'imdbface':
        dataset = IMDBFace(args.train, args.t_list)
    elif args.train_dataset == 'trp':
        dataset = TrillionPairs(args.train, args.t_list)
    else:
        dataset = MSCeleb1M(args.train, args.t_list)

    if dataset.have_landmarks:
        log.info('Use alignment for the train data')
        dataset.transform = t.Compose([augm.HorizontalFlipNumpy(p=.5),
                                       augm.CutOutWithPrior(p=0.05, max_area=0.1),
                                       augm.RandomRotationNumpy(10, p=.95),
                                       augm.ResizeNumpy(input_size),
                                       augm.BlurNumpy(k=5, p=.2),
                                       augm.NumpyToTensor(switch_rb=True)])
    else:
        dataset.transform = t.Compose([augm.ResizeNumpy(input_size),
                                       augm.HorizontalFlipNumpy(),
                                       augm.RandomRotationNumpy(10),
                                       augm.NumpyToTensor(switch_rb=True)])

    if args.weighted:
        train_weights = dataset.get_weights()
        train_weights = torch.DoubleTensor(train_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size,
                                                   sampler=sampler, num_workers=3, pin_memory=False)
    else:
        train_loader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True)

    lfw = LFW(args.val, args.v_list, args.v_land)
    if lfw.use_landmarks:
        log.info('Use alignment for the test data')
        lfw.transform = t.Compose([augm.ResizeNumpy(input_size),
                                   augm.NumpyToTensor(switch_rb=True)])
    else:
        lfw.transform = t.Compose([augm.ResizeNumpy((160, 160)),
                                   augm.CenterCropNumpy(input_size),
                                   augm.NumpyToTensor(switch_rb=True)])

    log_path = './logs/{:%Y_%m_%d_%H_%M}_{}'.format(datetime.datetime.now(), args.snap_prefix)
    writer = SummaryWriter(log_path)

    if not osp.exists(args.snap_folder):
        os.mkdir(args.snap_folder)

    model = models_backbones[args.model](embedding_size=args.embed_size,
                                         num_classes=dataset.get_num_classes(), feature=False)
    if args.snap_to_resume is not None:
        log.info('Resuming snapshot ' + args.snap_to_resume + ' ...')
        model = load_model_state(model, args.snap_to_resume, args.devices[0], eval_state=False)
        model = torch.nn.DataParallel(model, device_ids=args.devices)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.devices, output_device=args.devices[0])
        model.cuda()
        model.train()
        cudnn.benchmark = True

    log.info('Face Recognition model:')
    log.info(model)

    if args.mining_type == 'focal':
        softmax_criterion = AMSoftmaxLoss(gamma=args.gamma, m=args.m, margin_type=args.margin_type, s=args.s)
    else:
        softmax_criterion = AMSoftmaxLoss(t=args.t, m=0.35, margin_type=args.margin_type, s=args.s)
    aux_losses = MetricLosses(dataset.get_num_classes(), args.embed_size, writer)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [3, 6, 9, 13])
    for epoch_num in range(args.epoch_total_num):
        scheduler.step()
        if epoch_num > 6:
            model.module.set_dropout_ratio(0.)
        classification_correct = 0
        classification_total = 0

        for i, data in enumerate(train_loader, 0):
            iteration = epoch_num * len(train_loader) + i

            if iteration % args.val_step == 0:
                snapshot_name = osp.join(args.snap_folder, args.snap_prefix + '_{0}.pt'.format(iteration))
                if iteration > 0:
                    log.info('Saving Snapshot: ' + snapshot_name)
                    save_model_cpu(model, optimizer, snapshot_name, epoch_num)

                log.info('Evaluating Snapshot: ' + snapshot_name)
                model.eval()
                same_acc, diff_acc, all_acc, auc = evaluate(args, lfw, model, compute_embeddings_lfw,
                                                            args.val_batch_size, verbose=False)

                model.train()

                log.info('Validation accuracy: {0:.4f}, {1:.4f}'.format(same_acc, diff_acc))
                log.info('Validation accuracy mean: {0:.4f}'.format(all_acc))
                log.info('Validation AUC: {0:.4f}'.format(auc))
                writer.add_scalar('Accuracy/Val_same_accuracy', same_acc, iteration)
                writer.add_scalar('Accuracy/Val_diff_accuracy', diff_acc, iteration)
                writer.add_scalar('Accuracy/Val_accuracy', all_acc, iteration)
                writer.add_scalar('Accuracy/AUC', auc, iteration)

            data, label = data['img'], data['label'].cuda()
            features, sm_outputs = model(data)

            optimizer.zero_grad()
            aux_losses.init_iteration()
            aux_loss, aux_log = aux_losses(features, label, epoch_num, iteration)
            loss_sm = softmax_criterion(sm_outputs, label)
            loss = loss_sm + aux_loss
            loss.backward()
            aux_losses.end_iteration()
            optimizer.step()

            _, predicted = torch.max(sm_outputs.data, 1)
            classification_total += int(label.size(0))
            classification_correct += int(torch.sum(predicted.eq(label)))
            train_acc = float(classification_correct) / classification_total

            if i % 10 == 0:
                log.info('Iteration %d, Softmax loss: %.4f, Total loss: %.4f' % (iteration, loss_sm, loss) + aux_log)
                log.info('Learning rate: %f' % scheduler.get_lr()[0])
                writer.add_scalar('Loss/train_loss', loss, iteration)
                writer.add_scalar('Loss/softmax_loss', loss_sm, iteration)
                writer.add_scalar('Learning_rate', scheduler.get_lr()[0], iteration)
                writer.add_scalar('Accuracy/classification', train_acc, iteration)


def main():
    """Creates a command line parser and starts training"""
    parser = ArgumentParserWithYaml(description='Training Face Recognition in PyTorch',
                                    fromfile_prefix_chars='@',
                                    epilog="Please, note that you can parse parameters from a yaml file if \
                                    you add @<path_to_yaml_file> to command line")

    #datasets configuration
    parser.add_argument('--train_dataset', choices=['vgg', 'ms1m', 'trp', 'imdbface'],
                        type=str, default='vgg', help='Name of the train dataset.')
    parser.add_argument('--train_data_root', dest='train', required=True, type=str, help='Path to train data.')
    parser.add_argument('--train_list', dest='t_list', required=False, type=str, help='Path to train data image list.')
    parser.add_argument('--train_landmarks', default='', dest='t_land', required=False, type=str,
                        help='Path to landmarks for the train images.')

    parser.add_argument('--val_data_root', dest='val', required=True, type=str, help='Path to val data.')
    parser.add_argument('--val_step', type=int, default=1000, help='Evaluate model each val_step during each epoch.')
    parser.add_argument('--val_list', dest='v_list', required=True, type=str, help='Path to test data image list.')
    parser.add_argument('--val_landmarks', dest='v_land', default='', required=False, type=str,
                        help='Path to landmarks for test images.')

    #model configuration
    parser.add_argument('--model', choices=models_backbones.keys(), type=str, default='mobilenet', help='Model type.')
    parser.add_argument('--embed_size', type=int, default=256, help='Size of the face embedding.')

    #optimizer configuration
    parser.add_argument('--train_batch_size', type=int, default=170, help='Train batch size.')
    parser.add_argument('--epoch_total_num', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')

    #loss configuration
    parser.add_argument('--mining_type', choices=['focal', 'sv'],
                        type=str, default='sv', help='Hard mining method in loss.')
    parser.add_argument('--t', type=float, default=1.1, help='t in support vector softmax. See https://arxiv.org/abs/1812.11317 for details')
    parser.add_argument('--gamma', type=float, default=2., help='Gamma in focal loss. See https://arxiv.org/abs/1708.02002 for details')
    parser.add_argument('--m', type=float, default=0.35, help='Margin size for AMSoftmax.')
    parser.add_argument('--s', type=float, default=30., help='Scale for AMSoftmax.')
    parser.add_argument('--margin_type', choices=['cos', 'arc'],
                        type=str, default='cos', help='Margin type for AMSoftmax loss.')

    #other parameters
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='CUDA devices to use.')
    parser.add_argument('--val_batch_size', type=int, default=20, help='Validation batch size.')
    parser.add_argument('--snap_folder', type=str, default='./snapshots/', help='Folder to save snapshots.')
    parser.add_argument('--snap_prefix', type=str, default='FaceReidNet', help='Prefix for snapshots.')
    parser.add_argument('--snap_to_resume', type=str, default=None, help='Snapshot to resume.')
    parser.add_argument('--weighted', action='store_true')

    args = parser.parse_args()
    log.info('Arguments:\n' + pformat(args.__dict__))

    with torch.cuda.device(args.devices[0]):
        train(args)

if __name__ == '__main__':
    main()
