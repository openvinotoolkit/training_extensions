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

import glog as log
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as t
from tqdm import tqdm

from datasets import VGGFace2, CelebA, NDG

from model.common import models_landmarks
from utils.landmarks_augmentation import Rescale, ToTensor
from utils.utils import load_model_state


def evaluate(val_loader, model):
    """Calculates average error"""
    total_loss = 0.
    total_pp_error = 0.
    failures_num = 0
    items_num = 0
    for _, data in enumerate(tqdm(val_loader), 0):
        data, gt_landmarks = data['img'].cuda(), data['landmarks'].cuda()
        predicted_landmarks = model(data)
        loss = predicted_landmarks - gt_landmarks
        items_num += loss.shape[0]
        n_points = loss.shape[1] // 2
        per_point_error = loss.data.view(-1, n_points, 2)
        per_point_error = torch.norm(per_point_error, p=2, dim=2)
        avg_error = torch.sum(per_point_error, 1) / n_points
        eyes_dist = torch.norm(gt_landmarks[:, 0:2] - gt_landmarks[:, 2:4], p=2, dim=1).reshape(-1)

        per_point_error = torch.div(per_point_error, eyes_dist.view(-1, 1))
        total_pp_error += torch.sum(per_point_error, 0)

        avg_error = torch.div(avg_error, eyes_dist)
        failures_num += torch.nonzero(avg_error > 0.1).shape[0]
        total_loss += torch.sum(avg_error)

    return total_loss / items_num, (total_pp_error / items_num).data.cpu().numpy(), float(failures_num) / items_num


def start_evaluation(args):
    """Launches the evaluation process"""

    if args.dataset == 'vgg':
        dataset = VGGFace2(args.val, args.v_list, args.v_land, landmarks_training=True)
    elif args.dataset == 'celeb':
        dataset = CelebA(args.val, args.v_land, test=True)
    else:
        dataset = NDG(args.val, args.v_land)

    if dataset.have_landmarks:
        log.info('Use alignment for the train data')
        dataset.transform = t.Compose([Rescale((48, 48)), ToTensor(switch_rb=True)])
    else:
        exit()

    val_loader = DataLoader(dataset, batch_size=args.val_batch_size, num_workers=4, shuffle=False, pin_memory=True)

    model = models_landmarks['landnet']
    assert args.snapshot is not None
    log.info('Testing snapshot ' + args.snapshot + ' ...')
    model = load_model_state(model, args.snapshot, args.device, eval_state=True)
    model.eval()
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model, device_ids=[args.device], )

    log.info('Face landmarks model:')
    log.info(model)

    avg_err, per_point_avg_err, failures_rate = evaluate(val_loader, model)

    log.info('Avg RMSE error: {}'.format(avg_err))
    log.info('Per landmark RMSE error: {}'.format(per_point_avg_err))
    log.info('Failure rate: {}'.format(failures_rate))


def main():
    """Creates a cl parser"""
    parser = argparse.ArgumentParser(description='Evaluation script for landmarks detection network')
    parser.add_argument('--device', '-d', default=0, type=int)
    parser.add_argument('--val_data_root', dest='val', required=True, type=str, help='Path to val data.')
    parser.add_argument('--val_list', dest='v_list', required=False, type=str, help='Path to test data image list.')
    parser.add_argument('--val_landmarks', dest='v_land', default='', required=False, type=str,
                        help='Path to landmarks for test images.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size.')
    parser.add_argument('--snapshot', type=str, default=None, help='Snapshot to evaluate.')
    parser.add_argument('--dataset', choices=['vgg', 'celeb', 'ngd'], type=str, default='vgg', help='Dataset.')
    arguments = parser.parse_args()

    with torch.cuda.device(arguments.device):
        start_evaluation(arguments)

if __name__ == '__main__':
    main()
