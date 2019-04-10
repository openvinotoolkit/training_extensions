"""
 Copyright (c) 2019 Intel Corporation

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
import logging
import os.path as osp
from pydoc import locate
import yaml

import torch

from segmentoly.data.dataparallel import collate
from segmentoly.data.transforms import *
from segmentoly.datasets.factory import get_dataset
from segmentoly.utils.lr_scheduler import MultiStepLRWithWarmUp
from segmentoly.utils.logging import setup_logging, TextLogger, TensorboardLogger
from segmentoly.utils.training_engine import DefaultMaskRCNNTrainingEngine

setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Main parameters
    parser.add_argument('--model', default='segmentoly.rcnn.model_zoo.resnet_fpn_mask_rcnn.ResNet50FPNMaskRCNN',
                        help='Path to model')
    parser.add_argument('--dataset', default='coco2017',
                        help='Dataset name')
    parser.add_argument('--load_cfg',
                        help='Path to loading a configuration file')
    parser.add_argument('--display_interval', type=int, default=20,
                        help='Display training info every N iterations')
    parser.add_argument('--nw', type=int, default=4, dest='num_workers',
                        help='Number of workers to load data')
    parser.add_argument('--bs', type=int, default=16, dest='batch_size',
                        help='Total batch size')
    parser.add_argument('--bs_per_gpu', type=int, default=2,  dest='batch_size_per_gpu',
                        help='Number of images per GPU')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from a checkpoint')
    parser.add_argument('--output_dir', default=osp.join(osp.dirname(osp.realpath(__file__)), '../outputs'),
                        help='Directory with output data')
    parser.add_argument('--load_ckpt',
                        help='Checkpoint path to load')
    parser.add_argument('--load_backbone',
                        help='Path to backbone weights file')
    parser.add_argument('--save_cfg',
                        help='Save current configuration to file')
    parser.add_argument('--checkpoint_interval', type=int, default=10000,
                        help='Save checkpoints every N iterations')
    parser.add_argument('--test_interval', type=int, default=10000,
                        help='Test net every N iterations')
    # Optimizer parameters
    parser.add_argument('--lr', default=0.02, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum')
    # Warmup parameters
    parser.add_argument('--warmup_iters', default=1000, type=int,
                        help='Number of iterations for warm up')
    parser.add_argument('--warmup_factor', default=0.33, type=float,
                        help='Warm up factor')
    parser.add_argument('--warmup_method', default='linear', choices=['linear', 'constant'],
                        help='Warm up method')
    # Schedule parameters
    parser.add_argument('--max_iter', default=90000, type=int,
                        help='Maximum number of iterations')
    parser.add_argument('--drop_lr', default=(60000, 80000), nargs='+', type=int, metavar='<num>',
                        help='Milestones to drop learning rate')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma for learning rate decay')
    # Input image parameters
    parser.add_argument('--max_image_size', default=(800, 1333), type=int, nargs=2, metavar='<num>',
                        help='Max processed image size in a format (max short side, max long side)')
    parser.add_argument('--mean_pixel', default=(102.9801, 115.9465, 122.7717), type=float, nargs=3, metavar='<num>',
                        help='Mean pixel value to subtract from image')

    args = parser.parse_args()

    # Load config
    if args.load_cfg:
        logging.info('Loading configuration file "{}"...'.format(args.load_cfg))
        with open(args.load_cfg) as f:
            config = yaml.load(f)
        for k, v in config.items():
            v = None if v == 'None' else v
            setattr(args, k, v)

    # Save config
    if args.save_cfg and not args.load_cfg:
        with open(args.save_cfg, 'w') as f:
            config = {}
            for arg in vars(args):
                config[arg] = getattr(args, arg)
            yaml.dump(config, f)
        logging.info('Configuration file saved to "{}"'.format(args.save_cfg))

    args_line = ''
    for arg in vars(args):
        args_line += '\n -' + str(arg) + ': ' + str(getattr(args, arg))
    logging.info('Called with args: {}'.format(args_line))

    return args


def main(args):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    train_tool = DefaultMaskRCNNTrainingEngine()

    train_tool.identifier = 'MaskRCNN'
    train_tool.description = ''

    train_tool.set_random_seeds()

    train_tool.root_directory = osp.join(args.output_dir, train_tool.identifier)
    train_tool.run_directory = train_tool.create_run_directory(train_tool.root_directory)

    train_tool.batch_size = args.batch_size
    train_tool.virtual_iter_size = \
        train_tool.adjust_virtual_iteration_size(num_gpus, args.batch_size, args.batch_size_per_gpu)

    transforms = Compose(
        [
            Resize(max_size=args.max_image_size),
            ToTensor(),
            Normalize(mean=args.mean_pixel, std=[1., 1., 1.], rgb=False),
        ]
    )

    if args.dataset == 'coco2017':
        train_dataset_name = 'coco_2017_train'
        val_dataset_name = 'coco_2017_val'
    else:
        raise ValueError('Invalid dataset name "{}"'.format(args.dataset))

    train_dataset = get_dataset(train_dataset_name, True, True, transforms)
    val_dataset = get_dataset(val_dataset_name, False, False, transforms)
    assert train_dataset.classes_num == val_dataset.classes_num

    train_tool.training_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate
    )

    train_tool.validation_data_loaders = [
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=collate
        )
    ]

    train_tool.validate_every = args.test_interval

    train_tool.model = locate(args.model)(train_dataset.classes_num)

    train_tool.training_iterations_num = args.max_iter
    train_tool.lr_scheduler_milestones = args.drop_lr

    params = train_tool.setup_optimizer(train_tool.model, args.lr, args.weight_decay)
    train_tool.optimizer = torch.optim.SGD(params, momentum=args.momentum)

    start_step = 0
    if args.load_ckpt or args.load_backbone:
        start_step = train_tool.load_checkpoint(train_tool.model, train_tool.optimizer,
                                                args.load_ckpt, args.load_backbone, args.resume)

    train_tool.lr_scheduler = MultiStepLRWithWarmUp(
        train_tool.optimizer,
        milestones=args.drop_lr,
        warmup_iters=args.warmup_iters,
        warmup_method=args.warmup_method,
        warmup_factor_base=args.warmup_factor,
        gamma=args.gamma,
        last_epoch=start_step
    )

    text_log = TextLogger(logging.getLogger(__name__))
    tensorboard_log = TensorboardLogger(train_tool.run_directory)
    train_tool.loggers = [text_log, tensorboard_log]

    train_tool.log_every = args.display_interval

    train_tool.checkpoint_every = args.checkpoint_interval

    # Begin training
    train_tool.run(start_step)


if __name__ == '__main__':
    setup_logging()
    args = parse_args()
    main(args)
