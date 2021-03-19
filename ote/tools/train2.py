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
import logging
import os

from ote.interfaces.parameters import BaseTaskParameters
from ote.tasks.classification.dataset import ClassificationImageFolder
from ote.tasks.classification.task import ClassificationTask
from ote.utils import load_config
from ote.utils.misc import download_snapshot_if_not_yet, run_through_shell


def build_train_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-ann-files', required=True,
                        help='Comma-separated paths to training annotation files.')
    parser.add_argument('--train-data-roots', required=True,
                        help='Comma-separated paths to training data folders.')
    parser.add_argument('--val-ann-files', required=True,
                        help='Comma-separated paths to validation annotation files.')
    parser.add_argument('--val-data-roots', required=True,
                        help='Comma-separated paths to validation data folders.')
    parser.add_argument('--resume-from', default='',
                        help='Resume training from previously saved checkpoint')
    parser.add_argument('--load-weights', default='',
                        help='Load only weights from previously saved checkpoint')
    parser.add_argument('--save-checkpoints-to', default='/tmp/checkpoints',
                        help='Location where checkpoints will be stored')
    parser.add_argument('--batch-size', type=int,
                        default=10,
                        help='Size of a single batch during training per GPU.')
    parser.add_argument('--gpu-num', type=int,
                        default=1,
                        help='Number of GPUs that will be used in training, 0 is for CPU mode.')
    parser.add_argument('--tensorboard-dir',
                        help='Location where tensorboard logs will be stored.')
    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='Number of epochs during training')
    parser.add_argument('--base-learning-rate', type=float,
                        default=0.01,
                        help='Starting value of learning rate that might be changed during '
                                'training according to learning rate schedule that is usually '
                                'defined in detailed training configuration.')
    parser.add_argument('--template', default='', help=argparse.SUPPRESS)
    parser.add_argument('--work-dir', default='./logs', help=argparse.SUPPRESS)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    arg_parser = build_train_arg_parser()
    args = arg_parser.parse_args()


    template_config = load_config(args.template)
    os.makedirs(args.work_dir, exist_ok=True)
    download_snapshot_if_not_yet(args.template, args.work_dir)

    train_dataset = ClassificationImageFolder(args.train_data_roots)
    val_dataset = ClassificationImageFolder(args.val_data_roots)

    env_params = BaseTaskParameters.BaseEnvironmentParameters()
    env_params.snapshot_path = args.load_weights
    env_params.config_path = os.path.join(os.path.dirname(args.template), template_config['config'])
    env_params.gpus_num = args.gpu_num
    env_params.work_dir = args.work_dir

    train_params = BaseTaskParameters.BaseTrainingParameters()
    train_params.batch_size = args.batch_size
    train_params.learning_rate = args.base_learning_rate

    task = ClassificationTask(env_params)
    task.train(train_dataset, val_dataset, train_params)


if __name__ == '__main__':
    main()
