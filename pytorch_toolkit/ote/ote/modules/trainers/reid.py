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

from pathlib import Path
import logging
import os
import yaml
import sys

import torch

from ote import REID_TOOLS
from ote.utils import run_with_termination

from .base import BaseTrainer
from ..registry import TRAINERS


@TRAINERS.register_module()
class ReidTrainer(BaseTrainer):
    parameter_train_dir = 'train_data_roots'
    parameter_val_dir = 'val_data_roots'

    def __init__(self):
        super(ReidTrainer, self).__init__()

    def _get_tools_dir(self):
        return REID_TOOLS

    def __call__(self, config, gpu_num, out, update_config, tensorboard_dir):
        logging.basicConfig(level=logging.INFO)
        logging.info(f'Commandline:\n{" ".join(sys.argv)}')

        data_path_args = f'--custom-roots {update_config[self.parameter_train_dir]} {update_config[self.parameter_val_dir]} --root _ '
        del update_config[self.parameter_train_dir]
        del update_config[self.parameter_val_dir]
        update_config = data_path_args + ' '.join([f'{k} {v}' for k, v in update_config.items() if str(v) and str(k)])
        logging.info('Training started ...')
        training_info = self._train_internal(config, gpu_num, update_config, tensorboard_dir)
        logging.info('... training completed.')

        with open(out, 'a+') as dst_file:
            yaml.dump(training_info, dst_file)

    def _train_internal(self, config, gpu_num, update_config, tensorboard_dir):
        tools_dir = self._get_tools_dir()
        if tensorboard_dir is not None:
            update_config += f' data.tb_log_dir {tensorboard_dir}'

        training_info = {'training_gpu_num': 0}
        if torch.cuda.is_available():
            logging.info('Training on GPUs started ...')
            available_gpu_num = torch.cuda.device_count()
            if available_gpu_num < gpu_num:
                logging.warning(f'available_gpu_num < args.gpu_num: {available_gpu_num} < {gpu_num}')
                logging.warning(f'decreased number of gpu to: {available_gpu_num}')
                gpu_num = available_gpu_num
                sys.stdout.flush()
            training_info['training_gpu_num'] = gpu_num
            logging.info('... training on GPUs completed.')
        else:
            gpu_num = 0
            logging.info('Training on CPU started ...')

        run_with_termination(f'python {tools_dir}/main.py'
                             f' --config-file {config}'
                             f' --gpu-num {gpu_num}'
                             f' {update_config}'.split(' '))

        if torch.cuda.is_available():
            logging.info('... training on GPUs completed.')
        else:
            logging.info('... training on CPU completed.')

        return training_info
