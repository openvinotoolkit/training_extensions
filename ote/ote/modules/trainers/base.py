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

import glob
import logging
import os
import sys
from abc import ABCMeta, abstractmethod


from ote.utils import get_cuda_device_count, run_with_termination

class BaseTrainer(metaclass=ABCMeta):
    parameter_work_dir = 'work_dir'
    latest_file_name = 'latest.pth'

    def __init__(self):
        self.work_dir = None

    def __call__(self, config, gpu_num, update_config, tensorboard_dir):
        logging.basicConfig(level=logging.INFO)
        logging.info(f'Commandline:\n{" ".join(sys.argv)}')

        # This is required to skip the parameters that were not set in the template
        # (e.g. base_learning_rate or epochs) -- they will have default value None in
        # the parser
        update_config = {k: v for k, v in update_config.items() if v is not None}

        self.work_dir = update_config.get(self.parameter_work_dir)

        update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
        update_config = f' --update_config {update_config}' if update_config else ''

        logging.info('Training started ...')
        self._train_internal(config, gpu_num, update_config, tensorboard_dir)
        logging.info('... training completed.')

    def _train_internal(self, config, gpu_num, update_config, tensorboard_dir):
        tools_dir = self._get_tools_dir()
        tensorboard_dir = f' --tensorboard-dir {tensorboard_dir}' if tensorboard_dir is not None else ''

        if os.getenv('MASTER_ADDR') is not None and os.getenv('MASTER_PORT') is not None:
            # Distributed training is handled by Kubeflow’s PyTorchJob at a higher level.
            logging.info('Distributed training started ...')
            run_with_termination(f'python3 {tools_dir}/train.py'
                                 f' --launcher=pytorch'
                                 f' {config}'
                                 f'{tensorboard_dir}'
                                 f'{update_config}'.split(' '))
            logging.info('... distributed training completed.')
        elif get_cuda_device_count() > 0 and gpu_num:
            logging.info('Training on GPUs started ...')
            available_gpu_num = get_cuda_device_count()
            if available_gpu_num < gpu_num:
                logging.warning(f'available_gpu_num < args.gpu_num: {available_gpu_num} < {gpu_num}')
                logging.warning(f'decreased number of gpu to: {available_gpu_num}')
                gpu_num = available_gpu_num
                sys.stdout.flush()
            run_with_termination(f'{tools_dir}/dist_train.sh'
                                 f' {config}'
                                 f' {gpu_num}'
                                 f'{tensorboard_dir}'
                                 f'{update_config}'.split(' '))
            logging.info('... training on GPUs completed.')
        else:
            logging.info('Training on CPU started ...')
            run_with_termination(f'python3 {tools_dir}/train.py'
                                 f' {config}'
                                 f'{tensorboard_dir}'
                                 f'{update_config}'.split(' '))
            logging.info('... training on CPU completed.')

    @abstractmethod
    def _get_tools_dir(self):
        pass

    def get_latest_snapshot(self):
        if not self.work_dir:
            logging.warning('Cannot return final checkpoint: work_dir is not set')
            return None
        glob_filter = f'{self.work_dir}/**/{self.latest_file_name}'
        latest_snapshots = list(glob.glob(glob_filter, recursive=True))
        if not latest_snapshots:
            logging.warning(f'Cannot find the latest snapshot {self.latest_file_name} in the work_dir {self.work_dir}')
            return None
        if len(latest_snapshots) > 1:
            raise RuntimeError(f'Find more than one latest snapshots {self.latest_file_name} '
                               f'in the work_dir {self.work_dir}')
        return latest_snapshots[0]
