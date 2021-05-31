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
import sys

from text_recognition.utils.get_config import get_config
from text_recognition.utils.trainer import Trainer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    args.add_argument('--work_dir')
    args.add_argument('--local_rank', default=0, type=int)
    return args.parse_args()


if __name__ == '__main__':
    assert sys.version_info[0] == 3
    arguments = parse_args()

    train_config = get_config(arguments.config, section='train')

    experiment = Trainer(work_dir=arguments.work_dir, config=train_config, rank=arguments.local_rank)
    experiment.train()
