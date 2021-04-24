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

import logging
import os
import tempfile
import json
import yaml
from abc import ABCMeta, abstractmethod

from mmcv.utils import Config

from ote.utils import get_file_size_and_sha256
from ote.utils.misc import run_through_shell


class BaseEvaluator(metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, config, snapshot, out, update_config, **kwargs):
        logging.basicConfig(level=logging.INFO)

        metrics_functions = self._get_metric_functions()
        self._evaluate_internal(config, snapshot, out, update_config, metrics_functions, **kwargs)

    def _evaluate_internal(self, config_path, snapshot, out, update_config, metrics_functions, **kwargs):
        assert isinstance(update_config, dict)

        cfg = Config.fromfile(config_path)

        work_dir = tempfile.mkdtemp()
        print('results are stored in:', work_dir)

        if os.path.islink(snapshot):
            snapshot = os.path.join(os.path.dirname(snapshot), os.readlink(snapshot))

        metrics = []
        metrics.extend(self._get_complexity_and_size(cfg, config_path, work_dir, update_config))

        metric_args = {
            'config_path': config_path,
            'work_dir': work_dir,
            'snapshot': snapshot,
            'update_config': update_config
        }
        metric_args.update(kwargs)
        for func in metrics_functions:
            metrics.extend(func(**metric_args))

        outputs = {
            'files': [get_file_size_and_sha256(snapshot)],
            'metrics': self._round_metrics(metrics)
        }

        if os.path.exists(out):
            with open(out) as read_file:
                content = yaml.load(read_file, Loader=yaml.SafeLoader)
            content.update(outputs)
            outputs = content

        with open(out, 'w') as write_file:
            yaml.dump(outputs, write_file)

    def _get_complexity_and_size(self, cfg, config_path, work_dir, update_config):
        image_shape = self._get_image_shape(cfg)
        tools_dir = self._get_tools_dir()

        res_complexity = os.path.join(work_dir, 'complexity.json')
        update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
        update_config = f' --update_config {update_config}' if update_config else ''
        update_config = update_config.replace('"', '\\"')
        run_through_shell(
            f'python3 {tools_dir}/analysis_tools/get_flops.py'
            f' {config_path}'
            f' --shape {image_shape}'
            f' --out {res_complexity}'
            f'{update_config}')

        with open(res_complexity) as read_file:
            content = json.load(read_file)

        return content

    @staticmethod
    def _round_metrics(metrics, num_digits=3):
        for metric in metrics:
            metric['value'] = round(metric['value'], num_digits) if metric['value'] else metric['value']

        return metrics

    @abstractmethod
    def _get_tools_dir(self):
        pass

    @abstractmethod
    def _get_metric_functions(self):
        pass

    @abstractmethod
    def _get_image_shape(self, cfg):
        pass
