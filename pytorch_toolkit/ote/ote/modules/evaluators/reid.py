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
import json
import os
import subprocess
import tempfile
import yaml

from ote import REID_TOOLS
from ote.utils import get_file_size_and_sha256
from ote.metrics.classification.reid import mean_accuracy_eval

from .base import BaseEvaluator
from ..registry import EVALUATORS


@EVALUATORS.register_module()
class ReidEvaluator(BaseEvaluator):
    parameter_test_dir = 'test_data_roots'

    def __init__(self):
        super(ReidEvaluator, self).__init__()

    def _evaluate_internal(self, config_path, snapshot, out, update_config, metrics_functions, **kwargs):
        assert isinstance(update_config, dict)
        work_dir = tempfile.mkdtemp()
        print('results are stored in:', work_dir)

        if os.path.islink(snapshot):
            snapshot = os.path.join(os.path.dirname(snapshot), os.readlink(snapshot))

        data_path_args = f'--custom-roots {update_config[self.parameter_test_dir]} {update_config[self.parameter_test_dir]} --root _ '
        del update_config[self.parameter_test_dir]
        update_config = data_path_args + ' '.join([f'{k} {v}' for k, v in update_config.items() if str(v) and str(k)])
        update_config = update_config if update_config else ''

        metrics = []
        metrics.extend(self._get_complexity_and_size(config_path, work_dir, update_config))

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

    def _get_complexity_and_size(self, config_path, work_dir, update_config):
        tools_dir = self._get_tools_dir()

        res_complexity = os.path.join(work_dir, "complexity.json")

        subprocess.run(
            f'python {tools_dir}/get_flops.py'
            f' --config-file {config_path}'
            f' --out {res_complexity}'
            f' {update_config}'.split(' '), check=True)

        with open(res_complexity) as read_file:
            content = json.load(read_file)

        return content


    def _get_tools_dir(self):
        return REID_TOOLS

    def _get_metric_functions(self):
        return [mean_accuracy_eval]

    def _get_image_shape(self, cfg):
        pass
