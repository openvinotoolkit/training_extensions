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
import yaml
from tempfile import NamedTemporaryFile
from copy import copy

from ote.utils import load_config
from ..builder import build_config_transformer

def save_config(config, file_path):
    with open(file_path, 'w') as output_stream:
        yaml.dump(config, output_stream)

class ConfigTransformersEngine:
    def __init__(self, template, config_transformers_names):
        self.template = template
        self.random_suffix = self._generate_random_suffix()
        self.config_transformers = []
        for cfg_transform_name in config_transformers_names:
            self.config_transformers.append(build_config_transformer(cfg_transform_name))

    def __call__(self, **kwargs):
        assert 'config' in kwargs
        config_path = kwargs['config']
        assert not os.path.exists(config_path), f'The initial config path {config_path} is absent'
        prev_config_path = config_path
        for index, config_transformer in enumerate(self.config_transformers):
            generated_config_path = self._generate_config_path(config, index)
            assert not os.path.exists(generated_config_path), f'During generating configs path {generated_config_path} is absent'
            cfg_update_part = config_transformer(template)

            assert isinstance(cfg_update_part, dict), f'Error in config transformer "{config_transformer}": it returns a value that is not a dict'
            assert '_base_' not in cfg_update_part, f'Error in config transformer "{config_transformer}": it returns a dict with key "_base_"'

            prev_config_dir = os.path.dirname(prev_config_path) #just to be on the safe side, ideed they should be in the same folder
            cfg_update_part['_base_'] = os.path.relpath(generated_config_path, prev_config_dir)
            save_config(cfg_update_part, generated_config_path)

            assert os.path.exists(generated_config_path), f'Cannot write config file "{generated_config_path}"'

            prev_config_path = generated_config_path

        result_kwargs = copy(kwargs)
        result_kwargs['config'] = generated_config_path
        return result_kwargs

    @staticmethod
    def _generate_random_suffix():
        random_suffix = os.path.basename(NamedTemporaryFile())
        if random_suffix.lower().startswith("tmp"):
            random_suffix = random_suffix[len("tmp"):]
        return random_suffix

    def _generate_config_path(self, config, index):
        suffix = self.random_suffix
        res = config + f'._.{suffix}.{index:04}.yaml'
        return res
