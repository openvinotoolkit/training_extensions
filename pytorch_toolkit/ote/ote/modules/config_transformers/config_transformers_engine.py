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
import yaml
import datetime
from tempfile import NamedTemporaryFile
from copy import copy

from ote.utils import load_config
from ..builder import build_config_transformer

def save_config(config, file_path):
    with open(file_path, 'w') as output_stream:
        yaml.dump(config, output_stream)

class _ConfigTransformersHandler:
    def __init__(self, template_path, config_transformers_names):
        self.template_path = template_path
        self.timestamp = self._generate_timestamp_suffix()

        if config_transformers_names is None:
            self.config_transformers = None
            return

        if not isinstance(config_transformers_names, list):
            config_transformers_names = [config_transformers_names]
        self.config_transformers = []
        for cfg_transform_name in config_transformers_names:
            self.config_transformers.append(build_config_transformer(cfg_transform_name))

    def __call__(self, config_path):
        if not self.config_transformers:
            return config_path
        logging.debug(f'_ConfigTransformersHandler: config_path={config_path}')
        assert os.path.exists(config_path), f'The initial config path {config_path} is absent'
        prev_config_path = config_path
        for index, config_transformer in enumerate(self.config_transformers):
            generated_config_path = self._generate_config_path(config_path, index)
            logging.debug(f'_ConfigTransformersHandler: index={index}, generated_config_path={generated_config_path}')
            assert not os.path.exists(generated_config_path), f'During generating configs path {generated_config_path} is present'
            cfg_update_part = config_transformer(template_path=self.template_path)

            assert isinstance(cfg_update_part, dict), (
                    f'Error in config transformer #{index} "{config_transformer}": it returns a value that is not a dict:'
                    f' type(cfg_update_part)={type(cfg_update_part)}')
            assert '_base_' not in cfg_update_part, (
                    f'Error in config transformer #{index} "{config_transformer}": it returns a dict with key "_base_"')

            generated_config_dir = os.path.dirname(generated_config_path) #just to be on the safe side, indeed they should be in the same folder
            cfg_update_part['_base_'] = os.path.relpath(prev_config_path, generated_config_dir)
            save_config(cfg_update_part, generated_config_path)

            assert os.path.exists(generated_config_path), f'Cannot write config file "{generated_config_path}"'

            prev_config_path = generated_config_path

        return generated_config_path



    @staticmethod
    def _generate_timestamp_suffix():
        return datetime.datetime.now().strftime('%y%m%d%H%M%S')

    def _generate_config_path(self, config_path, index):
        suffix = self.timestamp
        #config_ext = os.path.splitext(config_path)[-1]
        config_ext = 'yaml'
        res = config_path + f'._.{suffix}.{index:04}.{config_ext}'
        return res

class ConfigTransformersEngine:
    CONFIG_ARG_TO_SUBSTITUTE = 'config'
    def __init__(self, template_path, config_transformers_names):
        self.config_transformers_handler = _ConfigTransformersHandler(template_path, config_transformers_names)

    def process_args(self, kwargs):
        # NB: at the moment it is one function, using one variable self.CONFIG_ARG_TO_SUBSTITUTE,
        #     since all scripts (train, eval, export) use the same parameter 'config'
        assert self.CONFIG_ARG_TO_SUBSTITUTE in kwargs, (
                f'Error: kwargs does not contain {self.CONFIG_ARG_TO_SUBSTITUTE}, kwargs={kwargs}')
        config_path = kwargs[self.CONFIG_ARG_TO_SUBSTITUTE]

        res_config_path = self.config_transformers_handler(config_path)

        kwargs = copy(kwargs)
        kwargs[self.CONFIG_ARG_TO_SUBSTITUTE] = res_config_path
        return kwargs
