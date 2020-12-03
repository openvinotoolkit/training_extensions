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
from tempfile import NamedTemporaryFile
from copy import copy

import mmcv

from ote.utils import load_config
from ..registry import COMPRESSION
from .nncf_config_generator import NNCFConfigGenerator

def _save_config(config_dict, base_rel_path, file_path):
    assert file_path.endswith('.py')
    with open(file_path, 'w') as f:
        f.write(f'_base_ = "{base_rel_path}"\n')
        cfg = mmcv.Config(config_dict)
        f.write(cfg.pretty_text)

def _generate_random_suffix():
    random_suffix = os.path.basename(NamedTemporaryFile().name)
    prefix = 'tmp'
    if random_suffix.startswith(prefix):
        random_suffix = random_suffix[len(prefix):]
    return random_suffix


# TODO(LeonidBeynenson): implement unit tests on NNCFConfigTransformer

@COMPRESSION.register_module()
class NNCFConfigTransformer:
    CONFIG_ARG_TO_SUBSTITUTE = 'config'

    def process_args(self, template_path, kwargs):
        # NB: at the moment it is one function, using one variable self.CONFIG_ARG_TO_SUBSTITUTE,
        #     since all scripts (train, eval, export) use the same parameter 'config'
        assert self.CONFIG_ARG_TO_SUBSTITUTE in kwargs, (
                f'Error: kwargs does not contain {self.CONFIG_ARG_TO_SUBSTITUTE}, kwargs={kwargs}')
        config_path = kwargs[self.CONFIG_ARG_TO_SUBSTITUTE]

        res_config_path = self._create_derived_config_file(template_path, config_path)

        kwargs[self.CONFIG_ARG_TO_SUBSTITUTE] = res_config_path
        return kwargs

    def _create_derived_config_file(self, template_path, config_path):
        assert os.path.exists(config_path), f'The initial config path {config_path} is absent'

        generated_config_path = self._generate_config_path(config_path)
        assert not os.path.exists(generated_config_path), f'Generated configs path {generated_config_path} exists'

        config_generator = NNCFConfigGenerator()
        cfg_update_part = config_generator(template_path=template_path)

        assert '_base_' not in cfg_update_part, 'Error in config generator: it returns a dict with key "_base_"'

        #just to be on the safe side, indeed they should be in the same folder
        generated_config_dir = os.path.dirname(generated_config_path)

        base_rel_path = os.path.relpath(config_path, generated_config_dir)
        _save_config(cfg_update_part, base_rel_path, generated_config_path)

        assert os.path.exists(generated_config_path), f'Cannot write config file "{generated_config_path}"'

        return generated_config_path

    @staticmethod
    def _generate_config_path(config_path):
        suffix = _generate_random_suffix()
        config_ext = 'py'
        res = config_path + f'._.{suffix}.{config_ext}'
        return res
