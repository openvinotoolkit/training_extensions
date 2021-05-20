"""
 Copyright (c) 2020-2021 Intel Corporation

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

import json
import os
import yaml

from ote.utils.misc import generate_random_suffix

from ..registry import COMPRESSION
from .nncf_config_generator import NNCFConfigGenerator, POSSIBLE_NNCF_PARTS

def save_config(config, file_path):
    # TODO(LeonidBeynenson): add possibility to write python instead of yaml
    #     (otherwise mmcv.Config tries to write its result config as yaml file too)

    assert '.' in file_path
    ext = file_path[file_path.rindex('.') + 1:]
    if ext in ('yaml', 'yml'):
        backend = yaml
    elif ext in ('json', 'jsn'):
        backend = json
    else:
        raise RuntimeError(f'Unknown extension {ext} of path {file_path}')

    with open(file_path, 'w') as output_stream:
        backend.dump(config, output_stream)

def generate_config_path(config_path, config_ext='yaml'):
    suffix = generate_random_suffix()
    res = config_path + f'._.{suffix}.{config_ext}'
    return res

# TODO(LeonidBeynenson): implement unit tests on NNCFConfigTransformer

@COMPRESSION.register_module()
class NNCFConfigTransformer:
    CONFIG_ARG_TO_SUBSTITUTE = 'config'
    NNCF_PARAMETERS = list(POSSIBLE_NNCF_PARTS)

    def process_args(self, template_path, kwargs):
        # NB: at the moment it is one function, using one variable self.CONFIG_ARG_TO_SUBSTITUTE,
        #     since all scripts (train, eval, export) use the same parameter 'config'
        assert self.CONFIG_ARG_TO_SUBSTITUTE in kwargs, (
                f'Error: kwargs does not contain {self.CONFIG_ARG_TO_SUBSTITUTE}, kwargs={kwargs}')
        config_path = kwargs[self.CONFIG_ARG_TO_SUBSTITUTE]
        kwargs_nncf = {k:kwargs.get(k) for k in self.NNCF_PARAMETERS}
        is_optimisation_enabled = any(kwargs.get(k) for k in self.NNCF_PARAMETERS)

        if is_optimisation_enabled:
            res_config_path = self._create_derived_config_file(template_path, config_path, kwargs_nncf)
            kwargs[self.CONFIG_ARG_TO_SUBSTITUTE] = res_config_path

        for k in self.NNCF_PARAMETERS:
            if k in kwargs:
                del kwargs[k]

        return kwargs, is_optimisation_enabled

    def _create_derived_config_file(self, template_path, config_path, kwargs_nncf):
        assert os.path.exists(config_path), f'The initial config path {config_path} is absent'

        generated_config_path = generate_config_path(config_path)
        assert not os.path.exists(generated_config_path), f'Generated configs path {generated_config_path} exists'

        config_generator = NNCFConfigGenerator()
        cfg_update_part = config_generator(template_path, kwargs_nncf)

        assert '_base_' not in cfg_update_part, 'Error in config generator: it returns a dict with key "_base_"'

        #just to be on the safe side, indeed they should be in the same folder
        generated_config_dir = os.path.dirname(generated_config_path)

        cfg_update_part['_base_'] = os.path.relpath(config_path, generated_config_dir)
        save_config(cfg_update_part, generated_config_path)

        assert os.path.exists(generated_config_path), f'Cannot write config file "{generated_config_path}"'

        return generated_config_path
