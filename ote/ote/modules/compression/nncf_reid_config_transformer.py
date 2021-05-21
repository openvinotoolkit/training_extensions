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
import logging
import yaml

from ..registry import COMPRESSION
from .nncf_config_transformer import (save_config,
                                      generate_config_path,
                                      NNCFConfigTransformer)


@COMPRESSION.register_module()
class NNCFReidConfigTransformer:
    CONFIG_ARG_TO_SUBSTITUTE = 'config'
    NAME_FIELD_TO_EXTRACT_TO_NNCF_CONFIG = 'nncf_config'
    NAME_FIELD_FOR_EXTRACTED_NNCF_CONFIG_PATH = 'nncf_config_path'

    NAME_FIELD_TO_EXTRACT_TO_AUX_CONFIG_CHANGE = 'changes_aux_config'
    NAME_FIELD_FOR_EXTRACTED_AUX_CONFIG_CHANGE_PATH = 'changes_in_aux_train_config'

    def process_args(self, template_path, kwargs):
        assert self.CONFIG_ARG_TO_SUBSTITUTE in kwargs, (
                f'Error: kwargs does not contain {self.CONFIG_ARG_TO_SUBSTITUTE}, kwargs={kwargs}')

        kwargs, is_optimisation_enabled = NNCFConfigTransformer().process_args(template_path, kwargs)

        if not is_optimisation_enabled:
            return kwargs, is_optimisation_enabled

        assert self.CONFIG_ARG_TO_SUBSTITUTE in kwargs, (
                f'Error: kwargs after NNCFConfigTransformer does not contain {self.CONFIG_ARG_TO_SUBSTITUTE}, '
                f'kwargs={kwargs}')

        cur_config_path = kwargs[self.CONFIG_ARG_TO_SUBSTITUTE]
        with open(cur_config_path) as f:
            cur_config = yaml.safe_load(f)
        if ( (self.NAME_FIELD_TO_EXTRACT_TO_NNCF_CONFIG not in cur_config) or
             (self.NAME_FIELD_TO_EXTRACT_TO_AUX_CONFIG_CHANGE not in cur_config) ):
            raise RuntimeError(f'The fields {self.NAME_FIELD_TO_EXTRACT_TO_NNCF_CONFIG} '
                               f'and {self.NAME_FIELD_TO_EXTRACT_TO_AUX_CONFIG_CHANGE} may be absent '
                               f'in generated config file {cur_config_path} -- but they are required for '
                               f'{type(self).__name__} -- check the original config file '
                               f'and nncf config')
        nncf_part_to_extract = cur_config[self.NAME_FIELD_TO_EXTRACT_TO_NNCF_CONFIG]
        del cur_config[self.NAME_FIELD_TO_EXTRACT_TO_NNCF_CONFIG]

        aux_changes_part_to_extract = cur_config[self.NAME_FIELD_TO_EXTRACT_TO_AUX_CONFIG_CHANGE]
        del cur_config[self.NAME_FIELD_TO_EXTRACT_TO_AUX_CONFIG_CHANGE]

        new_config_path = generate_config_path(cur_config_path, 'yml')
        extracted_nncf_cfg_path = generate_config_path(cur_config_path, 'nncf_part.json')
        changes_aux_config_path = generate_config_path(cur_config_path, 'aux_changes.yml')

        save_config(nncf_part_to_extract, extracted_nncf_cfg_path)
        logging.debug(f'Extracted NNCF part of config saved to the file {extracted_nncf_cfg_path}')

        save_config(aux_changes_part_to_extract, changes_aux_config_path)
        logging.debug(f'Extracted aux changes of config saved to the file {changes_aux_config_path}')

        cur_config['nncf'][self.NAME_FIELD_FOR_EXTRACTED_NNCF_CONFIG_PATH] = extracted_nncf_cfg_path
        cur_config['nncf'][self.NAME_FIELD_FOR_EXTRACTED_AUX_CONFIG_CHANGE_PATH] = changes_aux_config_path
        save_config(cur_config, new_config_path)
        logging.debug(f'After extracting NNCF part: saved new config to the file {new_config_path}')

        kwargs[self.CONFIG_ARG_TO_SUBSTITUTE] = new_config_path

        return kwargs, is_optimisation_enabled
