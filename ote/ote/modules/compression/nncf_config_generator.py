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

import json
import logging
from copy import copy

from ote.utils import load_config
from .merger import merge_dicts_and_lists_b_into_a

# TODO(LeonidBeynenson): implement unit tests on NNCFConfigGenerator

OPTIMISATION_PART_NAME = 'optimisations'
POSSIBLE_NNCF_PARTS = {'nncf_quantization', 'nncf_sparsity', 'nncf_pruning', 'nncf_binarization'}
COMPRESSION_CONFIG_KEY = 'config'

def _get_optimisation_configs_from_template(model_template):
    optimisation_template = model_template.get(OPTIMISATION_PART_NAME)
    cur_keys = set(optimisation_template.keys()) & POSSIBLE_NNCF_PARTS
    opt_sections = [optimisation_template[k] for k in cur_keys]
    optimisation_configs = [v.get(COMPRESSION_CONFIG_KEY) for v in opt_sections]
    optimisation_configs = [v for v in optimisation_configs if v is not None]
    optimisation_configs = list(set(optimisation_configs))
    return optimisation_configs

def get_optimisation_config_from_template(model_template):
    configs = _get_optimisation_configs_from_template(model_template)
    assert len(configs) == 1
    return configs[0]

def is_optimisation_enabled_in_template(template):
    """
    The function returns if a model template contains
    'optimisation' section; also the function
    validates if the section is correct
    The function receives as the parameter either
    template path or template dict read from file
    """
    if isinstance(template, str):
        template = load_config(template)
    optimisation_template = template.get(OPTIMISATION_PART_NAME)
    if not optimisation_template:
        return False
    assert isinstance(optimisation_template, dict), (
            f'Error: optimisation part of template is not a dict: template["optimisation"]={optimisation_template}')
    unknown_keys = set(optimisation_template.keys()) - POSSIBLE_NNCF_PARTS
    if unknown_keys:
        raise RuntimeError(f'Optimisation parameters contain unknown keys: {list(unknown_keys)}')
    optimisation_configs = _get_optimisation_configs_from_template(template)
    if not optimisation_configs:
        raise RuntimeError(f'Optimisation parameters do not contain the field "{COMPRESSION_CONFIG_KEY}"')
    if len(optimisation_configs) > 1:
        raise RuntimeError('Wrong config: the optimisation config contains different config files: '
                           f'{optimisation_configs}')
    return True

class NNCFConfigGenerator:

    def __call__(self, template_path, kwargs_nncf):
        if not is_optimisation_enabled_in_template(template_path):
            logging.warning('WARNING: optimisation class is called for a template that does not enable optimisation.'
                            ' This must not be happened in OTE.')
            return {}
        template = load_config(template_path)
        optimisation_template = template[OPTIMISATION_PART_NAME]

        optimisation_template = copy(optimisation_template)
        optimisation_config_path = get_optimisation_config_from_template(template)

        optimisation_parts_to_choose = []
        for k in POSSIBLE_NNCF_PARTS:
            should_pick = bool(kwargs_nncf.get(k))
            if should_pick:
                optimisation_parts_to_choose.append(k)

        if not optimisation_parts_to_choose:
            return {}

        nncf_config_part = self._merge_nncf_optimisation_parts(optimisation_config_path, optimisation_parts_to_choose)
        return nncf_config_part

    @staticmethod
    def _load_optimisation_config(optimisation_config_path, nostrict=False):
        assert optimisation_config_path.endswith('.json'), (
                f'Only json files are allowed as optimisation configs,'
                f' optimisation_config_path={optimisation_config_path}')
        with open(optimisation_config_path) as f_src:
            optimisation_parts  = json.load(f_src)
        return optimisation_parts

    @staticmethod
    def _merge_nncf_optimisation_parts(optimisation_config_path, optimisation_parts_to_choose):
        optimisation_parts = NNCFConfigGenerator._load_optimisation_config(optimisation_config_path)

        if 'order_of_parts' in optimisation_parts:
            # The result of applying the changes from optimisation parts
            # may depend on the order of applying the changes
            # (e.g. if for nncf_quantization it is sufficient to have `total_epochs=2`,
            #  but for sparsity it is required `total_epochs=50`)
            # So, user can define `order_of_parts` in the optimisation_config
            # to specify the order of applying the parts.
            order_of_parts = optimisation_parts['order_of_parts']
            assert isinstance(order_of_parts, list), \
                'The field "order_of_parts" in optimisation config should be a list'

            for part in optimisation_parts_to_choose:
                assert part in order_of_parts, (
                        f'The part {part} is selected, but it is absent in order_of_parts={order_of_parts},'
                        f' see the optimisation config file {optimisation_config_path}')

            optimisation_parts_to_choose = [part for part in order_of_parts if part in optimisation_parts_to_choose]

        assert 'base' in optimisation_parts, 'Error: the optimisation config does not contain the "base" part'
        nncf_config_part = optimisation_parts['base']

        for part in optimisation_parts_to_choose:
            assert part in optimisation_parts, (
                    f'Error: the optimisation config does not contain the part "{part}", '
                    f'whereas it was selected; see the optimisation config file "{optimisation_config_path}"')
            optimisation_part_dict = optimisation_parts[part]
            try:
                nncf_config_part = merge_dicts_and_lists_b_into_a(nncf_config_part, optimisation_part_dict)
            except AssertionError as cur_error:
                err_descr = (f'Error during merging the parts of nncf configs from file "{optimisation_config_path}":\n'
                    f'the current part={part}, '
                    f'the order of merging parts into base is {optimisation_parts_to_choose}.\n'
                    f'The error is:\n{cur_error}')
                raise RuntimeError(err_descr) from None

        return nncf_config_part
