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
from copy import copy

from mmcv import Config

from ote.utils import load_config
from .merger import merge_dicts_and_lists_b_into_a

# TODO(LeonidBeynenson): implement unit tests on NNCFConfigGenerator

POSSIBLE_NNCF_PARTS = {'int8', 'sparsity', 'pruning'}
COMPRESSION_CONFIG_KEY = 'compression_config'

def is_compression_enabled_in_template(template_path):
    template = load_config(template_path)
    compression_template = template.get('compression')
    if not compression_template:
        return False
    assert isinstance(compression_template, dict), (
            f'Error: compression part of template is not a dict: template["compression"]={compression_template}')
    possible_keys = POSSIBLE_NNCF_PARTS | {COMPRESSION_CONFIG_KEY}
    unknown_keys = set(compression_template.keys()) - possible_keys
    if unknown_keys:
        raise RuntimeError(f'Compression parameters contain unknown keys: {list(unknown_keys)}')
    if COMPRESSION_CONFIG_KEY not in compression_template:
        raise RuntimeError(f'Compression parameters do not contain the field "{COMPRESSION_CONFIG_KEY}"')
    is_compression_enabled = any(compression_template.get(key) for key in POSSIBLE_NNCF_PARTS)
    return is_compression_enabled

class NNCFConfigGenerator:

    def __call__(self, template_path):
        assert is_compression_enabled_in_template(template_path), (
                'Error: compression class is called for a template that does not enable compression.'
                ' This must not be happened in OTE.')
        template = load_config(template_path)
        compression_template = template['compression']

        compression_template = copy(compression_template)
        compression_config_path = compression_template.pop(COMPRESSION_CONFIG_KEY)

        compression_parts_to_choose = []
        for k, v in compression_template.items():
            should_pick = bool(v)
            if should_pick:
                compression_parts_to_choose.append(k)

        if not compression_parts_to_choose:
            return {}

        nncf_config_part = self._merge_nncf_compression_parts(compression_config_path, compression_parts_to_choose)
        return nncf_config_part

    @staticmethod
    def _load_compression_config(compression_config_path, nostrict=False):
        assert compression_config_path.endswith('.json'), (
                f'Only json files are allowed as compression configs,'
                f' compression_config_path={compression_config_path}')
        with open(compression_config_path) as f_src:
            compression_parts  = json.load(f_src)
        return compression_parts

    @staticmethod
    def _merge_nncf_compression_parts(compression_config_path, compression_parts_to_choose):
        compression_parts = NNCFConfigGenerator._load_compression_config(compression_config_path)

        if 'order_of_parts' in compression_parts:
            # The result of applying the changes from compression parts
            # may depend on the order of applying the changes
            # (e.g. if for int8 it is sufficient to have `total_epochs=2`,
            #  but for sparsity it is required `total_epochs=50`)
            # So, user can define `order_of_parts` in the compression_config
            # to specify the order of applying the parts.
            order_of_parts = compression_parts['order_of_parts']
            assert isinstance(order_of_parts, list), 'The field "order_of_parts" in compression config should be a list'

            for part in compression_parts_to_choose:
                assert part in order_of_parts, (
                        f'The part {part} is selected, but it is absent in order_of_parts={order_of_parts},'
                        f' see the compression config file {compression_config_path}')

            compression_parts_to_choose = [part for part in order_of_parts if part in compression_parts_to_choose]

        assert 'base' in compression_parts, f'Error: the compression config does not contain the "base" part'
        nncf_config_part = compression_parts['base']

        for part in compression_parts_to_choose:
            assert part in compression_parts, (
                    f'Error: the compression config does not contain the part "{part}", '
                    f'whereas it was selected; see the compression config file "{compression_config_path}"')
            compression_part_dict = compression_parts[part]
            try:
                nncf_config_part = merge_dicts_and_lists_b_into_a(nncf_config_part, compression_part_dict)
            except AssertionError as cur_error:
                err_descr = (f'Error during merging the parts of nncf configs from file "{compression_config_path}":\n'
                    f'the current part={part}, '
                    f'the order of merging parts into base is {compression_parts_to_choose}.\n'
                    f'The error is:\n{cur_error}')
                raise RuntimeError(err_descr) from None

        return nncf_config_part
