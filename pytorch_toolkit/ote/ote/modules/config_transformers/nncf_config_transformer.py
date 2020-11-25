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

from copy import copy
from mmcv import Config

from ote.utils import load_config
from ..registry import CONFIG_TRANSFORMERS
from .utils import merge_dicts_and_lists_b_into_a

@CONFIG_TRANSFORMERS.register_module()
class NNCFConfigTransformer:
    POSSIBLE_NNCF_PARTS = ('int8', 'sparsity', 'pruning')
    COMPRESSION_CONFIG_KEY = 'compression_config'

    def __call__(self, template_path, config_path):
        config_path = None #is not used
        template = load_config(template_path)
        compression_template = template.get('compression')
        if compression_template is None or compression_template == {}:
            return {}
        assert isinstance(compression_template, dict), f'Error: compression part of template is not a dict: template["compression"]={compression_template}'
        possible_keys = set(self.POSSIBLE_NNCF_PARTS) | set([self.COMPRESSION_CONFIG_KEY])
        unknown_keys = set(compression_template.keys()) - set(possible_keys)
        if unknown_keys:
            raise RuntimeError(f'Compression parameters contain unknown keys: {list(unknown_keys)}')

        compression_template = copy(compression_template)
        if self.COMPRESSION_CONFIG_KEY not in compression_template:
            raise RuntimeError(f'Error: compression part of template does not contain the key {self.COMPRESSION_CONFIG_KEY}')
        compression_config_path = compression_template.pop(self.COMPRESSION_CONFIG_KEY)

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
    def _merge_nncf_compression_parts(compression_config_path, compression_parts_to_choose):
        compression_parts = Config.fromfile(compression_config_path)

        if 'order_of_parts' in compression_parts:
            order_of_parts = compression_parts['order_of_parts']
            assert isinstance(order_of_parts, list), 'The field "order_of_parts" in compression config should be a list'

            for part in compression_parts_to_choose:
                assert part in order_of_parts, (
                        f'The part {part} is selected, but it is absent in order_of_parts={order_of_parts},'
                        f' see the compression config file {compression_config_path}')

            compression_parts_to_choose = [part for part in order_of_parts if part in compression_parts_to_choose]

        assert 'base' in compression_parts, f'Error: the compression config does not contain the "base" part'
        nncf_config_part = compression_parts['base'].to_dict()

        for part in compression_parts_to_choose:
            assert part in compression_parts, (
                    f'Error: the compression config does not contain the part "{part}", '
                    f'whereas it was selected; see the compression config file "{compression_config_path}"')
            compression_part_dict = compression_parts[part].to_dict()
            try:
                nncf_config_part = merge_dicts_and_lists_b_into_a(nncf_config_part, compression_part_dict)
                cur_error = None
            except AssertionError as e:
                cur_error = e

            if cur_error:
                raise RuntimeError(
                        f'Error during merging the parts of nncf configs from file "{compression_config_path}":\n'
                        f'the current part={part}, '
                        f'the order of merging parts into base is {compression_parts_to_choose}.\n'
                        f'The error is:\n{cur_error}')
        return nncf_config_part
