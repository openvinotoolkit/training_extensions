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

from ote.utils import load_config
from ..registry import CONFIG_TRANSFORMERS

def _convert_value_to_bool(v, key):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower().strip() in ("on", "true", "yes", "y"):
            return True
        if v.lower().strip() in ("off", "false", "no", "n"):
            return False
    elif isinstance(v, int):
        if v == 1:
            return True
        if v == 0:
            return False
    err_str = f'Cannot convert to bool the value `{v}`'
    if key is not None:
        err_str += ' for the key "{key}"'
    raise RuntimeError(err_str)

def _merge_dicts_and_lists_b_into_a(a, b, cur_key=""):
    """The function is inspired by mmcf.Config._merge_a_into_b,
    but it
    * supports merging of lists (by '+' the lists)
    * works with usual dicts and lists, won't work with derived types
    * makes recursive merging for dict + dict case
    * overwrites when merging scalar into scalar

    Note that we merge b into a (whereas Config makes merge a into b),
    since otherwise the order of list merging is counter-intuitive.
    """
    err_str = (
            f'Error in merging parts of config: different types for key={cur_key}, '
            f'type of left part is {type(a)}, '
            f'type of right part is {type(b)}')

    assert isinstance(a, (dict, list)) and type(a) == type(b), err_str
    if isinstance(a, list):
        # the main feature -- merging of lists
        return a + b
    a = copy(a)
    for k in b.keys():
        if k not in a:
            a[k] = copy(b[k])
            continue
        if isinstance(a[k], (dict, list)):
            new_cur_key = cur_key + '.' + k if cur_key else k
            a[k] = _merge_dicts_and_lists_b_into_a(a[k], b[k], new_cur_key)
            continue

        assert not isinstance(b[k], (dict, list)), err_str

        # suppose here that a[k] and b[k] are scalars
        a[k] = b[k]
    return a



@CONFIG_TRANSFORMERS.register_module()
class NNCFConfigTransformer(BaseConfigTransformer):
    POSSIBLE_NNCF_PARTS = ('int8', 'sparsity', 'pruning')
    COMPRESSION_CONFIG_KEY = 'compression_config'

    def __init__(self):
        super().__init__()

    def process(self, template):
        compression_template = template.get('compression')
        if compression_template is None or compression_template == {}:
            return {}
        assert isinstance(compression_template, dict), f'Error: compression part of template is not a dict: template["compression"]={compression_template}'
        possible_keys = set(self.POSSIBLE_NNCF_PARTS) + set([self.COMPRESSION_CONFIG_KEY])
        unknown_keys = set(compression_template.keys()) - set(possible_keys)
        if not unknown_keys:
            raise RuntimeError(f'Compression parameters contain unknown keys: {list(unknown_keys)}')

        compression_template = copy(compression_template)
        if self.COMPRESSION_CONFIG_KEY not in compression_template:
            raise RuntimeError(f'Error: compression part of template does not contain the key {self.COMPRESSION_CONFIG_KEY}')
        compression_config_path = compression_template.pop(self.COMPRESSION_CONFIG_KEY)

        compression_parts_to_choose = []
        for k, v in compression_template.items():
            should_pick = _convert_value_to_bool(v, f'compression.{k}')
            if should_pick:
                compression_parts_to_choose.append(k)

        if not compression_parts_to_choose:
            return {}

        nncf_config = self._merge_nncf_compression_parts(compression_config_path, compression_parts_to_choose)
        return {'nncf_config': nncf_config}

    @staticmethod
    def _merge_nncf_compression_parameters(compression_config_path, compression_parts_to_choose):
        compression_parts = Config.fromfile(compression_config_path)
        assert 'base' in compression_parts, f'Error: the NNCF compression config does not contain the "base" part'
        nncf_config = compression_parts['base']
        for part in compression_parts_to_choose:
            assert part in compression_parts, (
                    f'Error: NNCF compression config does not contain the part "{part}", '
                    f'whereas it was selected; see the NNCF config file "{compression_config_path}"')
            try:
                nncf_config = _merge_dicts_and_lists_b_into_a(nncf_config, compression_parts[part])
                cur_error = None
            except AssertionError as e:
                cur_error = e

            if cur_error:
                raise RuntimeError(
                        f'Error during merging the parts of nncf configs from file "{compression_config_path}":\n'
                        f'the current part={part}, '
                        f'the order of merging parts into base is {compression_parts_to_choose}.\n'
                        f'The error is:\n{cur_error}')
        return nncf_config










