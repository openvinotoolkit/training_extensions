"""
 Copyright (c) 2019 Intel Corporation
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

import re


def load_state(model, saved_state_dict, is_resume=False):
    def key_normalizer(key):
        new_key = key
        match = re.search('(pre_ops|post_ops)\\.(\\d+?)\\.op', key)
        return new_key if not match else new_key.replace(match.group(), 'operation')

    if 'state_dict' in saved_state_dict:
        saved_state_dict = saved_state_dict['state_dict']
    state_dict = model.state_dict()

    new_dict, num_loaded_layers, problematic_keys = match_keys(is_resume, saved_state_dict, state_dict, key_normalizer)
    num_saved_layers = len(saved_state_dict.items())
    process_problematic_keys(is_resume, problematic_keys, num_loaded_layers == num_saved_layers)
    print("Loaded {}/{} layers".format(num_loaded_layers, len(state_dict.items())))

    model.load_state_dict(new_dict, strict=False)
    return num_loaded_layers


def process_problematic_keys(is_resume, issues, is_all_saved_loaded):
    error_msgs = []

    def add_error_msg(name, keys):
        error_msgs.insert(
            0, '{} key(s):\n{}. '.format(name,
                                         ',\n'.join('\t\t"{}"'.format(k) for k in keys)))

    for name, keys in issues.items():
        is_missing = name == 'Missing'
        if keys and (not is_missing or is_missing and (is_resume or not is_all_saved_loaded)):
            add_error_msg(name, keys)
    if error_msgs:
        error_msg = 'Error(s) when loading model parameters:\n\t{}'.format("\n\t".join(error_msgs))
        if is_resume:
            raise RuntimeError(error_msg)
        print(error_msg)


def match_keys(is_resume, saved_state_dict, state_dict, key_normalizer):
    skipped_keys = []
    num_loaded_layers = 0
    new_dict = {}

    def check_parameter_size(key, saved_value, num_loaded_layers):
        saved_size = saved_value.size()
        size = state_dict[key].size()
        if saved_size == size:
            new_dict[key] = saved_value
            return num_loaded_layers + 1
        print("Different size of value of '{}' in dictionary ({}) and in resuming model ({})"
              .format(key, saved_size, size, ))
        skipped_keys.append(key)
        return num_loaded_layers

    clipped_keys = {k.replace('module.', ''): k for k in state_dict.keys()}
    norm_clipped_keys = {}
    collisions = []
    for clipped_key, orig_key in clipped_keys.items():
        normalized_key = key_normalizer(clipped_key)
        if normalized_key in norm_clipped_keys:
            collisions.append(clipped_key)
        norm_clipped_keys[normalized_key] = orig_key

    unexpected_keys = []

    for (saved_key, saved_value) in saved_state_dict.items():
        clipped_saved_key = saved_key.replace('module.', '')
        if clipped_saved_key in clipped_keys:
            key = clipped_keys[clipped_saved_key]
            num_loaded_layers = check_parameter_size(key, saved_value, num_loaded_layers)
        else:
            norm_clipped_saved_key = key_normalizer(clipped_saved_key)
            if norm_clipped_saved_key in norm_clipped_keys and clipped_saved_key not in collisions and not is_resume:
                key = norm_clipped_keys[norm_clipped_saved_key]
                num_loaded_layers = check_parameter_size(key, saved_value, num_loaded_layers)
            else:
                unexpected_keys.append(saved_key)
    missing_keys = [k for k in state_dict.keys() if k not in new_dict and k not in skipped_keys]
    problematic_keys = {'Missing': missing_keys,
                        'Unexpected': unexpected_keys,
                        'Skipped': skipped_keys}
    return new_dict, num_loaded_layers, problematic_keys
