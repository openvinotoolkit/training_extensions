"""
 Copyright (c) 2019-2020 Intel Corporation
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

import torch

from nncf.nncf_network import MODEL_WRAPPED_BY_NNCF_ATTR_NAME
from nncf.nncf_logger import logger as nncf_logger


def load_state(model: torch.nn.Module, saved_state_dict: dict, is_resume: bool = False) -> int:
    """
    Used to load a checkpoint containing a compressed model into an NNCFNetwork object, but can
    be used for any PyTorch module as well. Will do matching of saved_state_dict parameters to
    the model's state_dict parameters while discarding irrelevant prefixes added during wrapping
    in NNCFNetwork or DataParallel/DistributedDataParallel objects, and load the matched parameters
    from the saved_state_dict into the model's state dict.
    :param model: The target module for the saved_state_dict to be loaded to.
    :param saved_state_dict: A state dict containing the parameters to be loaded into the model.
    :param is_resume: Determines the behavior when the function cannot do a successful parameter match
    when loading. If True, the function will raise an exception if it cannot match the saved_state_dict
    parameters to the model's parameters (i.e. if some parameters required by model are missing in
    saved_state_dict, or if saved_state_dict has parameters that could not be matched to model parameters,
    or if the shape of parameters is not matching). If False, the exception won't be raised.
    Usually is_resume is specified as False when loading uncompressed model's weights into the model with
    compression algorithms already applied, and as True when loading a compressed model's weights into the model
    with compression algorithms applied to evaluate the model.
    :return: The number of saved_state_dict entries successfully matched and loaded into model.
    """
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
    nncf_logger.info("Loaded {}/{} layers".format(num_loaded_layers, len(state_dict.items())))

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
        nncf_logger.warning(error_msg)


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
        nncf_logger.warning("Different size of value of '{}' in dictionary ({}) and in resuming model ({})"
                            .format(key, saved_size, size, ))
        skipped_keys.append(key)
        return num_loaded_layers

    clip_patterns = [MODEL_WRAPPED_BY_NNCF_ATTR_NAME + '.',
                     'module.']

    clipped_keys = list(state_dict.keys())
    for pattern in clip_patterns:
        for i, _ in enumerate(clipped_keys):
            clipped_keys[i] = clipped_keys[i].replace(pattern, '')

    clipped_key_to_model_key_dict = dict(zip(clipped_keys, state_dict.keys()))

    norm_clipped_keys = {}
    collisions = []
    for clipped_key, orig_key in clipped_key_to_model_key_dict.items():
        normalized_key = key_normalizer(clipped_key)
        if normalized_key in norm_clipped_keys:
            collisions.append(clipped_key)
        norm_clipped_keys[normalized_key] = orig_key

    unexpected_keys = []

    for (saved_key, saved_value) in saved_state_dict.items():
        clipped_saved_key = saved_key
        for pattern in clip_patterns:
            clipped_saved_key = clipped_saved_key.replace(pattern, '')

        if clipped_saved_key in clipped_key_to_model_key_dict:
            key = clipped_key_to_model_key_dict[clipped_saved_key]
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
