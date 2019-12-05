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

import logging
import math
import operator
from functools import reduce

try:
    from termcolor import colored
except ImportError:
    colored = lambda x, *args, **kwargs: x

import torch
from torch.nn import init

logger = logging.getLogger(__name__)


def load_checkpoint(model, ckpt_path, mapping=None, verbose=False, skip_prefix=''):
    if mapping is None:
        mapping = {}

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']

    from collections import OrderedDict
    source_state = OrderedDict([(mapping.get(k, k), v) for k, v in ckpt.items()])
    if skip_prefix:
        source_state = OrderedDict([(k, v) for k, v in source_state.items() if not k.startswith(skip_prefix)])

    target_state = model.state_dict()
    if verbose:
        logger.info(colored('missed weights:', 'red'))
        logger.info('\n'.join([colored(x, 'red') for x in target_state.keys() if x not in source_state]))
        logger.info(colored('extra weights:', 'magenta'))
        logger.info('\n'.join([colored(x, 'magenta') for x in source_state.keys() if x not in target_state]))

    source_state = OrderedDict({k:v for k, v in source_state.items() if k in target_state.keys()})

    for name in source_state:
        # Avoid error when number of classes was changed and sizes of tensors are different
        tensor = operator.attrgetter(name)(model)
        if tensor.data.shape != source_state[name].shape:
            if tensor.data.numel() == source_state[name].numel():
                source_state[name] = source_state[name].reshape(tensor.data.shape)
            else:
                logger.warning('Model and checkpoint have different sizes of tensors in \'{}\': {} vs {}. '
                               'Weights will not be loaded'.format(name, tensor.data.shape, source_state[name].shape))
                continue

    model.load_state_dict(state_dict=source_state, strict=False)


def load_rcnn_ckpt(model, ckpt):
    """Load checkpoint"""
    state_dict = {}
    for name in ckpt:
        # Avoid error when number of classes was changed and sizes of tensors are different
        tensor = operator.attrgetter(name)(model)
        if tensor.data.shape != ckpt[name].shape:
            if tensor.data.numel() == ckpt[name].numel():
                ckpt[name] = ckpt[name].reshape(tensor.data.shape)
            else:
                logger.warning('Model and checkpoint have different sizes of tensors in \'{}\': {} vs {}. '
                               'Weights will not be loaded'.format(name, tensor.data.shape, ckpt[name].shape))
                continue
        state_dict[name] = ckpt[name]
    model.load_state_dict(state_dict, strict=False)


def xavier_fill(tensor):
    """Caffe2 XavierFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_in = size / tensor.shape[0]
    scale = math.sqrt(3 / fan_in)
    return init.uniform_(tensor, -scale, scale)


def msra_fill(tensor):
    """Caffe2 MSRAFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_out = size / tensor.shape[1]
    scale = math.sqrt(2 / fan_out)
    return init.normal_(tensor, 0, scale)


def freeze_params(root_module, layer_type, freeze=True):
    for m in root_module.modules():
        if isinstance(m, layer_type):
            for p in m.parameters():
                p.requires_grad = not freeze


def set_train_mode(root_module, layer_type, mode=True):
    for m in root_module.modules():
        if isinstance(m, layer_type):
            m.train(mode)


def get_group_gn(dim, dim_per_gp=-1, num_groups=32):
    """Get number of groups used by GroupNorm, based on number of channels."""

    assert dim_per_gp == -1 or num_groups == -1, \
        'GroupNorm: can only specify G or C/G.'

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn
