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

from copy import deepcopy

import pytest
import torch
from pytest import approx

from nncf.algo_selector import create_compression_algorithm
from nncf.operations import UpdateWeight
from nncf.sparsity.magnitude.algo import MagnitudeSparsity
from nncf.sparsity.magnitude.functions import normed_magnitude
from nncf.sparsity.magnitude.layers import MagnitudeSparsifyingWeight
from nncf.utils import get_all_modules_by_type
from tests.quantization.test_functions import check_equal
from tests.sparsity.const.test_algo import ref_mask_2, ref_mask_1
from tests.sparsity.magnitude.test_helpers import MagnitudeTestModel, get_basic_magnitude_sparsity_config


@pytest.mark.parametrize('update_mask_on_forward', [True, False],
                         ids=['Mask Update on Forward', 'Mask Update by Scheduler'])
def test_can_create_magnitude_sparse_algo__with_defaults(update_mask_on_forward):
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = \
        {'schedule': 'multistep', 'update_mask_on_forward': update_mask_on_forward}
    compression_algo = create_compression_algorithm(deepcopy(model), config)

    assert isinstance(compression_algo, MagnitudeSparsity)
    sparse_model = compression_algo.model
    assert compression_algo.sparsity_level == approx(0.1)
    assert len(list(sparse_model.modules())) == 11

    model_conv = get_all_modules_by_type(model, 'Conv2d')
    sparse_model_conv = get_all_modules_by_type(sparse_model, 'NNCFConv2d')
    assert len(model_conv) == len(sparse_model_conv)

    i = 0
    for module_name in model_conv:
        scope = module_name.split('/')
        scope[-1] = scope[-1].replace('Conv2d', 'NNCFConv2d')
        sparse_module_name = '/'.join(scope)
        assert sparse_module_name in sparse_model_conv

        store = []
        sparse_module = sparse_model_conv[sparse_module_name]
        ref_mask = torch.ones_like(sparse_module.weight) if update_mask_on_forward or i == 0 else ref_mask_2
        i += 1
        for op in sparse_module.pre_ops.values():
            if isinstance(op, UpdateWeight) and isinstance(op.operand, MagnitudeSparsifyingWeight):
                assert op.operand.threshold == approx(0.24, 0.1)
                assert torch.allclose(op.operand.binary_mask, ref_mask)
                assert isinstance(op.operand.weight_importance, type(normed_magnitude))
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)


@pytest.mark.parametrize(
    ('weight_importance', 'sparsity_level', 'threshold'),
    (
        ('normed_abs', None, 0.219),
        ('abs', None, 9),
        ('normed_abs', 0.5, 0.243),
        ('abs', 0.5, 10),
    )
)
def test_magnitude_sparse_algo_sets_threshold(weight_importance, sparsity_level, threshold):
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['weight_importance'] = weight_importance
    config['compression']['params'] = {'schedule': 'multistep'}
    compression_algo = create_compression_algorithm(model, config)
    if sparsity_level:
        compression_algo.set_sparsity_level(sparsity_level)
    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.threshold == pytest.approx(threshold, 0.01)


def test_can_not_set_sparsity_more_than_one_for_magnitude_sparse_algo():
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    compression_algo = create_compression_algorithm(model, config)
    with pytest.raises(AttributeError):
        compression_algo.set_sparsity_level(1)
        compression_algo.set_sparsity_level(1.2)


def test_can_not_create_magnitude_algo__without_steps():
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'sparsity_levels': [0.1]}
    with pytest.raises(AttributeError):
        create_compression_algorithm(model, config)


def test_can_create_magnitude_algo__without_levels():
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'steps': [1]}
    compression_algo = create_compression_algorithm(model, config)
    assert compression_algo.sparsity_level == approx(0.1)


def test_can_not_create_magnitude_algo__with_not_matched_steps_and_levels():
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'sparsity_levels': [0.1], 'steps': [1, 2]}
    with pytest.raises(AttributeError):
        create_compression_algorithm(model, config)


def test_magnitude_algo_set_binary_mask_on_forward():
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['weight_importance'] = 'abs'
    compression_algo = create_compression_algorithm(model, config)
    compression_algo.set_sparsity_level(0.3)
    model = compression_algo.model
    with torch.no_grad():
        model(torch.ones([1, 1, 10, 10]))

    op = model.conv1.pre_ops['0']
    check_equal(ref_mask_1, op.operand.binary_mask)

    op = model.conv2.pre_ops['0']
    check_equal(ref_mask_2, op.operand.binary_mask)
