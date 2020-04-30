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

from copy import deepcopy

import pytest
import torch
from pytest import approx

from nncf.module_operations import UpdateWeight
from nncf.sparsity.layers import BinaryMask
from nncf.sparsity.magnitude.algo import MagnitudeSparsityController
from nncf.sparsity.magnitude.functions import normed_magnitude
from tests.quantization.test_functions import check_equal
from tests.sparsity.const.test_algo import ref_mask_2, ref_mask_1
from tests.sparsity.magnitude.test_helpers import MagnitudeTestModel, get_basic_magnitude_sparsity_config
from tests.test_helpers import create_compressed_model_and_algo_for_test, MockModel, BasicConvTestModel, \
    get_empty_config, check_correct_nncf_modules_replacement


def test_can_create_magnitude_sparse_algo__with_defaults():
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = \
        {'schedule': 'multistep'}
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(deepcopy(model), config)

    assert isinstance(compression_ctrl, MagnitudeSparsityController)
    assert compression_ctrl.sparsity_level == approx(0.1)
    assert len(list(sparse_model.modules())) == 12

    _, sparse_model_conv = check_correct_nncf_modules_replacement(model, sparse_model)

    i = 0
    for sparse_module in sparse_model_conv.values():
        store = []
        ref_mask = torch.ones_like(sparse_module.weight) if i == 0 else ref_mask_2
        i += 1
        for op in sparse_module.pre_ops.values():
            if isinstance(op, UpdateWeight) and isinstance(op.operand, BinaryMask):
                assert compression_ctrl.threshold == approx(0.24, 0.1)
                assert torch.allclose(op.operand.binary_mask, ref_mask)
                assert isinstance(compression_ctrl.weight_importance, type(normed_magnitude))
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
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if sparsity_level:
        compression_ctrl.set_sparsity_level(sparsity_level)
    assert compression_ctrl.threshold == pytest.approx(threshold, 0.01)


def test_can_not_set_sparsity_more_than_one_for_magnitude_sparse_algo():
    config = get_basic_magnitude_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    with pytest.raises(AttributeError):
        compression_ctrl.set_sparsity_level(1)
        compression_ctrl.set_sparsity_level(1.2)


def test_can_not_create_magnitude_algo__without_steps():
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'sparsity_levels': [0.1]}
    with pytest.raises(AttributeError):
        _, _ = create_compressed_model_and_algo_for_test(MockModel(), config)


def test_can_create_magnitude_algo__without_levels():
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'steps': [1]}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert compression_ctrl.sparsity_level == approx(0.1)


def test_can_not_create_magnitude_algo__with_not_matched_steps_and_levels():
    config = get_basic_magnitude_sparsity_config()
    config['compression']['params'] = {'schedule': 'multistep', 'sparsity_levels': [0.1], 'steps': [1, 2]}
    with pytest.raises(AttributeError):
        _, _ = create_compressed_model_and_algo_for_test(MockModel(), config)


def test_magnitude_algo_set_binary_mask_on_forward():
    config = get_basic_magnitude_sparsity_config()
    config['compression']['weight_importance'] = 'abs'
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    compression_ctrl.set_sparsity_level(0.3)
    with torch.no_grad():
        sparse_model(torch.ones([1, 1, 10, 10]))

    op = sparse_model.conv1.pre_ops['0']
    check_equal(ref_mask_1, op.operand.binary_mask)

    op = sparse_model.conv2.pre_ops['0']
    check_equal(ref_mask_2, op.operand.binary_mask)


def test_magnitude_algo_binary_masks_are_applied():
    model = BasicConvTestModel()
    config = get_empty_config()
    config['compression']['algorithm'] = "magnitude_sparsity"
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    minfo_list = compression_ctrl.sparsified_module_info  # type: List[SparseModuleInfo]
    minfo = minfo_list[0]  # type: SparseModuleInfo

    minfo.operand.binary_mask = torch.ones_like(minfo.module.weight)  # 1x1x2x2
    input_ = torch.ones(size=(1, 1, 5, 5))
    ref_output_1 = -4 * torch.ones(size=(2, 4, 4))
    output_1 = compressed_model(input_)
    assert torch.all(torch.eq(output_1, ref_output_1))

    minfo.operand.binary_mask[0][0][0][1] = 0
    minfo.operand.binary_mask[1][0][1][0] = 0
    ref_output_2 = - 3 * torch.ones_like(ref_output_1)
    output_2 = compressed_model(input_)
    assert torch.all(torch.eq(output_2, ref_output_2))

    minfo.operand.binary_mask[1][0][0][1] = 0
    ref_output_3 = ref_output_2.clone()
    ref_output_3[1] = -2 * torch.ones_like(ref_output_1[1])
    output_3 = compressed_model(input_)
    assert torch.all(torch.eq(output_3, ref_output_3))
