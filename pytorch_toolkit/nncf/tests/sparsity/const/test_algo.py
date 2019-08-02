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

import torch

from examples.common.model_loader import load_state
from nncf.algo_selector import create_compression_algorithm
from nncf.dynamic_graph import reset_context
from nncf.operations import UpdateWeight
from nncf.sparsity.const.algo import ConstSparsity
from nncf.sparsity.layers import BinaryMask
from nncf.utils import get_all_modules_by_type
from tests.quantization.test_functions import check_equal
from tests.sparsity.magnitude.test_helpers import MagnitudeTestModel
from tests.test_helpers import BasicConvTestModel, get_empty_config

sub_tensor = torch.tensor([[[[1., 0.],
                             [0., 1.]]]])
ref_mask_1 = torch.cat((sub_tensor, sub_tensor), 0)
sub_tensor = torch.tensor([[[[0., 1., 1.],
                             [1., 0., 1.],
                             [1., 1., 0.]]]])
ref_mask_2 = torch.cat((sub_tensor, sub_tensor), 1)


def test_can_create_const_sparse_algo__with_default():
    model = BasicConvTestModel()
    config = get_empty_config()
    config["compression"] = {"algorithm": "const_sparsity"}
    compression_algo = create_compression_algorithm(deepcopy(model), config)

    assert isinstance(compression_algo, ConstSparsity)
    sparse_model = compression_algo.model
    assert len(list(sparse_model.modules())) == 6

    model_conv = get_all_modules_by_type(model, 'Conv2d')
    sparse_model_conv = get_all_modules_by_type(sparse_model, 'NNCFConv2d')
    assert len(model_conv) == len(sparse_model_conv)

    for module_name in model_conv:
        scope = module_name.split('/')
        scope[-1] = scope[-1].replace('Conv2d', 'NNCFConv2d')
        sparse_module_name = '/'.join(scope)
        assert sparse_module_name in sparse_model_conv

        store = []
        sparse_module = sparse_model_conv[sparse_module_name]
        for op in sparse_module.pre_ops.values():
            if isinstance(op, UpdateWeight) and isinstance(op.operand, BinaryMask):
                ref_mask = torch.ones_like(sparse_module.weight)
                assert torch.allclose(op.operand.binary_mask, ref_mask)
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)


def test_can_restore_binary_mask_on_magnitude_algo_resume():
    config = get_empty_config()
    config['compression'] = {"algorithm": "magnitude_sparsity", "weight_importance": "abs",
                             "params": {"schedule": "multistep", "sparsity_levels": [0.3, 0.5]}}
    magnitude_algo = create_compression_algorithm(MagnitudeTestModel(), config)
    sparse_model = magnitude_algo.model
    with torch.no_grad():
        sparse_model(torch.ones([1, 1, 10, 10]))

    config = get_empty_config()
    config["compression"] = {"algorithm": "const_sparsity"}
    const_algo = create_compression_algorithm(MagnitudeTestModel(), config)
    const_sparse_model = const_algo.model

    load_state(const_sparse_model, sparse_model.state_dict())

    op = const_sparse_model.conv1.pre_ops['0']
    check_equal(ref_mask_1, op.operand.binary_mask)

    op = const_sparse_model.conv2.pre_ops['0']
    check_equal(ref_mask_2, op.operand.binary_mask)


def test_can_restore_binary_mask_on_magnitude_quant_algo_resume():
    config = get_empty_config()
    config["compression"] = [
        {"algorithm": "magnitude_sparsity", "weight_importance": "abs",
         "params": {"schedule": "multistep", "sparsity_levels": [0.3, 0.5]}},
        {"algorithm": "quantization"}]
    reset_context('orig')
    reset_context('quantized_graphs')
    magnitude_quant_algo = create_compression_algorithm(MagnitudeTestModel(), config)
    # load_state doesn't support CPU + Quantization
    sparse_model = torch.nn.DataParallel(magnitude_quant_algo.model)
    sparse_model.cuda()
    with torch.no_grad():
        sparse_model(torch.ones([1, 1, 10, 10]))

    reset_context('orig')
    reset_context('quantized_graphs')
    config = get_empty_config()
    config["compression"] = [{"algorithm": "const_sparsity"}, {"algorithm": "quantization"}]
    const_algo = create_compression_algorithm(MagnitudeTestModel(), config)
    const_sparse_model = const_algo.model

    load_state(const_sparse_model, sparse_model.state_dict())

    op = const_sparse_model.module.conv1.pre_ops['0']
    check_equal(ref_mask_1, op.operand.binary_mask)

    op = const_sparse_model.module.conv2.pre_ops['0']
    check_equal(ref_mask_2, op.operand.binary_mask)
