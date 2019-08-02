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
from torch import nn

from nncf.algo_selector import create_compression_algorithm
from nncf.config import Config
from nncf.operations import UpdateWeight
from nncf.sparsity import RBSparsity
from nncf.sparsity.rb.layers import RBSparsifyingWeight
from nncf.sparsity.rb.loss import SparseLoss
from nncf.sparsity.schedulers import PolynomialSparseScheduler
from nncf.utils import get_all_modules_by_type
from tests.test_helpers import BasicConvTestModel, TwoConvTestModel


def get_basic_sparsity_config(model_size=4, input_sample_size=(1, 1, 4, 4),
                              sparsity_init=0.02, sparsity_target=0.5, sparsity_steps=2, sparsity_training_steps=3):
    config = Config()
    config.update({
        "model": "basic_sparse_conv",
        "model_size": model_size,
        "input_sample_size": input_sample_size,
        "compression":
            {
                "algorithm": "rb_sparsity",
                "params":
                    {
                        "schedule": "polynomial",
                        "sparsity_init": sparsity_init,
                        "sparsity_target": sparsity_target,
                        "sparsity_steps": sparsity_steps,
                        "sparsity_training_steps": sparsity_training_steps
                    },

                "layers":
                    {
                        "conv": {"sparsify": True},
                    }
            }
    })
    return config


def test_can_load_sparse_algo__with_defaults():
    model = BasicConvTestModel()
    config = get_basic_sparsity_config()
    compression_algo = create_compression_algorithm(deepcopy(model), config)
    assert isinstance(compression_algo, RBSparsity)
    sparse_model = compression_algo.model

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
            if isinstance(op, UpdateWeight) and isinstance(op.operand, RBSparsifyingWeight):
                assert torch.allclose(op.operand.binary_mask, torch.ones_like(sparse_module.weight))
                assert op.operand.sparsify
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)


def test_can_set_sparse_layers_to_loss():
    model = BasicConvTestModel()
    config = get_basic_sparsity_config()
    config['compression']['train_phase'] = ''
    compression_algo = create_compression_algorithm(model, config)
    loss = compression_algo.loss
    assert isinstance(loss, SparseLoss)
    for layer in loss.sparse_layers:
        assert isinstance(layer, RBSparsifyingWeight)


def test_sparse_algo_does_not_replace_not_conv_layer():
    class TwoLayersTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.bn = nn.BatchNorm2d(1)

        def forward(self, x):
            return self.bn(self.conv(x))

    model = TwoLayersTestModel()
    config = get_basic_sparsity_config()
    config['compression']['train_phase'] = ''
    compression_algo = create_compression_algorithm(model, config)
    assert isinstance(compression_algo, RBSparsity)
    for m in compression_algo.sparsified_module_info:
        assert isinstance(m.operand, RBSparsifyingWeight)


def test_can_create_sparse_loss_and_scheduler():
    model = BasicConvTestModel()

    config = get_basic_sparsity_config()
    compression_algo = create_compression_algorithm(model, config)

    loss = compression_algo.loss
    assert isinstance(loss, SparseLoss)
    assert not loss.disabled
    assert loss.target_sparsity_rate == approx(0.02)
    assert loss.p == approx(0.05)

    scheduler = compression_algo.scheduler
    assert isinstance(scheduler, PolynomialSparseScheduler)
    assert scheduler.current_sparsity_level == approx(0.02)
    assert scheduler.max_sparsity == approx(0.5)
    assert scheduler.max_step == 2
    assert scheduler.sparsity_training_steps == 3


def test_sparse_algo_can_calc_sparsity_rate__for_basic_model():
    model = BasicConvTestModel()

    config = get_basic_sparsity_config()
    compression_algo = create_compression_algorithm(model, config)

    assert compression_algo.sparsified_weights_count == model.weights_num
    assert compression_algo.sparsity_rate_for_model == (
        1 - (model.nz_weights_num + model.nz_bias_num) / (model.weights_num + model.bias_num)
    )
    assert compression_algo.sparsity_rate_for_sparsified_modules == 1 - model.nz_weights_num / model.weights_num
    assert len(compression_algo.sparsified_module_info) == 1


def test_sparse_algo_can_collect_sparse_layers():
    model = TwoConvTestModel()

    config = get_basic_sparsity_config()
    compression_algo = create_compression_algorithm(model, config)

    assert len(compression_algo.sparsified_module_info) == 2


def test_sparse_algo_can_calc_sparsity_rate__for_2_conv_model():
    model = TwoConvTestModel()

    config = get_basic_sparsity_config()
    compression_algo = create_compression_algorithm(model, config)

    assert compression_algo.sparsified_weights_count == model.weights_num
    assert compression_algo.sparsity_rate_for_model == (
        1 - (model.nz_weights_num + model.nz_bias_num) / (model.weights_num + model.bias_num)
    )
    assert compression_algo.sparsity_rate_for_sparsified_modules == 1 - model.nz_weights_num / model.weights_num


def test_scheduler_can_do_epoch_step__with_rb_algo():
    config = Config()
    config['compression']['algorithm'] = 'rb_sparsity'

    config['compression']["params"] = {
        'schedule': 'polynomial',
        'power': 1, 'sparsity_steps': 2, 'sparsity_init': 0.2, 'sparsity_target': 0.6,
        'sparsity_training_steps': 4
    }
    compression_algo = create_compression_algorithm(BasicConvTestModel(), config)
    scheduler = compression_algo.scheduler
    loss = compression_algo.loss

    assert pytest.approx(loss.target_sparsity_rate) == 0.2
    assert not loss.disabled

    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.sparsify
    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.4
    assert pytest.approx(loss().item(), abs=1e-3) == 64
    assert not loss.disabled

    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.6
    assert pytest.approx(loss().item(), abs=1e-3) == 144
    assert not loss.disabled

    scheduler.epoch_step()
    assert not loss.disabled
    assert loss.target_sparsity_rate == 0.6
    assert loss().item() == 144

    scheduler.epoch_step()
    assert loss.disabled
    assert loss.target_sparsity_rate == 0.6
    assert loss() == 0
    for module_info in compression_algo.sparsified_module_info:
        assert not module_info.operand.sparsify
