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

import pytest
import torch
from copy import deepcopy
from pytest import approx
from torch import nn

from nncf.config import Config
from nncf.module_operations import UpdateWeight
from nncf.sparsity.rb.algo import RBSparsityController
from nncf.sparsity.rb.layers import RBSparsifyingWeight
from nncf.sparsity.rb.loss import SparseLoss
from nncf.sparsity.schedulers import PolynomialSparseScheduler
from tests.test_helpers import BasicConvTestModel, TwoConvTestModel, create_compressed_model_and_algo_for_test, \
    check_correct_nncf_modules_replacement


def get_basic_sparsity_config(model_size=4, input_sample_size=(1, 1, 4, 4),
                              sparsity_init=0.02, sparsity_target=0.5, sparsity_steps=2, sparsity_training_steps=3):
    config = Config()
    config.update({
        "model": "basic_sparse_conv",
        "model_size": model_size,
        "input_info":
            {
                "sample_size": input_sample_size,
            },
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
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(deepcopy(model), config)
    assert isinstance(compression_ctrl, RBSparsityController)

    _, sparse_model_conv = check_correct_nncf_modules_replacement(model, sparse_model)

    for sparse_module in sparse_model_conv.values():
        store = []
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
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    loss = compression_ctrl.loss
    assert isinstance(loss, SparseLoss)
    #pylint: disable=protected-access
    for layer in loss._sparse_layers:
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
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, RBSparsityController)
    for m in compression_ctrl.sparsified_module_info:
        assert isinstance(m.operand, RBSparsifyingWeight)


def test_can_create_sparse_loss_and_scheduler():
    model = BasicConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    loss = compression_ctrl.loss
    assert isinstance(loss, SparseLoss)
    assert not loss.disabled
    assert loss.target_sparsity_rate == approx(0.02)
    assert loss.p == approx(0.05)

    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, PolynomialSparseScheduler)
    assert scheduler.current_sparsity_level == approx(0.02)
    assert scheduler.max_sparsity == approx(0.5)
    assert scheduler.max_step == 2
    assert scheduler.sparsity_training_steps == 3


def test_sparse_algo_can_calc_sparsity_rate__for_basic_model():
    model = BasicConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.sparsified_weights_count == model.weights_num
    assert compression_ctrl.sparsity_rate_for_model == (
        1 - (model.nz_weights_num + model.nz_bias_num) / (model.weights_num + model.bias_num)
    )
    assert compression_ctrl.sparsity_rate_for_sparsified_modules == 1 - model.nz_weights_num / model.weights_num
    assert len(compression_ctrl.sparsified_module_info) == 1


def test_sparse_algo_can_collect_sparse_layers():
    model = TwoConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert len(compression_ctrl.sparsified_module_info) == 2


def test_sparse_algo_can_calc_sparsity_rate__for_2_conv_model():
    model = TwoConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.sparsified_weights_count == model.weights_num
    assert compression_ctrl.sparsity_rate_for_model == (
        1 - (model.nz_weights_num + model.nz_bias_num) / (model.weights_num + model.bias_num)
    )
    assert compression_ctrl.sparsity_rate_for_sparsified_modules == 1 - model.nz_weights_num / model.weights_num


def test_scheduler_can_do_epoch_step__with_rb_algo():
    config = Config()
    config['input_info'] = [{"sample_size": [1, 1, 32, 32]}]
    config['compression']['algorithm'] = 'rb_sparsity'

    config['compression']["params"] = {
        'schedule': 'polynomial',
        'power': 1, 'sparsity_steps': 2, 'sparsity_init': 0.2, 'sparsity_target': 0.6,
        'sparsity_training_steps': 4
    }

    _, compression_ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    scheduler = compression_ctrl.scheduler
    loss = compression_ctrl.loss

    assert pytest.approx(loss.target_sparsity_rate) == 0.2
    assert not loss.disabled

    for module_info in compression_ctrl.sparsified_module_info:
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
    for module_info in compression_ctrl.sparsified_module_info:
        assert not module_info.operand.sparsify
