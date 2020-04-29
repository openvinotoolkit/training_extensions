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
import pytest
import torch

from examples.common.optimizer import make_optimizer, get_parameter_groups
from nncf.module_operations import UpdateWeight
from nncf.pruning.filter_pruning.algo import FilterPruningController
from nncf.pruning.filter_pruning.functions import l2_filter_norm
from nncf.pruning.filter_pruning.layers import FilterPruningBlock, apply_filter_binary_mask
from nncf.pruning.schedulers import BaselinePruningScheduler
from tests.pruning.test_helpers import get_basic_pruning_config, PruningTestModel, \
    BigPruningTestModel, create_dataloader
from tests.test_helpers import create_compressed_model_and_algo_for_test, check_correct_nncf_modules_replacement


def create_pruning_algo_with_config(config):
    """
    Create filter_pruning with default params.
    :param config: config for the algorithm
    :return pruned model, pruning_algo, nncf_modules
    """
    config['compression']['algorithm'] = 'filter_pruning'
    model = BigPruningTestModel()
    pruned_model, pruning_algo = create_compressed_model_and_algo_for_test(BigPruningTestModel(), config)

    # Check that all modules was correctly replaced by NNCF modules and return this NNCF modules
    _, nncf_modules = check_correct_nncf_modules_replacement(model, pruned_model)
    return pruned_model, pruning_algo, nncf_modules


def test_check_default_algo_params():
    """
    Test for default algorithm params. Creating empty config and check for valid default
    parameters.
    """
    # Creating algorithm with empty config
    config = get_basic_pruning_config()
    config['compression']['algorithm'] = 'filter_pruning'
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, FilterPruningController)
    scheduler = compression_ctrl.scheduler
    # Check default algo params
    assert compression_ctrl.prune_first is False
    assert compression_ctrl.prune_last is False
    assert compression_ctrl.prune_batch_norms is False
    assert compression_ctrl.filter_importance is l2_filter_norm

    assert compression_ctrl.all_weights is False
    assert compression_ctrl.zero_grad is True

    # Check default scheduler params
    assert isinstance(scheduler, BaselinePruningScheduler)


@pytest.mark.parametrize(
    ('prune_first', 'prune_last'),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]

)
def test_valid_modules_replacement_and_pruning(prune_first, prune_last):
    """
    Test that checks that all conv modules in model was replaced by nncf modules and
    pruning pre ops were added correctly.
    :param prune_first: whether to prune first convolution or not
    :param prune_last: whether to prune last convolution or not
    """

    def check_that_module_is_pruned(module):
        assert len(module.pre_ops.values()) == 1
        op = list(module.pre_ops.values())[0]
        assert isinstance(op, UpdateWeight)
        assert isinstance(op.operand, FilterPruningBlock)

    config = get_basic_pruning_config(input_sample_size=(1, 1, 8, 8))
    config['compression']['params']['prune_first_conv'] = prune_first
    config['compression']['params']['prune_last_conv'] = prune_last

    pruned_model, pruning_algo, nncf_modules = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_info
    pruned_modules = [minfo.module for minfo in pruned_module_info]

    # Check for conv1
    conv1 = pruned_model.conv1
    if prune_first:
        assert conv1 in pruned_modules
        assert conv1 in nncf_modules.values()
        check_that_module_is_pruned(conv1)

    # Check for conv2
    conv2 = pruned_model.conv2
    assert conv2 in pruned_modules
    assert conv2 in nncf_modules.values()
    check_that_module_is_pruned(conv2)

    # Check for conv3
    conv3 = pruned_model.conv3
    if prune_last:
        assert conv3 in pruned_modules
        assert conv3 in nncf_modules.values()
        check_that_module_is_pruned(conv3)


@pytest.mark.parametrize(('all_weights', 'prune_first', 'ref_masks'),
                         [(False, True, [torch.tensor([0.0] * 8 + [1.0] * 8), torch.tensor([0.0] * 16 + [1.0] * 16)]),
                          (True, True, [torch.tensor([0.0] * 7 + [1.0] * 9), torch.tensor([0.0] * 17 + [1.0] * 15)]),
                          (False, False, [torch.tensor([0.0] * 16 + [1.0] * 16)]),
                          (True, False, [torch.tensor([0.0] * 16 + [1.0] * 16)]),
                          ]
                         )
def test_pruning_masks_correctness(all_weights, prune_first, ref_masks):
    """
    Test for pruning masks check (_set_binary_masks_for_filters, _set_binary_masks_for_all_filters_together).
    :param all_weights: whether mask will be calculated for all weights in common or not
    :param prune_first: whether to prune first convolution or not
    :param ref_masks: reference masks values
    """

    def check_mask(module, num):
        op = list(module.pre_ops.values())[0]
        assert hasattr(op.operand, 'binary_filter_pruning_mask')
        assert torch.allclose(op.operand.binary_filter_pruning_mask, ref_masks[num])

    config = get_basic_pruning_config(input_sample_size=(1, 1, 8, 8))
    config['compression']['params']['all_weights'] = all_weights
    config['compression']['params']['prune_first_conv'] = prune_first

    pruned_model, pruning_algo, _ = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_info
    pruned_modules = [minfo.module for minfo in pruned_module_info]
    assert pruning_algo.pruning_rate == 0.5
    assert pruning_algo.all_weights is all_weights

    i = 0
    # Check for conv1
    conv1 = pruned_model.conv1
    if prune_first:
        assert conv1 in pruned_modules
        check_mask(conv1, i)
        i += 1

    # Check for conv2
    conv2 = pruned_model.conv2
    assert conv2 in pruned_modules
    check_mask(conv2, i)


@pytest.mark.parametrize('prune_bn',
                         [False,
                          True]
                         )
def test_applying_masks(prune_bn):
    config = get_basic_pruning_config(input_sample_size=(1, 1, 8, 8))
    config['compression']['params']['prune_batch_norms'] = prune_bn
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['prune_last_conv'] = True

    pruned_model, pruning_algo, nncf_modules = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_info
    pruned_modules = [minfo.module for minfo in pruned_module_info]

    assert len(pruned_modules) == len(nncf_modules)

    for module in pruned_modules:
        op = list(module.pre_ops.values())[0]
        mask = op.operand.binary_filter_pruning_mask
        masked_weight = apply_filter_binary_mask(mask, module.weight)
        masked_bias = apply_filter_binary_mask(mask, module.bias)
        assert torch.allclose(module.weight, masked_weight)
        assert torch.allclose(module.bias, masked_bias)

    # Have only one BN node in graph
    bn_module = pruned_model.bn
    conv_for_bn = pruned_model.conv2
    bn_mask = list(conv_for_bn.pre_ops.values())[0].operand.binary_filter_pruning_mask
    if prune_bn:
        masked_bn_weight = apply_filter_binary_mask(bn_mask, bn_module.weight)
        masked_bn_bias = apply_filter_binary_mask(bn_mask, bn_module.bias)
        assert torch.allclose(bn_module.weight, masked_bn_weight)
        assert torch.allclose(bn_module.bias, masked_bn_bias)


@pytest.mark.parametrize('zero_grad',
                         [True, False])
def test_zeroing_gradients(zero_grad):
    """
    Test for zeroing gradients functionality (zero_grads_for_pruned_modules in base algo)
    :param zero_grad: zero grad or not
    """
    config = get_basic_pruning_config(input_sample_size=(2, 1, 8, 8))
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['prune_last_conv'] = True
    config['compression']['params']['zero_grad'] = zero_grad

    pruned_model, pruning_algo, _ = create_pruning_algo_with_config(config)
    assert pruning_algo.zero_grad is zero_grad

    pruned_module_info = pruning_algo.pruned_module_info
    pruned_modules = [minfo.module for minfo in pruned_module_info]

    device = next(pruned_model.parameters()).device
    data_loader = create_dataloader(config)

    pruning_algo.initialize(data_loader)

    params_to_optimize = get_parameter_groups(pruned_model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    lr_scheduler.step(0)

    pruned_model.train()
    for input_, target in data_loader:
        input_ = input_.to(device)
        target = target.to(device).view(1)

        output = pruned_model(input_)

        loss = torch.sum(target.to(torch.float32) - output)

        optimizer.zero_grad()
        loss.backward()

        # In case of zero_grad = True gradients should be masked
        if zero_grad:
            for module in pruned_modules:
                op = list(module.pre_ops.values())[0]
                mask = op.operand.binary_filter_pruning_mask
                grad = module.weight.grad
                masked_grad = apply_filter_binary_mask(mask, grad)
                assert torch.allclose(masked_grad, grad)
