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
import numpy as np
import pytest

from nncf.pruning.schedulers import BaselinePruningScheduler, ExponentialWithBiasPruningScheduler
from tests.pruning.test_helpers import get_pruning_baseline_config, PruningTestModel, get_pruning_exponential_config
from tests.test_helpers import create_compressed_model_and_algo_for_test


def test_baseline_scheduler():
    """
    Test baseline scheduler parameters and changes of params during epochs.
    """
    config = get_pruning_baseline_config()
    config['compression']['algorithm'] = 'filter_pruning'
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler

    # Check default params
    assert isinstance(scheduler, BaselinePruningScheduler)
    assert pytest.approx(scheduler.pruning_target) == 0.5
    assert pytest.approx(scheduler.initial_pruning) == 0.0
    assert scheduler.num_init_steps == 1

    # Check pruning params on epoch 0
    assert pytest.approx(scheduler.current_pruning_level) == 0.0
    assert pytest.approx(compression_ctrl.pruning_rate) == 0.0
    assert scheduler.last_epoch == 0
    assert compression_ctrl.frozen is False

    # Check pruning params on epoch 1
    scheduler.epoch_step()
    assert pytest.approx(scheduler.current_pruning_level) == 0.5
    assert pytest.approx(compression_ctrl.pruning_rate) == 0.5
    assert scheduler.last_epoch == 1
    assert compression_ctrl.frozen is True


def test_exponential_scheduler():
    """
    Test exponential with bias scheduler parameters and changes of params during epochs.
    """
    config = get_pruning_exponential_config()
    config['compression']['algorithm'] = 'filter_pruning'
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler

    # Check default params
    assert isinstance(scheduler, ExponentialWithBiasPruningScheduler)
    assert pytest.approx(scheduler.pruning_target) == 0.5
    assert pytest.approx(scheduler.initial_pruning) == 0.0
    assert scheduler.num_init_steps == 1
    assert scheduler.pruning_steps == 20
    assert pytest.approx(scheduler.a, abs=1e-4) == -0.5
    assert pytest.approx(scheduler.b, abs=1e-4) == 0.5
    assert pytest.approx(scheduler.k, abs=1e-4) == 0.5544

    # Check pruning params on epoch 0
    assert pytest.approx(scheduler.current_pruning_level) == 0.0
    assert pytest.approx(compression_ctrl.pruning_rate) == 0.0
    assert compression_ctrl.frozen is False
    assert scheduler.last_epoch == 0

    # Check pruning params on epoch 1 - 20
    for i in range(20):
        # Check pruning params on epoch 2
        scheduler.epoch_step()
        pruning_rate = scheduler.a * np.exp(
            -scheduler.k * (scheduler.last_epoch - scheduler.num_init_steps)) + scheduler.b
        assert pytest.approx(scheduler.current_pruning_level) == pruning_rate
        assert pytest.approx(compression_ctrl.pruning_rate) == pruning_rate
        assert compression_ctrl.frozen is False
        assert scheduler.last_epoch == i + 1

    # Check pruning params on epoch 3
    scheduler.epoch_step()
    assert pytest.approx(scheduler.current_pruning_level) == 0.5
    assert pytest.approx(compression_ctrl.pruning_rate) == 0.5
    assert compression_ctrl.frozen is True
    assert scheduler.last_epoch == 21
