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

import pytest

from nncf.algo_selector import create_compression_algorithm
from nncf.sparsity.schedulers import MultiStepSparsityScheduler
from tests.sparsity.magnitude.test_helpers import MagnitudeTestModel, get_basic_magnitude_sparsity_config
from tests.test_helpers import get_empty_config


def get_multistep_normed_abs_config():
    config = get_basic_magnitude_sparsity_config()
    compression_config = config['compression']
    compression_config['weight_importance'] = 'normed_abs'
    compression_config['params'] = {
        'schedule': 'multistep',
        'steps': [1, 3],
        'sparsity_levels': [0.1, 0.5, 0.9]
    }
    return config


def test_magnitude_scheduler_can_do_epoch_step__with_norm():
    model = MagnitudeTestModel()
    config = get_multistep_normed_abs_config()
    compression_algo = create_compression_algorithm(model, config)
    scheduler = compression_algo.scheduler
    assert isinstance(scheduler, MultiStepSparsityScheduler)

    assert pytest.approx(compression_algo.sparsity_level) == 0.1
    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.threshold == pytest.approx(0.219, 0.01)
    assert scheduler.prev_ind == 0

    scheduler.epoch_step()
    assert compression_algo.sparsity_level == 0.5
    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.threshold == pytest.approx(0.243, 0.01)
    assert scheduler.prev_ind == 1

    scheduler.epoch_step()
    assert compression_algo.sparsity_level == 0.5
    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.threshold == pytest.approx(0.243, 0.01)
    assert scheduler.prev_ind == 1

    scheduler.epoch_step()
    assert compression_algo.sparsity_level == 0.9
    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.threshold == pytest.approx(0.371, 0.01)
    assert scheduler.prev_ind == 2


def test_magnitude_scheduler_can_do_epoch_step__with_last():
    model = MagnitudeTestModel()
    config = get_multistep_normed_abs_config()
    compression_algo = create_compression_algorithm(model, config)
    scheduler = compression_algo.scheduler

    scheduler.epoch_step(3)
    assert scheduler.prev_ind == 2
    assert compression_algo.sparsity_level == 0.9
    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.threshold == pytest.approx(0.371, 0.01)

    scheduler.epoch_step()
    assert scheduler.prev_ind == 2
    assert compression_algo.sparsity_level == 0.9
    for module_info in compression_algo.sparsified_module_info:
        assert module_info.operand.threshold == pytest.approx(0.371, 0.01)


def test_magnitude_scheduler_can_do_epoch_step__with_multistep():
    model = MagnitudeTestModel()
    config = get_empty_config()
    config["compression"] = {"algorithm": "magnitude_sparsity", "params": {"schedule": "multistep", 'steps': [1]}}
    compression_algo = create_compression_algorithm(model, config)
    scheduler = compression_algo.scheduler
    assert isinstance(scheduler, MultiStepSparsityScheduler)
    assert pytest.approx(compression_algo.sparsity_level) == 0.1
    assert scheduler.sparsity_levels == [0.1, 0.5]
    scheduler.epoch_step()
    assert compression_algo.sparsity_level == 0.5
    scheduler.epoch_step()
    assert compression_algo.sparsity_level == 0.5
