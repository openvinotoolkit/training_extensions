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

from nncf.pruning.schedulers import BaselinePruningScheduler, ExponentialPruningScheduler, \
    ExponentialWithBiasPruningScheduler
from tests.pruning.test_helpers import PruningTestModel, get_basic_pruning_config
from tests.test_helpers import create_compressed_model_and_algo_for_test


@pytest.mark.parametrize('algo',
                         ('filter_pruning', ))
@pytest.mark.parametrize(('scheduler', 'scheduler_class'),
                         (
                             ('baseline', BaselinePruningScheduler),
                             ('exponential', ExponentialPruningScheduler),
                             ('exponential_with_bias', ExponentialWithBiasPruningScheduler),
                         ))
def test_can_choose_scheduler(algo, scheduler, scheduler_class):
    config = get_basic_pruning_config()
    config['compression']['algorithm'] = algo
    config['compression']['params']['schedule'] = scheduler
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, scheduler_class)


@pytest.mark.parametrize(
    ("algo", "ref_scheduler", "ref_scheduler_params"),
    (('filter_pruning', BaselinePruningScheduler, {'num_init_steps': 0, "pruning_steps": 100,
                                                   "initial_pruning": 0, "pruning_target": 0.5}),)
)
def test_check_default_scheduler_params(algo, ref_scheduler, ref_scheduler_params):
    config = get_basic_pruning_config()
    config['compression']['algorithm'] = algo
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, ref_scheduler)
    for key, value in ref_scheduler_params.items():
        assert getattr(scheduler, key) == value
