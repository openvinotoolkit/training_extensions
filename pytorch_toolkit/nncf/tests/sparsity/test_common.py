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

from nncf.sparsity.schedulers import PolynomialSparseScheduler, ExponentialSparsityScheduler, \
    AdaptiveSparsityScheduler, MultiStepSparsityScheduler
from tests.test_helpers import BasicConvTestModel, get_empty_config, create_compressed_model_and_algo_for_test, \
    MockModel


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity', 'rb_sparsity'))
@pytest.mark.parametrize(('schedule_type', 'scheduler_class'),
                         (
                             ('polynomial', PolynomialSparseScheduler),
                             ('exponential', ExponentialSparsityScheduler),
                             ('multistep', MultiStepSparsityScheduler)
                         ))


def test_can_choose_scheduler(algo, schedule_type, scheduler_class):
    config = get_empty_config()
    config['compression']['algorithm'] = algo
    config['compression']["params"]["schedule"] = schedule_type
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert isinstance(compression_ctrl.scheduler, scheduler_class)


def test_can_create_rb_algo__with_adaptive_scheduler():
    config = get_empty_config()
    config['compression']['algorithm'] = 'rb_sparsity'
    config['compression']["params"]["schedule"] = 'adaptive'
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert isinstance(compression_ctrl.scheduler, AdaptiveSparsityScheduler)


def test_can_not_create_magnitude_algo__with_adaptive_scheduler():
    config = get_empty_config()
    config['compression']['algorithm'] = 'magnitude_sparsity'
    config['compression']["params"]["schedule"] = 'adaptive'
    with pytest.raises(TypeError):
        _, _ = create_compressed_model_and_algo_for_test(MockModel(), config)


def get_poly_params():
    return {
        'power': 1, 'sparsity_steps': 2, 'sparsity_init': 0.2, 'sparsity_target': 0.6,
        'sparsity_training_steps': 4
    }


def get_multistep_params():
    return {
        'steps': [2, 3, 4], 'sparsity_levels': [0.2, 0.4, 0.5, 0.6],
        'sparsity_training_steps': 4
    }


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity', 'rb_sparsity'))
class TestSparseModules:
    def test_can_create_sparse_scheduler__with_defaults(self, algo):
        config = get_empty_config()
        config['compression']['algorithm'] = algo
        config['compression']["params"]["schedule"] = 'polynomial'
        _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
        scheduler = compression_ctrl.scheduler
        assert scheduler.initial_sparsity == 0
        assert scheduler.max_sparsity == 0.5
        assert scheduler.max_step == 90
        assert scheduler.sparsity_training_steps == 100

    @pytest.mark.parametrize(('schedule', 'get_params', 'ref_levels'),
                             (('polynomial', get_poly_params, [0.2, 0.4, 0.6, 0.6, 0.6, 0.6]),
                              ('exponential', get_poly_params, [0.2, 0.4343145, 0.6, 0.6, 0.6, 0.6]),
                              ('multistep', get_multistep_params, [0.2, 0.2, 0.4, 0.5, 0.6, 0.6])))
    def test_scheduler_can_do_epoch_step(self, algo, schedule, get_params, ref_levels):
        model = BasicConvTestModel()
        config = get_empty_config()
        config['compression']['algorithm'] = algo
        config['compression']["params"] = get_params()
        config['compression']["params"]["schedule"] = schedule

        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        scheduler = compression_ctrl.scheduler

        assert pytest.approx(scheduler.current_sparsity_level) == ref_levels[0]
        for ref_level in ref_levels[1:]:
            scheduler.epoch_step()
            assert pytest.approx(scheduler.current_sparsity_level) == ref_level

        for m in compression_ctrl.sparsified_module_info:
            if hasattr(m.operand, "sparsify"):
                assert not m.operand.sparsify
