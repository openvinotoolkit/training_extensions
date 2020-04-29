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

import os
import pytest
import torch

from examples.common.distributed import configure_distributed
from examples.common.execution import ExecutionMode, prepare_model_for_execution, get_device
from examples.common.model_loader import load_model
from nncf.config import Config
from nncf.checkpoint_loading import load_state
from tests.conftest import TEST_ROOT
from tests.test_helpers import create_compressed_model_and_algo_for_test
from tests.test_sanity_sample import Command, create_command_line
from tests.test_compression_training import get_cli_dict_args, parse_best_acc1

GLOBAL_CONFIG = {
    TEST_ROOT.joinpath("data", "configs", "squeezenet1_1_cifar10_rb_sparsity_int8.json"): [
        {
            'checkpoint_name': 'squeezenet1_1_custom_cifar10_rb_sparsity_int8_dp.pth',
            'dataset': "cifar10",
            'execution_mode': ExecutionMode.GPU_DATAPARALLEL,
        },
        {
            'checkpoint_name': 'squeezenet1_1_custom_cifar10_rb_sparsity_int8_ddp.pth',
            'dataset': "cifar10",
            'execution_mode': ExecutionMode.MULTIPROCESSING_DISTRIBUTED,
        },
    ],
}

CONFIG_PARAMS = []
for config_path_, cases_list_ in GLOBAL_CONFIG.items():
    for case_params_ in cases_list_:
        CONFIG_PARAMS.append((config_path_, case_params_,))


@pytest.fixture(scope='module', params=CONFIG_PARAMS,
                ids=['-'.join([str(p[0]), p[1]['execution_mode']]) for p in CONFIG_PARAMS])
def _params(request, backward_compat_models_path):
    if backward_compat_models_path is None:
        pytest.skip('Path to models weights for backward compatibility testing is not set,'
                    ' use --backward-compat-models option.')
    config_path, case_params = request.param
    checkpoint_path = str(os.path.join(backward_compat_models_path, case_params['checkpoint_name']))
    return {
        'nncf_config_path': config_path,
        'checkpoint_path': checkpoint_path,
        'execution_mode': case_params['execution_mode'],
        'dataset': case_params['dataset']
    }


def test_model_can_be_loaded_with_resume(_params):
    p = _params
    config_path = p['nncf_config_path']
    checkpoint_path = p['checkpoint_path']

    config = Config.from_json(str(config_path))
    config.execution_mode = p['execution_mode']

    config.current_gpu = 0
    config.device = get_device(config)
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        config.dist_url = "tcp://127.0.0.1:9898"
        config.dist_backend = "nccl"
        config.rank = 0
        config.world_size = 1
        configure_distributed(config)

    model_name = config['model']
    model = load_model(model_name,
                       pretrained=False,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'))

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model, _ = prepare_model_for_execution(model, config)

    if config.distributed:
        compression_ctrl.distributed()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(model, checkpoint['state_dict'], is_resume=True)


def test_loaded_model_evals_according_to_saved_acc(_params, tmp_path):
    p = _params
    config_path = p['nncf_config_path']
    checkpoint_path = p['checkpoint_path']

    tmp_path = str(tmp_path)
    args = {}
    args['data'] = tmp_path + '/' + p['dataset']
    args['dataset'] = p['dataset']
    args['config'] = str(config_path)
    args['mode'] = 'test'
    args['log-dir'] = tmp_path
    args['workers'] = 4
    args['seed'] = 1
    args['resume'] = checkpoint_path

    if p['execution_mode'] == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        args['multiprocessing-distributed'] = ''
    else:
        pytest.skip("DataParallel eval takes too long for this test to be run during pre-commit")

    runner = Command(create_command_line(get_cli_dict_args(args), "classification"))
    res = runner.run()
    assert res == 0

    acc1 = parse_best_acc1(tmp_path)
    assert torch.load(checkpoint_path)['best_acc1'] == pytest.approx(acc1)
