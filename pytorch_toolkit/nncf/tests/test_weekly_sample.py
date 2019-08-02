'''
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''
import importlib
import json
import os
import tempfile

import pytest
import torch
from pytest import approx

from examples.common.utils import get_name
from nncf.dynamic_graph import reset_context
from tests.conftest import TEST_ROOT

# sample
# ├── dataset
# │   ├── path
# │   ├── batch
# │   ├── configs
# │   │     ├─── config_filename
# │   │     │       ├── expected_accuracy
# │   │     │       ├── absolute_tolerance
# │   │     │       ├── execution_arg
# │   │     │       ├── epochs
# │   │     │       ├── weights
GLOBAL_CONFIG = {
    'classification':
        {
            'cifar100':
                {
                    'batch': 256,
                    'configs': {
                        'mobilenetV2_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 64.53,
                            'weights': 'mobilenetV2_64.53.sd',
                            'absolute_tolerance': 1,  # it's default value, just to show usage
                        },
                        'inceptionV3_int8.json': {
                            'expected_accuracy': 77.53,
                            'weights': 'inceptionV3_77.53.sd',
                        },
                        'resnet50_int8.json': {
                            'expected_accuracy': 68.86,
                            'weights': 'resnet50_68.86.sd',
                        },
                        'mobilenetV2_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 64.53,
                            'weights': 'mobilenetV2_64.53.sd',
                            'execution_arg': {'multiprocessing-distributed', ''},
                        },
                        'mobilenetV2_rb_sparsity_int8.json': {
                            'expected_accuracy': 64.53,
                            'weights': 'mobilenetV2_64.53.sd',
                            'execution_arg': {'multiprocessing-distributed'},
                        }
                    }
                },
        }
}


def get_weights_path(dataset_name, sample_type, weights_name):
    if "WEEKLY_MODELS_PATH" not in os.environ:
        return None
    weights_path = os.path.join(os.environ["WEEKLY_MODELS_PATH"], sample_type, dataset_name, weights_name)
    if not os.path.exists(weights_path):
        raise FileExistsError('Weights file does not exist: {}'.format(weights_path))
    return weights_path


def get_cli_args(args):
    cli_args = []
    for key, val in args.items():
        cli_args.append('--{}'.format(str(key)))
        if val:
            cli_args.append(str(val))
    return cli_args


CONFIG_PARAMS = []
for sample_type_ in GLOBAL_CONFIG:
    datasets = GLOBAL_CONFIG[sample_type_]
    for dataset_name_ in datasets:
        dataset_path = datasets[dataset_name_].get('path', os.path.join(tempfile.gettempdir(), dataset_name_))
        batch_size = datasets[dataset_name_].get('batch', 256)
        configs = datasets[dataset_name_].get('configs', {})
        for config_name in configs:
            config_params = configs[config_name]
            execution_args = config_params.get('execution_arg', [''])
            expected_accuracy_ = config_params.get('expected_accuracy', 100)
            absolute_tolerance_ = config_params.get('absolute_tolerance', 1)
            weights_name_ = config_params.get('weights', None)
            epochs = config_params.get('epochs', None)
            for execution_arg_ in execution_args:
                weights_path_ = os.path.join(sample_type_, dataset_name_, weights_name_)
                config_path_ = TEST_ROOT.joinpath("data", "configs", "weekly", sample_type_, dataset_name_, config_name)
                jconfig = json.load(config_path_.open())
                args_ = {
                    'data': dataset_path,
                    'batch-size': batch_size,
                    'weights': weights_path_,
                    'config': str(config_path_)
                }
                if epochs:
                    args_['epochs'] = epochs
                test_config_ = {
                    'sample_type': sample_type_,
                    'expected_accuracy': expected_accuracy_,
                    'absolute_tolerance': absolute_tolerance_,
                    'checkpoint_name': get_name(jconfig)
                }
                CONFIG_PARAMS.append(tuple([test_config_, args_, execution_arg_]))


def get_config_name(config_path):
    base = os.path.basename(config_path)
    return os.path.splitext(base)[0]


@pytest.fixture(scope='module', params=CONFIG_PARAMS,
                ids=['-'.join([p[0]['sample_type'], get_config_name(p[1]['config']), p[2]]) for p in CONFIG_PARAMS])
def _params(request, tmp_path_factory, dataset_dir, weekly_models_path):
    if weekly_models_path is None:
        pytest.skip('Path to models weights for weekly testing is not set, use --weekly-models option.')
    test_config, args, execution_arg = request.param
    weights_path = os.path.join(weekly_models_path, args['weights'])
    if not os.path.exists(weights_path):
        raise FileExistsError('Weights file does not exist: {}'.format(weights_path))
    args['weights'] = weights_path
    if execution_arg:
        args[execution_arg] = None
    checkpoint_save_dir = str(tmp_path_factory.mktemp('models'))
    checkpoint_save_dir = os.path.join(checkpoint_save_dir, execution_arg.replace('-', '_'))
    args['checkpoint-save-dir'] = checkpoint_save_dir
    if dataset_dir:
        args['data'] = dataset_dir
    return {
        'test_config': test_config,
        'args': args,
    }


def test_compression_train(_params, tmp_path):
    p = _params
    args = p['args']
    tc = p['test_config']

    args['mode'] = 'train'
    args['log-dir'] = tmp_path
    args['workers'] = 4
    args['seed'] = 1

    sample_module = importlib.import_module('examples.{}.main'.format(tc['sample_type']))
    reset_context('orig')
    reset_context('quantized_graphs')
    # pylint: disable=global-variable-undefined
    global best_acc1
    best_acc1 = 0
    sample_module.main(get_cli_args(args))

    checkpoint_path = os.path.join(args['checkpoint-save-dir'], tc['checkpoint_name'] + '_best.pth')
    assert os.path.exists(checkpoint_path)
    assert torch.load(checkpoint_path)['best_acc1'] == approx(tc['expected_accuracy'], abs=tc['absolute_tolerance'])
