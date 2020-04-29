"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os
import re
import tempfile

import pytest
import torch
from pytest import approx

from examples.common.utils import get_name
from tests.conftest import TEST_ROOT
from tests.test_sanity_sample import Command, create_command_line

# sample
# ├── dataset
# │   ├── path
# │   ├── batch
# │   ├── configs
# │   │     ├─── config_filename
# │   │     │       ├── expected_accuracy
# │   │     │       ├── absolute_tolerance_train
# │   │     │       ├── absolute_tolerance_eval
# │   │     │       ├── execution_arg
# │   │     │       ├── epochs
# │   │     │       ├── weights
GLOBAL_CONFIG = {
    'classification':
        {
            'cifar100':
                {
                    'configs': {
                        'mobilenet_v2_sym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 64.53,
                            'weights': 'mobilenetV2_64.53.sd',
                            'absolute_tolerance_train': 1.5,
                        },
                        'mobilenet_v2_asym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed', 'cpu-only'},
                            'expected_accuracy': 64.53,
                            'weights': 'mobilenetV2_64.53.sd',
                            'absolute_tolerance_train': 1.5,
                        },
                        'inceptionV3_int8.json': {
                            'expected_accuracy': 77.53,
                            'weights': 'inceptionV3_77.53.sd',
                            'absolute_tolerance_eval': 6e-2,
                        },
                        'resnet50_int8.json': {
                            'expected_accuracy': 68.86,
                            'weights': 'resnet50_68.86.sd',
                            'absolute_tolerance_eval': 6e-2,
                        },
                        'mobilenet_v2_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 64.53,
                            'weights': 'mobilenetV2_64.53.sd',
                            'execution_arg': {'multiprocessing-distributed', ''},
                            'absolute_tolerance_train': 1.5,
                        },
                        'mobilenet_v2_rb_sparsity_int8.json': {
                            'expected_accuracy': 64.53,
                            'weights': 'mobilenetV2_64.53.sd',
                            'execution_arg': {'multiprocessing-distributed'},
                        }
                    }
                },
            'imagenet':
                {
                    'configs': {
                        'mobilenet_v2_imagenet_sym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                            'weights': 'mobilenet_v2.pth.tar'
                        },
                        'mobilenet_v2_imagenet_asym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                            'weights': 'mobilenet_v2.pth.tar',
                        },
                        'resnet50_imagenet_sym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                        },
                        'resnet50_imagenet_asym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                        },
                    }
                }
        }
}


def get_cli_args(args):
    cli_args = []
    for key, val in args.items():
        cli_args.append('--{}'.format(str(key)))
        if val:
            cli_args.append(str(val))
    return cli_args


def get_cli_dict_args(args):
    cli_args = dict()
    for key, val in args.items():
        cli_key = '--{}'.format(str(key))
        cli_args[cli_key] = None
        if val:
            cli_args[cli_key] = str(val)
    return cli_args


def parse_best_acc1(tmp_path):
    output_path = None
    for root, _, names in os.walk(str(tmp_path)):
        for name in names:
            if 'output' in name:
                output_path = os.path.join(root, name)

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        for line in reversed(f.readlines()):
            if line != '\n':
                matches = re.findall("\\d+\\.\\d+", line)
                if not matches:
                    raise RuntimeError("Could not parse output log for accuracy!")
                acc1 = float(matches[0])
                return acc1
    raise RuntimeError("Could not parse output log for accuracy!")



CONFIG_PARAMS = []
for sample_type_ in GLOBAL_CONFIG:
    datasets = GLOBAL_CONFIG[sample_type_]
    for dataset_name_ in datasets:
        dataset_path = datasets[dataset_name_].get('path', os.path.join(tempfile.gettempdir(), dataset_name_))
        batch_size = datasets[dataset_name_].get('batch', None)
        configs = datasets[dataset_name_].get('configs', {})
        for config_name in configs:
            config_params = configs[config_name]
            execution_args = config_params.get('execution_arg', [''])
            expected_accuracy_ = config_params.get('expected_accuracy', 100)
            absolute_tolerance_train_ = config_params.get('absolute_tolerance_train', 1)
            absolute_tolerance_eval_ = config_params.get('absolute_tolerance_eval', 1e-3)
            weights_path_ = config_params.get('weights', None)
            epochs = config_params.get('epochs', None)
            if weights_path_:
                weights_path_ = os.path.join(sample_type_, dataset_name_, weights_path_)
            for execution_arg_ in execution_args:
                config_path_ = TEST_ROOT.joinpath("data", "configs", "weekly", sample_type_, dataset_name_, config_name)
                jconfig = json.load(config_path_.open())
                args_ = {
                    'data': dataset_path,
                    'weights': weights_path_,
                    'config': str(config_path_)
                }
                if batch_size:
                    args_['batch-size'] = batch_size
                if epochs:
                    args_['epochs'] = epochs
                test_config_ = {
                    'sample_type': sample_type_,
                    'expected_accuracy': expected_accuracy_,
                    'absolute_tolerance_train': absolute_tolerance_train_,
                    'absolute_tolerance_eval': absolute_tolerance_eval_,
                    'checkpoint_name': get_name(jconfig)
                }
                CONFIG_PARAMS.append(tuple([test_config_, args_, execution_arg_, dataset_name_]))


def get_config_name(config_path):
    base = os.path.basename(config_path)
    return os.path.splitext(base)[0]


@pytest.fixture(scope='module', params=CONFIG_PARAMS,
                ids=['-'.join([p[0]['sample_type'], get_config_name(p[1]['config']), p[2]]) for p in CONFIG_PARAMS])
def _params(request, tmp_path_factory, dataset_dir, weekly_models_path, enable_imagenet):
    if weekly_models_path is None:
        pytest.skip('Path to models weights for weekly testing is not set, use --weekly-models option.')
    test_config, args, execution_arg, dataset_name = request.param
    test_config['timeout'] = 2 * 60 * 60  # 2 hours, because rb sparsity + int8 works 1.5-2 hours
    if enable_imagenet:
        test_config['timeout'] = None
    if 'imagenet' in dataset_name and not enable_imagenet:
        pytest.skip('ImageNet tests were intentionally skipped as it takes a lot of time')
    if args['weights']:
        weights_path = os.path.join(weekly_models_path, args['weights'])
        if not os.path.exists(weights_path):
            raise FileExistsError('Weights file does not exist: {}'.format(weights_path))
        args['weights'] = weights_path
    else:
        del args['weights']
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


@pytest.mark.dependency(name=["train"])
def test_compression_train(_params, tmp_path):
    p = _params
    args = p['args']
    tc = p['test_config']

    args['mode'] = 'train'
    args['log-dir'] = tmp_path
    args['workers'] = 4
    args['seed'] = 1

    runner = Command(create_command_line(get_cli_dict_args(args), tc['sample_type']))
    res = runner.run(timeout=tc['timeout'])

    assert res == 0
    checkpoint_path = os.path.join(args['checkpoint-save-dir'], tc['checkpoint_name'] + '_best.pth')
    assert os.path.exists(checkpoint_path)
    actual_acc = torch.load(checkpoint_path)['best_acc1']
    ref_acc = tc['expected_accuracy']
    better_accuracy_tolerance = 3
    tolerance = tc['absolute_tolerance_train'] if actual_acc < ref_acc else better_accuracy_tolerance
    assert actual_acc == approx(ref_acc, abs=tolerance)


@pytest.mark.dependency(depends=["train"])
def test_compression_eval_trained(_params, tmp_path):
    p = _params
    args = p['args']
    tc = p['test_config']

    args['mode'] = 'test'
    args['log-dir'] = tmp_path
    args['workers'] = 4
    args['seed'] = 1
    checkpoint_path = os.path.join(args['checkpoint-save-dir'], tc['checkpoint_name'] + '_best.pth')
    args['resume'] = checkpoint_path
    if 'weights' in args:
        del args['weights']

    runner = Command(create_command_line(get_cli_dict_args(args), tc['sample_type']))
    res = runner.run(timeout=tc['timeout'])
    assert res == 0

    acc1 = parse_best_acc1(tmp_path)
    assert torch.load(checkpoint_path)['best_acc1'] == approx(acc1, abs=tc['absolute_tolerance_eval'])
