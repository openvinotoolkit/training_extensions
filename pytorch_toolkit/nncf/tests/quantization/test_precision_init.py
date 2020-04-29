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
import itertools
import json
import math
from collections import namedtuple

import os
import pytest
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from torch.utils import model_zoo
from torchvision.models import MobileNetV2
from torchvision.transforms import transforms

from examples.classification.main import create_cifar
from examples.common.models import squeezenet1_1_custom, model_urls, OrderedDict
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.checkpoint_loading import load_state
from nncf.nncf_network import CompressionModuleType
from nncf.quantization.hessian_trace import HessianTraceEstimator
from nncf.quantization.init_precision import HessianAwarePrecisionInitializeRunner
from nncf.quantization.layers import QUANTIZATION_MODULES
from nncf.utils import get_all_modules_by_type, safe_thread_call
from tests.conftest import TEST_ROOT
from tests.quantization.test_algo_quantization import get_squeezenet_quantization_config, \
    get_basic_quantization_config, RankDatasetMock, compare_multi_gpu_dump
from tests.test_helpers import create_compressed_model_and_algo_for_test


def create_test_dataloaders(model_size, dataset_dir, batch_size):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(model_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dummy_config = type('dummy', (object,), {'dataset_dir': dataset_dir})()
    train_dataset = create_cifar(dummy_config, dataset_config='cifar10', is_train=True, transform=train_transforms)
    pin_memory = True
    workers = 1

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                               pin_memory=pin_memory)
    return train_loader, train_dataset


def get_bitwidth_per_scope(model):
    all_quantizations = OrderedDict()
    for class_type in QUANTIZATION_MODULES.registry_dict.values():
        quantization_type = class_type.__name__
        all_quantizations.update(
            get_all_modules_by_type(model.get_compression_modules_by_type(CompressionModuleType.ACTIVATION_QUANTIZER),
                                    quantization_type))
        all_quantizations.update(
            get_all_modules_by_type(model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER),
                                    quantization_type))
        all_quantizations.update(get_all_modules_by_type(model, quantization_type))

    all_quantizations = OrderedDict(sorted(all_quantizations.items(), key=lambda x: str(x[0])))
    full_bitwidth_per_scope = []
    for scope, quantizer in all_quantizations.items():
        full_bitwidth_per_scope.append([quantizer.num_bits, str(scope)])
    return full_bitwidth_per_scope


# TODO: split into 2 functions or rename
def compare_with_ref_if_exists(actual_state, path_to_ref):
    if os.path.exists(path_to_ref):
        with open(path_to_ref, 'r') as f:
            assert json.load(f) == actual_state
    else:
        with open(path_to_ref, 'w') as f:
            json.dump(actual_state, f)


def create_hawq_test_config(batch_size, num_data_points):
    config = get_squeezenet_quantization_config()
    config['batch_size'] = batch_size
    config['compression'].update({
        'initializer': {
            'precision': {
                "type": "hawq",
                "bits": [
                    4,
                    8,
                    6,
                    7,
                    5
                ],
                "num_data_points": num_data_points,
                "iter_number": 1,
                "tolerance": 1e-2
            },
            'range': {
                'num_init_steps': 1
            }
        }})
    return config


def test_hawq_precision_init(_seed, dataset_dir, tmp_path, mocker):
    num_data_points = 100
    batch_size = 10
    config = create_hawq_test_config(batch_size, num_data_points)
    model = squeezenet1_1_custom(num_classes=10, pretrained=False, dropout=0)

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    load_state(model, model_zoo.load_url(model_urls['squeezenet1_1']))
    model = model.cuda()
    device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss().cuda()

    if not dataset_dir:
        dataset_dir = str(tmp_path)
    train_loader, _ = create_test_dataloaders(config.model_size, dataset_dir, batch_size)
    mocked_trace = mocker.patch('nncf.quantization.hessian_trace.HessianTraceEstimator.get_average_traces')
    num_traces = len(get_all_modules_by_type(model, 'NNCFConv2d'))
    mock_avg_traces = [torch.Tensor([num_traces - i]).to(device) for i in range(num_traces)]
    mocked_trace.return_value = mock_avg_traces

    compression_ctrl.initialize(criterion=criterion, data_loader=train_loader)
    act_bitwidth_per_scope = get_bitwidth_per_scope(model)
    path_to_ref = str(TEST_ROOT / 'data/hawq_reference/squeezenet1_1_mixed_bitwidth_per_scope.json')
    compare_with_ref_if_exists(act_bitwidth_per_scope, path_to_ref)


HAWQTestParams = namedtuple('HAWQTestParams', ('iter_number', 'batch_size', 'num_data_points', 'ref_trace'))


@pytest.mark.parametrize("params",
                         (HAWQTestParams(200, 13, 100, 0.07957423478364944),
                          HAWQTestParams(2, 13, 100, 0.062167033553123474),
                          HAWQTestParams(2, 10, 10, 0.11200366914272308),
                          HAWQTestParams(2, 10, 5, 0.11200366914272308)),
                         ids=('until_threshold', 'until_num_iter', 'batch_eq_num_data', 'batch_larger_num_data'))
def test_hawq_on_single_conv_without_quantizers(_seed, dataset_dir, tmp_path, params: HAWQTestParams):
    config = get_squeezenet_quantization_config(batch_size=params.batch_size)
    iter_number = params.iter_number
    tolerance = 4e-4

    model = squeezenet1_1_custom(num_classes=10, pretrained=False, dropout=0)
    load_state(model, model_zoo.load_url(model_urls['squeezenet1_1']))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    if not dataset_dir:
        dataset_dir = str(tmp_path)
    data_loader, _ = create_test_dataloaders(config.model_size, dataset_dir, params.batch_size)
    device = next(model.parameters()).device

    for _, param in model.named_parameters():
        param.requires_grad = False
    first_conv = next(iter(get_all_modules_by_type(model, 'Conv2d').values()))
    first_conv.weight.requires_grad = True

    trace_estimator = HessianTraceEstimator(model, criterion, device, data_loader, params.num_data_points)
    actual_state = trace_estimator.get_average_traces(max_iter=iter_number, tolerance=tolerance)
    assert math.isclose(actual_state.item(), params.ref_trace, rel_tol=1e-09)


def get_size_of_search_space(m, L):
    def nCr(n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    ref_num = 0
    for j in range(1, m + 1):
        ref_num += nCr(m, j) * nCr(L - 1, j - 1)
    return ref_num


def test_constrained_bit_configs():
    bits = [4, 2, 8]
    L = 4
    m = len(bits)
    all_configs = list(itertools.product(bits, repeat=L))

    ref_configs = []
    for bit_config in all_configs:
        is_ok = True
        for i in range(L - 1):
            if bit_config[i + 1] < bit_config[i]:
                is_ok = False
                break
        if is_ok:
            ref_configs.append(list(bit_config))
    actual_config = HessianAwarePrecisionInitializeRunner.get_constrained_configs(bits, L)
    ref_num = get_size_of_search_space(m, L)
    assert len(ref_configs) == ref_num
    assert len(actual_config) == ref_num
    assert sorted(actual_config) == sorted(ref_configs)


def get_requires_grad_per_param(model):
    not_sorted = OrderedDict({param_name: param.requires_grad for param_name, param in model.named_parameters()})
    return OrderedDict(sorted(not_sorted.items()))


def test_disable_quantizer_gradients():
    config = get_basic_quantization_config()
    config['input_info'] = {
        "sample_size": (1, 3, 10, 10),
    }
    model = MobileNetV2(num_classes=10)
    model.eval()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)

    HessianAwarePrecisionInitializeRunner.disable_quantizer_gradients(
        all_quantizations,
        compression_ctrl.quantized_weight_modules_registry,
        model)
    actual_state = get_requires_grad_per_param(model)
    path_to_ref = str(TEST_ROOT / 'data/hawq_reference/mobilenet_v2_requires_grad_per_param.json')
    compare_with_ref_if_exists(actual_state, path_to_ref)


def test_enable_quantizer_gradients():
    config = get_basic_quantization_config()
    config['input_info'] = {
        "sample_size": (1, 3, 10, 10),
    }
    model = MobileNetV2(num_classes=10)
    model.eval()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    all_quantizations = get_all_modules_by_type(model, quantization_types)

    original = get_requires_grad_per_param(model)
    disabled = HessianAwarePrecisionInitializeRunner.disable_quantizer_gradients(
        all_quantizations,
        compression_ctrl.quantized_weight_modules_registry,
        model)
    HessianAwarePrecisionInitializeRunner.enable_quantizer_gradients(model, all_quantizations, disabled)
    actual = get_requires_grad_per_param(model)
    assert original == actual


def get_path_to_bitwidth_dump(tmp_path, rank):
    out_file_path = tmp_path / 'bitwidth_per_scope_gpu{}.pt'.format(rank)
    return out_file_path


def hawq_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    config.batch_size = 3
    config.workers = 3
    config.gpu = gpu
    config.ngpus_per_node = ngpus_per_node
    config.rank = gpu
    config.distributed = True

    torch.distributed.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:8899',
                                         world_size=config.world_size, rank=config.rank)

    model = safe_thread_call(partial(squeezenet1_1_custom, pretrained=True, dropout=0))

    quant_model, compression_algo = create_compressed_model_and_algo_for_test(model, config)
    compression_algo.distributed()

    torch.cuda.set_device(config.gpu)
    quant_model.cuda(config.gpu)
    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.workers = int(config.workers / ngpus_per_node)
    quant_model = torch.nn.parallel.DistributedDataParallel(quant_model, device_ids=[config.gpu])

    torch.backends.cudnn.benchmark = True

    input_infos_list = create_input_infos(config)
    input_sample_size = input_infos_list[0].shape
    data_loader = torch.utils.data.DataLoader(RankDatasetMock(input_sample_size[1:], config.rank),
                                              batch_size=3,
                                              num_workers=1,
                                              shuffle=False)
    criterion = torch.nn.MSELoss().cuda(config.gpu)

    # just to reproduce the same scale values without Dropout
    quant_model.eval()

    compression_algo.initialize(criterion=criterion, data_loader=data_loader)

    act_bitwidth_per_scope = get_bitwidth_per_scope(quant_model.module)
    out_file_path = get_path_to_bitwidth_dump(tmp_path, config.rank)
    torch.save(act_bitwidth_per_scope, str(out_file_path))


def test_hawq_broadcast_avg_traces_in_distributed_mode(tmp_path):
    num_data_points = 100
    batch_size = 10
    config = create_hawq_test_config(batch_size, num_data_points)

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(hawq_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config, tmp_path),
                                join=True)

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_to_bitwidth_dump)
