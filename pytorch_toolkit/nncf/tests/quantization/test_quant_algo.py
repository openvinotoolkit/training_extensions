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

import re
from copy import deepcopy
from functools import partial

import pytest
import torch
import torch.utils.data
from pytest import approx
from torchvision.models import resnet50

from examples.common.models.classification import squeezenet1_1_custom
from nncf import Quantization, SymmetricQuantizer
from nncf import utils
from nncf.algo_selector import create_compression_algorithm
from nncf.compression_method_api import CompressionLoss, CompressionScheduler
from nncf.config import Config
from nncf.dynamic_graph import reset_context, patch_torch_operators
from nncf.helpers import safe_thread_call, create_compressed_model, load_state
from nncf.initializers import InitializeDataLoader
from nncf.operations import UpdateWeight, UpdateInputs
from nncf.utils import get_all_modules_by_type
from tests.test_helpers import BasicConvTestModel, TwoConvTestModel, get_empty_config

patch_torch_operators()


def get_basic_quantization_config(model_size=4):
    config = Config()
    config.update({
        "model": "basic_quant_conv",
        "model_size": model_size,
        "input_sample_size": (1, 1, model_size, model_size),
        "compression":
            {
                "algorithm": "quantization",
                "initializer": {
                    "num_init_steps": 0
                },
                "params": {}
            }
    })
    return config


def get_basic_asym_quantization_config(model_size=4):
    config = get_basic_quantization_config(model_size)
    config['compression']['activations'] = {"mode": "asymmetric"}
    config['compression']['weights'] = {"mode": "asymmetric"}
    return config


def get_squeezenet_quantization_config(model_size=32):
    config = Config()
    config.update({
        "model": "squeezenet1_1_custom",
        "model_size": model_size,
        "input_sample_size": (3, 3, model_size, model_size),
        "compression":
            {
                "algorithm": "quantization",
                "initializer": {
                    "num_init_steps": 0
                }
            }
    })
    return config


def test_can_load_quant_algo__with_defaults():
    model = BasicConvTestModel()
    config = get_basic_quantization_config()
    reset_context('orig')
    reset_context('quantized_graphs')
    compression_algo = create_compression_algorithm(deepcopy(model), config)
    assert isinstance(compression_algo, Quantization)
    quant_model = compression_algo.model

    model_conv = get_all_modules_by_type(model, 'Conv2d')
    quant_model_conv = get_all_modules_by_type(quant_model.module, 'NNCFConv2d')
    assert len(model_conv) == len(quant_model_conv)

    for module_name in model_conv:
        scope = module_name.split('/')
        scope[-1] = scope[-1].replace('Conv2d', 'NNCFConv2d')
        quant_module_name = '/'.join(scope)
        assert quant_module_name in quant_model_conv

        store = []
        for op in quant_model_conv[quant_module_name].pre_ops.values():
            if isinstance(op, (UpdateInputs, UpdateWeight)) and isinstance(op.operand, SymmetricQuantizer):
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)
        assert UpdateWeight.__name__ in store


def test_can_create_quant_loss_and_scheduler():
    model = BasicConvTestModel()

    config = get_basic_quantization_config()
    reset_context('orig')
    reset_context('quantized_graphs')
    compression_algo = create_compression_algorithm(model, config)

    loss = compression_algo.loss
    assert isinstance(loss, CompressionLoss)

    scheduler = compression_algo.scheduler
    assert isinstance(scheduler, CompressionScheduler)


class RankDatasetMock:
    def __init__(self, input_size, rank):
        self.input_size = input_size
        self.rank = rank
        super().__init__()

    def __getitem__(self, index):
        dummy_input = torch.ones(self.input_size) * (self.rank - 1) * 3
        return dummy_input, torch.ones(1)

    def __len__(self):
        return 100


def scale_signed_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    config.batch_size = 3
    config.workers = 3
    config.gpu = gpu
    config.ngpus_per_node = ngpus_per_node
    config.rank = gpu
    config.distributed = True

    torch.distributed.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:8899',
                                         world_size=config.world_size, rank=config.rank)

    model = safe_thread_call(partial(squeezenet1_1_custom, pretrained=True))

    compression_algo = create_compression_algorithm(model, config)
    compression_algo.distributed()
    model = compression_algo.model
    compression_scheduler = compression_algo.scheduler

    torch.cuda.set_device(config.gpu)
    model.cuda(config.gpu)
    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.workers = int(config.workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])

    criterion = torch.nn.MSELoss().cuda(config.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    torch.backends.cudnn.benchmark = True

    input_sample_size = config.input_sample_size
    data_loader = torch.utils.data.DataLoader(RankDatasetMock(input_sample_size[1:], config.rank),
                                              batch_size=3,
                                              num_workers=1,
                                              shuffle=False)
    # just to reproduce the same scale values without Dropout
    model.eval()
    compression_algo.initialize(data_loader)

    act_sum = 0
    for layer in get_all_modules_by_type(model, "SymmetricQuantizer").values():
        act_sum += layer.scale
    ref_sum = 3518.797
    assert act_sum.item() == approx(ref_sum, 0.01), \
        'sum of scales is not expected {} vs {} rank {}'.format(act_sum.item(), ref_sum, config.rank)

    out_file_path = get_path_after_broadcast(tmp_path, config.rank)
    save_params(model, out_file_path)
    compression_scheduler.step()
    for i, (input_, _) in enumerate(data_loader):
        if i > 5:
            break
        output = model(input_)
        optimizer.zero_grad()
        dummy_target = torch.randn(1000).cuda(config.gpu, non_blocking=True)
        loss = criterion(output, dummy_target)
        compression_scheduler.step()
        loss.backward()
        optimizer.step()
        compression_scheduler.step()

    out_file_path = get_path_path_after_train_iters(tmp_path, config.rank)
    save_params(model, out_file_path)


def get_path_path_after_train_iters(tmp_path, rank):
    out_file_path = tmp_path / 'scale_signed_after_1_train_iter_gpu{}.pt'.format(rank)
    return out_file_path


def get_path_after_broadcast(tmp_path, rank):
    out_file_path = tmp_path / 'scale_signed_after_broadcast_gpu{}.pt'.format(rank)
    return out_file_path


def save_params(model, out_file_path):
    gpu_scale_signed_params = []
    for _, layer in utils.get_all_modules_by_type(model, 'SymmetricQuantizer').items():
        gpu_scale_signed_params.append((layer.scale.to(torch.device('cpu')),
                                        layer.signed_tensor.to(torch.device('cpu'))))
    with out_file_path.open('wb') as out_file:
        torch.save(gpu_scale_signed_params, out_file)


def compare_scales_and_sign(config, tmp_path, get_path_fn):
    mismatching = False
    ref_file_path = get_path_fn(tmp_path, 0)
    with ref_file_path.open('rb') as ref_scale_file:
        ref_scale_signed_params = torch.load(ref_scale_file)
        for other_rank in range(1, config.world_size):
            other_file_path = get_path_fn(tmp_path, other_rank)
            with other_file_path.open('rb') as in_file:
                gpu_scale_signed_params = torch.load(in_file)
                if ref_scale_signed_params != gpu_scale_signed_params:
                    mismatching = True
    return mismatching


def test_multiprocessing_distributed_shares_init_scales_signedness_across_gpus(tmp_path):
    num_init_steps = 10
    reset_context('orig')
    reset_context('quantized_graphs')

    config = get_squeezenet_quantization_config()
    config['compression']['initializer'] = {'num_init_steps': num_init_steps}

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(scale_signed_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config, tmp_path),
                                join=True)

    assert not compare_scales_and_sign(config, tmp_path, get_path_after_broadcast)
    assert not compare_scales_and_sign(config, tmp_path, get_path_path_after_train_iters)


class OnesDatasetMock:
    def __init__(self, input_size):
        self.input_size = input_size
        super().__init__()

    def __getitem__(self, index):
        return torch.ones(self.input_size), torch.ones(1)

    def __len__(self):
        return 1


@pytest.mark.parametrize("wrap_dataloader",
                         (True, False),
                         ids=['wrapped_dataloader', 'standard_dataloader'])
class TestInit:
    @staticmethod
    def create_algo(config):
        model = TwoConvTestModel()
        reset_context('orig')
        reset_context('quantized_graphs')
        compression_algo = create_compression_algorithm(model, config)
        return compression_algo

    @staticmethod
    def create_config():
        config = get_empty_config()
        config['compression'] = {'algorithm': 'quantization', 'initializer': {'num_init_steps': 1}}
        return config

    @staticmethod
    def create_dataloader(wrap_dataloader, config, algo):
        input_sample_size = config.input_sample_size
        data_loader = torch.utils.data.DataLoader(OnesDatasetMock(input_sample_size[1:]),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  shuffle=False)
        if wrap_dataloader:
            device = next(algo.model.parameters()).device
            data_loader = InitializeDataLoader(data_loader=data_loader,
                                               device=device,
                                               kwargs={})
        return data_loader

    @staticmethod
    def check_sign_and_scale(algo, ref_table):
        model_conv = get_all_modules_by_type(algo.model, 'SymmetricQuantizer')
        for name, module in model_conv.items():
            for pattern, ref_values in ref_table.items():
                match = re.search(pattern, name)
                if match:
                    assert isinstance(module, SymmetricQuantizer)
                    assert module.signed == ref_values[0], 'sign is not matched for {}'.format(name)
                    assert module.scale == ref_values[1], 'scale is not matched for {}'.format(name)

    def test_scale_and_sign_init_for_quant_algo(self, wrap_dataloader):
        config = self.create_config()
        algo = self.create_algo(config)
        data_loader = self.create_dataloader(wrap_dataloader, config, algo)

        algo.initialize(data_loader)

        self.check_sign_and_scale(algo, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (True, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 1),
            '.*activation_quantizers.*Sequential\\[0\\].*': (True, 4),
            '.*activation_quantizers.*Sequential\\[1\\].*': (True, 24)
        })

    def test_scale_and_sign_init_for_quant_algo__after_load_state(self, wrap_dataloader):
        config = self.create_config()
        algo = self.create_algo(config)
        load_state(algo.model, {
            'module.features.0.0.pre_ops.0.op.signed_tensor': torch.tensor([0.]),  # quantizer of 1st conv's weights
            'module.features.1.0.pre_ops.0.op.scale': torch.tensor([100])  # quantizer of 2nd conv's weights
        })
        data_loader = self.create_dataloader(wrap_dataloader, config, algo)

        algo.initialize(data_loader)

        self.check_sign_and_scale(algo, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (False, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 100),
            '.*activation_quantizers.*Sequential\\[0\\].*': (True, 4),
            '.*activation_quantizers.*Sequential\\[1\\].*': (True, 24)
        })


def get_path_to_keys(tmp_path, rank):
    return '{}_{}'.format(tmp_path, str(rank))


def activation_quantizers_dumping_worker(current_gpu, config, tmp_path):
    model = resnet50(pretrained=False)

    reset_context('orig')
    reset_context('quantized_graphs')

    algo = create_compression_algorithm(model, config)
    model = algo.model
    path = get_path_to_keys(tmp_path, current_gpu)
    print(path)
    with open(path, 'w') as f:
        f.writelines("%s\n" % key for key in model.activation_quantizers.keys())


def test_activation_quantizers_order_is_the_same__for_resnet50(tmp_path):
    config = get_empty_config(input_sample_size=[1, 3, 224, 224])
    config['compression'] = {'algorithm': 'quantization'}
    ngpus_per_node = torch.cuda.device_count()

    torch.multiprocessing.spawn(activation_quantizers_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(config, tmp_path),
                                join=True)

    with open(get_path_to_keys(tmp_path, 0), 'r') as f:
        ref_list = f.readlines()
    for i in range(1, ngpus_per_node):
        with open(get_path_to_keys(tmp_path, i), 'r') as f:
            curr_list = f.readlines()
            assert curr_list == ref_list


def test_load_state_sets_initialized_flag(tmp_path):
    config = get_basic_quantization_config()
    config.log_dir = str(tmp_path)
    reset_context('orig')
    reset_context('quantized_graphs')
    _, model = create_compressed_model(TwoConvTestModel(), config)

    load_state(model, {
        'module.features.0.0.pre_ops.0.op.signed_tensor': torch.tensor([1.0]),  # quantizer of 1st conv's weights
        'module.features.1.0.pre_ops.0.op.scale': torch.tensor([1.0])  # quantizer of 2nd conv's weights
    })

    quantizers = get_all_modules_by_type(model, 'SymmetricQuantizer')
    for name, module in quantizers.items():
        if 'activation_quantizers' in name or 'UpdateInputs' in name:
            assert not module.initialized
        else:
            assert module.initialized
