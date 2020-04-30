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
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from copy import deepcopy
from functools import partial
from pytest import approx
from torchvision.models import resnet50

from examples.common.models.classification import squeezenet1_1_custom
from nncf import utils
from nncf.algo_selector import create_compression_algorithm_builders
from nncf.compression_method_api import CompressionLoss, CompressionScheduler
from nncf.config import Config
from nncf.dynamic_graph.context import ScopeElement, Scope
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.checkpoint_loading import load_state
from nncf.hw_config import HWConfigType
from nncf.initialization import InitializingDataLoader
from nncf.layers import NNCFConv2d
from nncf.module_operations import UpdateWeight, UpdateInputs
from nncf.nncf_network import CompressionModuleType
from nncf.quantization.algo import QuantizationController, QuantizationBuilder
from nncf.quantization.layers import QuantizationMode, QuantizerConfig, SymmetricQuantizer, AsymmetricQuantizer, \
    INITIALIZABLE_MODULES, BaseQuantizer, QUANTIZATION_MODULES
from nncf.utils import get_all_modules_by_type, safe_thread_call
from tests.test_helpers import BasicConvTestModel, TwoConvTestModel, get_empty_config, \
    create_compressed_model_and_algo_for_test, MockModel, create_conv


def get_basic_quantization_config(model_size=4):
    config = Config()
    config.update({
        "model": "basic_quant_conv",
        "model_size": model_size,
        "input_info":
            {
                "sample_size": (1, 1, model_size, model_size),
            },
        "compression":
            {
                "algorithm": "quantization",
                "initializer": {
                    "range": {
                        "num_init_steps": 0
                    }
                },
                "params": {}
            }
    })
    return config


def get_basic_asym_quantization_config(model_size=4):
    config = get_basic_quantization_config(model_size)
    config['compression']['activations'] = {"mode": "asymmetric"}
    config['compression']['initializer']['range'] = {"mode": "asymmetric"}
    return config


def get_squeezenet_quantization_config(model_size=32, batch_size=3):
    config = get_basic_quantization_config(model_size)
    config['model'] = 'squeezenet1_1_custom'
    config['input_info'] = {
        "sample_size": (batch_size, 3, model_size, model_size),
    }
    return config


def split_quantizers(quant_model):
    quantizers = get_all_modules_by_type(quant_model, list(INITIALIZABLE_MODULES.registry_dict.keys()))
    weight_quantizers = []
    activation_quantizers = []
    for name, data in quantizers.items():
        if 'UpdateWeight' in name:
            weight_quantizers.append(data)
        else:
            activation_quantizers.append(data)
    return weight_quantizers, activation_quantizers


def compare_qconfigs(config: QuantizerConfig, quantizer: BaseQuantizer):
    assert config.is_weights == quantizer.is_weights
    assert config.bits == quantizer.num_bits
    assert isinstance(quantizer, QUANTIZATION_MODULES.get(config.mode))
    assert config.per_channel == quantizer.per_channel
    assert config.signedness_to_force == quantizer.signedness_to_force


def test_quantization_configs__with_defaults():
    model = BasicConvTestModel()
    config = get_basic_quantization_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizers = compression_ctrl.non_weight_quantizers

    ref_weight_qconfig = QuantizerConfig(8, QuantizationMode.SYMMETRIC, None, False, None, True)
    for wq in weight_quantizers.values():
        compare_qconfigs(ref_weight_qconfig, wq)

    ref_activation_qconfig = QuantizerConfig(8, QuantizationMode.SYMMETRIC, None, False, None, False)
    for wq in activation_quantizers.values():
        compare_qconfigs(ref_activation_qconfig, wq)


def test_quantization_configs__custom():
    model = BasicConvTestModel()

    config = get_basic_quantization_config()
    config['compression'].update({
        "weights": {
            "mode": "asymmetric",
            "per_channel": True,
            "bits": 4
        },
        "activations": {
            "mode": "asymmetric",
            "bits": 4,
            "signed": True,
        },
    })
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizers = compression_ctrl.non_weight_quantizers

    ref_weight_qconfig = QuantizerConfig(bits=4,
                                         mode=QuantizationMode.ASYMMETRIC,
                                         signedness_to_force=None,
                                         per_channel=True,
                                         input_shape=None,
                                         is_weights=True)
    for wq in weight_quantizers.values():
        compare_qconfigs(ref_weight_qconfig, wq)

    ref_activation_qconfig = QuantizerConfig(bits=4,
                                             mode=QuantizationMode.ASYMMETRIC,
                                             signedness_to_force=True,
                                             per_channel=False,
                                             input_shape=None,
                                             is_weights=False)
    for wq in activation_quantizers.values():
        compare_qconfigs(ref_activation_qconfig, wq)


#       fq_2
#        \
# fq_2 - conv_1 - fq_6
#                   \
#        fq_4       add
#         \         /
# fq_4 - conv_2 - fq_6
#
def test_quantization_configs__with_precisions_list():
    class ModelForTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = create_conv(1, 2, 2, -1, -2)
            self.conv2 = create_conv(1, 2, 2, -1, -2)

        def forward(self, x):
            return self.conv1(x) + self.conv2(x)

    model = ModelForTest()

    config = get_basic_quantization_config()
    config['compression'].update({
        "initializer": {
            "precision": {
                "bitwidth_per_scope":
                    [[2, 'ModelForTest/NNCFConv2d[conv1]'],
                     [4, 'ModelForTest/NNCFConv2d[conv2]']]
            },
        },
        "activations": {
            "bits": 6
        }
    })
    model, compression_ctrl = \
        create_compressed_model_and_algo_for_test(model, config)  # type: NNCFNetwork, QuantizationController

    device = next(model.parameters()).device
    data_loader = TestInit.create_dataloader(False, config, device)
    compression_ctrl.initialize(data_loader)

    ref_bits = [('ModelForTest/NNCFConv2d[conv1]module_weight', 2),
                ('ModelForTest/NNCFConv2d[conv2]module_weight', 4),
                ('ModelForTest/NNCFConv2d[conv2]/conv2d_0', 6),
                ('ModelForTest/NNCFConv2d[conv1]/conv2d_0', 6),
                ('ModelForTest/NNCFConv2d[conv1]module_input', 2),
                ('ModelForTest/NNCFConv2d[conv2]module_input', 4)]

    for key, quantizer in compression_ctrl.all_quantizations.items():
        expected_bit = [ref_bit for (name, ref_bit) in ref_bits if name == str(key)][0]
        assert quantizer.num_bits == expected_bit

    ref_rows = [['2', '16.667', '16.667', '33.333'],
                ['4', '16.667', '16.667', '33.333'],
                ['6', '0', '33.333', '33.333']]
    table = compression_ctrl.get_bit_stats()
    # pylint: disable=protected-access
    assert table._rows == ref_rows


def compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_name):
    def get_name(name):
        return '/'.join([model_name, name])

    all_quantizations = {str(key): quantizer for key, quantizer in algo.all_quantizations.items()}
    assert len(actual_pairs) == len(ref_pair_names)
    for (wqs, aq), (wqs_names, aq_name) in zip(actual_pairs, ref_pair_names):
        assert not aq.is_weights
        assert aq == all_quantizations[get_name(aq_name)]
        ref_weight_quantizers = [all_quantizations[get_name(name)] for name in wqs_names]
        for weight_quantizer in wqs:
            assert weight_quantizer.is_weights
            assert weight_quantizer in ref_weight_quantizers


#
#  fq           fq
#   \            \
# сonv0 - fq - conv1
#   /
# fq
#
def test_get_weight_activation_pairs():
    model_cls = TwoConvTestModel
    config = get_basic_quantization_config()
    _, algo = create_compressed_model_and_algo_for_test(model_cls(), config)

    actual_pairs = algo.get_weights_activation_quantizers_pairs()
    ref_pair_names = [(['Sequential[features]/Sequential[0]/NNCFConv2d[0]module_weight'],
                       'Sequential[features]/Sequential[0]/NNCFConv2d[0]module_input',
                       ),
                      (['Sequential[features]/Sequential[1]/NNCFConv2d[0]module_weight'],
                       'Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0',
                       )]

    compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_cls.__name__)


class DoubleWeightsPerActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.conv1 = create_conv(1, 2, 2, -1, -2)
        self.conv2 = create_conv(1, 2, 2, -1, -2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return self.conv1(x), self.conv2(x)


#              fq
#             /
#          conv2d
#         /
# relu - fq     fq
#         \    /
#         conv2d
#
def test_get_weight_activation_pairs__with_double_weights_per_activation():
    model_cls = DoubleWeightsPerActivation
    model_name = model_cls.__name__
    config = get_basic_quantization_config()

    _, algo = create_compressed_model_and_algo_for_test(model_cls(), config)

    actual_pairs = algo.get_weights_activation_quantizers_pairs()
    ref_pair_names = [(['NNCFConv2d[conv1]module_weight', 'NNCFConv2d[conv2]module_weight'],
                       'ReLU[relu]/RELU_0')]

    compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_name)


class DoubleWeightsPerActivationWithExtraModule(DoubleWeightsPerActivation):
    def forward(self, x):
        x = self.relu(x)
        return self.conv1(torch.sigmoid(x)), self.conv2(torch.sigmoid(x))


#                     fq
#                      \
#         sigmoid - conv1d
#         /
# relu - fq           fq
#         \            \
#         sigmoid - conv2d
#
def test_get_weight_activation_pairs__with_extra_module():
    model_cls = DoubleWeightsPerActivationWithExtraModule
    model_name = model_cls.__name__
    config = get_basic_quantization_config()
    config["compression"].update({
        "quantizable_subgraph_patterns": [["sigmoid", "conv2d"]],
        "quantize_inputs": False})

    _, algo = create_compressed_model_and_algo_for_test(model_cls(), config)

    actual_pairs = algo.get_weights_activation_quantizers_pairs()
    ref_pair_names = [(['NNCFConv2d[conv1]module_weight', 'NNCFConv2d[conv2]module_weight'],
                       'ReLU[relu]/RELU_0')]

    compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_name)


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

    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_ctrl.distributed()
    compression_scheduler = compression_ctrl.scheduler

    torch.cuda.set_device(config.gpu)
    quant_model.cuda(config.gpu)
    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.workers = int(config.workers / ngpus_per_node)
    quant_model = torch.nn.parallel.DistributedDataParallel(quant_model, device_ids=[config.gpu])

    criterion = torch.nn.MSELoss().cuda(config.gpu)
    optimizer = torch.optim.Adam(quant_model.parameters(), lr=0.01)

    torch.backends.cudnn.benchmark = True

    input_infos_list = create_input_infos(config)
    input_sample_size = input_infos_list[0].shape
    data_loader = torch.utils.data.DataLoader(RankDatasetMock(input_sample_size[1:], config.rank),
                                              batch_size=3,
                                              num_workers=1,
                                              shuffle=False)
    # just to reproduce the same scale values without Dropout
    quant_model.eval()
    compression_ctrl.initialize(data_loader)

    act_sum = 0
    for layer in get_all_modules_by_type(quant_model, "SymmetricQuantizer").values():
        act_sum += layer.scale
    ref_sum = 3467.322
    assert act_sum.item() == approx(ref_sum, 0.01), \
        'sum of scales is not expected {} vs {} rank {}'.format(act_sum.item(), ref_sum, config.rank)

    out_file_path = get_path_after_broadcast(tmp_path, config.rank)
    save_params(quant_model, out_file_path)
    compression_scheduler.step()
    for i, (input_, _) in enumerate(data_loader):
        if i > 5:
            break
        output = quant_model(input_)
        optimizer.zero_grad()
        dummy_target = torch.randn(1000).cuda(config.gpu, non_blocking=True)
        loss = criterion(output, dummy_target)
        compression_scheduler.step()
        loss.backward()
        optimizer.step()
        compression_scheduler.step()

    out_file_path = get_path_path_after_train_iters(tmp_path, config.rank)
    save_params(quant_model, out_file_path)


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


def compare_multi_gpu_dump(config, tmp_path, get_path_fn):
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


def test_can_load_quant_algo__with_defaults():
    model = BasicConvTestModel()
    config = get_basic_quantization_config()
    compression_algo_builder_list = create_compression_algorithm_builders(config)
    assert len(compression_algo_builder_list) == 1
    assert isinstance(compression_algo_builder_list[0], QuantizationBuilder)

    quant_model, _ = create_compressed_model_and_algo_for_test(deepcopy(model), config)

    model_conv = get_all_modules_by_type(model, 'Conv2d')
    quant_model_conv = get_all_modules_by_type(quant_model.get_nncf_wrapped_model(), 'NNCFConv2d')
    assert len(model_conv) == len(quant_model_conv)

    for module_scope, _ in model_conv.items():
        quant_scope = deepcopy(module_scope)  # type: Scope
        quant_scope.pop()
        quant_scope.push(ScopeElement('NNCFConv2d', 'conv'))
        assert quant_scope in quant_model_conv.keys()

        store = []
        for op in quant_model_conv[quant_scope].pre_ops.values():
            if isinstance(op, (UpdateInputs, UpdateWeight)) and isinstance(op.operand, SymmetricQuantizer):
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)
        assert UpdateWeight.__name__ in store


def test_can_create_quant_loss_and_scheduler():
    config = get_basic_quantization_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)

    loss = compression_ctrl.loss
    assert isinstance(loss, CompressionLoss)

    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, CompressionScheduler)


def test_multiprocessing_distributed_shares_init_scales_signedness_across_gpus(tmp_path):
    num_init_steps = 10

    config = get_squeezenet_quantization_config()
    config['compression']['initializer'] = {'range': {'num_init_steps': num_init_steps}}

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(scale_signed_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config, tmp_path),
                                join=True)

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_after_broadcast)
    assert not compare_multi_gpu_dump(config, tmp_path, get_path_path_after_train_iters)


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
    def create_algo_and_compressed_model(config):
        model = TwoConvTestModel()
        compressed_model, algo = create_compressed_model_and_algo_for_test(model, config)
        return algo, compressed_model

    @staticmethod
    def create_config():
        config = get_empty_config()
        config['compression'] = {'algorithm': 'quantization', 'initializer': {'range': {'num_init_steps': 1}}}
        return config

    @staticmethod
    def create_dataloader(wrap_dataloader, config, device):
        input_infos_list = create_input_infos(config)
        input_sample_size = input_infos_list[0].shape
        data_loader = torch.utils.data.DataLoader(OnesDatasetMock(input_sample_size[1:]),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  shuffle=False)
        if wrap_dataloader:
            data_loader = InitializingDataLoader(data_loader=data_loader,
                                                 device=device,
                                                 kwargs={})
        return data_loader

    @staticmethod
    def check_sign_and_scale(model, ref_table):
        model_conv = get_all_modules_by_type(model, 'SymmetricQuantizer')
        for scope, module in model_conv.items():
            for pattern, ref_values in ref_table.items():
                match = re.search(pattern, str(scope))
                if match:
                    assert isinstance(module, SymmetricQuantizer)
                    assert module.signed == ref_values[0], 'sign is not matched for {}'.format(str(scope))
                    assert module.scale == ref_values[1], 'scale is not matched for {}'.format(str(scope))

    def test_scale_and_sign_init_for_quant_algo(self, wrap_dataloader):
        config = self.create_config()
        algo, compressed_model = self.create_algo_and_compressed_model(config)
        device = next(compressed_model.parameters()).device
        data_loader = self.create_dataloader(wrap_dataloader, config, device)

        algo.initialize(data_loader)

        self.check_sign_and_scale(compressed_model, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (True, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 1),
            '.*activation_quantizers.*Sequential\\[0\\].*': (True, 4),
            '.*activation_quantizers.*Sequential\\[1\\].*': (True, 24)
        })

    def test_scale_and_sign_init_for_quant_algo__without_init_section(self, wrap_dataloader):
        config = get_empty_config()
        config['compression'] = {'algorithm': 'quantization'}

        algo, compressed_model = self.create_algo_and_compressed_model(config)
        device = next(compressed_model.parameters()).device
        data_loader = self.create_dataloader(wrap_dataloader, config, device)

        algo.initialize(data_loader)

        self.check_sign_and_scale(compressed_model, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (True, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 1),
            '.*activation_quantizers.*Sequential\\[0\\].*': (True, 4),
            '.*activation_quantizers.*Sequential\\[1\\].*': (True, 24)
        })

    def test_scale_and_sign_init_for_quant_algo__with_zero_init_steps(self, wrap_dataloader):
        config = self.create_config()
        config['compression']['initializer']['range']['num_init_steps'] = 0

        algo, compressed_model = self.create_algo_and_compressed_model(config)
        device = next(compressed_model.parameters()).device
        data_loader = self.create_dataloader(wrap_dataloader, config, device)

        algo.initialize(data_loader)

        self.check_sign_and_scale(compressed_model, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (False, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 1),
            '.*activation_quantizers.*Sequential\\[0\\].*': (False, 1),
            '.*activation_quantizers.*Sequential\\[1\\].*': (False, 1)
        })

    def test_scale_and_sign_init_for_quant_algo__after_load_state(self, wrap_dataloader):
        config = self.create_config()
        algo, compressed_model = self.create_algo_and_compressed_model(config)
        load_state(compressed_model, {
            'module.features.0.0.pre_ops.0.op.signed_tensor': torch.tensor([0.]),  # quantizer of 1st conv's weights
            'module.features.1.0.pre_ops.0.op.scale': torch.tensor([100])  # quantizer of 2nd conv's weights
        })

        device = next(compressed_model.parameters()).device
        data_loader = self.create_dataloader(wrap_dataloader, config, device)

        algo.initialize(data_loader)

        self.check_sign_and_scale(compressed_model, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (False, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 100),
            '.*activation_quantizers.*Sequential\\[0\\].*': (True, 4),
            '.*activation_quantizers.*Sequential\\[1\\].*': (True, 24)
        })

    def test_scope_overrides(self, wrap_dataloader):
        config = self.create_config()
        config["compression"]["scope_overrides"] = {
            r"{re}NNCFConv2d\[[0-9]*\]$": {
                "bits": 7,
                "mode": "asymmetric",
            },
            r"{re}NNCFConv2d\[[0-9]*\]/conv2d_0": {
                "bits": 7,
                "signed": False,
            }
        }
        algo, compressed_model = self.create_algo_and_compressed_model(config)

        device = next(compressed_model.parameters()).device
        data_loader = self.create_dataloader(wrap_dataloader, config, device)

        algo.initialize(data_loader)

        quantizers = get_all_modules_by_type(compressed_model, ['SymmetricQuantizer',
                                                                'AsymmetricQuantizer'])
        quantizer_str_dict = {str(k): v for k, v in quantizers.items()}
        group_1 = [quantizer_str_dict["NNCFNetwork/TwoConvTestModel[nncf_module]/Sequential[features]/"
                                      "Sequential[0]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateWeight[0]/"
                                      "AsymmetricQuantizer[op]"],
                   quantizer_str_dict["NNCFNetwork/TwoConvTestModel[nncf_module]/Sequential[features]/"
                                      "Sequential[0]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateInputs[1]/"
                                      "AsymmetricQuantizer[op]"],
                   quantizer_str_dict['NNCFNetwork/TwoConvTestModel[nncf_module]/Sequential[features]/'
                                      'Sequential[1]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateWeight[0]/'
                                      'AsymmetricQuantizer[op]']
                   ]
        group_2 = [quantizer_str_dict['NNCFNetwork/ModuleDict[activation_quantizers]/'
                                      'SymmetricQuantizer[TwoConvTestModel/Sequential[features]'
                                      '/Sequential[0]/NNCFConv2d[0]/conv2d_0]']]

        for quantizer in group_1:
            assert isinstance(quantizer, AsymmetricQuantizer)
            assert quantizer.levels == 2 ** 7
        for quantizer in group_2:
            assert isinstance(quantizer, SymmetricQuantizer)
            assert not quantizer.signed


def get_path_to_keys(tmp_path, rank):
    return '{}_{}'.format(tmp_path, str(rank))


def activation_quantizers_dumping_worker(current_gpu, config, tmp_path):
    model = resnet50(pretrained=False)
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)
    path = get_path_to_keys(tmp_path, current_gpu)
    print(path)
    with open(path, 'w') as f:
        f.writelines("%s\n" % key for key in quant_model.activation_quantizers.keys())


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


def test_load_state_sets_initialized_flag():
    config = get_basic_quantization_config()

    model = TwoConvTestModel()
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)

    load_state(quant_model, {
        'module.features.0.0.pre_ops.0.op.signed_tensor': torch.tensor([1.0]),  # quantizer of 1st conv's weights
        'module.features.1.0.pre_ops.0.op.scale': torch.tensor([1.0])  # quantizer of 2nd conv's weights
    })

    quantizers = get_all_modules_by_type(quant_model, 'SymmetricQuantizer')
    for scope, module in quantizers.items():
        if 'activation_quantizers' in str(scope) or 'UpdateInputs' in str(scope):
            assert not module.initialized
        else:
            assert module.initialized


def test_quantize_has_proper_is_weights_flag():
    class Model(nn.Module):
        def __init__(self, size=1):
            super().__init__()
            self.size = size
            self.conv = nn.Conv2d(size, size, size)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    config = get_basic_quantization_config(model_size=2)
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)

    for module in quant_model.modules():
        if isinstance(module, NNCFConv2d):
            for op in module.pre_ops.values():
                assert isinstance(op, (UpdateWeight, UpdateInputs))
                assert op.operand.is_weights == isinstance(op, UpdateWeight)
    for _, aq in quant_model.get_compression_modules_by_type(CompressionModuleType.ACTIVATION_QUANTIZER).items():
        assert aq.is_weights is False


def test_can_quantize_free_operators(mocker):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones([1]))
            self.bias = nn.Parameter(torch.ones([1]))

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    mod = Model()
    config = get_basic_quantization_config(model_size=1)

    config["compression"].update({"quantize_inputs": False})
    quant_model, _ = create_compressed_model_and_algo_for_test(mod, config)

    quantizer_list = quant_model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER).values()
    assert len(quantizer_list) == 2
    for quantizer in quantizer_list:
        mocker.spy(quantizer, 'quantize')

    quant_model.do_dummy_forward()
    for quantizer in quantizer_list:
        assert quantizer.quantize.call_count == 1


@pytest.fixture(name="hw_config_type", params=HWConfigType)
def hw_config_type_(request):
    return request.param


def test_hw_config_quantization_can_quantize_squeezenet(hw_config_type):
    config = get_squeezenet_quantization_config()
    config["hw_config"] = hw_config_type.value
    model = squeezenet1_1_custom()
    create_compressed_model_and_algo_for_test(model, config)


class QuantizeInputsTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=2)
        self.linear = nn.Linear(in_features=8, out_features=8)

    #    (1)     (2)      (3)    (4)   (5)
    #     |       |        |      |     |-----\
    #  (conv1)   (MP)     (MP)    (MP)  (MP)  |
    #     |       |        |      |     |     |
    #     |       |       (+)     |     |     |
    #     |       |--\     |      |     |     |
    #     |       |   \    |      |     |     |
    #     |    (conv2) | (conv3)  |     |     |
    #     |       |    |   |       \   /      |
    #     |     (AvP)  \   |       (cat)      |
    #     |       |     \  |         |        |
    #  (conv4) (linear)  \ |      (conv6)     |
    #     |       |      (cat)       |        |
    #     |       |        |        (+)------/
    #     |       |      (conv5)     |
    #   (AvP)     |        |         |
    #     |       |      (AvP)       |
    #      \      |        /         |
    #       \---(cat)---------------/

    def forward(self, input_1, input_2, input_3, input_4, input_5):
        x_1 = self.conv1(input_1)
        x_1 = self.conv4(x_1)
        x_1 = F.adaptive_avg_pool2d(x_1, output_size=1)
        x_1 = x_1.flatten(start_dim=1)

        x_2_br = F.max_pool2d(input_2, kernel_size=2)
        x_2 = self.conv2(x_2_br)
        x_2 = F.adaptive_avg_pool2d(x_2, output_size=1)
        x_2 = x_2.flatten(start_dim=1)
        x_2 = self.linear(x_2)

        x_3 = F.max_pool2d(input_3, kernel_size=2)
        x_3 = x_3 + torch.ones_like(x_3)
        x_3 = self.conv3(x_3)
        x_3 = x_3.flatten(start_dim=1)
        x_2_br = x_2_br.flatten(start_dim=1)
        x_3 = torch.cat([x_2_br, x_3], dim=-1)
        x_3 = self.conv5(x_3.unsqueeze(2).unsqueeze(3).transpose(1, 2))
        x_3 = F.adaptive_avg_pool2d(x_3, output_size=1)
        x_3 = x_3.flatten(start_dim=1)

        x_4 = F.max_pool2d(input_4, kernel_size=2)
        x_5 = F.max_pool2d(input_5, kernel_size=2)
        x_45 = torch.cat([x_4, x_5], dim=1)
        x_45 = self.conv6(x_45)
        x_45 = x_45.flatten(start_dim=1)
        in_5_flat = input_5.flatten(start_dim=1)
        x_45 += F.pad(input_5.flatten(start_dim=1), [0, x_45.shape[1] - in_5_flat.shape[1]])

        return torch.cat([x_1, x_2, x_3, x_45], dim=-1)


def test_quantize_inputs():
    model = QuantizeInputsTestModel()
    config = get_basic_quantization_config()
    config["input_info"] = [
        {
            "sample_size": (2, 3, 32, 32),
        },
        {
            "sample_size": (2, 3, 32, 32),
        },
        {
            "sample_size": (2, 3, 32, 32),
        },
        {
            "sample_size": (2, 3, 32, 32),
        },
        {
            "sample_size": (2, 3, 32, 32),
        }
    ]

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    REF_QUANTIZED_INPUT_MODULE_SCOPES = [
        "QuantizeInputsTestModel/NNCFConv2d[conv1]",
        "QuantizeInputsTestModel/NNCFConv2d[conv2]",
        "QuantizeInputsTestModel/NNCFConv2d[conv5]",
        "QuantizeInputsTestModel/NNCFConv2d[conv6]",
    ]
    for ref_qinput_module_scope_str in REF_QUANTIZED_INPUT_MODULE_SCOPES:
        scope = Scope.from_str(ref_qinput_module_scope_str)
        assert model.get_module_by_scope(scope) is not None
        assert ref_qinput_module_scope_str in compression_ctrl.quantized_inputs_modules_registry

    nncf_modules_dict = model.get_nncf_modules()
    for scope, nncf_module in nncf_modules_dict.items():
        scope_str = str(scope)
        update_inputs_count = sum(1 for pre_op in nncf_module.pre_ops.values() if isinstance(pre_op, UpdateInputs))
        if scope_str in REF_QUANTIZED_INPUT_MODULE_SCOPES:
            assert update_inputs_count == 1
        else:
            assert update_inputs_count == 0
