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
import os
from collections import namedtuple
from copy import deepcopy
from functools import partial

import networkx as nx
import pytest
import torch
import torch.nn as nn

from nncf import QuantizedNetwork
from nncf import SymmetricQuantizer
from nncf.algo_selector import create_compression_algorithm
from nncf.dynamic_graph import patch_torch_operators, reset_context, context
from nncf.dynamic_graph.context import get_version_agnostic_name
from nncf.dynamic_graph.graph import NNCFGraph
from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.helpers import replace_lstm
from nncf.layers import LSTMCellNNCF, NNCF_RNN
from nncf.quantization.layers import AsymmetricQuantizer, QuantizerConfig
from nncf.utils import get_all_modules_by_type
from tests import test_models
from tests.modules.seq2seq.gnmt import GNMT
from tests.test_helpers import get_empty_config

patch_torch_operators()


def get_version_agnostic_graph(nx_graph):
    done = False
    while not done:
        counter = 0
        for node_name, node_data in nx_graph.nodes().data():
            version_specific_name = node_data["type"]
            version_agnostic_name = get_version_agnostic_name(version_specific_name)
            if version_agnostic_name is not None:
                node_data["type"] = version_agnostic_name
                mapping = dict(zip(nx_graph, nx_graph))  # identity mapping
                new_node_name = node_name.replace(version_specific_name, version_agnostic_name)
                mapping[node_name] = new_node_name
                nx_graph = nx.relabel_nodes(nx_graph, mapping, copy=False)
                break  # Looks like iterators will be invalidated after relabel_nodes
            counter += 1
        if counter == len(nx_graph.nodes().data()):
            done = True

    return nx_graph


def sort_dot(path):
    with open(path, 'r') as f:
        content = f.readlines()
    start_line = 'strict digraph  {\n'
    end_line = '}\n'
    content.remove(start_line)
    content.remove(end_line)

    def graph_key(line, offset):
        key = line.split(' ')[0].replace('"', '')
        if '->' in line:
            key += line.split(' ')[3].replace('"', '')
            return int(key) + offset
        return int(key)

    sorted_content = sorted(content, key=partial(graph_key, offset=len(content)))
    with open(path, 'w') as f:
        f.write(start_line)
        f.writelines(sorted_content)
        f.write(end_line)


def check_graph(graph: NNCFGraph, path_to_dot, graph_dir):
    # pylint:disable=protected-access
    nx_graph = graph._get_graph_to_dump()
    data_dir = os.path.join(os.path.dirname(__file__), 'data/reference_graphs')
    path_to_dot = os.path.abspath(os.path.join(data_dir, graph_dir, path_to_dot))

    # validate .dot file manually!
    if not os.path.exists(path_to_dot):
        nx.drawing.nx_pydot.write_dot(nx_graph, path_to_dot)
        sort_dot(path_to_dot)

    load_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)
    load_graph = get_version_agnostic_graph(load_graph)

    # nx_graph is expected to have version-agnostic operator names already
    for k, attrs in nx_graph.nodes.items():
        attrs = {k: str(v) for k, v in attrs.items()}
        load_attrs = {k: str(v).strip('"') for k, v in load_graph.nodes[k].items()}
        assert attrs == load_attrs

    assert load_graph.nodes.keys() == nx_graph.nodes.keys()
    assert nx.DiGraph(load_graph).edges == nx_graph.edges


QuantizeConfig = namedtuple('QuantizeConfig', ['quantizer', 'graph_dir'])

QUANTIZERS = [
    QuantizeConfig(lambda _, is_weights=False, input_shape=None: SymmetricQuantizer(
        QuantizerConfig(signedness_to_force=is_weights, is_weights=is_weights, input_shape=input_shape)),
                   'symmetric'),
    QuantizeConfig(lambda _, is_weights, input_shape=None: AsymmetricQuantizer(QuantizerConfig()), 'asymmetric')
]


@pytest.fixture(scope='function', params=QUANTIZERS, ids=[pair.graph_dir for pair in QUANTIZERS])
def _quantize_config(request):
    config = request.param
    graph_dir = os.path.join('quantized', config.graph_dir)
    return QuantizeConfig(config.quantizer, graph_dir)


def default_forward_fn(model, input_size_):
    device = next(model.parameters()).device
    return model(torch.zeros(input_size_).to(device))


def gnmt_forward_fn(seq_len, batch_size, vocab_size):
    def forward_fn(model, seq_len_, batch_size_, vocab_size_, batch_first_):
        device = next(model.parameters()).device

        def gen_packed_sequence():
            seq_list = []
            seq_lens = torch.LongTensor(batch_size_).random_(1, seq_len_ + 1).to(device)
            seq_lens = torch.sort(seq_lens, descending=True).values
            for seq_size in seq_lens:
                seq_list.append(torch.LongTensor(seq_size.item()).random_(1, vocab_size_).to(device))
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=batch_first_)
            return padded_seq_batch, seq_lens

        x_data, seq_lens = gen_packed_sequence()
        input_encoder = x_data
        input_enc_len = seq_lens
        input_decoder = gen_packed_sequence()[0]
        model.forward(input_encoder, input_enc_len, input_decoder)

    return partial(forward_fn, seq_len_=seq_len, batch_size_=batch_size, vocab_size_=vocab_size, batch_first_=False)


TEST_MODELS_DEFAULT = [
    ("alexnet.dot", test_models.AlexNet, (1, 3, 32, 32)),
    ("lenet.dot", test_models.LeNet, (1, 3, 32, 32)),
    ("resnet18.dot", test_models.ResNet18, (1, 3, 32, 32)),
    ("resnet50.dot", test_models.ResNet50, (1, 3, 32, 32)),
    ("vgg16.dot", partial(test_models.VGG, 'VGG16'), (1, 3, 32, 32)),
    ("inception.dot", test_models.GoogLeNet, (1, 3, 32, 32)),
    ("densenet121.dot", test_models.DenseNet121, (1, 3, 32, 32)),
    ("inception_v3.dot", test_models.Inception3, (2, 3, 299, 299)),
    ("squeezenet1_0.dot", test_models.squeezenet1_0, (1, 3, 32, 32)),
    ("squeezenet1_1.dot", test_models.squeezenet1_1, (1, 3, 32, 32)),
    ("shufflenetv2.dot", partial(test_models.ShuffleNetV2, net_size=0.5), (1, 3, 32, 32)),
    ("shuflenet_g2.dot", test_models.ShuffleNetG2, (1, 3, 32, 32)),
    ("mobnetv2.dot", test_models.MobileNetV2, (1, 3, 32, 32)),
    ("resnext29_32x4d.dot", test_models.ResNeXt29_32x4d, (1, 3, 32, 32)),
    ("pnasnetb.dot", test_models.PNASNetB, (1, 3, 32, 32)),
    ("senet18.dot", test_models.SENet18, (1, 3, 32, 32)),
    ("ssd_vgg.dot", test_models.ssd_vgg300, (2, 3, 300, 300)),
    ("ssd_mobilenet.dot", test_models.ssd_mobilenet, (2, 3, 300, 300)),
    ("preresnet50.dot", test_models.PreActResNet50, (1, 3, 32, 32)),
    ("unet.dot", test_models.UNet, (1, 3, 360, 480)),
    ("lstm_cell.dot", LSTMCellNNCF, (1, 1)),
    ("lstm_uni_seq.dot", partial(NNCF_RNN, num_layers=1, bidirectional=False), (3, 1, 1)),
    ("lstm_uni_stacked.dot", partial(NNCF_RNN, num_layers=2, bidirectional=False), (3, 1, 1)),
    ("lstm_bi_seq.dot", partial(NNCF_RNN, num_layers=1, bidirectional=True), (3, 1, 1)),
    ("lstm_bi_stacked.dot", partial(NNCF_RNN, num_layers=2, bidirectional=True), (3, 1, 1))
]

TEST_MODELS = [(m[0], m[1], partial(default_forward_fn, input_size_=m[2])) for m in TEST_MODELS_DEFAULT]


def _get_model_name(dot_name):
    if isinstance(dot_name, tuple):
        dot_name = dot_name[0]
    return dot_name[:dot_name.find('.dot')]


@pytest.mark.parametrize(
    "model_name, model_builder, forward_fn_", TEST_MODELS, ids=[_get_model_name(m[0]) for m in TEST_MODELS]
)
class TestModelsGraph:
    device = torch.device('cuda')

    def test_build_graph(self, model_name, model_builder, forward_fn_):
        net = model_builder()
        net.to(self.device)
        ctx = reset_context('test')
        with context('test') as c:
            forward_fn_(net)
            c.reset_scope_operator_call_counters()
            forward_fn_(net)

        check_graph(ctx.graph, model_name, 'original')

    @pytest.mark.parametrize(
        ("algo", "params"),
        (
            ("rb_sparsity", {}),
            ("magnitude_sparsity", {}),
            ("const_sparsity", {})
        ), ids=['RB', 'Magnitude', 'Const']
    )
    def test_sparse_network(self, model_name, model_builder, forward_fn_, algo, params):
        model = model_builder()
        from nncf.layers import NNCF_MODULES_MAP
        sparsifiable_modules = list(NNCF_MODULES_MAP.values())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))
        ctx = reset_context('test')
        config = get_empty_config()
        config["compression"] = {"algorithm": algo, "params": params}
        compression_algo = create_compression_algorithm(model, config, dummy_forward_fn=forward_fn_)
        assert ref_num_sparsed == len(compression_algo.sparsified_module_info)
        model = compression_algo.model
        model.to(self.device)
        with context('test') as c:
            forward_fn_(model)
            c.reset_scope_operator_call_counters()
            forward_fn_(model)
        check_graph(ctx.graph, model_name, algo)

    def test_quantize_network(self, model_name, model_builder, forward_fn_, _quantize_config):
        net = model_builder()
        ctx = reset_context('orig')
        ctx = reset_context('quantized_graphs')
        qnet = QuantizedNetwork(net, _quantize_config.quantizer,
                                input_infos=[ModelInputInfo(forward_fn_.keywords["input_size_"]), ],
                                dummy_forward_fn=forward_fn_)
        qnet.to(self.device)
        forward_fn_(qnet)
        forward_fn_(qnet)
        check_graph(ctx.graph, model_name, _quantize_config.graph_dir)

    def test_sparse_quantize_network(self, model_name, model_builder, forward_fn_):
        model = model_builder()
        from nncf.layers import NNCF_MODULES_MAP
        sparsifiable_modules = list(NNCF_MODULES_MAP.values())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))
        ctx = reset_context('test')
        config = get_empty_config()
        config["compression"] = [
            {"algorithm": "rb_sparsity", "params": {}},
            {"algorithm": "quantization", "params": {}}
        ]
        ctx = reset_context('orig')
        ctx = reset_context('quantized_graphs')
        compression_algo = create_compression_algorithm(model, config, dummy_forward_fn=forward_fn_)
        assert ref_num_sparsed == len(compression_algo.child_algos[0].sparsified_module_info)
        model = compression_algo.model
        model.to(self.device)
        forward_fn_(model)
        forward_fn_(model)
        check_graph(ctx.graph, model_name, "quantized_rb_sparsity")


def test_gnmt_quantization(_quantize_config):
    net = GNMT(vocab_size=32)
    net = replace_lstm(net)
    forward_fn_ = gnmt_forward_fn(seq_len=10, batch_size=3, vocab_size=32)
    ctx = reset_context('orig')
    ctx = reset_context('quantized_graphs')
    qnet = QuantizedNetwork(
        net, _quantize_config.quantizer, dummy_forward_fn=forward_fn_,
        quantizable_subgraph_patterns=[["linear", "__add__"],
                                       ["sigmoid", "__mul__", "__add__"],
                                       ["__add__", "tanh", "__mul__"],
                                       ["sigmoid", "__mul__"]],
        scopes_without_shape_matching=
        ['GNMT/ResidualRecurrentDecoder[decoder]/RecurrentAttention[att_rnn]/BahdanauAttention[attn]'],
        disable_function_quantization_hooks=True,
    )
    forward_fn_(qnet)
    forward_fn_(qnet)

    check_graph(ctx.graph, 'gnmt_variable.dot', _quantize_config.graph_dir)


def test_resnet18__with_not_qinput(_quantize_config):
    net = test_models.ResNet18()
    ctx = reset_context('orig')
    ctx = reset_context('quantized_graphs')
    input_shape = (1, 3, 32, 32)
    qnet = QuantizedNetwork(net, _quantize_config.quantizer, [ModelInputInfo(input_shape), ],
                            quantize_inputs=False)
    _ = qnet(torch.zeros(*input_shape))
    _ = qnet(torch.zeros(*input_shape))

    check_graph(ctx.graph, 'resnet18_no_qinput.dot', _quantize_config.graph_dir)


def test_resnet18__with_ignore(_quantize_config):
    net = test_models.ResNet18()
    ctx = reset_context('orig')
    ctx = reset_context('quantized_graphs')
    input_shape = (1, 3, 32, 32)
    qnet = QuantizedNetwork(net, _quantize_config.quantizer, [ModelInputInfo(input_shape), ],
                            ignored_scopes=['ResNet/Sequential[layer3]'])
    _ = qnet(torch.zeros(*input_shape))
    _ = qnet(torch.zeros(*input_shape))

    check_graph(ctx.graph, 'resnet18_ignore.dot', _quantize_config.graph_dir)


def test_iterate_module_list():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ml = nn.ModuleList([nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)])

        def forward(self, x):
            return [self.ml[0](x), self.ml[1](x)]

    net = Net()

    ctx = reset_context('test')
    with context('test'):
        _ = net(torch.zeros(1, 1, 1, 1))

    check_graph(ctx.graph, 'case_iterate_module_list.dot', 'original')


def test_output_quantization(_quantize_config):
    net = test_models.UNet()
    ctx = reset_context('orig')
    ctx = reset_context('quantized_graphs')
    input_shape = (1, 3, 360, 480)
    qnet = QuantizedNetwork(net, _quantize_config.quantizer, [ModelInputInfo(input_shape), ],
                            quantize_outputs=True)
    _ = qnet(torch.zeros(*input_shape))
    _ = qnet(torch.zeros(*input_shape))

    check_graph(ctx.graph, 'unet_qoutput.dot', _quantize_config.graph_dir)


def test_custom_quantizable_subgraph_patterns(_quantize_config):
    net = test_models.SENet18()
    ctx = reset_context('orig')
    ctx = reset_context('quantized_graphs')
    input_shape = (1, 3, 32, 32)
    qnet = QuantizedNetwork(net, _quantize_config.quantizer, [ModelInputInfo(input_shape), ],
                            quantize_outputs=False,
                            quantizable_subgraph_patterns=(("sigmoid", "__mul__"),
                                                           ("__iadd__", "batch_norm")))
    _ = qnet(torch.zeros(*input_shape))
    _ = qnet(torch.zeros(*input_shape))

    check_graph(ctx.graph, 'senet_custom_patterns.dot', _quantize_config.graph_dir)


def test_disable_shape_matching(_quantize_config):
    class MatMulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.ones([1]))

        def forward(self, inputs):
            half1, half2 = torch.chunk(inputs, 2, dim=2)
            return torch.bmm(half1, half2.transpose(1, 2))

    model = MatMulModel()

    _ = reset_context('orig')
    _ = reset_context('quantized_graphs')
    input_shape_1 = (3, 32, 32)
    input_shape_2 = (4, 64, 64)

    qnet_no_shape = QuantizedNetwork(deepcopy(model), _quantize_config.quantizer, [ModelInputInfo(input_shape_1), ],
                                     scopes_without_shape_matching=['MatMulModel'])
    _ = qnet_no_shape(torch.zeros(*input_shape_1))
    graph_1 = deepcopy(qnet_no_shape.get_quantized_graph())

    _ = qnet_no_shape(torch.zeros(*input_shape_2))
    graph_2 = deepcopy(qnet_no_shape.get_quantized_graph())

    keys_1 = list(graph_1.get_all_node_keys())
    keys_2 = list(graph_2.get_all_node_keys())
    assert len(keys_1) == 1
    assert keys_1 == keys_2

    _ = reset_context('orig')
    _ = reset_context('quantized_graphs')
    qnet = QuantizedNetwork(model, _quantize_config.quantizer, [ModelInputInfo(input_shape_1), ])
    _ = qnet(torch.zeros(*input_shape_1))
    _ = qnet(torch.zeros(*input_shape_2))
    # The second forward run should have led to an increase in registered node counts
    # since disable_shape_matching was False and the network was run with a different
    # shape of input tensor
    assert qnet.get_quantized_graph().get_nodes_count() > graph_1.get_nodes_count()


def test_forward_trace_functor():
    from nncf.dynamic_graph.patch_pytorch import ForwardTraceOnly
    from nncf.dynamic_graph.trace_tensor import TracedTensor, TensorMeta

    func = ForwardTraceOnly()
    shape1, shape2 = ([32, 1, 4, 8], [1, 8, 12, 16])
    meta1, meta2 = (TensorMeta(5, 1, shape1), TensorMeta(3, 8, shape2))
    input_tensor1 = TracedTensor.from_torch_tensor(torch.Tensor(shape1), meta1)
    input_tensor2 = TracedTensor.from_torch_tensor(torch.Tensor(shape2), meta2)

    # 1 -> 1
    output_tensor = func(torch.Tensor.view, input_tensor1, [-1])
    assert output_tensor.tensor_meta == input_tensor1.tensor_meta

    # 1 -> N
    outputs = func(torch.Tensor.chunk, input_tensor1, 3)
    for out in outputs:
        assert out.tensor_meta == input_tensor1.tensor_meta

    # N -> N (2 -> 2)
    outputs = func(lambda x: x + [5], [input_tensor1, input_tensor2])
    assert outputs[0].tensor_meta == input_tensor1.tensor_meta
    assert outputs[1].tensor_meta == input_tensor2.tensor_meta

    # M -> N (2 -> 3)
    with pytest.raises(RuntimeError):
        outputs = func(lambda x: x + [torch.Tensor(shape2)], [input_tensor1, input_tensor2])

    # M -> N (2 -> 1)
    with pytest.raises(RuntimeError):
        outputs = func(lambda x: x[0], [input_tensor1, input_tensor2])
