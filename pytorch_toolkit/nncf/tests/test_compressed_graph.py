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
from functools import partial

import networkx as nx
import pytest
import torch
import torch.nn as nn
import torchvision

from nncf.dynamic_graph.patch_pytorch import nncf_model_input
from nncf.nncf_network import NNCFNetwork
from nncf.algo_selector import create_compression_algorithm_builders
from nncf.dynamic_graph.context import get_version_agnostic_name, TracingContext
from nncf.dynamic_graph.graph import NNCFGraph
from nncf.dynamic_graph.graph_builder import create_input_infos, GraphBuilder, create_dummy_forward_fn, ModelInputInfo
from nncf.layers import LSTMCellNNCF, NNCF_RNN
from nncf.utils import get_all_modules_by_type
from tests import test_models
from tests.modules.seq2seq.gnmt import GNMT
from tests.modules.test_rnn import replace_lstm
from tests.test_helpers import get_empty_config, create_compressed_model_and_algo_for_test


def get_basic_quantization_config(quantization_type, input_sample_size):
    config = get_empty_config(input_sample_size=input_sample_size)
    config["compression"] = {"algorithm": "quantization",
                             "activations": {
                                 "mode": quantization_type
                             },
                             "weights": {
                                 "mode": quantization_type
                             }}
    return config

def get_version_agnostic_graph(nx_graph):
    done = False
    while not done:
        counter = 0
        for node_name, node_data in nx_graph.nodes().data():
            version_specific_name = node_data["type"]
            version_agnostic_name = get_version_agnostic_name(version_specific_name)
            if version_agnostic_name != version_specific_name:
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


class QuantizeTestCaseConfiguration:
    def __init__(self, quant_type: str, graph_dir: str):
        self.quant_type = quant_type
        self.graph_dir = graph_dir

QUANTIZERS = ['symmetric', 'asymmetric']

@pytest.fixture(scope='function', params=QUANTIZERS)
def _case_config(request):
    quantization_type = request.param
    graph_dir = os.path.join('quantized', quantization_type)
    return QuantizeTestCaseConfiguration(quantization_type, graph_dir)


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

        nncf_model_input(input_encoder)
        nncf_model_input(input_enc_len)
        nncf_model_input(input_decoder)
        model(input_encoder, input_enc_len, input_decoder)

    return partial(forward_fn, seq_len_=seq_len, batch_size_=batch_size, vocab_size_=vocab_size, batch_first_=False)


TEST_MODELS_DEFAULT = [
    ("alexnet.dot", test_models.AlexNet, (1, 3, 32, 32)),
    ("lenet.dot", test_models.LeNet, (1, 3, 32, 32)),
    ("resnet18.dot", test_models.ResNet18, (1, 3, 32, 32)),
    ("resnet50.dot", test_models.ResNet50, (1, 3, 32, 32)),
    ("vgg16.dot", partial(test_models.VGG, 'VGG16'), (1, 3, 32, 32)),
    ("inception.dot", test_models.GoogLeNet, (1, 3, 32, 32)),
    ("densenet121.dot", test_models.DenseNet121, (1, 3, 32, 32)),
    ("inception_v3.dot", partial(test_models.Inception3, aux_logits=True, transform_input=True), (2, 3, 299, 299)),
    ("squeezenet1_0.dot", test_models.squeezenet1_0, (1, 3, 32, 32)),
    ("squeezenet1_1.dot", test_models.squeezenet1_1, (1, 3, 32, 32)),
    ("shufflenetv2.dot", partial(test_models.ShuffleNetV2, net_size=0.5), (1, 3, 32, 32)),
    ("shuflenet_g2.dot", test_models.ShuffleNetG2, (1, 3, 32, 32)),
    ("ssd_vgg.dot", test_models.ssd_vgg300, (2, 3, 300, 300)),
    ("ssd_mobilenet.dot", test_models.ssd_mobilenet, (2, 3, 300, 300)),
    ("mobilenet_v2.dot", torchvision.models.MobileNetV2, (2, 3, 32, 32)),
    ("resnext29_32x4d.dot", test_models.ResNeXt29_32x4d, (1, 3, 32, 32)),
    ("pnasnetb.dot", test_models.PNASNetB, (1, 3, 32, 32)),
    ("senet18.dot", test_models.SENet18, (1, 3, 32, 32)),
    ("preresnet50.dot", test_models.PreActResNet50, (1, 3, 32, 32)),
    ("unet.dot", test_models.UNet, (1, 3, 360, 480)),
    ("lstm_cell.dot", LSTMCellNNCF, (1, 1)),
    ("lstm_uni_seq.dot", partial(NNCF_RNN, num_layers=1, bidirectional=False), (3, 1, 1)),
    ("lstm_uni_stacked.dot", partial(NNCF_RNN, num_layers=2, bidirectional=False), (3, 1, 1)),
    ("lstm_bi_seq.dot", partial(NNCF_RNN, num_layers=1, bidirectional=True), (3, 1, 1)),
    ("lstm_bi_stacked.dot", partial(NNCF_RNN, num_layers=2, bidirectional=True), (3, 1, 1))
]

TEST_MODELS = [(m[0], m[1], m[2]) for m in TEST_MODELS_DEFAULT]


def _get_model_name(dot_name):
    if isinstance(dot_name, tuple):
        dot_name = dot_name[0]
    return dot_name[:dot_name.find('.dot')]


def check_model_graph(compressed_model: NNCFNetwork, ref_dot_file_name: str, ref_dot_file_directory: str):
    compressed_model.to('cuda')
    compressed_model.do_dummy_forward()
    compressed_model.do_dummy_forward()
    check_graph(compressed_model.get_graph(), ref_dot_file_name, ref_dot_file_directory)

@pytest.mark.parametrize(
    "model_name, model_builder, input_size", TEST_MODELS, ids=[_get_model_name(m[0]) for m in TEST_MODELS]
)
class TestModelsGraph:
    def test_build_graph(self, model_name, model_builder, input_size):
        net = model_builder()
        graph_builder = GraphBuilder(create_dummy_forward_fn([ModelInputInfo(input_size), ]))
        graph = graph_builder.build_graph(net)
        check_graph(graph, model_name, 'original')

    @pytest.mark.parametrize(
        ("algo", "params"),
        (
            ("rb_sparsity", {}),
            ("magnitude_sparsity", {}),
            ("const_sparsity", {})
        ), ids=['RB', 'Magnitude', 'Const']
    )
    def test_sparse_network(self, model_name, model_builder, input_size, algo, params):
        model = model_builder()
        from nncf.layers import NNCF_MODULES_MAP
        sparsifiable_modules = list(NNCF_MODULES_MAP.values())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))

        config = get_empty_config(input_sample_size=input_size)
        config["compression"] = {"algorithm": algo, "params": params}

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
        assert ref_num_sparsed == len(compression_ctrl.sparsified_module_info)
        check_model_graph(compressed_model, model_name, algo)

    def test_quantize_network(self, model_name, model_builder, input_size, _case_config):
        model = model_builder()
        config = get_basic_quantization_config(_case_config.quant_type, input_sample_size=input_size)
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
        check_model_graph(compressed_model, model_name, _case_config.graph_dir)

    def test_sparse_quantize_network(self, model_name, model_builder, input_size):
        model = model_builder()

        from nncf.layers import NNCF_MODULES_MAP
        sparsifiable_modules = list(NNCF_MODULES_MAP.values())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))
        config = get_empty_config(input_sample_size=input_size)
        config["compression"] = [
            {"algorithm": "rb_sparsity", "params": {}},
            {"algorithm": "quantization", "params": {}}
        ]

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        assert ref_num_sparsed == len(compression_ctrl.child_algos[0].sparsified_module_info)
        check_model_graph(compressed_model, model_name, "quantized_rb_sparsity")


def test_gnmt_quantization(_case_config):
    model = GNMT(vocab_size=32)
    model = replace_lstm(model)
    forward_fn_ = gnmt_forward_fn(seq_len=10, batch_size=3, vocab_size=32)

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_size=(3, 10))
    config["compression"].update({
        "quantizable_subgraph_patterns": [["linear", "__add__"],
                                          ["sigmoid", "__mul__", "__add__"],
                                          ["__add__", "tanh", "__mul__"],
                                          ["sigmoid", "__mul__"]],
        "disable_function_quantization_hooks": True,
        "ignored_scopes": ["GNMT/ResidualRecurrentEncoder[encoder]/Embedding[embedder]",
                           "GNMT/ResidualRecurrentDecoder[decoder]/Embedding[embedder]"]})

    compressed_model = NNCFNetwork(model,
                                   input_infos=create_input_infos(config),
                                   dummy_forward_fn=forward_fn_,
                                   scopes_without_shape_matching=
                                   ['GNMT/ResidualRecurrentDecoder[decoder]/RecurrentAttention[att_rnn]/'
                                    'BahdanauAttention[attn]'])

    compression_algo_builder_list = create_compression_algorithm_builders(config)

    for builder in compression_algo_builder_list:
        compressed_model = builder.apply_to(compressed_model)
    _ = compressed_model.commit_compression_changes()
    check_model_graph(compressed_model, 'gnmt_variable.dot', _case_config.graph_dir)


def test_resnet18__with_not_qinput(_case_config):
    model = test_models.ResNet18()
    input_shape = (1, 3, 32, 32)

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_size=input_shape)
    config["compression"].update({"quantize_inputs": False})

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'resnet18_no_qinput.dot', _case_config.graph_dir)


def test_resnet18__with_ignore(_case_config):
    model = test_models.ResNet18()
    input_shape = (1, 3, 32, 32)

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_size=input_shape)
    ignored_scopes = ['ResNet/Sequential[layer3]', ]
    config.update({"ignored_scopes": ignored_scopes})  # Global config ignored_scopes for NNCF module replacement
    config["compression"].update({"ignored_scopes": ignored_scopes})  # Local ignored_scopes for quantization

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'resnet18_ignore.dot', _case_config.graph_dir)


def test_iterate_module_list():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ml = nn.ModuleList([nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)])

        def forward(self, x):
            return [self.ml[0](x), self.ml[1](x)]

    net = Net()

    context = TracingContext()
    with context:
        _ = net(torch.zeros(1, 1, 1, 1))

    check_graph(context.graph, 'case_iterate_module_list.dot', 'original')


def test_output_quantization(_case_config):
    model = test_models.UNet()
    input_shape = (1, 3, 360, 480)

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_size=input_shape)
    config["compression"].update({"quantize_outputs": True})

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'unet_qoutput.dot', _case_config.graph_dir)


def test_custom_quantizable_subgraph_patterns(_case_config):
    model = test_models.SENet18()

    input_shape = (1, 3, 32, 32)

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_size=input_shape)

    config["compression"].update({"quantize_outputs": False,
                                  "quantizable_subgraph_patterns": [["sigmoid", "__mul__"],
                                                                    ["__iadd__", "batch_norm"]]})

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'senet_custom_patterns.dot', _case_config.graph_dir)
