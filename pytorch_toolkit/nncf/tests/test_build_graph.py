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
from functools import partial

import networkx as nx
import pytest
import torch
import torch.nn as nn

from nncf import QuantizedNetwork
from nncf.algo_selector import create_compression_algorithm
from nncf.dynamic_graph import patch_torch_operators, reset_context, context
from nncf.dynamic_graph.context import get_version_agnostic_name
from nncf.dynamic_graph.utils import to_networkx
from nncf.utils import get_all_modules_by_type
from tests import test_models
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


def check_graph(nx_graph, path_to_dot, graph_dir):
    data_dir = os.path.join(os.path.dirname(__file__), 'data/reference_graphs')
    path_to_dot = os.path.abspath(os.path.join(data_dir, graph_dir, path_to_dot))

    # validate .dot file manually!
    if not os.path.exists(path_to_dot):
        nx.drawing.nx_pydot.write_dot(nx_graph, path_to_dot)

    load_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)
    load_graph = get_version_agnostic_graph(load_graph)

    # nx_graph is expected to have version-agnostic operator names already
    for k, attrs in nx_graph.nodes.items():
        attrs = {k: str(v) for k, v in attrs.items()}
        load_attrs = {k: str(v).strip('"') for k, v in load_graph.nodes[k].items()}
        assert attrs == load_attrs

    assert load_graph.nodes.keys() == nx_graph.nodes.keys()
    assert nx.DiGraph(load_graph).edges == nx_graph.edges


TEST_MODELS = [
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
    pytest.param("shufflenetv2.dot", partial(test_models.ShuffleNetV2, net_size=0.5), (1, 3, 32, 32),
                 marks=pytest.mark.skip(reason="__getitem__ is not wrapped")),
    ("shuflenet_g2.dot", test_models.ShuffleNetG2, (1, 3, 32, 32)),
    ("mobnetv2.dot", test_models.MobileNetV2, (1, 3, 32, 32)),
    ("resnext29_32x4d.dot", test_models.ResNeXt29_32x4d, (1, 3, 32, 32)),
    ("pnasnetb.dot", test_models.PNASNetB, (1, 3, 32, 32)),
    ("senet18.dot", test_models.SENet18, (1, 3, 32, 32)),
    ("preresnet50.dot", test_models.PreActResNet50, (1, 3, 32, 32)),
    ("unet.dot", test_models.UNet, (1, 3, 360, 480))
]


def _get_model_name(dot_name):
    if isinstance(dot_name, tuple):
        dot_name = dot_name[0]
    return dot_name[:dot_name.find('.dot')]


@pytest.mark.parametrize(
    "model_name, model_builder, input_size", TEST_MODELS, ids=[_get_model_name(m[0]) for m in TEST_MODELS]
)
class TestModelsGraph:
    def test_build_graph(self, model_name, model_builder, input_size):
        net = model_builder()
        ctx = reset_context('test')
        with context('test') as c:
            _ = net(torch.zeros(input_size))
            c.reset_scope_operator_call_counters()
            _ = net(torch.zeros(input_size))

        check_graph(to_networkx(ctx), model_name, 'original')

    @pytest.mark.parametrize(
        ("algo", "params"),
        (
            ("rb_sparsity", {}),
            ("magnitude_sparsity", {"update_mask_on_forward": True}),
            ("magnitude_sparsity", {"update_mask_on_forward": False}),
            ("const_sparsity", {})
        ), ids=['RB', 'Magnitude', 'Magnitude_Frozen', 'Const']
    )
    def test_sparse_network(self, model_name, model_builder, input_size, algo, params):
        model = model_builder()
        from nncf.layers import NNCF_MODULES_MAP
        sparsifiable_modules = list(NNCF_MODULES_MAP.values())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))
        ctx = reset_context('test')
        config = get_empty_config(input_sample_size=input_size)
        config["compression"] = {"algorithm": algo, "params": params}
        compression_algo = create_compression_algorithm(model, config)
        assert ref_num_sparsed == len(compression_algo.sparsified_module_info)
        model = compression_algo.model
        with context('test') as c:
            _ = model(torch.zeros(input_size))
            c.reset_scope_operator_call_counters()
            _ = model(torch.zeros(input_size))
        if params and not params.get('update_mask_on_forward', None):
            algo += '_frozen'
        check_graph(to_networkx(ctx), model_name, algo)

    def test_quantize_network(self, model_name, model_builder, input_size):
        net = model_builder()
        ctx = reset_context('orig')
        ctx = reset_context('quantized_graphs')
        qnet = QuantizedNetwork(net, input_size)
        _ = qnet(torch.zeros(*input_size))
        _ = qnet(torch.zeros(*input_size))
        check_graph(to_networkx(ctx), model_name, 'quantized')


def test_resnet18__with_qinput():
    net = test_models.ResNet18()
    ctx = reset_context('orig')
    ctx = reset_context('quantized_graphs')
    input_shape = (1, 3, 32, 32)
    qnet = QuantizedNetwork(net, input_shape, quantize_inputs=True)
    _ = qnet(torch.zeros(*input_shape))
    _ = qnet(torch.zeros(*input_shape))

    check_graph(to_networkx(ctx), 'resnet18_qinput.dot', 'quantized')


def test_resnet18__with_ignore():
    net = test_models.ResNet18()
    ctx = reset_context('orig')
    ctx = reset_context('quantized_graphs')
    input_shape = (1, 3, 32, 32)
    qnet = QuantizedNetwork(net, input_shape, ignored_scopes=['ResNet/Sequential[layer3]'])
    _ = qnet(torch.zeros(*input_shape))
    _ = qnet(torch.zeros(*input_shape))

    check_graph(to_networkx(ctx), 'resnet18_ignore.dot', 'quantized')


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

    check_graph(to_networkx(ctx), 'case_iterate_module_list.dot', 'original')
