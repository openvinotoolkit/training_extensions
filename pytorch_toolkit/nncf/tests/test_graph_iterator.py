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

from collections import OrderedDict
import pytest

import torch.nn.functional as F
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, AvgPool2d, init

from nncf.dynamic_graph.version_agnostic_op_names import VersionAgnosticNames
from nncf.utils import get_all_modules_by_type, get_module_by_node_name, get_all_node_names, \
    apply_by_node_name, set_module_by_node_name, parse_node_name



class ModelForTest(Module):
    def __init__(self, size=1):
        super().__init__()
        self.size = size
        self.conv0 = Conv2d(size, size, size)
        self.conv1 = Conv2d(size, size, size)
        self.bn1 = BatchNorm2d(size)
        self.bn2 = BatchNorm2d(size)
        self.norm10 = BatchNorm2d(size)
        self.norm20 = BatchNorm2d(size)
        self.avgpool = AvgPool2d(size)
        self.layer1 = Sequential(OrderedDict([
            ('conv01', self.conv0),
            ('norm01', self.norm10),
            ('relu01', ReLU()),
            ('pool01', MaxPool2d(size))
        ]))
        self.layer2 = Sequential(OrderedDict([
            ('layer1', self.layer1),
            ('conv02', self.conv0),
            ('relu02', ReLU()),
            ('norm02', self.norm20),
            ('pool02', MaxPool2d(size))
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = ReLU()(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return x


def test_get_module_by_node_name__for_non_nested_module():
    model = ModelForTest()
    assert get_module_by_node_name(model, 'ModelForTest/BatchNorm2d[bn1]') == model.bn1


def test_get_module_by_node_name__for_nested_module():
    model = ModelForTest()
    assert get_module_by_node_name(model, 'ModelForTest/Sequential[layer2]/Sequential[layer1]') == model.layer1


def test_get_all_layers_by_type__for_standard_type():
    model = ModelForTest()
    act_bn = get_all_modules_by_type(model, 'BatchNorm2d')
    act_bn = OrderedDict((str(k), v) for k, v in act_bn.items())
    ref_bn = {
        'ModelForTest/BatchNorm2d[bn1]': model.bn1,
        'ModelForTest/BatchNorm2d[bn2]': model.bn2,
        'ModelForTest/BatchNorm2d[norm10]': model.norm10,
        'ModelForTest/BatchNorm2d[norm20]': model.norm20,
        'ModelForTest/Sequential[layer1]/BatchNorm2d[norm01]': model.norm10,
        'ModelForTest/Sequential[layer2]/BatchNorm2d[norm02]': model.norm20,
        'ModelForTest/Sequential[layer2]/Sequential[layer1]/BatchNorm2d[norm01]': model.norm10,
    }
    assert act_bn == ref_bn


def test_get_all_layers_by_type__for_multiple_type():
    model = ModelForTest()
    act_bn = get_all_modules_by_type(model, ['ReLU', 'AvgPool2d'])
    act_bn = OrderedDict((str(k), v) for k, v in act_bn.items())
    ref_bn = [
        'ModelForTest/AvgPool2d[avgpool]',
        'ModelForTest/Sequential[layer1]/ReLU[relu01]',
        'ModelForTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]',
        'ModelForTest/Sequential[layer2]/ReLU[relu02]']
    assert list(act_bn.keys()) == ref_bn
    assert isinstance(act_bn, OrderedDict)


def test_get_all_layers_by_type__for_not_exact_type():
    model = ModelForTest()
    l = get_all_modules_by_type(model, 'Avg')
    assert not l


def test_get_all_layers_by_type__for_subtype():
    model = ModelForTest()
    l = get_all_modules_by_type(model, 'AvgPool2d_dummy')
    assert not l


IGNORED_SCOPES = [
    ("single", ['ModelForTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]']),
    ("multiple", ['ModelForTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]',
                  'ModelForTest/Sequential[layer2]/Conv2d[conv02]']),
    ("common", ['ModelForTest/Sequential[layer1]'])
    ]


@pytest.mark.parametrize(
    "ignored_scopes", [s[1] for s in IGNORED_SCOPES], ids=[s[0] for s in IGNORED_SCOPES]
)
def test_get_all_layers_by_type__with_ignored_scope(ignored_scopes):
    model = ModelForTest()

    model_modules = set()
    for _, module in model.named_modules():
        model_modules.add(module.__class__.__name__)
    model_modules = list(model_modules)

    act_modules = get_all_modules_by_type(model, model_modules, ignored_scopes=ignored_scopes)

    for module_scope, _ in act_modules.items():
        for scope in ignored_scopes:
            assert not str(module_scope).startswith(str(scope))


def test_set_module_by_node_name__for_non_nested_module():
    model = ModelForTest()
    new_module = ReLU()
    set_module_by_node_name(model, 'ModelForTest/BatchNorm2d[bn1]', new_module)
    assert new_module == get_module_by_node_name(model, 'ModelForTest/ReLU[bn1]')


def test_set_module_by_node_name__for_nested_module():
    model = ModelForTest()
    new_module = ReLU()
    set_module_by_node_name(model, 'ModelForTest/Sequential[layer2]/Sequential[layer1]', new_module)
    assert new_module == get_module_by_node_name(model, 'ModelForTest/Sequential[layer2]/ReLU[layer1]')


def test_get_all_nodes():
    model = ModelForTest()
    ref_list = [
        'ModelForTest/Conv2d[conv1]/conv2d',
        'ModelForTest/BatchNorm2d[bn1]/batch_norm',
        'ModelForTest/ReLU/' + VersionAgnosticNames.RELU,
        'ModelForTest/' + VersionAgnosticNames.RELU,
        'ModelForTest/BatchNorm2d[bn2]/batch_norm',
        'ModelForTest/Sequential[layer2]/Sequential[layer1]/Conv2d[conv01]/conv2d',
        'ModelForTest/Sequential[layer2]/Sequential[layer1]/BatchNorm2d[norm01]/batch_norm',
        'ModelForTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]/' + VersionAgnosticNames.RELU,
        'ModelForTest/Sequential[layer2]/Sequential[layer1]/MaxPool2d[pool01]/max_pool2d',
        'ModelForTest/Sequential[layer2]/Conv2d[conv02]/conv2d',
        'ModelForTest/Sequential[layer2]/ReLU[relu02]/' + VersionAgnosticNames.RELU,
        'ModelForTest/Sequential[layer2]/BatchNorm2d[norm02]/batch_norm',
        'ModelForTest/Sequential[layer2]/MaxPool2d[pool02]/max_pool2d',
        'ModelForTest/AvgPool2d[avgpool]/avg_pool2d'
    ]

    act_list = get_all_node_names(model, (1, 1, 4, 4))
    assert ref_list == act_list


def test_apply_by_node_name():
    model = ModelForTest()
    node_name = 'ModelForTest/BatchNorm2d[bn1]'
    bn1 = get_module_by_node_name(model, node_name)
    bn1.weight.data.fill_(1)
    assert bn1.weight == 1
    apply_by_node_name(model, [node_name], command=lambda m: init.zeros_(m.weight))
    assert bn1.weight == 0


def test_parse_node_name():
    node_names = ["conv2d", "Conv2d[conv1]", "Conv2d[conv1]/RELU[relu]"]
    ref_class_name = ["conv2d", "Conv2d", "RELU"]
    ref_var_name = [None, "conv1", "relu"]
    ref_prefix = [None, None, "Conv2d[conv1]"]
    for i, node_name in enumerate(node_names):
        prefix, class_name, var_name = parse_node_name(node_name)
        assert class_name == ref_class_name[i]
        assert var_name == ref_var_name[i]
        assert prefix == ref_prefix[i]
