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
from collections import namedtuple
from typing import Dict
from typing import Tuple

import numpy as np
import pytest
import torch
from copy import deepcopy
from functools import partial
from torch import nn
from torch.nn import Module

from nncf.compression_method_api import CompressionAlgorithmController
from nncf.config import Config
from nncf.dynamic_graph.context import Scope
from nncf.model_creation import create_compressed_model
from nncf.layers import NNCF_MODULES_MAP
from nncf.nncf_network import NNCFNetwork
from nncf.utils import get_all_modules_by_type, objwalk


def fill_conv_weight(conv, value):
    conv.weight.data.fill_(value)
    with torch.no_grad():
        mask = torch.eye(conv.kernel_size[0])
        conv.weight += mask


def fill_bias(module, value):
    module.bias.data.fill_(value)


def fill_linear_weight(linear, value):
    linear.weight.data.fill_(value)
    with torch.no_grad():
        n = min(linear.in_features, linear.out_features)
        linear.weight[:n, :n] += torch.eye(n)


def create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


class BasicConvTestModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, kernel_size=2, weight_init=-1, bias_init=-2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.conv = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init)

    @staticmethod
    def default_weight():
        return torch.tensor([[[[0., -1.],
                               [-1., 0.]]], [[[0., -1.],
                                              [-1., 0.]]]])

    @staticmethod
    def default_bias():
        return torch.tensor([-2., -2])

    def forward(self, x):
        return self.conv(x)

    @property
    def weights_num(self):
        return self.out_channels * self.kernel_size ** 2

    @property
    def bias_num(self):
        return self.kernel_size

    @property
    def nz_weights_num(self):
        return self.kernel_size * self.out_channels

    @property
    def nz_bias_num(self):
        return self.kernel_size


def test_basic_model_has_expected_params():
    model = BasicConvTestModel()
    act_weights = model.conv.weight.data
    ref_weights = BasicConvTestModel.default_weight()
    act_bias = model.conv.bias.data
    ref_bias = BasicConvTestModel.default_bias()

    check_equal(act_bias, ref_bias)
    check_equal(act_weights, ref_weights)

    assert act_weights.nonzero().size(0) == model.nz_weights_num
    assert act_bias.nonzero().size(0) == model.nz_bias_num
    assert act_weights.numel() == model.weights_num
    assert act_bias.numel() == model.bias_num


def test_basic_model_is_valid():
    model = BasicConvTestModel()
    input_ = torch.ones([1, 1, 4, 4])
    ref_output = torch.ones((1, 2, 3, 3)) * (-4)
    act_output = model(input_)
    check_equal(ref_output, act_output)


class TwoConvTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(nn.Sequential(create_conv(1, 2, 2, -1, -2)))
        self.features.append(nn.Sequential(create_conv(2, 1, 3, 0, 0)))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

    @property
    def weights_num(self):
        return 8 + 18

    @property
    def bias_num(self):
        return 2 + 1

    @property
    def nz_weights_num(self):
        return 4 + 6

    @property
    def nz_bias_num(self):
        return 2


def test_two_conv_model_has_expected_params():
    model = TwoConvTestModel()
    act_weights_1 = model.features[0][0].weight.data
    act_weights_2 = model.features[1][0].weight.data
    act_bias_1 = model.features[0][0].bias.data
    act_bias_2 = model.features[1][0].bias.data

    ref_weights_1 = BasicConvTestModel.default_weight()
    channel = torch.eye(3, 3).reshape([1, 1, 3, 3])
    ref_weights_2 = torch.cat((channel, channel), 1)

    check_equal(act_weights_1, ref_weights_1)
    check_equal(act_weights_2, ref_weights_2)

    check_equal(act_bias_1, BasicConvTestModel.default_bias())
    check_equal(act_bias_2, torch.tensor([0]))

    assert act_weights_1.nonzero().size(0) + act_weights_2.nonzero().size(0) == model.nz_weights_num
    assert act_bias_1.nonzero().size(0) + act_bias_2.nonzero().size(0) == model.nz_bias_num
    assert act_weights_1.numel() + act_weights_2.numel() == model.weights_num
    assert act_bias_1.numel() + act_bias_2.numel() == model.bias_num


def test_two_conv_model_is_valid():
    model = TwoConvTestModel()
    input_ = torch.ones([1, 1, 4, 4])
    ref_output = torch.tensor([-24])
    act_output = model(input_)
    check_equal(ref_output, act_output)


def get_empty_config(model_size=4, input_sample_size=(1, 1, 4, 4)):
    config = Config()
    config.update({
        "model": "basic_sparse_conv",
        "model_size": model_size,
        "input_info":
            {
                "sample_size": input_sample_size,
            },
    })
    return config


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_equal(test, reference, rtol=1e-4):
    for i, (x, y) in enumerate(zip(test, reference)):
        y = y.cpu().detach().numpy()
        np.testing.assert_allclose(x, y, rtol=rtol, err_msg="Index: {}".format(i))


def create_compressed_model_and_algo_for_test(model: NNCFNetwork, config) -> Tuple[
        NNCFNetwork, CompressionAlgorithmController]:
    algo, model = create_compressed_model(model, config, dump_graphs=False)
    return model, algo


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = nn.Linear(1, 1)

    def forward(self, *input_, **kwargs):
        return None


def check_correct_nncf_modules_replacement(model: NNCFNetwork, compressed_model: NNCFNetwork) -> Tuple[
        Dict[Scope, Module],
        Dict[Scope, Module]]:
    """
    Check that all convolutions in model was replaced by NNCF convolution.
    :param model: original model
    :param compressed_model: compressed model
    :return: list of all convolutions in  original model and list of all NNCF convolutions from compressed model
    """
    NNCF_MODULES_REVERSED_MAP = {value: key for key, value in NNCF_MODULES_MAP.items()}
    original_modules = get_all_modules_by_type(model, list(NNCF_MODULES_MAP.values()))
    nncf_modules = get_all_modules_by_type(compressed_model.get_nncf_wrapped_model(),
                                           list(NNCF_MODULES_MAP.keys()))
    assert len(original_modules) == len(nncf_modules)
    print(original_modules, nncf_modules)
    for scope in original_modules.keys():
        sparse_scope = deepcopy(scope)
        elt = sparse_scope.pop()  # type: ScopeElement
        elt.calling_module_class_name = NNCF_MODULES_REVERSED_MAP[elt.calling_module_class_name]
        sparse_scope.push(elt)
        print(sparse_scope, nncf_modules)
        assert sparse_scope in nncf_modules
    return original_modules, nncf_modules


class ObjwalkTestClass:
    def __init__(self, field: int):
        self.field = field

    def member_fn(self, val):
        return ObjwalkTestClass(self.field + 1)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

NamedTuple = namedtuple("NamedTuple", ("field1", "field2"))

OBJWALK_INIT_VAL = 0
OBJWALK_REF_VAL = OBJWALK_INIT_VAL + 1
TEST_VS_REF_OBJECTS_TO_WALK = [
    (0,
     0),

    ("foo",
     "foo"),

    (ObjwalkTestClass(OBJWALK_INIT_VAL),
     ObjwalkTestClass(OBJWALK_REF_VAL)),

    ([0, ObjwalkTestClass(OBJWALK_INIT_VAL), "bar"],
     [0, ObjwalkTestClass(OBJWALK_REF_VAL), "bar"]),

    ([ObjwalkTestClass(OBJWALK_INIT_VAL), ObjwalkTestClass(OBJWALK_INIT_VAL), (5, 8)],
     [ObjwalkTestClass(OBJWALK_REF_VAL), ObjwalkTestClass(OBJWALK_REF_VAL), (5, 8)]),

    (
        {
            "obj1": ObjwalkTestClass(OBJWALK_INIT_VAL),
            "obj2": ObjwalkTestClass(OBJWALK_INIT_VAL)
        },
        {
            "obj1": ObjwalkTestClass(OBJWALK_REF_VAL),
            "obj2": ObjwalkTestClass(OBJWALK_REF_VAL)
        }
    ),

    ((ObjwalkTestClass(OBJWALK_INIT_VAL), 42),
     (ObjwalkTestClass(OBJWALK_REF_VAL), 42)),

    ([(ObjwalkTestClass(OBJWALK_INIT_VAL), 8), [ObjwalkTestClass(OBJWALK_INIT_VAL), "foo"],
      {"bar": ObjwalkTestClass(OBJWALK_INIT_VAL),
       "baz": (ObjwalkTestClass(OBJWALK_INIT_VAL), ObjwalkTestClass(OBJWALK_INIT_VAL)),
       "xyzzy": {1337: ObjwalkTestClass(OBJWALK_INIT_VAL),
                 31337: ObjwalkTestClass(OBJWALK_INIT_VAL)}}],
     [(ObjwalkTestClass(OBJWALK_REF_VAL), 8), [ObjwalkTestClass(OBJWALK_REF_VAL), "foo"],
      {"bar": ObjwalkTestClass(OBJWALK_REF_VAL),
       "baz": (ObjwalkTestClass(OBJWALK_REF_VAL), ObjwalkTestClass(OBJWALK_REF_VAL)),
       "xyzzy": {1337: ObjwalkTestClass(OBJWALK_REF_VAL),
                 31337: ObjwalkTestClass(OBJWALK_REF_VAL)}}]
    ),
    (
        (0, NamedTuple(field1=ObjwalkTestClass(OBJWALK_INIT_VAL), field2=-5.3), "bar"),
        (0, NamedTuple(field1=ObjwalkTestClass(OBJWALK_REF_VAL), field2=-5.3), "bar"),
    )
]


@pytest.fixture(name="objwalk_objects", params=TEST_VS_REF_OBJECTS_TO_WALK)
def objwalk_objects_(request):
    return request.param


def test_objwalk(objwalk_objects):
    start_obj = objwalk_objects[0]
    ref_obj = objwalk_objects[1]

    def is_target_class(obj):
        return isinstance(obj, ObjwalkTestClass)

    fn_to_apply = partial(ObjwalkTestClass.member_fn, val=OBJWALK_REF_VAL)

    test_obj = objwalk(start_obj, is_target_class, fn_to_apply)

    assert test_obj == ref_obj
