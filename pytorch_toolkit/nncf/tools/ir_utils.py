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

from collections import OrderedDict

import xml.etree.cElementTree as ET
import numpy as np


class Parameter:
    def __init__(self, data, size, offset, shape):
        self.shape = shape
        self.offset = offset
        self.size = size
        self.data = data


def get_ir_paths(model_arg, bin_arg):
    if model_arg.endswith(".xml"):
        model_xml = model_arg
    else:
        model_xml = model_arg + ".xml"
    if bin_arg is not None:
        model_bin = bin_arg
    else:
        model_bin = model_xml.replace(".xml", ".bin")
    return model_bin, model_xml


def find_all_parameters(buffer, model_xml):
    ir_tree = ET.parse(model_xml)
    all_parameters = OrderedDict()
    for layer in ir_tree.iter("layer"):
        if layer.get("type") not in {"Convolution", "FullyConnected", "ScaleShift"}:
            continue

        get_weight_shape_fn = get_conv_weight_shape
        if layer.get("type") == "FullyConnected":
            get_weight_shape_fn = get_fc_weight_shape
        if layer.get("type") == "ScaleShift":
            get_weight_shape_fn = get_ss_weight_shape

        extract_params(buffer, all_parameters, layer, get_weight_shape_fn)

    return all_parameters


def get_conv_weight_shape(layer, input_shape, output_shape):
    groups = int(layer.find('data').get("group", 1))
    kernel_size = [int(dim) for dim in layer.find('data').get("kernel").split(",")]
    return [output_shape[1], input_shape[1] // groups, *kernel_size]


def get_fc_weight_shape(layer, input_shape, output_shape):
    return [output_shape[1], input_shape[1]]


def get_ss_weight_shape(layer, input_shape, output_shape):
    return [output_shape[1]]


def extract_params(buffer, all_parameters, layer, get_weight_shape_fn):
    layer_name = layer.get("name")
    precision = np.float32 if layer.get("precision").lower() == 'fp32' else np.float16
    weight = layer.find("blobs/weights")
    biases = layer.find("blobs/biases")
    input_shape = [int(dim.text) for dim in layer.find("input/port")]
    output_shape = [int(dim.text) for dim in layer.find("output/port")]

    weight_shape = get_weight_shape_fn(layer, input_shape, output_shape)
    weight_offset = int(weight.get("offset"))
    weight_size = int(weight.get("size"))
    weight_data = get_blob(buffer, weight_offset, weight_size, weight_shape, precision)
    param = Parameter(weight_data, weight_size, weight_offset, weight_shape)
    all_parameters["{}.weight".format(layer_name)] = param
    if biases is not None:
        bias_shape = [output_shape[1]]
        bias_size = int(biases.get("size"))
        bias_offset = int(biases.get("offset"))

        bias_data = get_blob(buffer, bias_offset, bias_size, bias_shape, precision)
        bias_param = Parameter(bias_data, bias_size, bias_offset, bias_shape)
        all_parameters["{}.bias".format(layer_name)] = bias_param


def get_blob(buffer, offset, size, shape, dtype=np.float32):
    data = np.frombuffer(buffer[offset:offset + size], dtype=dtype).copy()
    if shape is not None:
        data = data.reshape(shape)
    return data
