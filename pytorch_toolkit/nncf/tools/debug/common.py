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

import numpy as np
import torch
from torch import nn

from examples.common.model_loader import load_model
from nncf.model_creation import create_compressed_model
from nncf.checkpoint_loading import load_state
from nncf.layers import NNCFConv2d, NNCFLinear
from examples.common.utils import print_statistics


def dump_in_out_hook(module, inputs, output):
    dump_out_hook(module, inputs, output)
    dump_path = get_dump_path(module)
    if dump_path:
        key = 0
        output_dir = os.path.abspath(os.path.join(dump_path, os.pardir))
        file_name = os.path.basename(dump_path)
        for input_ in inputs:
            key += 1
            input_data = input_.data.cpu().numpy().flatten()
            dump_name = '.'.join([file_name, "in", str(key)])
            npy_path, _ = save_dump(dump_name, output_dir, input_data)
            add_full_dump_path(module, npy_path)


def dump_out_hook(module, inputs, output):
    dump_path = get_dump_path(module)
    if dump_path:
        output_data = output.data.cpu().numpy().flatten()
        output_dir = os.path.abspath(os.path.join(dump_path, os.pardir))
        file_name = os.path.basename(dump_path)
        dump_name = '.'.join([file_name, "out"])
        npy_path, _ = save_dump(dump_name, output_dir, output_data, force=False)
        add_full_dump_path(module, npy_path)


def get_dump_path(module):
    if hasattr(module, "dump_path"):
        return module.dump_path
    return None


def set_dump_path(layer, path):
    layer.dump_path = path


def add_full_dump_path(layer, full_path):
    if not hasattr(layer, 'dump_full_paths'):
        layer.dump_full_paths = []
    layer.dump_full_paths.append(full_path)


def get_full_dump_paths(layer):
    if hasattr(layer, 'dump_full_paths'):
        return layer.dump_full_paths
    return None


def is_weightable(layer):
    return isinstance(layer, (nn.Conv2d, nn.Linear)) and \
           not isinstance(layer, (NNCFConv2d, NNCFLinear))


def has_sparse_quant_weights(layer, name):
    from nncf.quantization.layers import SymmetricQuantizer
    from nncf.sparsity.rb.layers import RBSparsifyingWeight
    return (isinstance(layer, RBSparsifyingWeight) and ('sparsified_weight' in name)) or \
           (isinstance(layer, SymmetricQuantizer) and ('quantized_weight' in name))


def save_dump_(path, ext, saver, data, force=False):
    full_path = '.'.join([path, ext])
    if not os.path.exists(full_path) or force:
        print("Saving dump to {}".format(full_path))
        saver(full_path, data)
    else:
        print("Dump already exists " + full_path)
    return full_path


def save_dump(dump_name, output_dir, data, force=False):
    path = os.path.join(output_dir, dump_name)
    npy_path = save_dump_(path, "npy", np.save, data, force)
    txt_path = save_dump_(path, "txt", partial(np.savetxt, fmt="%s"), data, force)
    return npy_path, txt_path


def register_print_hooks(path, model, data_to_compare, num_layers=-1, dump_activations=False, prefix='', idx=0):
    for name, children in model.named_children():
        name_full = "{}{}".format(prefix, name)
        idx = register_print_hooks(path, children, data_to_compare, num_layers, dump_activations,
                                   prefix=name_full + ".", idx=idx)

        within_range = (num_layers == -1) or idx < num_layers
        has_weights = has_sparse_quant_weights(children, name_full) or is_weightable(children)
        within_type = has_weights if not dump_activations else dump_activations
        if within_range and within_type:
            # always there for activations if dump_activation is enabled
            # always there for weights if dump_activation is disabled
            name_full = name_full.replace('/', '_')
            dump_path = os.path.join(path, '.'.join([str(idx), name_full]))
            idx += 1
            if is_weightable(children):
                output_dir = os.path.abspath(os.path.join(dump_path, os.pardir))
                file_name = os.path.basename(dump_path)

                def dump_attr(attr):
                    if hasattr(children, attr):
                        dump_name = '.'.join([file_name, attr])
                        data = children.weight.data.numpy()
                        save_dump(dump_name, output_dir, data, force=False)
                        data_to_compare[dump_name] = data

                dump_attr('weight')
                dump_attr('bias')
            else:
                set_dump_path(children, dump_path)
                hook = dump_in_out_hook if dump_activations else dump_out_hook
                children.register_forward_hook(hook)
    return idx


def load_torch_model(config, cuda=False):
    weights = config.get('weights')
    model = load_model(config.get('model'),
                       pretrained=config.get('pretrained', True) if weights is None else False,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params', {}))
    compression_ctrl, model = create_compressed_model(model, config)
    if weights:
        sd = torch.load(weights, map_location='cpu')
        load_state(model, sd)
    if cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    print_statistics(compression_ctrl.statistics())
    return model


def compare_activations(ir_dump_txt, torch_dump_npy):
    with open(ir_dump_txt, 'r') as fin:
        first_line = fin.readline()
        if "shape:" in first_line:
            data = fin.read().splitlines(True)
            with open(ir_dump_txt, 'w') as fout:
                fout.writelines(data)
    ie = np.loadtxt(ir_dump_txt, dtype=np.float32)
    pt = np.load(torch_dump_npy)
    print("Size, [ MIN, MAX ]")
    print_info = lambda np_array: print(
        "{} [{:.3f}, {:.3f}]".format(np_array.size, np_array.min().item(), np_array.max().item()))
    print_info(ie)
    print_info(pt)
    print("Maximum of absolute difference: {:.7f}".format(abs(ie - pt).max()))


def print_args(args):
    for arg in sorted(vars(args)):
        print("{: <27s}: {}".format(arg, getattr(args, arg)))
