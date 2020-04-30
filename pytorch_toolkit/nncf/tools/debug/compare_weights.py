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

import argparse
import json
import xml.etree.cElementTree as ET
from collections import OrderedDict

import os

import numpy as np
from torch import randn

from tools.ir_utils import get_ir_paths, find_all_parameters
from tools.debug.common import save_dump, register_print_hooks, load_torch_model, get_full_dump_paths, print_args


argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", help="input IR name", required=True)
argparser.add_argument("--bin", help="Input *.bin file name")
argparser.add_argument("-o", "--output-dir", help="Output directory to dump weights", required=True)
argparser.add_argument("-c", "--config", type=str, default='config.json', help="Model's config", required=True)
argparser.add_argument("-n", "--num-layers", type=int, default=-1,
                       help="Compare weights for given number of layers")
argparser.add_argument("--ignore", help="comma separated list of ignored layers", default="")
args = argparser.parse_args()
print_args(args)


def main():
    model_bin, model_xml = get_ir_paths(args.model, args.bin)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ir_weights = collect_IR_weights(os.path.join(args.output_dir, "IR"), model_xml, model_bin, args.num_layers)

    config = json.load(open(args.config))
    torch_weights = collect_torch_weights(os.path.join(args.output_dir, "PTH"), config, args.num_layers)

    assert len(ir_weights) == len(torch_weights), '{} vs {}'.format(len(ir_weights), len(torch_weights))
    print("Maximum of absolute difference - IR vs Torch")
    max_max = []
    for (k1, v1), (k2, v2) in zip(ir_weights.items(), torch_weights.items()):
        max_diff = abs(v1 - v2).max()
        max_max.append(max_diff)
        print("{0:.5} - max diff [{1:}]  vs  [{2:}]".format(max_diff, k1, k2))
    print("Global maximum:  {0:.5}".format(np.max(max_max)))


def collect_IR_weights(output_dir, model_xml, model_bin, num_layers):
    data_to_compare = OrderedDict()
    print("IR loaded from {}".format(model_bin))
    with open(model_bin, "rb") as f:
        buffer = f.read()

    ignored = args.ignore.split(",") + get_ignored_layers(model_xml, args.num_layers)

    all_parameters = find_all_parameters(buffer, model_xml)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    idx = 0

    for name, param in all_parameters.items():
        if name.split('.')[0] in ignored or 'bias' in name:
            continue
        if (num_layers > 0 and idx < num_layers) or (num_layers == -1):
            name = name.replace(os.path.sep, '_')
            dump_name = '.'.join([str(idx), name])
            output_data = param.data.flatten()
            save_dump(dump_name, output_dir, output_data)
            data_to_compare[dump_name] = output_data
            idx += 1
    return data_to_compare


def collect_torch_weights(output_dir, config, num_layers):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_torch_model(config)
    model_e = model.eval()

    data_to_compare = OrderedDict()

    register_print_hooks(output_dir, model_e, num_layers=num_layers, data_to_compare=data_to_compare,
                         dump_activations=False)
    input_ = randn(config['input_sample_size'])
    model_e(input_)

    for _, module in enumerate(model_e.modules()):
        paths = get_full_dump_paths(module)
        if paths is not None:
            for dump_path in paths:
                if os.path.isfile(dump_path):
                    data_to_compare[os.path.splitext(os.path.basename(dump_path))[0]] = np.load(dump_path)
    return data_to_compare


def get_ignored_layers(model_xml, num_layers=1):
    ir_tree = ET.parse(model_xml)
    ignored_layers = []
    all_supported = [l for l in ir_tree.iter("layer") if l.get("type") == ("Convolution", "FullyConnected")]
    if num_layers > 0:
        ignored_layers += [layer.get("name") for layer in all_supported[num_layers:]]
    all_bns = [l for l in ir_tree.iter("layer") if l.get("type") == "ScaleShift"]
    ignored_layers += [bn.get("name") for bn in all_bns]
    return ignored_layers


if __name__ == '__main__':
    main()
