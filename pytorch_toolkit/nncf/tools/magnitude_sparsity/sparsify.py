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
import xml.etree.cElementTree as ET

import numpy as np

from tools.ir_utils import find_all_parameters, get_ir_paths

argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", help="input IR name", required=True)
argparser.add_argument("-b", "--bin", help="Input *.bin file name")
argparser.add_argument("-o", "--output", help="Output *.bin file name")
argparser.add_argument("-s", "--sparsity-level", help="Desired number of zero parameters")
argparser.add_argument("--ignore", help="comma separated list of ignored layers", default="")
argparser.add_argument("--sparsify-first-conv", type=bool, default=False)
argparser.add_argument("--sparsify-fc", type=bool, default=True)
argparser.add_argument("--normed-threshold", type=bool, default=True)
args = argparser.parse_args()


def main():
    model_bin, model_xml = get_ir_paths(args.model, args.bin)

    with open(model_bin, "rb") as f:
        buffer = f.read()

    ignored = args.ignore.split(",") + get_ignored_layers(model_xml, args.sparsify_first_conv, args.sparsify_fc)

    all_parameters = find_all_parameters(buffer, model_xml)
    total_params = 0
    zero_params = 0
    for name, param in all_parameters.items():
        if name.split('.')[0] in ignored:
            continue
        total_params += param.data.size
        zero_params += (param.data == 0).sum()

    print("initial sparsity: {:.2f}%".format((zero_params / total_params) * 100))

    if args.sparsity_level:
        sparsify(all_parameters, ignored, args.normed_threshold)

    if args.output:
        print("saving new ir to: {}".format(args.output))

        with open(args.output, "wb") as f:
            f.write(generate_new_bin(buffer, all_parameters))


def get_ignored_layers(model_xml, sparsify_first_conv=False, sparsify_fc=True):
    ir_tree = ET.parse(model_xml)
    ignored_layers = []
    all_convs = [l for l in ir_tree.iter("layer") if l.get("type") == "Convolution"]
    all_fcs = [l for l in ir_tree.iter("layer") if l.get("type") == "FullyConnected"]
    all_bns = [l for l in ir_tree.iter("layer") if l.get("type") == "ScaleShift"]
    ignored_layers += [bn.get("name") for bn in all_bns]
    if not sparsify_first_conv:
        ignored_layers.append(all_convs[0].get("name"))
    if not sparsify_fc:
        ignored_layers.append(all_fcs[-1].get("name"))
    return ignored_layers


def generate_new_bin(buffer, parameters):
    new_buffer = bytearray(buffer)
    for param in parameters.values():
        new_buffer[param.offset:param.offset + param.size] = param.data.tobytes()
    return bytes(new_buffer)


def norm(data):
    return np.sqrt(np.sum(data ** 2))


def sparsify(parameters, ignored, normalize=True):
    if normalize:
        data_flat = [p.data.flatten() / norm(p.data) for k, p in parameters.items() if k.split(".")[0] not in ignored]
    else:
        data_flat = [p.data.flatten() for k, p in parameters.items() if k.split(".")[0] not in ignored]

    data_flat = np.concatenate(data_flat)
    data_flat = np.absolute(data_flat)
    data_flat.sort()
    sparsity_level = float(args.sparsity_level) / 100
    sparsity_threshold = data_flat[int(sparsity_level * len(data_flat))]
    for name, param in parameters.items():
        if name.split(".")[0] in ignored:
            continue

        if normalize:
            param.data[np.absolute(param.data / norm(param.data)) < sparsity_threshold] = 0
        else:
            param.data[np.absolute(param.data) < sparsity_threshold] = 0


if __name__ == '__main__':
    main()
