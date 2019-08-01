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

import sys
from argparse import ArgumentParser
from collections import OrderedDict

import torch

from nncf.registry import Registry

INCEPTION_NAME = 'inception'
RESNET_NAME = 'resnet'
MOBILENET_NAME = 'mobilenet'

KEYS_REPLACERS = Registry("keys_replacers")


@KEYS_REPLACERS.register(INCEPTION_NAME)
def inception_replacer(k):
    if 'RELU' in k:
        return k.replace('335', '0')
    return k


@KEYS_REPLACERS.register(MOBILENET_NAME)
def mobilenet_replacer(k):
    keywords = ['hardtanh', 'batch_norm', '__add__']
    if any(x in k for x in keywords):
        return k.replace('63', '0').replace('62', '0').replace('111', '0')
    return k


@KEYS_REPLACERS.register(RESNET_NAME)
def resnet_replacer(k):
    if 'RELU' in k:
        return k.replace('96', '0').replace('100', '1').replace('109', '2').replace('194', '0')
    if 'BatchNorm2d' in k:
        return k.replace('103', '0').replace('106', '0')
    return k


def main(argv):
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-model', help='Path to input model file', required=True)
    parser.add_argument('-o', '--output-model', help='Path to output model file', required=True)
    parser.add_argument('-n', '--name', help='Name of model', choices=[INCEPTION_NAME, RESNET_NAME, MOBILENET_NAME],
                        required=True)
    args = parser.parse_args(args=argv)

    pth = torch.load(args.input_model)
    sd = pth['state_dict']

    replace_key_fn = KEYS_REPLACERS.get(args.name)
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_k = replace_key_fn(k)
        if new_k != k:
            print('{}\n{}\n\n'.format(k, new_k))
        new_sd[replace_key_fn(k)] = v
    pth['state_dict'] = new_sd

    torch.save(pth, args.output_model)


if __name__ == '__main__':
    main(sys.argv[1:])
