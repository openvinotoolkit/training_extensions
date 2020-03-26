#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse

import torch
from ptflops import get_model_complexity_info
from mmdet.apis import init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet count flops')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args


def inp_fun(input_res):
    batch = torch.FloatTensor(1, 3, *input_res).cuda()
    return dict(img=[batch], img_meta=[[{'img_shape': (*input_res, 3),
                                         'ori_shape': (*input_res, 3),
                                         'scale_factor': 1.0}]],
                rescale=True, return_loss=False)


def main():
    args = parse_args()
    with torch.no_grad():
        model = init_detector(args.config)
        model.eval()
        input_res = model.cfg.data['test']['img_scale']
        flops, params = get_model_complexity_info(model, input_res,
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  input_constructor=inp_fun)
        print('Computational complexity: ' + flops)
        print('Number of parameters: ', params)


if __name__ == '__main__':
    main()
