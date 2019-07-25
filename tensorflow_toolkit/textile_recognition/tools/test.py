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
import logging as log
from textile.metrics import test_model

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model_weights', required=True, help='Path to model weights.')
    args.add_argument('--gallery', required=True, help='Gallery images folder.')
    args.add_argument('--test_data_path', required=True, help='Test images folder.')
    args.add_argument('--test_data_type', choices=['crops', 'cvat_annotation', 'videos'], required=True)
    args.add_argument('--test_annotation_path')
    args.add_argument('--input_size', default=224, type=int, help='Input image size.')
    args.add_argument('--model', choices=['resnet50', 'mobilenet_v2'], default='mobilenet_v2')
    args.add_argument('--imshow_delay', type=int, default=-1)
    args.add_argument('--ie', choices=['tf', 'ie'], required=True)

    return args.parse_args()

def main():
    LOG_FORMAT = '%(levelno)s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s|%(message)s'
    log.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    log.getLogger().setLevel(log.WARN)

    args = parse_args()

    test_model(model_path=args.model_weights,
               model_backend=args.ie,
               model=None,
               gallery_path=args.gallery,
               test_data_path=args.test_data_path,
               test_data_type=args.test_data_type,
               test_annotation_path=args.test_annotation_path,
               input_size=args.input_size,
               imshow_delay=args.imshow_delay)


if __name__ == '__main__':
    main()
