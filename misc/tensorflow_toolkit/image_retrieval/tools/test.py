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
import logging as log

from image_retrieval.metrics import test_model


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model_weights', required=True, help='Path to model weights.')
    args.add_argument('--gallery', required=True, help='Gallery images list.')
    args.add_argument('--test_images', required=True, help='Test images list.')
    args.add_argument('--input_size', default=224, type=int, help='Input image size.')
    args.add_argument('--model', choices=['resnet50', 'mobilenet_v2'], default='mobilenet_v2')
    args.add_argument('--ie', choices=['tf', 'ie'], required=True)
    args.add_argument('--cpu_extension', help='Path to lib cpu extensions.')

    return args.parse_args()


def main():
    LOG_FORMAT = '%(levelno)s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s|%(message)s'
    log.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    log.getLogger().setLevel(log.WARN)

    args = parse_args()

    test_model(model_path=args.model_weights,
               model_backend=args.ie,
               model=args.model,
               gallery_path=args.gallery,
               test_images=args.test_images,
               input_size=args.input_size,
               cpu_extension=args.cpu_extension)


if __name__ == '__main__':
    main()
