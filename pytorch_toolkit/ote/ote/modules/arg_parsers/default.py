"""
 Copyright (c) 2020 Intel Corporation

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
import yaml

from ..registry import ARG_PARSERS


@ARG_PARSERS.register_module()
class DefaultArgParser:
    def _get_base_parser_for_train_compression(self, model_template_path):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
        with open(model_template_path, 'r') as f:
            model_template = yaml.safe_load(f)

        parser.add_argument('--train-ann-files', required=True,
                            help='Comma-separated paths to training annotation files.')
        parser.add_argument('--train-data-roots', required=True,
                            help='Comma-separated paths to training data folders.')
        parser.add_argument('--val-ann-files', required=True,
                            help='Comma-separated paths to validation annotation files.')
        parser.add_argument('--val-data-roots', required=True,
                            help='Comma-separated paths to validation data folders.')
        parser.add_argument('--resume-from', default='',
                            help='Resume training from previously saved checkpoint')
        parser.add_argument('--load-weights', default='',
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save-checkpoints-to', default='/tmp/checkpoints',
                            help='Location where checkpoints will be stored')
        parser.add_argument('--batch-size', type=int,
                            default=model_template['hyper_parameters']['basic']['batch_size'],
                            help='Size of a single batch during training per GPU.')
        parser.add_argument('--gpu-num', type=int,
                            default=model_template['gpu_num'],
                            help='Number of GPUs that will be used in training, 0 is for CPU mode.')
        parser.add_argument('--tensorboard-dir',
                            help='Location where tensorboard logs will be stored.')
        parser.add_argument('--config', default=model_template['config'], help=argparse.SUPPRESS)

        return parser

    def get_train_parser(self, model_template_path):
        parser = self._get_base_parser_for_train_compression(model_template_path)

        with open(model_template_path, 'r') as f:
            model_template = yaml.safe_load(f)

        parser.add_argument('--epochs', type=int,
                            default=model_template['hyper_parameters']['basic']['epochs'],
                            help='Number of epochs during training')
        parser.add_argument('--base-learning-rate', type=float,
                            default=model_template['hyper_parameters']['basic']['base_learning_rate'],
                            help='Starting value of learning rate that might be changed during '
                                 'training according to learning rate schedule that is usually '
                                 'defined in detailed training configuration.')
        return parser

    def get_compression_parser(self, model_template_path):
        return self._get_base_parser_for_train_compression(model_template_path)

    def get_test_parser(self, model_template_path):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
        with open(model_template_path, 'r') as f:
            model_template = yaml.safe_load(f)

        parser.add_argument('--test-ann-files', required=True,
                            help='Comma-separated paths to test annotation files.')
        parser.add_argument('--test-data-roots', required=True,
                            help='Comma-separated paths to test data folders.')
        parser.add_argument('--load-weights', required=True,
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save-metrics-to', required=True,
                            help='Location where evaluated metrics values will be stored (yaml file).')
        parser.add_argument('--save-output-to', default='',
                            help='Location where output images (with displayed result of model work) will be stored.')
        parser.add_argument('--config', default=model_template['config'], help=argparse.SUPPRESS)

        return parser

    def get_export_parser(self, model_template_path):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
        with open(model_template_path, 'r') as f:
            model_template = yaml.safe_load(f)

        parser.add_argument('--load-weights', required=True,
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save-model-to', required='True',
                            help='Location where exported model will be stored.')
        parser.add_argument('--onnx', action='store_true', default=model_template['output_format']['onnx']['default'],
                            help='Enable onnx export.')
        parser.add_argument('--openvino', action='store_true',
                            default=model_template['output_format']['openvino']['default'],
                            help='Enable OpenVINO export.')
        parser.add_argument('--openvino-input-format',
                            default=model_template['output_format']['openvino']['input_format'],
                            help='Format of an input image for OpenVINO exported model.')
        parser.add_argument('--openvino-mo-args',
                            help='Additional args to OpenVINO Model Optimizer.')
        parser.add_argument('--config', default=model_template['config'], help=argparse.SUPPRESS)

        return parser
