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

from .default import DefaultArgParser
from ..registry import ARG_PARSERS


@ARG_PARSERS.register_module()
class CustomDetectorArgParser(DefaultArgParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_classes_to_parser(parser):
        parser.add_argument('--classes', required=True,
                            help='Comma-separated list of classes (e.g. "cat,dog,mouse").')
        return parser

    def get_train_parser(self, model_template_path):
        parser = super().get_train_parser(model_template_path)
        self._add_classes_to_parser(parser)
        return parser

    def get_test_parser(self, model_template_path):
        parser = super().get_test_parser(model_template_path)
        self._add_classes_to_parser(parser)
        return parser

    def get_compression_parser(self, model_template_path):
        parser = super().get_compression_parser(model_template_path)
        self._add_classes_to_parser(parser)
        return parser
