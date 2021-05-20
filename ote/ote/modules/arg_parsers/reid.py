"""
 Copyright (c) 2020-2021 Intel Corporation

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

from ote.api import (train_args_parser,
                     test_args_parser,
                     export_args_parser,
                     compression_args_parser)

from .default import DefaultArgParser
from ..registry import ARG_PARSERS


@ARG_PARSERS.register_module()
class ReidArgParser(DefaultArgParser):
    def get_train_parser(self, config_path):
        parser = argparse.ArgumentParser(parents=[train_args_parser(config_path)], add_help=False)
        parser = self._add_aux_weights_load_arg(parser)
        parser = self._add_classes_arg(parser)
        return parser

    def get_test_parser(self, config_path):
        parser = argparse.ArgumentParser(parents=[test_args_parser(config_path)], add_help=False)
        parser = self._add_aux_weights_load_arg(parser)
        parser = self._add_classes_arg(parser)
        return parser

    def get_export_parser(self, config_path):
        parser = argparse.ArgumentParser(parents=[export_args_parser(config_path)], add_help=False)
        parser = self._add_aux_weights_load_arg(parser)
        parser = self._add_classes_arg(parser)
        return parser

    def get_compression_parser(self, config_path):
        # Please, note that for image classification compression
        # batch_size should be set from the compression config file,
        # not from the command line arguments
        # (smaller batch size often gives better results with "regularization by noise")
        parser = argparse.ArgumentParser(parents=[compression_args_parser(config_path, with_batch_size=False)],
                                         add_help=False)
        parser = self._add_aux_weights_load_arg(parser)
        parser = self._add_classes_arg(parser)
        return parser

    @staticmethod
    def _add_classes_arg(parser):
        parser.add_argument('--classes', required=False,
                            help='Comma-separated list of classes (e.g. "cat,dog,mouse").')
        return parser

    @staticmethod
    def _add_aux_weights_load_arg(parser):
        parser.add_argument('--load-aux-weights', required=False,
                            help='Load weights for an auxiliary model if mutual learning with auxiliary model is used')
        return parser
