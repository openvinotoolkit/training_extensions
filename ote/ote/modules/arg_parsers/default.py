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

from ote.api import (train_args_parser,
                     test_args_parser,
                     export_args_parser,
                     compression_args_parser)

from ..registry import ARG_PARSERS


@ARG_PARSERS.register_module()
class DefaultArgParser:

    def get_train_parser(self, config_path):
        return train_args_parser(config_path)

    def get_test_parser(self, config_path):
        return test_args_parser(config_path)

    def get_export_parser(self, config_path):
        return export_args_parser(config_path)

    def get_compression_parser(self, config_path):
        return compression_args_parser(config_path)
