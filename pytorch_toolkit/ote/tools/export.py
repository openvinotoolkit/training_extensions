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

import logging

from ote import MODEL_TEMPLATE_FILENAME, MODULES_CONFIG_FILENAME
from ote.utils import load_config
from ote.modules import build_arg_parser, build_exporter, build_compression_arg_transformer
from ote.modules.compression import is_optimisation_enabled_in_template


def main():
    logging.basicConfig(level=logging.INFO)
    modules = load_config(MODULES_CONFIG_FILENAME)

    arg_parser = build_arg_parser(modules['arg_parser'])
    ote_args = vars(arg_parser.get_export_parser(MODEL_TEMPLATE_FILENAME).parse_args())

    if modules.get('compression') and is_optimisation_enabled_in_template(MODEL_TEMPLATE_FILENAME):
        compression_arg_transformer = build_compression_arg_transformer(modules['compression'])
        ote_args, _ = compression_arg_transformer.process_args(MODEL_TEMPLATE_FILENAME, ote_args)

    exporter = build_exporter(modules['exporter'])
    exporter(ote_args)


if __name__ == '__main__':
    main()
