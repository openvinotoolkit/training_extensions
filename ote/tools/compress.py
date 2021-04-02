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
from ote.modules import (build_arg_parser,
                         build_arg_converter,
                         build_trainer,
                         build_compression_arg_transformer)
from ote.modules.compression import is_optimisation_enabled_in_template


def main():
    logging.basicConfig(level=logging.INFO)
    modules = load_config(MODULES_CONFIG_FILENAME)

    arg_parser = build_arg_parser(modules['arg_parser'])
    ote_args = vars(arg_parser.get_compression_parser(MODEL_TEMPLATE_FILENAME).parse_args())

    if 'compression' not in modules:
        raise RuntimeError(f'Cannot make compression for the template that'
                           f' does not have "compression" field in its modules'
                           f' file {MODULES_CONFIG_FILENAME}')
    if not is_optimisation_enabled_in_template(MODEL_TEMPLATE_FILENAME):
        raise RuntimeError('Cannot make compression for the template that'
                           ' does not enable any of compression flags')

    arg_converter = build_arg_converter(modules['arg_converter_map'])
    compress_args = arg_converter.convert_compress_args(ote_args)

    compression_arg_transformer = build_compression_arg_transformer(modules['compression'])
    compress_args, is_optimisation_enabled = \
            compression_arg_transformer.process_args(MODEL_TEMPLATE_FILENAME, compress_args)

    if not is_optimisation_enabled:
        logging.warning('Optimization flags are not set -- compression is not made')
        return

    # Note that compression in this tool will be made by the same trainer,
    # as in the tool train.py
    # The difference is only in the argparser and in the NNCFConfigTransformer used to
    # transform the configuration file.
    trainer = build_trainer(modules['trainer'])
    trainer(**compress_args)


if __name__ == '__main__':
    main()
