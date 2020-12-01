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
from ote.modules import build_arg_parser, build_arg_converter, build_trainer, build_compression_arg_transformer
from ote.modules.compression import is_compression_enabled_in_template


def main():
    logging.basicConfig(level=logging.INFO)
    modules = load_config(MODULES_CONFIG_FILENAME)

    arg_parser = build_arg_parser(modules['arg_parser'])
    ote_args = vars(arg_parser.get_train_parser(MODEL_TEMPLATE_FILENAME).parse_args())

    arg_converter = build_arg_converter(modules['arg_converter'])
    train_args = arg_converter.convert_train_args(MODEL_TEMPLATE_FILENAME, ote_args)

    # Note that compression args transformer is not applied here,
    # since NNCF compression (if it is enabled) will be applied
    # later, when the training is finished.

    trainer = build_trainer(modules['trainer'])
    trainer(**train_args)

    if modules.get('compression') and is_compression_enabled_in_template(MODEL_TEMPLATE_FILENAME):
        # TODO: think on the case if compression is enabled in template.yaml, but modules does not contain 'compression'

        latest_snapshot = trainer.get_latest_snapshot()
        if not latest_snapshot:
            raise RuntimeError('Cannot find latest snapshot to make compression after training')

        compress_args = arg_converter.convert_train_args_to_compress_args(MODEL_TEMPLATE_FILENAME, ote_args)
        arg_converter.update_converted_args_to_load_from_snapshot(compress_args, latest_snapshot)

        compression_arg_transformer = build_compression_arg_transformer(modules['compression'])
        compress_args = compression_arg_transformer.process_args(MODEL_TEMPLATE_FILENAME, compress_args)

        compress_trainer = build_trainer(modules['trainer'])
        compress_trainer(**compress_args)


if __name__ == '__main__':
    main()
