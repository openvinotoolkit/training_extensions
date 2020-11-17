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

from ote import MODEL_TEMPLATE_FILENAME, MODULES_CONFIG_FILENAME
from ote.utils import load_config
from ote.modules import build_arg_parser, build_arg_converter, build_evaluator


def main():
    template = load_config(MODEL_TEMPLATE_FILENAME)
    modules = load_config(MODULES_CONFIG_FILENAME)

    arg_parser = build_arg_parser(modules['arg_parser'])
    ote_args = vars(arg_parser.get_test_parser(template).parse_args())

    arg_converter = build_arg_converter(modules['arg_converter'])
    eval_args = arg_converter.convert_test_args(MODEL_TEMPLATE_FILENAME, ote_args)

    evaluator = build_evaluator(modules['evaluator'])
    evaluator(**eval_args)


if __name__ == '__main__':
    main()
