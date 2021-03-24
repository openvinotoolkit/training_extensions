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
from importlib import import_module
import logging
import os
import shutil
import sys

from ote.api import template_filename_parser
from ote.utils import load_config
from ote.utils import download_snapshot_if_not_yet, copy_config_dependencies


from ote.modules import (build_arg_parser,
                         build_arg_converter)


def main():

    logging.basicConfig(level=logging.INFO)
    template_name_parser = template_filename_parser()
    args, extra_args = template_name_parser.parse_known_args()

    template_path = args.template
    template_config = load_config(args.template)

    arg_parser = build_arg_parser(template_config['modules']['arg_parser'])
    ote_args = arg_parser.get_test_parser(template_path).parse_args(extra_args)

    shutil.copytree(os.path.dirname(template_path), ote_args.work_dir, dirs_exist_ok=True)

    copy_config_dependencies(template_config, template_path, ote_args.work_dir)
    task_module = import_module('ote.tasks.' + template_config['modules']['task'])

    test_dataset = task_module.Dataset(ote_args.test_data_roots, ote_args.test_ann_files)

    args_converter = build_arg_converter(template_config['modules']['arg_converter_map'])
    env_params, test_params = task_module.build_test_parameters(
                    args_converter.convert_test_args(vars(ote_args)), ote_args.work_dir)

    task = task_module.Task(env_params)
    _, result_metrics = task.test(test_dataset, test_params)
    for name in result_metrics:
        print(f'{name} : {result_metrics[name]}')


if __name__ == '__main__':
    main()
