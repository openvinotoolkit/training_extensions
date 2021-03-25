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
import tempfile

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
    ote_args = arg_parser.get_export_parser(template_path).parse_args(extra_args)

    with tempfile.TemporaryDirectory() as work_dir:
        shutil.copytree(os.path.dirname(template_path), work_dir, dirs_exist_ok=True)
        copy_config_dependencies(template_config, template_path, work_dir)

        task_module = import_module('ote.tasks.' + template_config['modules']['task'])
        env_params, export_params = task_module.build_export_parameters(vars(ote_args), work_dir)

        task = task_module.Task(env_params)
        task.export(export_params)


if __name__ == '__main__':
    main()
