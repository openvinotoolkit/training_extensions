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
from ote.utils.misc import download_snapshot_if_not_yet


from ote.modules import (build_arg_parser,
                         build_arg_converter)


def main():

    logging.basicConfig(level=logging.INFO)
    template_name_parser = template_filename_parser()
    args, extra_args = template_name_parser.parse_known_args()

    template_path = args.template
    template_config = load_config(args.template)

    arg_parser = build_arg_parser(template_config['modules']['arg_parser'])
    ote_args = arg_parser.get_train_parser(template_path).parse_args(extra_args)

    shutil.copytree(os.path.dirname(template_path), ote_args.work_dir, dirs_exist_ok=True)

    if not ote_args.do_not_load_snapshot:
        download_snapshot_if_not_yet(template_path, ote_args.work_dir)

    for dependency in template_config['dependencies']:
        source = dependency['source']
        destination = dependency['destination']
        if destination != 'snapshot.pth':
            rel_source = os.path.join(os.path.dirname(template_path), source)
            cur_dst = os.path.join(ote_args.work_dir, destination)
            os.makedirs(os.path.dirname(cur_dst), exist_ok=True)
            if os.path.isdir(rel_source):
                shutil.copytree(rel_source, cur_dst, dirs_exist_ok=True)
            else:
                shutil.copy(rel_source, destination)

    module_path = os.path.abspath(os.path.join(ote_args.work_dir, 'packages/ote/ote/tasks'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    task_module = import_module(template_config['modules']['task'])

    train_dataset = task_module.Dataset(ote_args.train_data_roots, ote_args.train_ann_files)
    val_dataset = task_module.Dataset(ote_args.val_data_roots, ote_args.val_ann_files)

    args_converter = build_arg_converter(template_config['modules']['arg_converter_map'])
    env_params, train_params = task_module.build_train_parameters(
                    args_converter.convert_train_args(vars(ote_args)), ote_args.work_dir)

    task = task_module.Task(env_params)
    task.train(train_dataset, val_dataset, train_params)


if __name__ == '__main__':
    main()
