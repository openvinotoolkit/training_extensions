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

from importlib import import_module
import logging
import os
import shutil

from ote.api import template_filename_parser
from ote.monitors.base_monitors import MetricsMonitor, PerformanceMonitor
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
    ote_args = arg_parser.get_train_parser(template_path).parse_args(extra_args)

    shutil.copytree(os.path.dirname(template_path), ote_args.work_dir, dirs_exist_ok=True)

    if not ote_args.do_not_load_snapshot:
        download_snapshot_if_not_yet(template_path, ote_args.work_dir)

    copy_config_dependencies(template_config, template_path, ote_args.work_dir)
    task_module = import_module('ote.tasks.' + template_config['modules']['task'])

    train_dataset = task_module.Dataset(ote_args.train_data_roots, ote_args.train_ann_files)
    val_dataset = task_module.Dataset(ote_args.val_data_roots, ote_args.val_ann_files)

    args_converter = build_arg_converter(template_config['modules']['arg_converter_map'])
    env_params, train_params = task_module.build_train_parameters(
                    args_converter.convert_train_args(vars(ote_args)), ote_args.work_dir)

    metrics_monitor = MetricsMonitor(env_params.work_dir)
    perf_monitor = PerformanceMonitor()
    task = task_module.Task(env_params, metrics_monitor)
    task.train(train_dataset, val_dataset, train_params, perf_monitor)


if __name__ == '__main__':
    main()
