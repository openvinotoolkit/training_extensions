"""Utils for working with Configurable parameters."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os

import yaml


def override_parameters(overrides, parameters):
    """Overrides parameters values by overrides."""

    allowed_keys = {"default_value", "value"}
    for k, val in overrides.items():
        if isinstance(val, dict):
            if k in parameters.keys():
                override_parameters(val, parameters[k])
            else:
                raise ValueError(f'The "{k}" is not in original parameters.')
        else:
            if k in allowed_keys:
                parameters[k] = val
            else:
                raise ValueError(f'The "{k}" is not in allowed_keys: {allowed_keys}')


def configure_dataset(args, train=False):
    """Configure dataset args."""

    # Create instances of Task, ConfigurableParameters and Dataset.
    data_config = {"data": {}}
    if os.path.exists(args.data):
        with open(args.data, "r", encoding="UTF-8") as stream:
            data_config = yaml.safe_load(stream)
        stream.close()
        if train:
            args.save_model_to = "./models"
            args.save_logs_to = "./logs"

    # The command's args are overridden and use first
    if "train_ann_file" in args and args.train_ann_files:
        data_config["data"]["train"]["ann-files"] = args.train_ann_files
    if "train_data_roots" in args and args.train_data_roots:
        data_config["data"]["train"]["data-roots"] = args.train_data_roots
    if "val_ann_files" in args and args.val_ann_files:
        data_config["data"]["val"]["ann-files"] = args.val_ann_files
    if "val_data_roots" in args and args.val_data_roots:
        data_config["data"]["val"]["data-roots"] = args.val_data_roots
    if "unlabeled_file_list" in args and args.unlabeled_file_list:
        data_config["data"]["unlabeled"]["file-list"] = args.unlabeled_file_list
    if "unlabeled_data_roots" in args and args.unlabeled_data_roots:
        data_config["data"]["unlabeled"]["data-roots"] = args.unlabeled_data_roots
    if "test_ann_files" in args and args.test_ann_files:
        data_config["data"]["test"]["ann-files"] = args.test_ann_files
    if "test_data_roots" in args and args.test_data_roots:
        data_config["data"]["test"]["data-roots"] = args.test_data_roots
    return data_config
