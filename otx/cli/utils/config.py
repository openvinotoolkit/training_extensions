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


def configure_dataset(args):
    """Configure dataset args."""

    # Create instances of Task, ConfigurableParameters and Dataset.
    data_config = {"data-root": None, "train": None, "val": None, "test": None}
    data_config["unlabeled"] = {"file-list": None, "data-roots": None}
    if args.data is not None and os.path.exists(args.data):
        with open(args.data, "r", encoding="UTF-8") as stream:
            data_config = yaml.safe_load(stream)
        stream.close()

    # The command's args are overridden and use first
    if "data_root" in args and args.data_root:
        data_config["data-root"] = args.data_root
    if "train_ann_file" in args and args.train_ann_file:
        data_config["train"] = args.train_ann_file
    if "val_ann_file" in args and args.val_ann_file:
        data_config["val"] = args.val_ann_file
    if "test_ann_file" in args and args.test_ann_file:
        data_config["test"] = args.test_ann_file
    if "unlabeled_file_list" in args and args.unlabeled_file_list:
        data_config["unlabeled"]["file-list"] = args.unlabeled_file_list
    if "unlabeled_data_roots" in args and args.unlabeled_data_roots:
        data_config["unlabeled"]["data-roots"] = args.unlabeled_data_roots
    return data_config
