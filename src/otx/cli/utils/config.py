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

from pathlib import Path

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
        elif k in allowed_keys:
            parameters[k] = val
        else:
            raise ValueError(f'The "{k}" is not in allowed_keys: {allowed_keys}')


def configure_dataset(args, data_yaml_path=None):
    """Configure dataset args."""

    # Create instances of Task, ConfigurableParameters and Dataset.
    data_subset_format = {"ann-files": None, "data-roots": None}
    data_config = {"data": {subset: data_subset_format.copy() for subset in ("train", "val", "test")}}
    data_config["data"]["unlabeled"] = {"file-list": None, "data-roots": None}
    if data_yaml_path and Path(data_yaml_path).exists():
        with open(Path(data_yaml_path), "r", encoding="UTF-8") as stream:
            data_config = yaml.safe_load(stream)

    # The command's args are overridden and use first
    if "train_ann_files" in args and args.train_ann_files:
        data_config["data"]["train"]["ann-files"] = str(Path(args.train_ann_files).absolute())
    if "train_data_roots" in args and args.train_data_roots:
        data_config["data"]["train"]["data-roots"] = str(Path(args.train_data_roots).absolute())
    if "val_ann_files" in args and args.val_ann_files:
        data_config["data"]["val"]["ann-files"] = str(Path(args.val_ann_files).absolute())
    if "val_data_roots" in args and args.val_data_roots:
        data_config["data"]["val"]["data-roots"] = str(Path(args.val_data_roots).absolute())
    if "unlabeled_file_list" in args and args.unlabeled_file_list:
        data_config["data"]["unlabeled"]["file-list"] = str(Path(args.unlabeled_file_list).absolute())
    if "unlabeled_data_roots" in args and args.unlabeled_data_roots:
        data_config["data"]["unlabeled"]["data-roots"] = str(Path(args.unlabeled_data_roots).absolute())
    if "test_ann_files" in args and args.test_ann_files:
        data_config["data"]["test"]["ann-files"] = str(Path(args.test_ann_files).absolute())
    if "test_data_roots" in args and args.test_data_roots:
        data_config["data"]["test"]["data-roots"] = str(Path(args.test_data_roots).absolute())
    return data_config
