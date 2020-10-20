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

import yaml


def get_config(config_path, split):
    """Function reads config from config file and prepares config
    by merging common parts and specific split of the config (train, eval, demo, export)


    Args:
        config_path (str or path): path to a config file
        split (str): specific part of the config to merge with common part (e.g. 'train')
    """
    assert split in ['train', 'export', 'demo', 'eval']
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        specific_config = config.get(split)
        common_config = config.get("common")
        conflict_config_keys = set(specific_config.keys()) & set(common_config.keys())
        if conflict_config_keys:
            raise RuntimeError(
                "Error: the following config parameters are set both in demo config and common config sections: {}"
                .format(conflict_config_keys))
        specific_config.update(common_config)
    return specific_config
