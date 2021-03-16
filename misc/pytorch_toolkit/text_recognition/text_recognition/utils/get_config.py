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

import os
import yaml


def check_and_resolve_path(key, parameter):
    """Checks if given parameter could be path and tries to resolve it as relative path.
    If obtained path exists, returs it, else returns original input parameter.
    """
    if 'paths' in key:
        return [resolve_relative_path(p) for p in parameter]
    if 'path' in key:
        return resolve_relative_path(parameter)
    return parameter


def resolve_relative_path(path):
    """Resolves relative paths respectively to directory of this project (formula_recognition)

    Args:
        path (path-like or str): input path

    Returns:
        path-like or str: resolved path
    """
    if os.path.isabs(path):
        return path
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(root_dir, path)


def get_config(config_path, section):
    """Function reads config from config file and prepares config
    by merging common parts and specific section of the config (train, eval, demo, export)


    Args:
        config_path (str or path): path to a config file
        section (str): specific part of the config to merge with common part (e.g. 'train')
    """
    assert section in ['train', 'export', 'demo', 'eval']
    config_path = resolve_relative_path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        specific_config = config.get(section)
        common_config = config
        conflict_config_keys = set(specific_config.keys()) & set(common_config.keys())
        if conflict_config_keys:
            raise RuntimeError(
                f'Error: the following config parameters are set both in {section} config'
                f'and common config sections\n: {conflict_config_keys}')
        specific_config.update(common_config)
    specific_config = {k: check_and_resolve_path(k, v) for k, v in specific_config.items()}
    return specific_config
