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
import subprocess
import tempfile


def reload_hyper_parameters(model_template):
    """ This function copies template.yaml file and its configuration.yaml dependency to temporal folder.
        Then it re-loads hyper parameters from copied template.yaml file.
        This function should not be used in general case, it is assumed that
        the 'configuration.yaml' should be in the same folder as 'template.yaml' file.
    """

    template_file = model_template.model_template_path
    template_dir = os.path.dirname(template_file)
    temp_folder = tempfile.mkdtemp()
    conf_yaml = [dep.source for dep in model_template.dependencies if dep.destination == model_template.hyper_parameters.base_path][0]
    conf_yaml = os.path.join(template_dir, conf_yaml)
    subprocess.run(f'cp {conf_yaml} {temp_folder}', check=True, shell=True)
    subprocess.run(f'cp {template_file} {temp_folder}', check=True, shell=True)
    model_template.hyper_parameters.load_parameters(os.path.join(temp_folder, 'template.yaml'))
    assert model_template.hyper_parameters.data

def override_parameters(overrides, parameters, allow_value=False):
    allowed_keys = {'default_value'}
    if allow_value:
        allowed_keys.add('value')
    for k, v in overrides.items():
        if isinstance(v, dict):
            if k in parameters.keys():
                override_parameters(v, parameters[k], allow_value)
            else:
                raise ValueError(f'The "{k}" is not in original parameters.')
        else:
            if k in allowed_keys:
                parameters[k] = v
            else:
                raise ValueError(f'The "{k}" is not in allowed_keys: {allowed_keys}')

def set_values_as_default(parameters):
    for k, v in parameters.items():
        if isinstance(v, dict) and 'value' not in v:
          set_values_as_default(v)
        elif isinstance(v, dict) and 'value' in v:
            if v['value'] != v['default_value']:
                v['value'] = v['default_value']
