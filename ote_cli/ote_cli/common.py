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


import importlib

import yaml
from sc_sdk.utils.project_factory import ProjectFactory

MODEL_TEMPLATE_FILENAME = 'template.yaml'

def load_config(path):
    with open(path) as read_file:
        return yaml.safe_load(read_file)

def load_model_weights(path):
    with open(path, 'rb') as read_file:
        return read_file.read()

def create_project(classes):
    project_name = 'My project name'
    task_name = 'My task name'

    project = ProjectFactory().create_project_single_task(name=project_name, description="",
        label_names=classes, task_name=task_name)
    return project


def get_task_impl_class(config):
    task_impl_module_name, task_impl_class_name  = config['task']['impl'].rsplit('.', 1)
    task_impl_module = importlib.import_module(task_impl_module_name)
    task_impl_class = getattr(task_impl_module, task_impl_class_name)

    return task_impl_class


def gen_param_help(hyper_parameters):
    def _gen_param_help(prefix, d):
        xx = {}
        for k, v in d.items():
            if isinstance(v, dict) and 'value' not in v.keys():
                x = _gen_param_help(prefix + f'{k}.', v)
                xx.update(x)
            elif isinstance(v, dict) and 'value' in v.keys():
                assert isinstance(v['value'], (int, float, str))
                help_str = ', '.join([f'{kk} = {vv}' for kk,vv in v.items() if kk != 'value'])
                xx.update({prefix + f'{k}': {'default': v['value'], 'help': help_str}})
            else:
                xx.update({prefix + f'{k}': {'default': v, 'help': ''}})
        return xx
    return _gen_param_help('', hyper_parameters['params'])

def gen_params_dict_from_args(args):
    params_dict = {}
    for param_name in dir(args):
        if param_name.startswith('params.'):
            d = params_dict
            split_param_name = param_name.split('.')[1:]
            for i, k in enumerate(split_param_name):
                if k not in d:
                    d[k] = {}
                if i < len(split_param_name) - 1:
                    d = d[k]
                else:
                    d[k] = getattr(args, param_name)

    if params_dict:
        return {'params': params_dict}

    return None


def add_hyper_parameters_sub_parser(parser, config):
    params = gen_param_help(config['hyper_parameters'])

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('params', help=f'Hyper parameters defined in {MODEL_TEMPLATE_FILENAME}.')
    for k, v in params.items():
        parser_a.add_argument(f'--{k}', default=v['default'], help=v['help'], dest=f'params.{k}')
