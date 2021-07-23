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

import argparse


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


class ShortDefaultsHelpFormatter(argparse.HelpFormatter):

    def _get_default_metavar_for_optional(self, action):
        return action.dest.split('.')[-1].upper()


def add_hyper_parameters_sub_parser(parser, config):
    params = gen_param_help(config['hyper_parameters'])

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('params',
                                     help=f'Hyper parameters defined in template file.',
                                     formatter_class=ShortDefaultsHelpFormatter)
    for k, v in params.items():
        parser_a.add_argument(f'--{k}', default=v['default'], help=v['help'], dest=f'params.{k}')
