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
    type_map = {
        'FLOAT': float,
        'INTEGER': int,
        'BOOLEAN': bool
    }

    help_keys = ('header', 'type', 'default_value', 'max_value', 'min_value')
    def _gen_param_help(prefix, d):
        xx = {}
        for k, v in d.items():
            if isinstance(v, dict) and 'value' not in v.keys():
                if 'visible_in_ui' in v and v['visible_in_ui']:
                    x = _gen_param_help(prefix + f'{k}.', v)
                    xx.update(x)
            elif isinstance(v, dict) and 'value' in v.keys():
                assert isinstance(v['value'], (int, float, str))
                help_str = '\n'.join([f'{kk}: {v[kk]}' for kk in help_keys if kk in v.keys()])
                assert '.' not in k
                xx.update({prefix + f'{k}': {'default': v['default_value'],
                                             'help': help_str,
                                             'type': type_map[v['type']],
                                             'affects_outcome_of': v['affects_outcome_of']}})
        return xx
    return _gen_param_help('', hyper_parameters)


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
                    d[k] = {'value': getattr(args, param_name)}

    return params_dict


class ShortDefaultsHelpFormatter(argparse.RawTextHelpFormatter):

    def _get_default_metavar_for_optional(self, action):
        return action.dest.split('.')[-1].upper()


def add_hyper_parameters_sub_parser(parser, config, modes=None):
    default_modes = ('TRAINING', 'INFERENCE')
    if modes is None:
        modes = default_modes
    assert isinstance(modes, tuple)
    for mode in modes:
        assert mode in default_modes
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1'):
            return True
        if v.lower() in ('false', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    params = gen_param_help(config)

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('params',
                                     help=f'Hyper parameters defined in template file.',
                                     formatter_class=ShortDefaultsHelpFormatter)
    for k, v in params.items():
        param_type = v['type']
        if v['affects_outcome_of'] not in modes:
            continue
        if param_type == bool:
            param_type = str2bool
        parser_a.add_argument(f'--{k}', default=v['default'], help=v['help'], dest=f'params.{k}', type=param_type)
