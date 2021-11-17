"""
Utils for parsing command line arguments.
"""

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
    """
    Generates help for hyper parameters section.
    """

    type_map = {"FLOAT": float, "INTEGER": int, "BOOLEAN": bool}

    help_keys = ("header", "type", "default_value", "max_value", "min_value")

    def _gen_param_help(prefix, d):
        cur_help = {}
        for k, val in d.items():
            if isinstance(val, dict) and "default_value" not in val.keys():
                if "visible_in_ui" in val and val["visible_in_ui"]:
                    x = _gen_param_help(prefix + f"{k}.", val)
                    cur_help.update(x)
            elif isinstance(val, dict) and "default_value" in val.keys():
                assert isinstance(val["default_value"], (int, float, str))
                help_str = "\n".join(
                    [f"{kk}: {val[kk]}" for kk in help_keys if kk in val.keys()]
                )
                assert "." not in k

                if val["type"] == "SELECTABLE":
                    continue

                cur_help.update(
                    {
                        prefix
                        + f"{k}": {
                            "default": val["default_value"],
                            "help": help_str,
                            "type": type_map[val["type"]],
                            "affects_outcome_of": val["affects_outcome_of"],
                        }
                    }
                )
        return cur_help

    return _gen_param_help("", hyper_parameters)


def gen_params_dict_from_args(args):
    """
    Generates hyper parameters dict from parsed command line arguments.
    """

    params_dict = {}
    for param_name in dir(args):
        if param_name.startswith("params."):
            cur_dict = params_dict
            split_param_name = param_name.split(".")[1:]
            for i, k in enumerate(split_param_name):
                if k not in cur_dict:
                    cur_dict[k] = {}
                if i < len(split_param_name) - 1:
                    cur_dict = cur_dict[k]
                else:
                    cur_dict[k] = {"value": getattr(args, param_name)}

    return params_dict


class ShortDefaultsHelpFormatter(argparse.RawTextHelpFormatter):
    """
    Text Help Formatter that shortens
    """

    def _get_default_metavar_for_optional(self, action):
        return action.dest.split(".")[-1].upper()


def add_hyper_parameters_sub_parser(parser, config, modes=None):
    """
    Adds hyper parameters sub parser.
    """

    default_modes = ("TRAINING", "INFERENCE")
    if modes is None:
        modes = default_modes
    assert isinstance(modes, tuple)
    for mode in modes:
        assert mode in default_modes

    def str2bool(val):
        if isinstance(val, bool):
            return val
        if val.lower() in ("true", "1"):
            return True
        if val.lower() in ("false", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    params = gen_param_help(config)

    subparsers = parser.add_subparsers(help="sub-command help")
    parser_a = subparsers.add_parser(
        "params",
        help="Hyper parameters defined in template file.",
        formatter_class=ShortDefaultsHelpFormatter,
    )
    for k, val in params.items():
        param_type = val["type"]
        if val["affects_outcome_of"] not in modes:
            continue
        if param_type == bool:
            param_type = str2bool
        parser_a.add_argument(
            f"--{k}",
            default=val["default"],
            help=val["help"],
            dest=f"params.{k}",
            type=param_type,
        )
