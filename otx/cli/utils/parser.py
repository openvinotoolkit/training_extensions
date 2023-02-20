"""Utils for parsing command line arguments."""

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
import sys
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Union

from otx.api.entities.model_template import ModelTemplate, parse_model_template
from otx.cli.registry import find_and_parse_model_template


def gen_param_help(hyper_parameters: Dict) -> Dict:
    """Generates help for hyper parameters section."""

    type_map = {"FLOAT": float, "INTEGER": int, "BOOLEAN": bool, "SELECTABLE": str}

    help_keys = ("header", "type", "default_value", "max_value", "min_value")

    def _gen_param_help(prefix: str, cur_params: Dict) -> Dict:
        cur_help = {}
        for k, val in cur_params.items():
            if not isinstance(val, dict):
                continue

            if "default_value" not in val.keys():
                if "visible_in_ui" in val and val["visible_in_ui"]:
                    x = _gen_param_help(prefix + f"{k}.", val)
                    cur_help.update(x)
            else:
                assert isinstance(val["default_value"], (int, float, str))
                help_str = "\n".join([f"{kk}: {val[kk]}" for kk in help_keys if kk in val.keys()])
                assert "." not in k

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


def gen_params_dict_from_args(
    args, override_param: Optional[List] = None, type_hint: Optional[dict] = None
) -> Dict[str, dict]:
    """Generates hyper parameters dict from parsed command line arguments."""

    params_dict: Dict[str, dict] = {}
    for param_name in dir(args):
        if not param_name.startswith("params."):
            continue
        if override_param and param_name not in override_param:
            continue

        value_type = None
        cur_dict = params_dict
        split_param_name = param_name.split(".")[1:]
        if type_hint:
            origin_key = ".".join(split_param_name)
            value_type = type_hint[origin_key].get("type", None)
        for i, k in enumerate(split_param_name):
            if k not in cur_dict:
                cur_dict[k] = {}
            if i < len(split_param_name) - 1:
                cur_dict = cur_dict[k]
            else:
                value = getattr(args, param_name)
                cur_dict[k] = {"value": value_type(value) if value_type else value}

    return params_dict


def str2bool(val: Union[str, bool]) -> bool:
    """If input type is string, convert it to boolean.

    Args:
        val (Union[str, bool]): value to convert to boolean.

    Raises:
        argparse.ArgumentTypeError: If type is neither string and boolean, raise an error.

    Returns:
        bool: return converted boolean value.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.lower() in ("true", "1"):
            return True
        if val.lower() in ("false", "0"):
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class ShortDefaultsHelpFormatter(argparse.RawTextHelpFormatter):
    """Text Help Formatter that shortens."""

    def _get_default_metavar_for_optional(self, action):
        return action.dest.split(".")[-1].upper()


def add_hyper_parameters_sub_parser(
    parser, config, modes=None, return_sub_parser=False
) -> Optional[argparse.ArgumentParser]:
    """Adds hyper parameters sub parser."""

    default_modes = ("TRAINING", "INFERENCE")
    if modes is None:
        modes = default_modes
    assert isinstance(modes, tuple)
    for mode in modes:
        assert mode in default_modes

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
    if return_sub_parser:
        return parser_a
    return None


def get_parser_and_hprams_data():
    """A function to distinguish between when there is template input and when there is no template input.

    Inspect the template using pre_parser to get the template's hyper_parameters information.
    Finally, it returns the parser used in the actual main.
    """
    # TODO: Declaring pre_parser to get the template
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("template", nargs="?", default=None)
    parsed, _ = pre_parser.parse_known_args()
    params = []
    if "params" in sys.argv:
        params = sys.argv[sys.argv.index("params") :]

    template = parsed.template
    hyper_parameters = {}
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    template_config = find_and_parse_model_template(template)
    template_help_str = (
        "Enter the path or ID or name of the template file. \n"
        "This can be omitted if you have train-data-roots or run inside a workspace."
    )

    if isinstance(template_config, ModelTemplate):
        sys.argv[sys.argv.index(template)] = template_config.model_template_path
        hyper_parameters = template_config.hyper_parameters.data
        parser.add_argument("template", help=template_help_str)
    elif Path("./template.yaml").exists():
        # Workspace Environments
        template_config = parse_model_template("./template.yaml")
        hyper_parameters = template_config.hyper_parameters.data
        parser.add_argument("template", nargs="?", default="./template.yaml", help=template_help_str)
    else:
        parser.add_argument("template", nargs="?", default=None, help=template_help_str)

    return parser, hyper_parameters, params
