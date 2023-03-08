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
import re
import sys
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Union

from otx.api.entities.model_template import ModelTemplate, parse_model_template
from otx.cli.registry import find_and_parse_model_template


class MemSizeAction(argparse.Action):
    """Parser add on to parse memory size string."""

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        expected_dest = "params.algo_backend.mem_cache_size"
        if dest != expected_dest:
            raise ValueError(f"dest should be {expected_dest}, but dest={dest}.")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse and set the attribute of namespace."""
        setattr(namespace, self.dest, self._parse_mem_size_str(values))

    @staticmethod
    def _parse_mem_size_str(mem_size: str) -> int:
        assert isinstance(mem_size, str)

        match = re.match(r"^([\d\.]+)\s*([a-zA-Z]{0,3})$", mem_size.strip())

        if match is None:
            raise ValueError(f"Cannot parse {mem_size} string.")

        units = {
            "": 1,
            "B": 1,
            "KB": 2**10,
            "MB": 2**20,
            "GB": 2**30,
            "KIB": 10**3,
            "MIB": 10**6,
            "GIB": 10**9,
            "K": 2**10,
            "M": 2**20,
            "G": 2**30,
        }

        number, unit = int(match.group(1)), match.group(2).upper()

        if unit not in units:
            raise ValueError(f"{mem_size} has disallowed unit ({unit}).")

        return number * units[unit]


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

    def _get_leaf_node(curr_dict: Dict[str, dict], curr_key: str):
        split_key = curr_key.split(".")
        node_key = split_key[0]

        if len(split_key) == 1:
            # It is leaf node
            return curr_dict, node_key

        # Dive deeper
        curr_key = ".".join(split_key[1:])
        if node_key not in curr_dict:
            curr_dict[node_key] = {}
        return _get_leaf_node(curr_dict[node_key], curr_key)

    _prefix = "params."
    params_dict: Dict[str, dict] = {}
    for param_name in dir(args):
        value = getattr(args, param_name)

        if not param_name.startswith(_prefix) or value is None:
            continue
        if override_param and param_name not in override_param:
            continue

        # param_name.removeprefix(_prefix)
        origin_key = param_name[len(_prefix) :]
        value_type = None
        if type_hint is not None:
            value_type = type_hint.get(origin_key, {}).get("type", None)

        leaf_node_dict, node_key = _get_leaf_node(params_dict, origin_key)
        leaf_node_dict[node_key] = {"value": value_type(value) if value_type else value}

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
