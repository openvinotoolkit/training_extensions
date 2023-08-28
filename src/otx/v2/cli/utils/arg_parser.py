"""OTX CLI ArgParser."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
from typing import Any, List, Optional

import docstring_parser
import yaml
from jsonargparse import (
    ArgumentParser,
    Namespace,
    class_from_function,
)


# Custom Constructor for Python tuples
def tuple_constructor(loader, node):
    if isinstance(node, yaml.SequenceNode):
        # Load the elements as a list
        elements = loader.construct_sequence(node)
        # Return the tuple
        return tuple(elements)
    return None


def pre_parse_arguments():
    arguments = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:]
            value = None
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                value = sys.argv[i + 1]
                i += 1
            arguments[key] = value
        i += 1
    return arguments


def get_short_docstring(component: object) -> Optional[str]:
    """Gets the short description from the docstring.

    Args:
        component (object): The component to get the docstring from

    Returns:
        Optional[str]: The short description
    """
    if component.__doc__ is None:
        return None
    docstring = docstring_parser.parse(component.__doc__)
    return docstring.short_description


class OTXArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for OTX."""

    def __init__(
        self,
        *args: Any,
        description: str = "OpenVINO Training-Extension command line tool",
        env_prefix: str = "otx",
        default_env: bool = False,
        default_config_files: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            description=description,
            env_prefix=env_prefix,
            default_env=default_env,
            default_config_files=default_config_files,
            **kwargs,
        )

    def add_core_class_args(
        self,
        api_class,
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
        instantiate: bool = False,
        dataclass_mode: bool = False,
    ) -> List[str]:
        """Adds arguments from a class to a nested key of the parser.

        Args:
            api_class: A callable or any subclass.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.

        Returns:
            A list with the names of the class arguments added.
        """
        if callable(api_class) and not isinstance(api_class, type):
            api_class = class_from_function(api_class)

        if isinstance(api_class, type):
            if subclass_mode:
                return self.add_subclass_arguments(api_class, nested_key, fail_untyped=False, required=required)
            if dataclass_mode:
                return self.add_dataclass_arguments(
                    api_class,
                    nested_key,
                    fail_untyped=False,
                )
            return self.add_class_arguments(
                api_class,
                nested_key,
                fail_untyped=False,
                instantiate=instantiate,
                sub_configs=True,
            )
        raise NotImplementedError()

    def check_config(
        self,
        cfg: Namespace,
        skip_none: bool = True,
        skip_required: bool = True,
        branch: Optional[str] = None,
    ):
        # Skip This one for Flexible Configuration
        pass
