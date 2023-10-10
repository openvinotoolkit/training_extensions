"""Utilities for OTX API."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import yaml


# Prevent ConstructorError: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
def construct_tuple(loader: yaml.SafeLoader, node: yaml.SequenceNode) -> tuple:
    """Construct a tuple from a YAML sequence node.

    Args:
    ----
        loader (yaml.SafeLoader): The YAML loader object.
        node (yaml.SequenceNode): The YAML sequence node.

    Returns:
    -------
        tuple: The tuple constructed from the sequence node.
    """
    return tuple(loader.construct_sequence(node))


def set_tuple_constructor() -> None:
    """Set the tuple constructor for the YAML loader."""
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_tuple)
