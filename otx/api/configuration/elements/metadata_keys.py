"""This module contains the keys that can be used to retrieve parameter metadata."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import List

DEFAULT_VALUE = "default_value"
MIN_VALUE = "min_value"
MAX_VALUE = "max_value"
DESCRIPTION = "description"
HEADER = "header"
WARNING = "warning"
EDITABLE = "editable"
VISIBLE_IN_UI = "visible_in_ui"
AFFECTS_OUTCOME_OF = "affects_outcome_of"
UI_RULES = "ui_rules"
TYPE = "type"
OPTIONS = "options"
ENUM_NAME = "enum_name"
AUTO_HPO_STATE = "auto_hpo_state"
AUTO_HPO_VALUE = "auto_hpo_value"


def allows_model_template_override(keyword: str) -> bool:
    """Returns True if the metadata element described by `keyword` can be overridden in a model template file.

    Args:
        keyword (str): Name of the metadata key to check.

    Returns:
        bool: True if the metadata indicated by `keyword` can be overridden in a model template .yaml file, False
        otherwise.
    """
    overrideable_keys = [
        DEFAULT_VALUE,
        MIN_VALUE,
        MAX_VALUE,
        DESCRIPTION,
        HEADER,
        EDITABLE,
        WARNING,
        VISIBLE_IN_UI,
        OPTIONS,
        ENUM_NAME,
        UI_RULES,
        AFFECTS_OUTCOME_OF,
        AUTO_HPO_STATE,
    ]
    return keyword in overrideable_keys


def allows_dictionary_values(keyword: str) -> bool:
    """Returns True if the metadata element described by `keyword` allows having a dictionary as its value.

    Args:
        keyword (str): Name of the metadata key to check.

    Returns:
        bool: True if the metadata indicated by `keyword` allows having a dictionary as its value, False otherwise.
    """
    keys_allowing_dictionary_values = [OPTIONS, UI_RULES]
    return keyword in keys_allowing_dictionary_values


def all_keys() -> List[str]:
    """Returns a list of all metadata keys.

    Returns:
        List[str]: List of all available metadata keys
    """
    return [
        DEFAULT_VALUE,
        MIN_VALUE,
        MAX_VALUE,
        DESCRIPTION,
        HEADER,
        WARNING,
        EDITABLE,
        VISIBLE_IN_UI,
        AFFECTS_OUTCOME_OF,
        UI_RULES,
        TYPE,
        OPTIONS,
        ENUM_NAME,
        AUTO_HPO_STATE,
        AUTO_HPO_VALUE,
    ]
