# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
This module contains base elements that make up OTE ConfigurableParameters, as well as a collection of helper functions
to interact with them.

The configuration helper module can be imported as `ote_config_helper` and implements the following:

.. automodule:: ote_sdk.configuration.helper
   :members:

"""

# TODO: Remove cfg_helper once https://jira.devtools.intel.com/browse/CVS-67869 is done:
import ote_sdk.configuration.helper as ote_config_helper
import ote_sdk.configuration.helper as cfg_helper  # pylint: disable=reimported
from ote_sdk.configuration.elements import metadata_keys
from ote_sdk.configuration.elements.configurable_enum import ConfigurableEnum
from ote_sdk.configuration.enums.model_lifecycle import ModelLifecycle
from ote_sdk.configuration.ui_rules import Action, NullUIRules, Operator, Rule, UIRules

from .configurable_parameters import ConfigurableParameters
from .default_model_parameters import DefaultModelParameters

__all__ = [
    "metadata_keys",
    "cfg_helper",
    "ote_config_helper",
    "ConfigurableEnum",
    "ModelLifecycle",
    "Action",
    "NullUIRules",
    "Operator",
    "Rule",
    "UIRules",
    "DefaultModelParameters",
    "ConfigurableParameters",
]
