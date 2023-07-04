"""This module contains the base class to define ConfigurableParameters within OTX."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from attr import attrib, attrs, setters

from otx.api.entities.id import ID

from .elements.parameter_group import ParameterGroup
from .elements.utils import convert_string_to_id
from .enums.config_element_type import ConfigElementType


@attrs(auto_attribs=True, order=False, eq=False)
class ConfigurableParameters(ParameterGroup):
    """Base class representing a generic set of configurable parameters.

    A ConfigurableParameters instance is essentially a parameter group with an id
    attached to it, so that it can be uniquely identified in the repositories.

    Attributes:
        id (ID): ID that uniquely identifies the ConfigurableParameters
        type (ConfigElementType): Type of the ConfigurableParameters
    """

    id: ID = attrib(default=ID(), kw_only=True, converter=convert_string_to_id)
    type: ConfigElementType = attrib(
        default=ConfigElementType.CONFIGURABLE_PARAMETERS,
        repr=False,
        init=False,
        on_setattr=setters.frozen,
    )
