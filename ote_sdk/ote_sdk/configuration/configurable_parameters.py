"""
This module contains the base class to define ConfigurableParameters within OTE
"""
# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.
#
from attr import attrib, attrs, setters

from ote_sdk.entities.id import ID

from .elements.parameter_group import ParameterGroup
from .elements.utils import convert_string_to_id
from .enums.config_element_type import ConfigElementType


@attrs(auto_attribs=True, order=False, eq=False)
class ConfigurableParameters(ParameterGroup):
    """
    Base class representing a generic set of configurable parameters. A
    ConfigurableParameters instance is essentially a parameter group with an id
    attached to it, so that it can be uniquely identified in the repositories.

    :var id: ID that uniquely identifies the ConfigurableParameters
    :var header: User friendly name for the ConfigurableParameters, that will be
        displayed in the UI

    """

    id: ID = attrib(default=ID(), kw_only=True, converter=convert_string_to_id)
    type: ConfigElementType = attrib(
        default=ConfigElementType.CONFIGURABLE_PARAMETERS,
        repr=False,
        init=False,
        on_setattr=setters.frozen,
    )
