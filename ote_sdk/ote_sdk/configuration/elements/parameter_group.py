# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


"""
This module contains the definition of a ParameterGroup, which is the main class responsible for grouping configurable
parameters together.
"""

from typing import List, Type, TypeVar

import attr

from ote_sdk.configuration.enums.config_element_type import (
    ConfigElementType,
    ElementCategory,
)


@attr.s(auto_attribs=True, order=False, eq=False)
class ParameterGroup:
    """
    A group of configuration elements. Parameters living within the parameter group are typed attrs Attributes. The
    schema for each parameter is defined in its metadata, which can be retrieved using the `get_metadata` method from
    the parent ParameterGroup instance.

    :var header: User friendly name for the parameter group, that will be displayed in the UI
    :var description: User friendly string describing what the parameter group represents, that will be displayed in
        the UI.
    :var visible_in_ui: Boolean that controls whether or not this parameter group will be exposed through the REST API
        and shown in the UI. Set to False to hide this group. Defaults to True
    """

    header: str = attr.ib()
    description: str = attr.ib(default="Default parameter group description")
    visible_in_ui: bool = attr.ib(default=True)
    type: ConfigElementType = attr.ib(
        default=ConfigElementType.PARAMETER_GROUP,
        repr=False,
        init=False,
        on_setattr=attr.setters.frozen,
    )

    def __attrs_post_init__(self):
        """
        This method is called after the __init__ method to update the parameter and group fields of the ParameterGroup
        instance
        """
        groups: List[str] = []
        parameters: List[str] = []
        for attribute_or_method_name in dir(self):
            # Go over all attributes and methods of the class instance
            attribute_or_method = getattr(self, attribute_or_method_name)

            metadata = self.get_metadata(attribute_or_method_name)
            if metadata:
                # If the attribute or method has metadata, it might be a configurable parameter. In that case, check
                # its type to make sure it is one of the primitive parameter types
                _type = metadata.get("type", None)
                if _type is not None:
                    if _type.category == ElementCategory.PRIMITIVES:
                        # Add the parameter name to the `parameters` attribute
                        parameters.append(attribute_or_method_name)

            if isinstance(attribute_or_method, ParameterGroup):
                # Add the parameter group name to the `groups` attribute
                groups.append(attribute_or_method_name)
                # Also run the post_init for all groups in the group, in case of nested groups
                attribute_or_method.__attrs_post_init__()

        self.groups = groups  # pylint:disable=attribute-defined-outside-init
        self.parameters = parameters  # pylint:disable=attribute-defined-outside-init

    def get_metadata(self, parameter_name: str) -> dict:
        """
        Retrieve the metadata for a particular parameter from the group.
        :param parameter_name: name of the parameter for which to get the metadata
        :return: dictionary containing the metadata for the requested parameter. Returns an empty dict if no metadata
            was found for the parameter, or if the parameter was not found in the group.
        """
        parameter = getattr(attr.fields(type(self)), parameter_name, None)
        if parameter is not None:
            parameter_metadata = getattr(parameter, "metadata", {})
            return dict(parameter_metadata)
        return {}

    def __eq__(self, other):
        """
        Override default implementation of __eq__ to enable comparison of
        ParameterGroups generated dynamically via the config helper.
        """
        other_type = getattr(other, "type", None)
        if other_type == self.type:
            return self.__dict__ == other.__dict__
        return False


TParameterGroup = TypeVar("TParameterGroup", bound=ParameterGroup)


def add_parameter_group(group: Type[TParameterGroup]) -> TParameterGroup:
    """
    Wrapper to attr.ib to add nested parameter groups to a configuration.
    """
    return attr.ib(factory=group, type=group)
