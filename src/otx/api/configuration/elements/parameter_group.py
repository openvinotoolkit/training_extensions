"""ParameterGroup is the main class responsible for grouping configurable parameters together."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from enum import Enum
from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from otx.api.configuration.elements import metadata_keys
from otx.api.configuration.enums import AutoHPOState
from otx.api.configuration.enums.config_element_type import (
    ConfigElementType,
    ElementCategory,
)


@attr.s(auto_attribs=True, order=False, eq=False)
class ParameterGroup:
    """A group of configuration elements.

    Parameters living within the parameter group are typed attrs Attributes. The schema for each parameter is defined
    in its metadata, which can be retrieved using the `get_metadata` method from the parent ParameterGroup instance.

    Attributes:
        header (str): User friendly name for the parameter group, that will be
            displayed in the UI
        description (str): User friendly string describing what the parameter
            group represents, that will be displayed in the UI.
        visible_in_ui (bool): Boolean that controls whether or not this
            parameter group will be exposed through the REST API and
            shown in the UI. Set to False to hide this group. Defaults
            to True
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

    def __attrs_post_init__(self) -> None:
        """Update parameter and group after __init__.

        This method is called after the __init__ method to update the parameter and group fields of the ParameterGroup
        instance.
        """
        groups: List[str] = []
        parameters: List[str] = []
        self.__metadata_overrides: Dict[str, Any] = {}  # pylint:disable=attribute-defined-outside-init

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
        """Retrieve the metadata for a particular parameter from the group.

        Args:
            parameter_name (str): name of the parameter for which to get the
                metadata

        Returns:
            dict: dictionary containing the metadata for the requested
                parameter. Returns an empty dict if no metadata was found
                for the parameter, or if the parameter was not found in the
                group.
        """
        parameter = getattr(attr.fields(type(self)), parameter_name, None)
        if parameter is not None:
            parameter_metadata = getattr(parameter, "metadata", {})
            metadata_dict = dict(parameter_metadata)
            parameter_overrides = self.__metadata_overrides.get(parameter_name, None)
            if parameter_overrides is not None:
                for metadata_key, value_override in parameter_overrides.items():
                    metadata_dict.update({metadata_key: value_override})
            return metadata_dict
        return {}

    def set_metadata_value(
        self,
        parameter_name: str,
        metadata_key: str,
        value: Union[int, float, str, bool, Enum],
    ) -> bool:
        """Sets the value of a specific metadata item `metadata_key` for the parameter named `parameter_name`.

        Args:
            parameter_name (str): name of the parameter for which to get the
                metadata item
            metadata_key (str): name of the metadata value to set
            value (Union[int, float, str, bool, Enum]): New value to assign to the metadata item accessed by
                `metadata_key`. The type of `value` has to exactly match
                the type of the current value of the metadata item

        Returns:
            True if the metadata item was successfully updated, False
            otherwise
        """
        parameter = getattr(attr.fields(type(self)), parameter_name, None)
        if parameter is None:
            return False
        parameter_metadata = dict(getattr(parameter, "metadata", {}))
        if metadata_key not in metadata_keys.all_keys():
            return False
        metadata_value = parameter_metadata[metadata_key]
        if metadata_value is not None and type(metadata_value) is not type(
            value
        ):  # pylint: disable=unidiomatic-typecheck
            return False
        existing_overrides = self.__metadata_overrides.get(parameter_name, None)
        if existing_overrides is None:
            self.__metadata_overrides[parameter_name] = {metadata_key: value}
        else:
            existing_overrides.update({metadata_key: value})
        return True

    def update_auto_hpo_states(self):
        """Update hpo state based on teh value of parameters.

        Updates the `auto_hpo_state` metadata field for all parameters in the parameter
        group, based on the values of the parameters and the values of their
        `auto_hpo_value` metadata fields.
        """
        for parameter_name in self.parameters:
            metadata = self.get_metadata(parameter_name)
            if metadata[metadata_keys.AUTO_HPO_STATE] == AutoHPOState.NOT_POSSIBLE:
                continue
            auto_hpo_value = metadata[metadata_keys.AUTO_HPO_VALUE]
            if auto_hpo_value is None:
                continue
            if auto_hpo_value != getattr(self, parameter_name):
                auto_hpo_state = AutoHPOState.OVERRIDDEN
            else:
                auto_hpo_state = AutoHPOState.OPTIMIZED
            self.set_metadata_value(
                parameter_name=parameter_name,
                metadata_key=metadata_keys.AUTO_HPO_STATE,
                value=auto_hpo_state,
            )
        for group_name in self.groups:
            group = getattr(self, group_name)
            group.update_auto_hpo_states()

    def __eq__(self, other):
        """Comparison with support for dynamically generated ParameterGroups.

        Override default implementation of __eq__ to enable comparison of
        ParameterGroups generated dynamically via the config helper.
        """
        other_type = getattr(other, "type", None)
        if other_type == self.type:
            return self.__dict__ == other.__dict__
        return False


_ParameterGroup = TypeVar("_ParameterGroup", bound=ParameterGroup)


def add_parameter_group(group: Type[_ParameterGroup]) -> _ParameterGroup:
    """Wrapper to attr.ib to add nested parameter groups to a configuration."""
    return attr.ib(factory=group, type=group)
