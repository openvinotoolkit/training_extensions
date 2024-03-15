"""Enums for configuration element types used to construct/interact with OTX configuration objects."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from __future__ import annotations

from enum import Enum, auto


class ElementCategory(Enum):
    """This Enum represents the categories of configuration elements that are known in OTX."""

    PRIMITIVES = auto()
    GROUPS = auto()
    RULES = auto()

    def __str__(self):
        """Retrieves the string representation of an instance of the Enum."""
        return self.name


class ConfigElementType(Enum):
    """This Enum represents the available elements to compose a configuration.

    Each instance holds a name, value and category representing a configuration element.
    """

    # Because this Enum takes both a value and a category, the auto() mechanism cannot be used to assign values. Hence,
    # they are assigned manually.
    INTEGER = 0, ElementCategory.PRIMITIVES
    FLOAT = 1, ElementCategory.PRIMITIVES
    BOOLEAN = 2, ElementCategory.PRIMITIVES
    FLOAT_SELECTABLE = 3, ElementCategory.PRIMITIVES
    SELECTABLE = 4, ElementCategory.PRIMITIVES
    PARAMETER_GROUP = 5, ElementCategory.GROUPS
    CONFIGURABLE_PARAMETERS = 6, ElementCategory.GROUPS
    RULE = 7, ElementCategory.RULES
    UI_RULES = 8, ElementCategory.RULES

    def __new__(cls, value: int, category: ElementCategory):  # pylint: disable=unused-argument
        """Creates a new instance of the Enum.

        The ConfigElementType Enum holds both a value and a category. In this method the `value` argument is parsed and
        assigned.
        """
        obj = object.__new__(cls)
        # Only the value is assigned here, since the _category_ attribute does not exists yet.
        obj._value_ = value
        return obj

    def __init__(self, value: int, category: ElementCategory):  # pylint: disable=unused-argument
        """Upon initialization, the Enum category is assigned."""
        # We cannot assign to _category_ in the __new__ method since it is not a valid attribute yet until the Enum is
        # initialized
        self._category_ = category

    @property
    def category(self) -> ElementCategory:
        """Returns the element category which the ConfigElementType belongs to.

        Categories are instances of the `otx.api.configuration.configuration_types.ElementCategory` Enum.
        """
        return self._category_

    def __str__(self):
        """Retrieves the string representation of an instance of the Enum."""
        return self.name
