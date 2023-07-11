"""This module contains the ConfigurableEnum, that is used to define Enums which can be configured by the user."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum
from typing import List

from .metadata_keys import ENUM_NAME, OPTIONS


class ConfigurableEnum(Enum):
    """This class is used as the basis for defining `selectable` configurable parameters in the OTX API.

    Enums reflecting `selectable` options should inherit from thisclass.
    """

    def __str__(self):
        """Retrieves the string representation of an instance of the ConfigurableEnum (or subclasses thereof)."""
        return self.value

    def __eq__(self, other) -> bool:
        """Returns True if the ConfigurableEnum instance is equal to the other ConfigurableEnum instance.

        Checks whether one ConfigurableEnum instance (or instance of a subclass thereof) is equal to the `other`
        object. Comparison is made based on class name, instance value and instance name.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        if isinstance(other, ConfigurableEnum) and self.__class__.__name__ == other.__class__.__name__:
            return self.value == other.value and self.name == other.name
        return False

    def __hash__(self):
        """Computes hash for the ConfigurableEnum instance."""
        return hash(self.name)

    @classmethod
    def get_class_info(cls) -> dict:
        """Creates a dictionary representation of the ConfigurableEnum.

        This includes the name of the enum and the (name, value) pairs representing its members.

        Returns:
            dict: Dictionary representation of the ConfigurableEnum.
        """
        options_dict = {name: instance.value for name, instance in cls.__members__.items()}

        return {ENUM_NAME: cls.__name__, OPTIONS: options_dict}

    @classmethod
    def get_names(cls) -> List[str]:
        """Returns a list of names that can be used to index the Enum."""
        return [x.name for x in cls]

    @classmethod
    def get_values(cls) -> List[str]:
        """Returns a list of values that can be used to index the Enum."""
        return [x.value for x in cls]
