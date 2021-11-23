"""This module defines classes representing metadata information."""
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
import abc
from enum import Enum, auto
from typing import Optional

from ote_sdk.entities.model import ModelEntity


class IMetadata(metaclass=abc.ABCMeta):
    """
    This interface represents any additional metadata information which can be connected to an IMedia
    """

    __name = Optional[str]

    @property
    def name(self):
        """Gets or sets the name of the Metadata item"""
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value


class FloatType(Enum):
    """
    Represents the use of the FloatMetadata
    """

    FLOAT = auto()  # Regular float, without particular context
    EMBEDDING_VALUE = auto()
    ACTIVE_SCORE = auto()

    def __str__(self):
        return str(self.name)


class FloatMetadata(IMetadata):
    """
    This class represents metadata of type float.
    """

    def __init__(
        self, name: str, value: float, float_type: FloatType = FloatType.FLOAT
    ):
        self.name = name
        self.value = value
        self.float_type = float_type

    def __repr__(self):
        return f"FloatMetadata({self.name}, {self.value}, {self.float_type})"

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.value == other.value
            and self.float_type == other.float_type
        )


class MetadataItemEntity:
    """
    This class is a wrapper class which connects the metadata value to model,
    which was used to generate it.
    """

    def __init__(
        self,
        data: IMetadata,
        model: Optional[ModelEntity] = None,
    ):
        self.data = data
        self.model = model

    def __repr__(self):
        return f"MetadataItemEntity(model={self.model}, data={self.data})"

    def __eq__(self, other):
        return self.model == other.model and self.data == other.data
