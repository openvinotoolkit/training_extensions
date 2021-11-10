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

"""This module define the scored label entity."""

import datetime

from ote_sdk.entities.color import Color
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Domain, LabelEntity


class ScoredLabel:
    """
    This represents a label along with a probability. This is used inside :class:`Annotation`.

    :param label: a label. See :class:`Label`
    :param probability: a float denoting the probability of the shape belonging to the label.
    """

    def __init__(self, label: LabelEntity, probability: float = 0.0):
        self.label = label
        self.probability = probability

    @property
    def name(self) -> str:
        """
        Name of the label.
        """
        return self.label.name

    @property
    def id(self) -> ID:
        """
        Returns the label id.
        """
        return self.label.id

    @property
    def color(self) -> Color:
        """
        Color of the label.
        """
        return self.label.color

    @property
    def hotkey(self) -> str:
        """
        Hotkey of the label.
        """
        return self.label.hotkey

    @property
    def domain(self) -> Domain:
        """
        Domain of the label.
        """
        return self.label.domain

    @property
    def is_empty(self) -> bool:
        """
        Check if the label is empty
        """
        return self.label.is_empty

    @property
    def creation_date(self) -> datetime.datetime:
        """
        Creation data of the label
        """
        return self.label.creation_date

    def get_label(self) -> LabelEntity:
        """
        Gets the label that the ScoredLabel object was initialized with.
        """
        return self.label

    def __repr__(self):
        return (
            f"ScoredLabel({self.id}, name={self.name}, probability={self.probability}, "
            f"domain={self.domain}, color={self.color}, hotkey={self.hotkey})"
        )

    def __eq__(self, other):
        if isinstance(other, ScoredLabel):
            return (
                self.id == other.id
                and self.name == other.name
                and self.color == other.color
                and self.hotkey == other.hotkey
                and self.probability == other.probability
                and self.domain == other.domain
            )
        return False

    def __hash__(self):
        return hash(str(self))
