# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""This module define the scored label entity."""

import datetime
import math

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
        if math.isnan(probability) or (not 0 <= probability <= 1.0) :
            raise ValueError(f"Probability should be in range [0, 1], {probability} is given")
        self.label = label
        self.probability = probability

    @property
    def name(self) -> str:
        """
        Name of the label.
        """
        return self.label.name

    @property
    def id_(self) -> ID:
        """
        Returns the label id.
        """
        return self.label.id_

    @property
    def id(self) -> ID:
        """DEPRECATED"""
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
            f"ScoredLabel({self.id_}, name={self.name}, probability={self.probability}, "
            f"domain={self.domain}, color={self.color}, hotkey={self.hotkey})"
        )

    def __eq__(self, other):
        if isinstance(other, ScoredLabel):
            return (
                self.id_ == other.id_
                and self.name == other.name
                and self.color == other.color
                and self.hotkey == other.hotkey
                and self.probability == other.probability
                and self.domain == other.domain
            )
        return False

    def __hash__(self):
        return hash(str(self))
