# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""This module define the scored label entity."""

import datetime
import math
from dataclasses import dataclass
from typing import Optional

from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity


@dataclass
class LabelSource:
    """This dataclass contains information about the source of a scored label.

    For annotations, the id of the user who created the label and for predictions, the
    id and model storage id of the model that created the prediction. When a user has
    accepted a predictions as is, both the user id of the user who accepted and the
    model/model storage id of the model that predicted should be filled in.
    """

    user_id: str = ""
    model_id: ID = ID()
    model_storage_id: ID = ID()


class ScoredLabel:
    """This represents a label along with a probability. This is used inside `Annotation` class.

    Args:
        label (LabelEntity): Label entity to which probability and source are attached.
        probability (float): a float denoting the probability of the shape belonging to the label.
        label_source (LabelSource): a LabelSource dataclass containing the id of the user who created
            or the model that predicted this label.
    """

    def __init__(
        self,
        label: LabelEntity,
        probability: float = 0.0,
        label_source: Optional[LabelSource] = None,
    ):
        if math.isnan(probability) or (not 0 <= probability <= 1.0):
            raise ValueError(f"Probability should be in range [0, 1], {probability} is given")

        self.label = label
        self.probability = probability
        self.label_source = label_source if label_source is not None else LabelSource()

    @property
    def name(self) -> str:
        """Name of the label."""
        return self.label.name

    @property
    def id_(self) -> ID:
        """Returns the label id."""
        return self.label.id_

    @property
    def id(self) -> ID:
        """DEPRECATED."""
        return self.label.id

    @property
    def color(self) -> Color:
        """Color of the label."""
        return self.label.color

    @property
    def hotkey(self) -> str:
        """Hotkey of the label."""
        return self.label.hotkey

    @property
    def domain(self) -> Domain:
        """Domain of the label."""
        return self.label.domain

    @property
    def is_empty(self) -> bool:
        """Check if the label is empty."""
        return self.label.is_empty

    @property
    def creation_date(self) -> datetime.datetime:
        """Creation data of the label."""
        return self.label.creation_date

    def get_label(self) -> LabelEntity:
        """Gets the label that the ScoredLabel object was initialized with."""
        return self.label

    def __repr__(self):
        """String representation of the label."""
        return (
            f"ScoredLabel({self.id_}, name={self.name}, probability={self.probability}, "
            f"domain={self.domain}, color={self.color}, hotkey={self.hotkey}, "
            f"label_source={self.label_source})"
        )

    def __eq__(self, other: object) -> bool:
        """Checks if the label is equal to the other label.

        Args:
            other (ScoredLabel): Label to compare with

        Returns:
            bool: True if the labels are equal, False otherwise
        """
        if isinstance(other, ScoredLabel):
            return (
                self.id_ == other.id_
                and self.name == other.name
                and self.color == other.color
                and self.hotkey == other.hotkey
                and self.probability == other.probability
                and self.domain == other.domain
                and self.label_source == other.label_source
            )
        return False

    def __hash__(self):
        """Returns hash of the label."""
        return hash(str(self))
