"""This module define the label entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import datetime
import os
from enum import Enum, auto
from typing import Optional

from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.utils.time_utils import now


class Domain(Enum):
    """Describes an algorithm domain like classification, detection, etc."""

    NULL = auto()
    CLASSIFICATION = auto()
    DETECTION = auto()
    SEGMENTATION = auto()
    ANOMALY_CLASSIFICATION = auto()
    ANOMALY_DETECTION = auto()
    ANOMALY_SEGMENTATION = auto()
    INSTANCE_SEGMENTATION = auto()
    ROTATED_DETECTION = auto()
    if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
        ACTION_CLASSIFICATION = auto()
        ACTION_DETECTION = auto()
    if os.getenv("FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS", "0") == "1":
        VISUAL_PROMPTING = auto()

    def __str__(self):
        """Returns Domain name."""
        return str(self.name)


class LabelEntity:
    """This represents a label. The Label is the object that the user annotates and the tasks predict.

    For example, a label with name "car" can be constructed as follows.

    >>> car = LabelEntity(name="car", domain=Domain.DETECTION)

    .. rubric:: About Empty Label

    In addition to representing the presence of a certain object, the label can also
    be used to represent the absence of objects in the image (or other media types).
    Such a label is referred to as empty label.
    The empty label is constructed as follows:

    >>> empty = LabelEntity(name="empty", domain=Domain.DETECTION, is_empty=True)

    Empty label is used to declare that there is nothing of interest inside this image.
    For example, let's assume a car detection project. During annotation process,
    for positive images (images with cars), the users are asked to annotate the images
    with bounding boxes with car label. However, when the user sees a negative image
    (no car), the user needs to annotate this image with an empty label.

    The empty label is particularly useful to distinguish images with no objects
    of interest from images that have not been annotated, especially in task-chain
    scenario. Let's assume car detection task that is followed with with another
    detection task which detects the driver inside the car. There are two issues here:

    1. The user can (intentionally or unintentionally) miss to annotate
        the driver inside a car.
    2. There is no driver inside the car.

    Without empty label, these two cases cannot be distinguished.
    This is why an empty label is introduced. The empty label makes an explicit
    distinction between missing annotations and "negative" images.

    Args:
        name: the name of the label
        domain: the algorithm domain this label is associated to
        color: the color of the label (See :class:`Color`)
        hotkey: key or combination of keys to select this label in the
            UI
        creation_date: the date time of the label creation
        is_empty: set to True if the label is an empty label.
        id: the ID of the label. Set to ID() so that a new unique ID
            will be assigned upon saving. If the argument is None, it
            will be set to ID()
        is_anomalous: boolean that indicates whether the label is the
            Anomalous label. Always set to False for non- anomaly
            projects.
    """

    # pylint: disable=redefined-builtin, too-many-instance-attributes, too-many-arguments; Requires refactor
    def __init__(
        self,
        name: str,
        domain: Domain,
        color: Optional[Color] = None,
        hotkey: str = "",
        creation_date: Optional[datetime.datetime] = None,
        is_empty: bool = False,
        id: Optional[ID] = None,
        is_anomalous: bool = False,
    ):
        id = ID() if id is None else id
        color = Color.random() if color is None else color
        creation_date = now() if creation_date is None else creation_date

        self._name = name
        self._color = color
        self._hotkey = hotkey
        self._domain = domain
        self._is_empty = is_empty
        self._creation_date = creation_date
        self.__id_ = id
        self.is_anomalous = is_anomalous

    @property
    def name(self):
        """Returns the label name."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def color(self) -> Color:
        """Returns the Color object for the label."""
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    @property
    def hotkey(self) -> str:
        """Returns the hotkey for the label."""
        return self._hotkey

    @hotkey.setter
    def hotkey(self, value):
        self._hotkey = value

    @property
    def domain(self):
        """Returns the algorithm domain associated to this label."""
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value

    @property
    def is_empty(self) -> bool:
        """Returns a boolean indicating if the label is an empty label."""
        return self._is_empty

    @property
    def creation_date(self) -> datetime.datetime:
        """Returns the creation date of the label."""
        return self._creation_date

    @property
    def id_(self) -> ID:
        """Returns the label id."""
        return self.__id_

    @id_.setter
    def id_(self, value: ID):
        self.__id_ = value

    @property
    def id(self) -> ID:
        """DEPRECATED."""
        return self.__id_

    @id.setter
    def id(self, value: ID):
        """DEPRECATED."""
        self.__id_ = value

    def __repr__(self):
        """String representation of the label."""
        return (
            f"LabelEntity({self.id_}, name={self.name}, hotkey={self.hotkey}, "
            f"domain={self.domain}, color={self.color}, is_anomalous={self.is_anomalous})"
        )

    def __eq__(self, other):
        """Returns True if the two labels are equal."""
        if isinstance(other, LabelEntity):
            return (
                self.id_ == other.id_
                and self.name == other.name
                and self.color == other.color
                and self.hotkey == other.hotkey
                and self.domain == other.domain
                and self.is_anomalous == other.is_anomalous
            )
        return False

    def __lt__(self, other):
        """Returns True if self.id < other.id."""
        if isinstance(other, LabelEntity):
            return self.id_ < other.id_
        return False

    def __gt__(self, other):
        """Returns True if self.id is greater than other.id."""
        if isinstance(other, LabelEntity):
            return self.id_ > other.id_
        return False

    def __hash__(self):
        """Returns hash of the label."""
        return hash(str(self))
