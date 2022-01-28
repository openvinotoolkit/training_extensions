# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""This module define the label entity."""

import datetime
import warnings
from enum import Enum
from typing import Optional

from ote_sdk.entities.color import Color
from ote_sdk.entities.id import ID
from ote_sdk.utils.argument_checks import check_required_and_optional_parameters_type
from ote_sdk.utils.time_utils import now


class Domain(Enum):
    """
    Describes an algorithm domain like classification, detection, ...
    """

    CLASSIFICATION = 0
    DETECTION = 1
    SEGMENTATION = 2
    ANOMALY_CLASSIFICATION = 3
    ANOMALY_DETECTION = 4
    ANOMALY_SEGMENTATION = 5

    def __str__(self):
        return str(self.name)


class LabelEntity:
    """
    This represents a label. The Label is the object that the user annotates
    and the tasks predict.

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

    :param name: the name of the label
    :param domain: the algorithm domain this label is associated to
    :param color: the color of the label (See :class:`Color`)
    :param hotkey: key or combination of keys to select this label in the UI
    :param creation_date: the date time of the label creation
    :param is_empty: set to True if the label is an empty label.
    :param id: the ID of the label. Set to ID() so that a new unique ID
        will be assigned upon saving. If the argument is None, it will be set to ID()
    """

    # pylint: disable=redefined-builtin, too-many-arguments; Requires refactor
    def __init__(
        self,
        name: str,
        domain: Domain,
        color: Optional[Color] = None,
        hotkey: str = "",
        creation_date: Optional[datetime.datetime] = None,
        is_empty: bool = False,
        id: Optional[ID] = None,
    ):
        # Initialization parameters validation
        check_required_and_optional_parameters_type(
            required_parameters=[(name, "name", str), (domain, "domain", Domain)],
            optional_parameters=[
                (color, "color", Color),
                (hotkey, "hotkey", str),
                (creation_date, "creation_date", datetime.datetime),
                (is_empty, "is_empty", bool),
                (id, "id", (ID, int)),
            ],
        )
        if isinstance(id, int):
            warnings.warn(
                "ID-type object is expected as 'id' LabelEntity initialization parameter"
            )

        id = ID() if id is None else id
        color = Color.random() if color is None else color
        creation_date = now() if creation_date is None else creation_date

        self._name = name
        self._color = color
        self._hotkey = hotkey
        self._domain = domain
        self._is_empty = is_empty
        self._creation_date = creation_date
        self._id = id

    @property
    def name(self):
        """
        Returns the label name.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def color(self) -> Color:
        """
        Returns the Color object for the label.
        """
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    @property
    def hotkey(self) -> str:
        """
        Returns the hotkey for the label
        """
        return self._hotkey

    @hotkey.setter
    def hotkey(self, value):
        self._hotkey = value

    @property
    def domain(self):
        """
        Returns the algorithm domain associated to this label
        """
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value

    @property
    def is_empty(self) -> bool:
        """
        Returns a boolean indicating if the label is an empty label
        """
        return self._is_empty

    @property
    def creation_date(self) -> datetime.datetime:
        """
        Returns the creation date of the label
        """
        return self._creation_date

    @property
    def id(self) -> ID:
        """
        Returns the label id.
        """
        return self._id

    @id.setter
    def id(self, value: ID):
        self._id = value

    def __repr__(self):
        return (
            f"LabelEntity({self.id}, name={self.name}, hotkey={self.hotkey}, "
            f"domain={self.domain}, color={self.color})"
        )

    def __eq__(self, other):
        if isinstance(other, LabelEntity):
            return (
                self.id == other.id
                and self.name == other.name
                and self.color == other.color
                and self.hotkey == other.hotkey
                and self.domain == other.domain
            )
        return False

    def __lt__(self, other):
        if isinstance(other, LabelEntity):
            return self.id < other.id
        return False

    def __gt__(self, other):
        if isinstance(other, LabelEntity):
            return self.id > other.id
        return False

    def __hash__(self):
        return hash(str(self))
