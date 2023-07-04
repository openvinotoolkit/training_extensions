"""This module define the annotation entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from bson import ObjectId

from otx.api.entities.id import ID
from otx.api.entities.label import LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.shape import ShapeEntity
from otx.api.utils.time_utils import now


class Annotation(metaclass=abc.ABCMeta):
    """Base class for annotation objects.

    Args:
        shape (ShapeEntity): the shape of the annotation
        labels (List[ScoredLabel]): the labels of the annotation
        id (Optional[ID]): the id of the annotation
    """

    # pylint: disable=redefined-builtin;
    def __init__(self, shape: ShapeEntity, labels: List[ScoredLabel], id: Optional[ID] = None):
        self.__id_ = ID(ObjectId()) if id is None else id
        self.__shape = shape
        self.__labels = labels

    def __repr__(self):
        """String representation of the annotation."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"labels={self.get_labels(include_empty=True)}, "
            f"id={self.id_})"
        )

    @property
    def id_(self):
        """Returns the id for the annotation."""
        return self.__id_

    @id_.setter
    def id_(self, value):
        self.__id_ = value

    @property
    def id(self):
        """DEPRECATED."""
        return self.__id_

    @id.setter
    def id(self, value):
        """DEPRECATED."""
        self.__id_ = value

    @property
    def shape(self) -> ShapeEntity:
        """Returns the shape that is in the annotation."""
        return self.__shape

    @shape.setter
    def shape(self, value) -> None:
        self.__shape = value

    def get_labels(self, include_empty: bool = False) -> List[ScoredLabel]:
        """Get scored labels that are assigned to this annotation.

        Args:
            include_empty (bool): set to True to include empty label (if exists) in the output. Defaults to False.

        Returns:
            List of labels in annotation
        """
        return [label for label in self.__labels if include_empty or (not label.is_empty)]

    def get_label_ids(self, include_empty: bool = False) -> Set[ID]:
        """Get a set of ID's of labels that are assigned to this annotation.

        Args:
            include_empty (bool): set to True to include empty label (if exists) in the output. Defaults to False.

        Returns:
            Set of label id's in annotation
        """
        return {label.id_ for label in self.__labels if include_empty or (not label.is_empty)}

    def append_label(self, label: ScoredLabel) -> None:
        """Appends the scored label to the annotation.

        Args:
            label (ScoredLabel): the scored label to be appended to the annotation
        """
        self.__labels.append(label)

    def set_labels(self, labels: List[ScoredLabel]) -> None:
        """Sets the labels of the annotation to be the input of the function.

        Args:
            labels (List[ScoredLabel]): the scored labels to be set as annotation labels
        """
        self.__labels = labels

    def __eq__(self, other: object) -> bool:
        """Checks if the two annotations are equal.

        Args:
            other (Annotation): Annotation to compare with.

        Returns:
            bool: True if the two annotations are equal, False otherwise.
        """
        if isinstance(other, Annotation):
            return (
                self.id_ == other.id_ and self.get_labels(True) == other.get_labels(True) and self.shape == other.shape
            )
        return False


class AnnotationSceneKind(Enum):
    """AnnotationSceneKinds for an Annotation object."""

    #:  NONE represents NULLAnnotationScene's (See :class:`NullAnnotationScene`)
    NONE = 0
    #:  ANNOTATION represents user annotation
    ANNOTATION = 1
    #:  PREDICTION represents analysis result, which will be shown to the user
    PREDICTION = 2
    #:  EVALUATION represents analysis result for evaluation purposes, which will NOT be shown to the user
    EVALUATION = 3
    #:  INTERMEDIATE represents intermediary state.
    #:  This is used when the analysis is being transferred from one task to another.
    #:  This will not be shown to the user.
    #:  This state will be changed to either PREDICTION or EVALUATION at the end of analysis process.
    INTERMEDIATE = 4
    #:  TASK_PREDICTION represents analysis results for a single task
    TASK_PREDICTION = 5

    def __str__(self):
        """String representation of the AnnotationSceneKind."""
        return str(self.name)


class AnnotationSceneEntity(metaclass=abc.ABCMeta):
    """This class represents a user annotation or a result (prediction).

    It serves as a collection of shapes, with a relation to the media entity.

    Example:
        Creating an annotation:

        >>> from otx.api.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
        >>> from otx.api.entities.shapes.rectangle import Rectangle
        >>> box = Rectangle(x1=0.0, y1=0.0, x2=0.5, y2=0.5)  # Box covering top-left quart of image
        >>> AnnotationSceneEntity(annotations=[Annotation(shape=box, labels=[])], kind=AnnotationSceneKind.ANNOTATION)

    Args:
        annotations (List[Annotation]): List of annotations in the scene
        kind (AnnotationSceneKind): Kind of the annotation scene. E.g. `AnnotationSceneKind.ANNOTATION`.
        editor (str): The user that made this annotation scene object.
        creation_date (Optional[datetime.datetime]): Creation date of annotation scene entity. If None, current time is
            used. Defaults to None.
        id (Optional[ID]): ID of AnnotationSceneEntity. If None a new `ID` is created. Defaults to None.
    """

    # pylint: disable=too-many-arguments, redefined-builtin
    def __init__(
        self,
        annotations: List[Annotation],
        kind: AnnotationSceneKind,
        editor: str = "",
        creation_date: Optional[datetime.datetime] = None,
        id: Optional[ID] = None,
    ):
        self.__annotations = annotations
        self.__kind = kind
        self.__editor = editor
        self.__creation_date = now() if creation_date is None else creation_date
        self.__id_ = ID() if id is None else id

    def __repr__(self):
        """String representation of the annotation scene."""
        return (
            f"{self.__class__.__name__}("
            f"annotations={self.annotations}, "
            f"kind={self.kind}, "
            f"editor={self.editor_name}, "
            f"creation_date={self.creation_date}, "
            f"id={self.id_})"
        )

    @property
    def id_(self) -> ID:
        """Returns the ID of the AnnotationSceneEntity."""
        return self.__id_

    @id_.setter
    def id_(self, value) -> None:
        self.__id_ = value

    @property
    def id(self):
        """DEPRECATED."""
        return self.__id_

    @id.setter
    def id(self, value):
        """DEPRECATED."""
        self.__id_ = value

    @property
    def kind(self) -> AnnotationSceneKind:
        """Returns the AnnotationSceneKind of the AnnotationSceneEntity."""
        return self.__kind

    @kind.setter
    def kind(self, value) -> None:
        self.__kind = value

    @property
    def editor_name(self) -> str:
        """Returns the editor's name that made the AnnotationSceneEntity object."""
        return self.__editor

    @editor_name.setter
    def editor_name(self, value) -> None:
        self.__editor = value

    @property
    def creation_date(self) -> datetime.datetime:
        """Returns the creation date of the AnnotationSceneEntity object."""
        return self.__creation_date

    @creation_date.setter
    def creation_date(self, value) -> None:
        self.__creation_date = value

    @property
    def annotations(self) -> List[Annotation]:
        """Return the Annotations that are present in the AnnotationSceneEntity."""
        return self.__annotations

    @annotations.setter
    def annotations(self, value: List[Annotation]):
        self.__annotations = value

    @property
    def shapes(self) -> List[ShapeEntity]:
        """Returns all shapes that are inside the annotations of the AnnotationSceneEntity."""
        return [annotation.shape for annotation in self.annotations]

    def contains_any(self, labels: List[LabelEntity]) -> bool:
        """Checks whether the annotation contains any labels in the input parameter.

        Args:
            labels (List[LabelEntity]): List of labels to compare to.

        Returns:
            bool: True if there is any intersection between self.get_labels(include_empty=True) with labels.
        """
        label_names = {label.name for label in labels}
        return len({label.name for label in self.get_labels(include_empty=True)}.intersection(label_names)) != 0

    def append_annotation(self, annotation: Annotation) -> None:
        """Appends the passed annotation to the list of annotations present in the AnnotationSceneEntity object."""
        self.annotations.append(annotation)

    def append_annotations(self, annotations: List[Annotation]) -> None:
        """Adds a list of annotations to the annotation scene."""
        self.annotations.extend(annotations)

    def get_labels(self, include_empty: bool = False) -> List[LabelEntity]:
        """Returns a list of unique labels which appear in this annotation scene.

        Args:
            include_empty (bool): Set to True to include empty label (if exists) in the output. Defaults to False.

        Returns:
            List[LabelEntity]: a list of labels which appear in this annotation.
        """

        labels: Dict[str, LabelEntity] = {}
        for annotation in self.annotations:
            for label in annotation.get_labels(include_empty=include_empty):
                id_ = label.id_
                if id_ not in labels:
                    labels[id_] = label.get_label()
        return list(labels.values())

    def get_label_ids(self, include_empty: bool = False) -> Set[ID]:
        """Returns a set of the ID's of unique labels which appear in this annotation scene.

        Args:
            include_empty (bool): Set to True to include empty label (if exists) in the output. Defaults to False.

        Returns:
            Set[ID]: a set of the ID's of labels which appear in this annotation.
        """

        output: Set[ID] = set()
        for annotation in self.annotations:
            output.update(set(annotation.get_label_ids(include_empty=include_empty)))
        return output


class NullAnnotationSceneEntity(AnnotationSceneEntity):
    """Represents 'AnnotationSceneEntity not found."""

    def __init__(self) -> None:
        super().__init__(
            id=ID(),
            kind=AnnotationSceneKind.NONE,
            editor="",
            creation_date=datetime.datetime.now(),
            annotations=[],
        )

    def __repr__(self):
        """String representation NullAnnotationSceneEntity."""
        return "NullAnnotationSceneEntity()"
