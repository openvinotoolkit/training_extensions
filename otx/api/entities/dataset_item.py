"""This module implements the dataset item entity."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=cyclic-import

import abc
import copy
import itertools
import logging
from threading import Lock
from typing import List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from otx.api.entities.annotation import Annotation, AnnotationSceneEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.media import IMedia2DEntity
from otx.api.entities.metadata import IMetadata, MetadataItemEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.shape_factory import ShapeFactory

logger = logging.getLogger(__name__)


class DatasetItemEntity(metaclass=abc.ABCMeta):
    """DatasetItemEntity represents an item in the DatasetEntity.

    It holds a media item, annotation and an ROI. The ROI determines the region of interest for the dataset item, and
    is described by a shape entity.

    The fundamental properties of a dataset item are:

    - A 2d media entity (e.g. Image)
    - A 2d annotation entity for the full resolution media entity
    - An ROI, describing the region of interest.
    - The subset it belongs to
    - Metadata for the media entity (e.g. saliency map or active score)
    - A list of labels to ignore

    .. rubric:: Getting data from dataset item

    The first step is to fetch the input data for the network.

    >>> dataset_item = DatasetItemEntity()
    >>> media_numpy = dataset_item.numpy  # RGB media data (Height, Width, Channels)

    This returns the numpy data for the assigned ROI. But it is possible to extract any arbitrary region.

    >>> from otx.api.entities.shapes.rectangle import Rectangle
    >>> top_left_quart_roi = Annotation(Rectangle(x1=0.0, y1=0.0, x2=0.5, y2=0.5), labels=[])
    >>> top_left_quart_numpy = dataset_item.roi_numpy(roi=top_left_quart_roi)

    Get the subset of labels for the item ROI:

    >>> labels = dataset_item.get_roi_labels(labels=...)

    Get the annotations __visible__ in the ROI:

    >>> dataset_item.get_annotations()

    .. rubric:: Adding output data to dataset item

    It is possible to add shapes or just labels for the ROI.

    Add shapes to dataset item:

    >>> box = Rectangle(x1=0.2, y1=0.3, x2=0.6, y2=0.5)
    >>> dataset_item.append_annotations(annotations=[Annotation(box, labels=[...])])

    Add labels to ROI:

    >>> dataset_item.append_labels(labels=[...])

    Args:
        media (IMedia2DEntity): Media item
        annotation_scene (AnnotationSceneEntity): Annotation scene
        roi (Optional[Annotation]): Region Of Interest
        metadata (Optional[List[MetadataItemEntity]]): Metadata attached to dataset item
        subset (Subset): `Subset` for item. E.g. `Subset.VALIDATION`
        ignored_labels (Optional[Union[List[LabelEntity], Tuple[LabelEntity, ...], Set[LabelEntity]]]): Collection of
            labels that should be ignored in this dataset item. For instance, in a training scenario, this parameter is
            used to ignore certain labels within the existing annotations because their status becomes uncertain
            following a label schema change.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        media: IMedia2DEntity,
        annotation_scene: AnnotationSceneEntity,
        roi: Optional[Annotation] = None,
        metadata: Optional[List[MetadataItemEntity]] = None,
        subset: Subset = Subset.NONE,
        ignored_labels: Optional[Union[List[LabelEntity], Tuple[LabelEntity, ...], Set[LabelEntity]]] = None,
    ):
        self.__media: IMedia2DEntity = media
        self.__annotation_scene: AnnotationSceneEntity = annotation_scene
        self.__subset: Subset = subset
        self.__roi_lock = Lock()

        # set ROI
        if roi is None:
            for annotation in annotation_scene.annotations:
                # if there is a full box in annotation.shapes, set it as ROI
                if Rectangle.is_full_box(annotation.shape):
                    roi = annotation
                    break
        if roi is None:
            roi = Annotation(Rectangle.generate_full_box(), labels=[])
        self.__roi = roi

        self.__metadata: List[MetadataItemEntity] = []
        if metadata is not None:
            self.__metadata = metadata

        self.__ignored_labels: Set[LabelEntity] = set() if ignored_labels is None else set(ignored_labels)

    def set_metadata(self, metadata: List[MetadataItemEntity]):
        """Sets the metadata."""
        self.__metadata = metadata

    def get_metadata(self) -> List[MetadataItemEntity]:
        """Returns the metadata."""
        return self.__metadata

    @property
    def ignored_labels(self) -> Set[LabelEntity]:
        """Get the IDs of the labels to ignore in this dataset item."""
        return self.__ignored_labels

    @ignored_labels.setter
    def ignored_labels(self, value: Union[List[LabelEntity], Tuple[LabelEntity, ...], Set[LabelEntity]]):
        self.__ignored_labels = set(value)

    def __repr__(self):
        """String representation of the dataset item."""
        return (
            f"{self.__class__.__name__}("
            f"media={self.media}, "
            f"annotation_scene={self.annotation_scene}, "
            f"roi={self.roi}, "
            f"subset={self.subset}), "
            f"meta={self.get_metadata()}"
        )

    @property
    def roi(self) -> Annotation:
        """Region Of Interest."""
        with self.__roi_lock:
            return self.__roi

    @roi.setter
    def roi(self, roi: Optional[Annotation]):
        with self.__roi_lock:
            if roi is None:
                roi = Annotation(Rectangle.generate_full_box(), labels=[])
            self.__roi = roi

    @property
    def subset(self) -> Subset:
        """Returns the subset that the IDatasetItem belongs to. e.g. Subset.TRAINING."""
        return self.__subset

    @subset.setter
    def subset(self, value: Subset):
        self.__subset = value

    @property
    def media(self) -> IMedia2DEntity:
        """Media."""
        return self.__media

    def roi_numpy(self, roi: Optional[Annotation] = None) -> np.ndarray:
        """Gives the numpy data for the media, given an ROI.

        This function allows to take a crop of any arbitrary region of the media in the Dataset entity.
        If the ROI is not given, the ROI assigned to the DatasetItem will be used as default.

        Args:
            roi (Optional[Annotation]): Shape entity. The shape will be converted if needed, to extract the ROI numpy.

        Returns:
            np.ndarray: Numpy array with media data
        """
        if roi is None:
            roi = self.roi

        if roi is not None:
            roi.shape = ShapeFactory.shape_as_rectangle(roi.shape)

        return self.media.roi_numpy(roi=roi)

    @property
    def numpy(self) -> np.ndarray:
        """Returns the numpy data for the media, taking ROI into account.

        Returns:
            np.ndarrray: Numpy array. RGB array of shape (Height, Width, Channels)
        """
        return self.roi_numpy()

    @property
    def width(self) -> int:
        """The width of the dataset item, taking into account the ROI."""
        roi_shape_as_box = ShapeFactory.shape_as_rectangle(self.roi.shape)
        roi_shape_as_box = roi_shape_as_box.clip_to_visible_region()
        width = self.media.width

        # Note that we cannot directly use roi_shape_as_box.width due to the rounding
        # because round(x2 - x1) is not always equal to round(x2) - round(x1)
        x1 = int(round(roi_shape_as_box.x1 * width))
        x2 = int(round(roi_shape_as_box.x2 * width))
        return x2 - x1

    @property
    def height(self) -> int:
        """The height of the dataset item, taking into account the ROI."""
        roi_shape_as_box = ShapeFactory.shape_as_rectangle(self.roi.shape)
        roi_shape_as_box = roi_shape_as_box.clip_to_visible_region()
        height = self.media.height

        # Note that we cannot directly use roi_shape_as_box.height due to the rounding
        # because round(y2 - y1) is not always equal to round(y2) - round(y1)
        y1 = int(round(roi_shape_as_box.y1 * height))
        y2 = int(round(roi_shape_as_box.y2 * height))
        return y2 - y1

    @property
    def annotation_scene(self) -> AnnotationSceneEntity:
        """Access to annotation scene."""
        return self.__annotation_scene

    @annotation_scene.setter
    def annotation_scene(self, value: AnnotationSceneEntity):
        self.__annotation_scene = value

    def get_annotations(
        self,
        labels: Optional[List[LabelEntity]] = None,
        include_empty: bool = False,
        include_ignored: bool = False,
    ) -> List[Annotation]:
        """Returns a list of annotations that exist in the dataset item (wrt. ROI).

        This is done by checking that the center of the annotation is located in the ROI.

        Args:
            labels (Optional[LabelEntity]): Subset of input labels to filter with; if ``None``, all the shapes within
                the ROI are returned.
            include_empty (bool): if True, returns both empty and non-empty labels
            include_ignored (bool): if True, includes the labels in ignored_labels

        Returns:
            List[Annotation]: The intersection of the input label set and those present within the ROI
        """
        is_full_box = Rectangle.is_full_box(self.roi.shape)
        annotations = []
        if is_full_box and labels is None and include_empty and include_ignored:
            # Fast path for the case where we do not need to change the shapes
            annotations = self.annotation_scene.annotations
        else:
            # Todo: improve speed. This is O(n) for n shapes.
            roi_as_box = ShapeFactory.shape_as_rectangle(self.roi.shape)

            labels_set = {label.name for label in labels} if labels is not None else set()

            for annotation in self.annotation_scene.annotations:
                if not is_full_box and not self.roi.shape.contains_center(annotation.shape):
                    continue

                shape_labels = annotation.get_labels(include_empty)

                check_labels = False
                if not include_ignored:
                    shape_labels = [label for label in shape_labels if label.label not in self.ignored_labels]
                    check_labels = True

                if labels is not None:
                    shape_labels = [label for label in shape_labels if label.name in labels_set]
                    check_labels = True

                if check_labels and len(shape_labels) == 0:
                    continue

                if not is_full_box:
                    # Create a denormalized copy of the shape.
                    shape = annotation.shape.denormalize_wrt_roi_shape(roi_as_box)
                else:
                    # Also create a copy of the shape, so that we can safely modify the labels
                    # without tampering with the original shape.
                    shape = copy.deepcopy(annotation.shape)

                annotations.append(Annotation(shape=shape, labels=shape_labels))
        return annotations

    def append_annotations(self, annotations: Sequence[Annotation]):
        """Adds a list of shapes to the annotation."""
        roi_as_box = ShapeFactory.shape_as_rectangle(self.roi.shape)

        validated_annotations = [
            Annotation(
                shape=annotation.shape.normalize_wrt_roi_shape(roi_as_box),
                labels=annotation.get_labels(),
            )
            for annotation in annotations
            if ShapeFactory().shape_produces_valid_crop(
                shape=annotation.shape,
                media_width=self.media.width,
                media_height=self.media.height,
            )
        ]

        n_invalid_shapes = len(annotations) - len(validated_annotations)
        if n_invalid_shapes > 0:
            logger.info(
                "%d shapes will not be added to the dataset item as they "
                "would produce invalid crops (this is expected for some tasks, "
                "such as segmentation).",
                n_invalid_shapes,
            )

        self.annotation_scene.append_annotations(validated_annotations)

    def get_roi_labels(
        self,
        labels: Optional[List[LabelEntity]] = None,
        include_empty: bool = False,
        include_ignored: bool = False,
    ) -> List[LabelEntity]:
        """Return the subset of the input labels which exist in the dataset item (wrt. ROI).

        Args:
            labels (Optional[List[LabelEntity]]): Subset of input labels to filter with; if ``None``, all the labels
                within the ROI are returned.
            include_empty (bool): if True, returns both empty and non-empty labels
            include_ignored (bool): if True, includes the labels in ignored_labels

        Return:
            List[LabelEntity]: The intersection of the input label set and those present within the ROI.
        """
        filtered_labels = set()
        for label in self.roi.get_labels(include_empty):
            if labels is None or label.get_label() in labels:
                filtered_labels.add(label.get_label())
        if not include_ignored:
            filtered_labels -= self.ignored_labels
        return sorted(list(filtered_labels), key=lambda x: x.name)

    def get_shapes_labels(
        self,
        labels: Optional[List[LabelEntity]] = None,
        include_empty: bool = False,
        include_ignored: bool = False,
    ) -> List[LabelEntity]:
        """Get the labels of the shapes present in this dataset item.

        if a label list is supplied, only labels present within that list are returned. if include_empty is True,
        present empty labels are returned as well.

        Args:
            labels (Optional[List[LabelEntity]]): if supplied only labels present in this list are returned.
                Defaults to None.
            include_empty (bool): if True, returns both empty and non-empty labels. Defaults to False.
            include_ignored (bool): if True, includes the labels in ignored_labels. Defaults to False.

        Returns:
            List[LabelEntity]: a list of labels from the shapes within the roi of this dataset item
        """
        annotations = self.get_annotations(labels=labels, include_empty=include_empty, include_ignored=include_ignored)
        scored_label_set = set(itertools.chain(*[annotation.get_labels(include_empty) for annotation in annotations]))
        label_set = {scored_label.get_label() for scored_label in scored_label_set}
        if not include_ignored:
            label_set -= self.ignored_labels
        if labels is None:
            return list(label_set)
        return [label for label in label_set if label in labels]

    def append_labels(self, labels: List[ScoredLabel]):
        """Appends labels to the DatasetItem and adds it to the the annotation label as well if it's not yet there.

        Args:
            labels (List[ScoredLabel]): list of labels to be appended.
        """
        if len(labels) == 0:
            return

        roi_annotation = None
        for annotation in self.annotation_scene.annotations:
            if annotation.shape == self.roi.shape:
                roi_annotation = annotation
                break

        if roi_annotation is None:  # no annotation found with shape
            roi_annotation = self.roi
            self.annotation_scene.append_annotation(roi_annotation)

        for label in labels:
            if label not in self.roi.get_labels(include_empty=True):
                self.roi.append_label(label)
            if label not in roi_annotation.get_labels(include_empty=True):
                roi_annotation.append_label(label)

    def __eq__(self, other):
        """Compares if two DatasetItems are equal.

        Args:
            other ("DatasetItems"): other DatasetItem to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, DatasetItemEntity):
            return (
                self.media == other.media
                and self.annotation_scene == other.annotation_scene
                and self.roi == other.roi
                and self.subset == other.subset
                and self.ignored_labels == other.ignored_labels
            )
        return False

    def __deepcopy__(self, memo):
        """Avoids copying the lock and unintentional ID sharing among AnnotationSceneEntity instances.

        When we deepcopy this object, be sure not to deep copy the lock, as this is not possible,
        make a new lock instead. In addition, we prevent deepcopy of AnnotationSceneEntity member
        variable to avoid unintentional ID sharing among instances. Same instance reference is
        copied to the output instead.
        """
        # Call ROI getter to ensure original object has an ROI.
        _ = self.roi

        clone = copy.copy(self)

        for name, value in vars(self).items():
            if "__roi_lock" in name:
                setattr(clone, name, Lock())
            elif "__annotation_scene" in name:
                pass  # Keep the same instance
            else:
                setattr(clone, name, copy.deepcopy(value, memo))
        return clone

    def append_metadata_item(self, data: IMetadata, model: Optional[ModelEntity] = None):
        """Appends metadata produced by some model to the dataset item.

        .. rubric:: Adding visualization heatmap (ResultMediaEntity) to DatasetItemEntity

        >>> from otx.api.entities.image import Image
        >>> from otx.api.entities.result_media import ResultMediaEntity
        >>> media = Image(file_path='image.jpeg')
        >>> annotation = NullAnnotationSceneEntity()
        >>> dataset_item = DatasetItem(media=media, annotation_scene=annotation)
        >>> data = np.ones((120, 120, 3)).astype(np.uint8) * 255 # Saliency numpy
        >>> result_media = ResultMediaEntity(name="Gradcam++",
        ...                                  type="Gradcam++",
        ...                                  annotation_scene=annotation,
        ...                                  numpy=data)
        >>> dataset_item.append_metadata_item(result_media)

        .. rubric:: Representation vector for active learning

        >>> from otx.api.entities.tensor import TensorEntity
        >>> tensor = TensorEntity(name="representation_vector", numpy=data)
        >>> dataset_item.append_metadata_item(data=tensor, model=model)

        Args:
            data (IMetadata): any object of a class inherited from IMetadata. (e.g., FloatMetadata, Tensor)
            model (Optional[ModelEntity]): model that was used to generated metadata
        """
        self.__metadata.append(MetadataItemEntity(data=data, model=model))

    def get_metadata_by_name_and_model(self, name: str, model: Optional[ModelEntity]) -> Sequence[MetadataItemEntity]:
        """Returns a metadata item with `name` and generated by `model`.

        Args:
            name (str): the name of the metadata
            model (Optional[ModelEntity]): the model which was used to generate the metadata.

        Returns:
            Sequence[MetadataItemEntity]: a list of metadata items with `name` and generated by `model`.
        """
        return [meta for meta in self.get_metadata() if meta.data.name == name and meta.model == model]
