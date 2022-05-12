"""This module implements the Dataset entity"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=redefined-builtin, invalid-name

import collections.abc
import copy
import itertools
import logging
from enum import Enum
from typing import Iterator, List, Optional, Sequence, Union, overload

from bson.objectid import ObjectId

from ote_sdk.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.subset import Subset

logger = logging.getLogger(__name__)


class DatasetPurpose(Enum):
    """
    Describes the purpose for the dataset.
    This makes it possible to identify datasets for a particular use.
    """

    #: used for user inference.
    # Prediction will generate AnnotationSceneKind.PREDICTION
    INFERENCE = 0
    #: used for training.
    # AnnotationScene contains user annotation of type AnnotationSceneKind.ANNOTATION
    TRAINING = 1
    #: used for pre-evaluation, evaluation, or testing.
    # Prediction will generate AnnotationSceneKind.EVALUATION
    EVALUATION = 2
    #: used for generating output stage.
    # Prediction will generate AnnotationSceneKind.PREDICTION
    GENERATING_OUTPUT = 3
    #: used for dataset slices which are used for analysis.
    # Prediction will generate AnnotationSceneKind.INTERMEDIATE
    TEMPORARY_DATASET = 4
    #: used for task analysis.
    # Prediction will generate AnnotationSceneKind.TASK_PREDICTION
    TASK_INFERENCE = 5

    def __str__(self):
        return str(self.name)


class DatasetIterator(collections.abc.Iterator):
    """
    This DatasetIterator iterates over the dataset lazily.
    Implements collections.abc.Iterator.
    """

    def __init__(self, dataset: "DatasetEntity"):
        self.dataset = dataset
        self.index = 0

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        item = self.dataset[self.index]
        self.index += 1
        return item


class DatasetEntity:
    """
    A dataset consists of a list of DatasetItemEntities and a purpose.

    .. rubric:: With dataset items

    This way assumes the dataset item entities are constructed before the dataset entity is made.

    >>> from ote_sdk.entities.image import Image
    >>> from ote_sdk.entities.annotation import NullAnnotationSceneEntity
    >>> from ote_sdk.entities.dataset_item import DatasetItemEntity
    >>> item = DatasetItemEntity(media=Image(file_path="image.jpg"), annotation_scene=NullAnnotationSceneEntity())
    >>> dataset = DatasetEntity(items=[item])

    .. rubric:: Iterate over dataset

    Regardless of the instantiation method chosen, the Dataset will work the same.
    The dataset can be iterated:

    >>> dataset = DatasetEntity(items=[item_1])
    >>> for dataset_item in dataset:
    ...     print(dataset_item)
    DatasetItemEntity(
        media=Image(image.jpg, width=640, height=480),
        annotation_scene=NullAnnotationSceneEntity(),
        roi=Annotation(shape=Rectangle(x=0.0, y=0.0, width=1.0, height=1.0), labels=[], id=6149e454893b7ebbe3a8faf6),
        subset=NONE
    )

    A particular item can also be fetched:

    >>> first_item = dataset[0]

    Or a slice:

    >>> first_ten = dataset[:10]
    >>> last_ten = dataset[-10:]

    .. rubric:: Get a subset of Dataset

    To get the test data for validating the network:

    >>> dataset = DatasetEntity()
    >>> testing_subset = dataset.get_subset(Subset.TESTING)

    This subset is also a DatasetEntity. The entities in the subset dataset refer to the same entities as
    in the original dataset. Altering one of the objects in the subset, will also alter them in the original.

    :param items: A list of dataset items to create dataset with.
    :param purpose: Purpose for dataset. Refer to :class:`DatasetPurpose` for more info.
    """

    def __init__(
        self,
        items: Optional[List[DatasetItemEntity]] = None,
        purpose: DatasetPurpose = DatasetPurpose.INFERENCE,
    ):
        self._items = [] if items is None else items
        self._purpose = purpose

    @property
    def purpose(self) -> DatasetPurpose:
        """
        Returns the DatasetPurpose. For example DatasetPurpose.ANALYSIS.

        :return: DatasetPurpose
        """
        return self._purpose

    @purpose.setter
    def purpose(self, value: DatasetPurpose) -> None:
        self._purpose = value

    def _fetch(self, key):
        """
        Fetch the given entity/entities from the items.
        Helper function for __getitem__

        :param key: Key called on the dataset. E.g. int (dataset[0]) or slice (dataset[5:9])

        :return: List of DatasetItemEntity or a single DatasetItemEntity.
        """
        if isinstance(key, list):
            return [self._fetch(ii) for ii in key]
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self._fetch(ii) for ii in range(*key.indices(len(self._items)))]
        if isinstance(key, int):
            return self._items[key]
        raise TypeError(
            f"Instance of type `{type(key).__name__}` cannot be used to access Dataset items. "
            f"Only slice and int are supported"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(items={self._items}, purpose={self.purpose})"

    def __str__(self):
        return f"{self.__class__.__name__}(size={len(self)}, purpose={self.purpose})"

    def __len__(self):
        return len(self._items)

    def __eq__(self, other) -> bool:
        """
        Checks whether the dataset is equal to the operand.

        :param other: the dataset operand to the equal operator
        :return: True if the datasets are equal
        """
        if (
            isinstance(other, DatasetEntity)
            and len(self) == len(other)
            and self.purpose == other.purpose
        ):
            return all(self[i] == other[i] for i in range(len(self)))
        return False

    def __add__(self, other: Union["DatasetEntity", List[DatasetItemEntity]]):
        """
        Returns a new dataset which contains the items of self added with the input dataset.
        Note that additional info of the dataset might be incoherent to the addition operands.

        :param other: dataset to be added to output

        :return: new dataset
        """
        items: List[DatasetItemEntity]

        if isinstance(other, DatasetEntity):
            items = self._items + list(other)
        elif isinstance(other, list):
            items = self._items + [o for o in other if isinstance(o, DatasetItemEntity)]
        else:
            raise ValueError(f"Cannot add other of type {type(other)}")

        return DatasetEntity(items=items, purpose=self.purpose)

    @overload
    def __getitem__(self, key: int) -> DatasetItemEntity:
        return self._fetch(key)

    @overload  # Overload for proper type hinting of indexing on dataset.
    def __getitem__(self, key: slice) -> Sequence[DatasetItemEntity]:
        return self._fetch(key)

    def __getitem__(self, key):
        """
        Return a DatasetItemEntity or a list of DatasetItemEntity, given a slice or an integer.

        Given an integer index:

        >>> dataset = DatasetEntity(items=[...])
        >>> first_item = dataset[0]

        Or a slice:

        >>> first_ten = dataset[0:9]
        >>> last_ten = dataset[-9:]

        :param key: key to fetch. Should be `slice` or `int`
        :return: List of DatasetItemEntity or single DatasetItemEntity
        """
        return self._fetch(key)

    def __iter__(self) -> Iterator[DatasetItemEntity]:
        """
        Return an iterator for the DatasetEntity.
        This iterator is able to iterate over the DatasetEntity lazily.

        :return: DatasetIterator instance
        """
        return DatasetIterator(self)

    def with_empty_annotations(
        self, annotation_kind: AnnotationSceneKind = AnnotationSceneKind.PREDICTION
    ) -> "DatasetEntity":
        """
        Produces a new dataset with empty annotation objects (no shapes or labels).
        This is a convenience function to generate a dataset with empty annotations from another dataset.
        This is particularly useful for evaluation on validation data and to build resultsets.

        Assume a dataset containing user annotations.

        >>> labeled_dataset = Dataset()  # user annotated dataset

        Then, we want to see the performance of our task on this labeled_dataset,
        which means we need to create a new dataset to be passed for analysis.

        >>> prediction_dataset = labeled_dataset.with_empty_annotations()

        Later, we can pass this prediction_dataset to the task analysis function.
        By pairing the labeled_dataset and the prediction_dataset, the resultset can then be constructed.
        Refer to :class:`~ote_sdk.entities.resultset.ResultSetEntity` for more info.

        :param annotation_kind: Sets the empty annotation to this kind. Default value: AnnotationSceneKind.PREDICTION
        :return: a new dataset containing the same items, with empty annotation objects.
        """
        new_dataset = DatasetEntity(purpose=self.purpose)
        for dataset_item in self:
            if isinstance(dataset_item, DatasetItemEntity):
                empty_annotation = AnnotationSceneEntity(
                    annotations=[], kind=annotation_kind
                )

                # reset ROI
                roi = copy.copy(dataset_item.roi)
                roi.id_ = ID(ObjectId())
                roi.set_labels([])

                new_dataset_item = DatasetItemEntity(
                    media=dataset_item.media,
                    annotation_scene=empty_annotation,
                    roi=roi,
                    subset=dataset_item.subset,
                )
                new_dataset.append(new_dataset_item)
        return new_dataset

    def get_subset(self, subset: Subset) -> "DatasetEntity":
        """
        Returns a new DatasetEntity with just the dataset items matching the subset.

        >>> dataset = DatasetEntity()
        >>> training_subset = dataset.get_subset(Subset.TRAINING)

        This subset is also a DatasetEntity. The dataset items in the subset dataset are the same dataset items as
        in the original dataset.
        Altering one of the objects in the output of this function, will also alter them in the original.

        :param subset: `Subset` to return

        :return: DatasetEntity with items matching subset
        """
        dataset = DatasetEntity(
            items=[item for item in self._items if item.subset == subset],
            purpose=self.purpose,
        )
        return dataset

    def remove(self, item: DatasetItemEntity) -> None:
        """
        Remove an item from the items.
        This function calls remove_at_indices function.

        :raises ValueError: if the input item is not in the dataset
        :param item: the item to be deleted
        """
        index = self._items.index(item)
        self.remove_at_indices([index])

    def append(self, item: DatasetItemEntity) -> None:
        """
        Append a DatasetItemEntity to the dataset

        :example: Appending a dataset item to a dataset

        >>> from ote_sdk.entities.image import Image
        >>> from ote_sdk.entities.annotation import NullAnnotationSceneEntity
        >>> from ote_sdk.entities.dataset_item import DatasetItemEntity
        >>> dataset = DatasetEntity()
        >>> media = Image(file_path='image.jpg')
        >>> annotation = NullAnnotationSceneEntity()
        >>> dataset_item = DatasetItemEntity(media=media, annotation_scene=annotation)
        >>> dataset.append(dataset_item)

        :param item: item to append
        """

        if item.media is None:
            raise ValueError("Media in dataset item cannot be None")
        self._items.append(item)

    def sort_items(self) -> None:
        """
        Order the dataset items. Does nothing here, but may be overrided in child classes.

        :return: None
        """

    def remove_at_indices(self, indices: List[int]) -> None:
        """
        Delete items based on the `indices`.

        :param indices: the indices of the items that will be deleted from the items.
        """
        indices.sort(reverse=True)  # sort in descending order
        for i_item in indices:
            del self._items[i_item]

    def get_labels(self, include_empty: bool = False) -> List[LabelEntity]:
        """
        Returns the list of all unique labels that are in the dataset, this does not respect the ROI of the dataset
        items.

        :param include_empty: set to True to include empty label (if exists) in the output.
        :return: list of labels that appear in the dataset
        """
        label_set = set(
            itertools.chain(
                *[item.annotation_scene.get_labels(include_empty) for item in self]
            )
        )
        return list(label_set)
