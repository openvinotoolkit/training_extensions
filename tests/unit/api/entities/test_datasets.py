# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List

import pytest

from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity, DatasetPurpose
from otx.api.entities.label import LabelEntity
from otx.api.entities.subset import Subset
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements
from tests.unit.api.entities.test_dataset_item import DatasetItemParameters


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDatasetPurpose:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_purpose(self):
        """
        <b>Description:</b>
        Check DatasetPurpose Enum class elements

        <b>Expected results:</b>
        Test passes if DatasetPurpose Enum class length is equal to expected value, its elements have expected
        sequence numbers and value returned by __str__ method is equal to expected
        """
        assert len(DatasetPurpose) == 6
        assert DatasetPurpose.INFERENCE.value == 0
        assert str(DatasetPurpose.INFERENCE) == "INFERENCE"
        assert DatasetPurpose.TRAINING.value == 1
        assert str(DatasetPurpose.TRAINING) == "TRAINING"
        assert DatasetPurpose.EVALUATION.value == 2
        assert str(DatasetPurpose.EVALUATION) == "EVALUATION"
        assert DatasetPurpose.GENERATING_OUTPUT.value == 3
        assert str(DatasetPurpose.GENERATING_OUTPUT) == "GENERATING_OUTPUT"
        assert DatasetPurpose.TEMPORARY_DATASET.value == 4
        assert str(DatasetPurpose.TEMPORARY_DATASET) == "TEMPORARY_DATASET"
        assert DatasetPurpose.TASK_INFERENCE.value == 5
        assert str(DatasetPurpose.TASK_INFERENCE) == "TASK_INFERENCE"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDatasetEntity:
    @staticmethod
    def generate_random_image():
        return DatasetItemParameters.generate_random_image()

    @staticmethod
    def labels() -> List[LabelEntity]:
        return DatasetItemParameters.labels()

    @staticmethod
    def annotations_entity() -> AnnotationSceneEntity:
        return DatasetItemParameters().annotations_entity()

    @staticmethod
    def metadata():
        return DatasetItemParameters.metadata()

    @staticmethod
    def default_values_dataset_item() -> DatasetItemEntity:
        return DatasetItemParameters().default_values_dataset_item()

    @staticmethod
    def dataset_item() -> DatasetItemEntity:
        return DatasetItemParameters().dataset_item()

    def dataset(self) -> DatasetEntity:
        other_dataset_item = DatasetItemEntity(
            media=self.generate_random_image(),
            annotation_scene=self.annotations_entity(),
            metadata=self.metadata(),
            subset=Subset.VALIDATION,
        )
        items = [
            self.default_values_dataset_item(),
            self.dataset_item(),
            other_dataset_item,
        ]
        return DatasetEntity(items, DatasetPurpose.TEMPORARY_DATASET)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_initialization(self):
        """
        <b>Description:</b>
        Check DatasetEntity class object initialization

        <b>Input data:</b>
        DatasetEntity class objects with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if "items" and "purpose" attributes of DatasetEntity object are equal to expected values

        <b>Steps</b>
        1. Check attributes of DatasetItemEntity object initialized with default optional parameters
        2. Check attributes of DatasetItemEntity object initialized with specified optional parameters
        """
        # Checking attributes of DatasetItemEntity object initialized with default optional parameters
        default_parameters_dataset = DatasetEntity()
        assert default_parameters_dataset._items == []
        assert default_parameters_dataset.purpose == DatasetPurpose.INFERENCE
        # Checking attributes of DatasetItemEntity object initialized with specified optional parameters
        items = [
            self.default_values_dataset_item(),
            self.dataset_item(),
            self.dataset_item(),
        ]
        purpose = DatasetPurpose.TEMPORARY_DATASET
        optional_parameters_dataset = DatasetEntity(items, purpose)
        assert optional_parameters_dataset._items == items
        assert optional_parameters_dataset.purpose == purpose

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_purpose_setter(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "purpose" setter

        <b>Input data:</b>
        DatasetEntity class objects with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if assigned value of "purpose" property is equal to expected

        <b>Steps</b>
        1. Check value returned by "purpose" property after using @purpose.setter for DatasetEntity with
        default optional parameters
        2. Check value returned by "purpose" property after using @purpose.setter for DatasetEntity initialized with
        specified optional parameters
        """
        # Checking "purpose" property after using @purpose.setter for DatasetEntity with default optional parameters
        default_parameters_dataset = DatasetEntity()
        expected_purpose = DatasetPurpose.TRAINING
        default_parameters_dataset.purpose = expected_purpose
        assert default_parameters_dataset.purpose == expected_purpose
        # Checking "purpose" property after using @purpose.setter for DatasetEntity with specified optional parameters
        optional_parameters_dataset = self.dataset()
        expected_purpose = DatasetPurpose.TASK_INFERENCE
        optional_parameters_dataset.purpose = expected_purpose
        assert optional_parameters_dataset.purpose == expected_purpose

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_fetch(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "_fetch" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if value returned by "_fetch" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "_fetch" method when list of DatasetItems indexes is specified as "key" parameter
        2. Check value returned by "_fetch" method when slice of DatasetItems indexes is specified as "key" parameter
        3. Check value returned by "_fetch" method when DatasetItem index is specified as "key" parameter
        4. Check TypeError exception is raised when unexpected type object is specified as "key" parameter
        """
        dataset = self.dataset()
        dataset_items = dataset._items
        # Checking  "_fetch" method when list of DatasetItems indexes is specified as "key" parameter
        assert dataset._fetch([0, 2]) == [dataset_items[0], dataset_items[2]]
        # Checking  "_fetch" method when slice of DatasetItems indexes is specified as "key" parameter
        assert dataset._fetch(slice(0, 3, 2)) == [dataset_items[0], dataset_items[2]]
        assert dataset._fetch(slice(-1, -4, -1)) == [
            dataset_items[2],
            dataset_items[1],
            dataset_items[0],
        ]
        # Checking  "_fetch" method when DatasetItem index is specified as "key" parameter
        assert dataset._fetch(1) == dataset_items[1]
        # Checking that TypeError exception is raised when unexpected type object is specified as "key" parameter
        with pytest.raises(TypeError):
            dataset._fetch(str)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_repr(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "__repr__" method

        <b>Input data:</b>
        DatasetEntity class objects with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if value returned by "__repr__" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "__repr__" method for DatasetEntity with default optional parameters
        2. Check value returned by "__repr__" method for DatasetEntity with specified optional parameters
        """
        # Checking value returned by "__repr__" method for DatasetEntity with default optional parameters
        default_parameters_dataset = DatasetEntity()
        assert repr(default_parameters_dataset) == "DatasetEntity(items=[], purpose=INFERENCE)"
        # Checking value returned by "__repr__" method for DatasetEntity with specified optional parameters
        optional_parameters_dataset = self.dataset()
        assert (
            repr(optional_parameters_dataset) == f"DatasetEntity(items={optional_parameters_dataset._items}, "
            f"purpose=TEMPORARY_DATASET)"
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_str(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "__str__" method

        <b>Input data:</b>
        DatasetEntity class objects with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if value returned by "__str__" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "__str__" method for DatasetEntity with default optional parameters
        2. Check value returned by "__str__" method for DatasetEntity with specified optional parameters
        """
        # Checking value returned by "__str__" method for DatasetEntity with default optional parameters
        default_parameters_dataset = DatasetEntity()
        assert str(default_parameters_dataset) == "DatasetEntity(size=0, purpose=INFERENCE)"
        # Checking value returned by "__str__" method for DatasetEntity with specified optional parameters
        optional_parameters_dataset = self.dataset()
        assert str(optional_parameters_dataset) == "DatasetEntity(size=3, purpose=TEMPORARY_DATASET)"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_len(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "__len__" method

        <b>Input data:</b>
        DatasetEntity class objects with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if value returned by "__len__" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "__len__" method for DatasetEntity with default optional parameters
        2. Check value returned by "__len__" method for DatasetEntity with specified optional parameters
        """
        # Checking value returned by "__str__" method for DatasetEntity with default optional parameters
        default_parameters_dataset = DatasetEntity()
        assert len(default_parameters_dataset) == 0
        # Checking value returned by "__str__" method for DatasetEntity with specified optional parameters
        optional_parameters_dataset = self.dataset()
        assert len(optional_parameters_dataset) == 3

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_eq(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "__eq__" method

        <b>Input data:</b>
        DatasetEntity class objects with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if value returned by "__eq__" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "__eq__" method for equal DatasetEntity objects
        2. Check value returned by "__eq__" method for DatasetEntity objects with unequal length
        3. Check value returned by "__eq__" method for DatasetEntity objects with equal length, but unequal
        DatasetItem objects
        4. Check value returned by "__eq__" method for DatasetEntity objects with unequal "purpose" attributes
        5. Check value returned by "__eq__" method for comparing DatasetEntity object with object of different type
        """
        # Checking value returned by "__eq__" method for equal DatasetEntity objects
        items = [
            self.default_values_dataset_item(),
            self.dataset_item(),
            self.dataset_item(),
        ]
        purpose = DatasetPurpose.TEMPORARY_DATASET
        dataset = DatasetEntity(items, purpose)
        equal_dataset = DatasetEntity(items, purpose)
        assert dataset == equal_dataset
        # Checking value returned by "__eq__" method for DatasetEntity objects with unequal length
        unequal_items = list(items)
        unequal_items.pop(-1)
        unequal_dataset = DatasetEntity(unequal_items, purpose)
        assert dataset != unequal_dataset
        # Checking value returned by "__eq__" method for DatasetEntity objects with equal length, but unequal
        # DatasetItem objects
        unequal_items.append(self.dataset_item())
        unequal_dataset = DatasetEntity(unequal_items, purpose)
        assert dataset != unequal_dataset
        # Checking value returned by "__eq__" method for DatasetEntity objects with unequal "purpose" attributes
        unequal_dataset = DatasetEntity(items, DatasetPurpose.EVALUATION)
        assert dataset != unequal_dataset
        # Checking value returned by "__eq__" method for comparing DatasetEntity object with object of different type
        assert dataset != str

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_add(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "__add__" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if DatasetEntity object returned by "__add__"" method is equal to expected

        <b>Steps</b>
        1. Check DatasetEntity object returned by "__add__"" method with DatasetEntity specified as "other" parameter
        2. Check DatasetEntity object returned by "__add__"" method with list of DatasetItemEntity objects specified
        as "other" parameter
        3. Check ValueError exception is raised when unexpected type object is specified in "other" parameter of
        "__add__" method
        """
        dataset = self.dataset()
        dataset_items = list(dataset._items)
        # Checking DatasetEntity object returned by "__add__"" method with DatasetEntity specified as "other" parameter
        other_dataset_items = [self.dataset_item(), self.dataset_item()]
        other_dataset = DatasetEntity(other_dataset_items, DatasetPurpose.TRAINING)
        new_dataset = dataset.__add__(other_dataset)
        assert new_dataset._items == dataset_items + other_dataset_items
        assert new_dataset.purpose == DatasetPurpose.TEMPORARY_DATASET
        # Checking DatasetEntity object returned by "__add__"" method with list of DatasetItemEntity objects specified
        # as "other" parameter
        items_to_add = [
            self.dataset_item(),
            self.dataset_item(),
            "unexpected type object",
        ]
        new_dataset = dataset.__add__(items_to_add)
        # Expected that str object will not be added to new_dataset._items
        assert new_dataset._items == dataset_items + items_to_add[0:2]
        assert new_dataset.purpose == DatasetPurpose.TEMPORARY_DATASET
        # Checking ValueError exception is raised when unexpected type object is specified in "other" parameter of
        # "__add__" method
        with pytest.raises(ValueError):
            dataset.__add__(str)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_getitem(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "__getitem__" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if value returned by "__getitem__" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "__getitem__" method when index is specified as "key" parameter
        2. Check value returned by "__getitem__" method when slice is specified as "key" parameter
        """
        dataset = self.dataset()
        dataset_items = dataset._items
        # Checking value returned by "__getitem__" method when index is specified as "key" parameter
        assert dataset[1] == dataset_items[1]
        # Checking value returned by "__getitem__" method when slice is specified as "key" parameter
        assert dataset[slice(0, 3, 2)] == [dataset_items[0], dataset_items[2]]
        assert dataset[slice(-1, -4, -1)] == [
            dataset_items[2],
            dataset_items[1],
            dataset_items[0],
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_iter(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "__iter__" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if DatasetItemEntity returned by "__iter__" method is equal to expected
        """
        dataset = self.dataset()
        dataset_items = list(dataset._items)
        dataset_iterator = dataset.__iter__()
        expected_index = 0
        for expected_dataset_item in dataset_items:
            assert dataset_iterator.index == expected_index
            assert next(dataset_iterator) == expected_dataset_item
            expected_index += 1
        with pytest.raises(StopIteration):
            next(dataset_iterator)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_with_empty_annotations(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "with_empty_annotations" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if DatasetEntity object returned by "with_empty_annotations" method is equal to expected

        <b>Steps</b>
        1. Check DatasetEntity object returned by "with_empty_annotations" with non-specified "annotation_kind"
        parameter
        2. Check DatasetEntity object returned by "with_empty_annotations" with specified "annotation_kind" parameter
        """

        def check_empty_annotations_dataset(actual_dataset, expected_dataset, expected_kind):
            expected_items = expected_dataset._items
            actual_items = actual_dataset._items
            assert actual_dataset.purpose is expected_dataset.purpose
            for i in range(len(expected_items) - 1):
                actual_item = actual_items[i]
                expected_item = expected_items[i]
                assert actual_item.media is expected_item.media
                assert actual_item.annotation_scene.annotations == []
                assert actual_item.annotation_scene.kind == expected_kind
                assert actual_item.roi.id_ != expected_item.roi.id_
                assert actual_item.roi.shape is expected_item.roi.shape
                assert actual_item.roi.get_labels() == []
                assert actual_item.subset is expected_item.subset

        dataset = self.dataset()
        # Checking DatasetEntity object returned by "with_empty_annotations" with non-specified "annotation_kind"
        # parameter
        empty_annotations_dataset = dataset.with_empty_annotations()
        check_empty_annotations_dataset(empty_annotations_dataset, dataset, AnnotationSceneKind.PREDICTION)
        # Checking DatasetEntity object returned by "with_empty_annotations" with specified "annotation_kind" parameter
        kind = AnnotationSceneKind.ANNOTATION
        empty_annotations_dataset = dataset.with_empty_annotations(kind)
        check_empty_annotations_dataset(empty_annotations_dataset, dataset, kind)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_get_subset(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "get_subset" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if DatasetEntity returned by "get_subset" method is equal to expected

        <b>Steps</b>
        1. Check DatasetEntity object returned by "get_subset" method with "subset" parameter that in items of base
        dataset
        2. Check DatasetEntity object returned by "get_subset" method with "subset" parameter that not in items of base
        dataset
        """
        validation_item = self.dataset_item()
        validation_item.subset = Subset.VALIDATION
        dataset = self.dataset()
        dataset._items.append(validation_item)
        # Checking DatasetEntity object returned by "get_subset" method with "subset" parameter that in items of base
        # dataset
        validation_dataset = dataset.get_subset(Subset.VALIDATION)
        assert validation_dataset.purpose is dataset.purpose
        assert validation_dataset._items == [dataset._items[2]] + [validation_item]
        # Checking DatasetEntity object returned by "get_subset" method with "subset" parameter that not in items of
        # base dataset
        empty_items_dataset = dataset.get_subset(Subset.UNLABELED)
        assert empty_items_dataset.purpose is dataset.purpose
        assert empty_items_dataset._items == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_remove(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "remove" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if "items" attribute of DatasetEntity object is equal to expected after using "remove" method
        """
        dataset = self.dataset()
        dataset_items = list(dataset._items)
        # Removing DatasetItemEntity included in DatasetEntity
        dataset.remove(dataset_items[1])
        dataset_items.pop(1)
        assert dataset._items == dataset_items
        non_included_dataset_item = self.dataset_item()
        # Check that ValueError exception is raised when removing non-included DatasetItemEntity
        with pytest.raises(ValueError):
            dataset.remove(non_included_dataset_item)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_append(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "append" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if "items" attribute of DatasetEntity object is equal to expected after using "append" method

        <b>Steps</b>
        1. Check "items" attribute of DatasetEntity object after adding new DatasetEntity object
        2. Check "items" attribute of DatasetEntity object after adding existing DatasetEntity object
        3. Check that ValueError exception is raised when appending DatasetEntity with "media" attribute is equal to
        "None"
        """
        dataset = self.dataset()
        expected_items = list(dataset._items)
        # Checking "items" attribute of DatasetEntity object after adding new DatasetEntity object
        item_to_add = self.dataset_item()
        dataset.append(item_to_add)
        expected_items.append(item_to_add)
        assert dataset._items == expected_items
        # Checking "items" attribute of DatasetEntity object after adding existing DatasetEntity object
        dataset.append(item_to_add)
        expected_items.append(item_to_add)
        assert dataset._items == expected_items
        # Checking that ValueError exception is raised when appending DatasetEntity with "media" is "None" attribute
        no_media_item = DatasetItemEntity(None, self.annotations_entity())
        with pytest.raises(ValueError):
            dataset.append(no_media_item)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_sort_items(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "sort_items" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if "items" attribute of DatasetEntity object is remained the same after using "sort_items" method
        """
        dataset = self.dataset()
        expected_items = list(dataset._items)
        dataset.sort_items()
        assert dataset._items == expected_items
        assert dataset.purpose == DatasetPurpose.TEMPORARY_DATASET

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_remove_at_indices(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "remove_at_indices" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if "items" attribute of DatasetEntity object is equal to expected after using "remove_at_indices"
        method
        """
        dataset = self.dataset()
        expected_items = list(dataset._items)
        # Removing DatasetItemEntity included in DatasetEntity
        dataset.remove_at_indices([0, 2])
        expected_items.pop(2)
        expected_items.pop(0)
        assert dataset._items == expected_items
        # Check that IndexError exception is raised when removing DatasetItemEntity with non-included index
        with pytest.raises(IndexError):
            dataset.remove_at_indices([20])

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dataset_entity_get_labels(self):
        """
        <b>Description:</b>
        Check DatasetEntity class "get_labels" method

        <b>Input data:</b>
        DatasetEntity class object with specified "items" and "purpose" parameters

        <b>Expected results:</b>
        Test passes if list returned by "get_labels" method is equal to expected

         <b>Steps</b>
        1. Check list returned by "get_labels" method with "include_empty" parameter is "False"
        2. Check list returned by "get_labels" method with "include_empty" parameter is "True"
        """
        labels = self.labels()
        detection_label = labels[0]
        segmentation_empty_label = labels[1]
        dataset = self.dataset()
        # Checking list returned by "get_labels" method with "include_empty" parameter is "False"
        assert dataset.get_labels() == [detection_label]
        # Checking list returned by "get_labels" method with "include_empty" parameter is "True"
        actual_empty_labels = dataset.get_labels(include_empty=True)
        assert len(actual_empty_labels) == 2
        assert isinstance(actual_empty_labels, list)
        assert segmentation_empty_label in actual_empty_labels
        assert detection_label in actual_empty_labels
