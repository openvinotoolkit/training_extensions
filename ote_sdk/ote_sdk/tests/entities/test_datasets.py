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

import numpy as np
import pytest

from ote_sdk.entities.datasets import DatasetPurpose, DatasetEntity
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.entities.image import Image
from ote_sdk.entities.annotation import NullAnnotationSceneEntity
from ote_sdk.entities.dataset_item import DatasetItemEntity


item = DatasetItemEntity(media=Image(file_path="image.jpg"), annotation_scene=NullAnnotationSceneEntity())
item_2 = DatasetItemEntity(media=Image(file_path="image_2.jpg"), annotation_scene=NullAnnotationSceneEntity())
dataset = DatasetEntity(items=[item,item_2])
item_3 = DatasetItemEntity(media=Image(file_path="image_3.jpg"), annotation_scene=NullAnnotationSceneEntity())
item_4 = DatasetItemEntity(media=Image(file_path="image_4.jpg"), annotation_scene=NullAnnotationSceneEntity())
dataset_3 = DatasetEntity(items=[item_3,item_4])
item_5 = DatasetItemEntity(media=Image(file_path="image_5.jpg"), annotation_scene=NullAnnotationSceneEntity())
item_6 = DatasetItemEntity(media=Image(file_path="image_6.jpg"), annotation_scene=NullAnnotationSceneEntity())
dataset_4 = DatasetEntity(items=[item_5,item_6])
dataset_common = DatasetEntity(items=[item_3,item_4,item_5,item_6])


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestDatasetPurpose:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_datasets_purpose(self):
        """
        <b>Description:</b>
        To test datasets purpose

        <b>Input data:</b>
        Initiated instance of datasets purpose

        <b>Expected results:</b>
        Enum members return correct values:

        INFERENCE = 0
        TRAINING = 1
        EVALUATION = 2
        GENERATING_OUTPUT = 3
        TEMPORARY_DATASET = 4
        TASK_INFERENCE = 5

        <b>Steps</b>
        1. Initiate enum instance
        2. Check members return correct value
        """

        assert DatasetPurpose.INFERENCE.value == 0
        assert DatasetPurpose.TRAINING.value == 1
        assert DatasetPurpose.EVALUATION.value == 2
        assert DatasetPurpose.GENERATING_OUTPUT.value == 3
        assert DatasetPurpose.TEMPORARY_DATASET.value == 4
        assert DatasetPurpose.TASK_INFERENCE.value == 5
        assert len(DatasetPurpose) == 6


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestDatasetEntity:
    de = DatasetEntity()

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_repr(self):
        """
        <b>Description:</b>
        To test representation of dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        Return value is same.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Envolve __repr__ with dataset
        3. __repr__ is same as returning value
        """
        repr = self.de.__repr__()
        assert repr == f"{self.de.__class__.__name__}(items={self.de._items}, purpose={self.de.purpose})"
        repr = dataset.__repr__()
        assert repr == DatasetEntity(items=[item,item_2])

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_str(self):
        """
        <b>Description:</b>
        To test string of dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        Return value is same.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Envolve __str__ with dataset
        3. __str__ is same as returning value
        """
        str_ = self.de.__str__()
        assert str_  == f"{self.de.__class__.__name__}(size={len(self.de)}, purpose={self.de.purpose})"


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_len(self):
        """
        <b>Description:</b>
        To test lenght of dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        Return value is same as items in dataset.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Envolve __len__ with dataset
        3. __len__ is same as items in dataset
        """
        len_ = self.de.__len__()
        assert len_ == len(self.de._items)
        len_ = dataset.__len__()
        assert len_ == 2

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_eq(self):
        """
        <b>Description:</b>
        To test equal of datasets

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        Return value is same as items in dataset.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Envolve __eq__ with dataset
        3. __eq__ is false by default
        4. Add same dataset
        5. Equal 2 datasets is True
        """
        assert self.de.__eq__(dataset) == False
        dataset_2 = DatasetEntity(items=[item,item_2])
        assert dataset_2.__eq__(dataset) == True

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_add(self):
        """
        <b>Description:</b>
        To test add new dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        dataset1 is equal dataset.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Envolve __add__ with dataset
        3. dataset1 is equal with dataset
        """
        dataset_2 = self.de.__add__(dataset_3)
        assert dataset_2.__len__() == 2
        assert dataset_2 == dataset_3
        
        dataset_2 = self.de.__add__(dataset_3).__add__(dataset_4)
        assert dataset_2.__len__() == 4
        assert dataset_2 == dataset_common

        dataset_2 = self.de.__add__([item_3, item_4, item_5, item_6])
        assert dataset_2.__len__() == 4
        assert dataset_2 == dataset_common
        


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_with_empty_annotations(self):
        """
        <b>Description:</b>
        To test add new dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        dataset1 is equal dataset.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Add prediction_dataset dataset with method with_empty_annotations
        3. prediction_dataset is same as parent dataset
        """
        prediction_dataset = dataset.with_empty_annotations()
        assert prediction_dataset.__len__() == 2
        assert prediction_dataset.purpose.value == 0
    
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_remove(self):
        """
        <b>Description:</b>
        To test remove items in dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        Annotations reduced.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Remove item in dataset
        3. Items is reduced
        """
        dataset_common.remove(item_3)
        assert dataset_common.__len__() == 3
        assert dataset_common == DatasetEntity(items=[item_4,item_5,item_6])

        dataset_common.remove(item_4)
        dataset_common.remove(item_5)
        dataset_common.remove(item_6)
        assert dataset_common.__len__() == 0
        assert dataset_common == DatasetEntity()

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_append(self):
        """
        <b>Description:</b>
        To test remove items in dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        Annotations reduced.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Remove item in dataset
        3. Items is reduced
        """
        check_dataset = DatasetEntity(items=[item,item_2,item_3])
        assert dataset.__len__() == 2
        dataset.append(item_3)
        assert dataset.__len__() == 3
        assert dataset == check_dataset

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_get_labels(self):
        """
        <b>Description:</b>
        To test get_labels in dataset

        <b>Input data:</b>
        Initiated instance of DatasetEntity

        <b>Expected results:</b>
        Labels is empty.

        <b>Steps</b>
        1. Initiate DatasetEntity
        2. Invoke get_labels from dataset
        3. labels is empty
        """
        pytest.xfail("Not yet implemented")
