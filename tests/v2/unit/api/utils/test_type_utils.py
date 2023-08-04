"""OTX V2 API-utils Unit-Test codes (Type Utils)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.type_utils import str_to_subset_type, str_to_task_type, str_to_train_type


class TestTypeUtils:
    def test_str_to_task_type(self) -> None:
        """Test the str_to_task_type function.

        This function tests whether the str_to_task_type function returns the correct TaskType
        enum value for a given string input. It also tests whether the function raises a ValueError
        when an invalid task type string is provided.
        """
        assert str_to_task_type("CLASSIFICATION") == TaskType.CLASSIFICATION
        assert str_to_task_type("classification") == TaskType.CLASSIFICATION
        assert str_to_task_type("DETECTION") == TaskType.DETECTION
        assert str_to_task_type("detection") == TaskType.DETECTION
        with pytest.raises(ValueError, match="is not supported task."):
            str_to_task_type("invalid_task_type")


    def test_str_to_train_type(self) -> None:
        """Test the str_to_train_type function.

        This function tests whether the str_to_train_type function returns the correct TrainType
        enum value for a given string input. It also tests whether the function raises a ValueError
        when an invalid train type string is provided.
        """
        assert str_to_train_type("Incremental") == TrainType.Incremental
        assert str_to_train_type("incremental") == TrainType.Incremental
        assert str_to_train_type("Semisupervised") == TrainType.Semisupervised
        assert str_to_train_type("semisupervised") == TrainType.Semisupervised
        with pytest.raises(ValueError, match="is not supported train type."):
            str_to_train_type("invalid_train_type")


    def test_str_to_subset_type(self) -> None:
        """Test the str_to_subset_type function.

        This function tests whether the str_to_subset_type function returns the correct Subset
        enum value for a given string input. It also tests whether the function raises a ValueError
        when an invalid subset type string is provided.
        """
        assert str_to_subset_type("train") == Subset.TRAINING
        assert str_to_subset_type("training") == Subset.TRAINING
        assert str_to_subset_type("TRAINING") == Subset.TRAINING
        assert str_to_subset_type("val") == Subset.VALIDATION
        assert str_to_subset_type("validation") == Subset.VALIDATION
        assert str_to_subset_type("VALIDATION") == Subset.VALIDATION
        assert str_to_subset_type("test") == Subset.TESTING
        assert str_to_subset_type("testing") == Subset.TESTING
        assert str_to_subset_type("TESTING") == Subset.TESTING
        assert str_to_subset_type("unlabel") == Subset.UNLABELED
        assert str_to_subset_type("unlabeled") == Subset.UNLABELED
        assert str_to_subset_type("UNLABELED") == Subset.UNLABELED
        with pytest.raises(ValueError, match="is not supported subset type."):
            str_to_subset_type("invalid_subset_type")
