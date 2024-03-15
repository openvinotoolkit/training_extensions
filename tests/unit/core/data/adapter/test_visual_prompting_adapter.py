"""Unit-Test case for otx.core.data.adapter.visual_prompting_dataset_adapter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Union

import numpy as np
import pytest

from otx.api.entities.image import Image
from otx.api.entities.model_template import TaskType
from otx.api.entities.shapes.polygon import Polygon
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter
from otx.core.data.adapter.visual_prompting_dataset_adapter import (
    VisualPromptingDatasetAdapter,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)


class TestVisualPromptingDatasetAdapter:
    def setup_method(self):
        self.root_path: str = os.getcwd()
        self.task: str = "visual_prompting"
        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[self.task]

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "data_format, use_mask, expected_shape",
        [
            ("coco", True, Image),
            ("coco", False, Polygon),
            ("voc", True, Image),
            ("voc", False, Polygon),
            ("common_semantic_segmentation", True, Image),
            ("common_semantic_segmentation", False, Polygon),
        ],
    )
    def test_get_otx_dataset(self, data_format: str, use_mask: bool, expected_shape: Union[Image, Polygon]) -> None:
        """Test get_otx_dataset."""
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[self.task][data_format]
        train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        dataset_adapter: VisualPromptingDatasetAdapter = VisualPromptingDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=train_data_roots,
            use_mask=use_mask,
        )

        results = dataset_adapter.get_otx_dataset()

        assert len(results) > 0
        for result in results:
            assert isinstance(result.media, Image)
            assert isinstance(result.media.numpy, np.ndarray)
            for annotation in result.annotation_scene.annotations:
                assert isinstance(annotation.shape, expected_shape)
