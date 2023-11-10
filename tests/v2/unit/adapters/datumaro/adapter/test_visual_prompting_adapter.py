"""Unit-Test case for otx.core.data.adapter.visual_prompting_dataset_adapter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import TYPE_CHECKING, Union
from datumaro.components.annotation import Mask, Polygon
from datumaro.components.media import Image

import numpy as np
import pytest
from otx.v2.adapters.datumaro.adapter.visual_prompting_dataset_adapter import (
    VisualPromptingDatasetAdapter,
)

from tests.v2.unit.adapters.datumaro.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)

if TYPE_CHECKING:
    from otx.v2.api.entities.task_type import TaskType


class TestVisualPromptingDatasetAdapter:
    def setup_method(self) -> None:
        self.root_path: str = Path.cwd()
        self.task: str = "visual_prompting"
        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[self.task]

    @pytest.mark.parametrize(
        ("data_format", "use_mask", "expected_shape"),
        [
            ("coco", True, Mask),
            ("coco", False, Polygon),
            ("voc", True, Mask),
            ("voc", False, Polygon),
        ],
    )
    def test_get_otx_dataset(self, data_format: str, use_mask: bool, expected_shape: Union[Image, Polygon]) -> None:
        """Test get_otx_dataset."""
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[self.task][data_format]
        train_data_roots: str = str(self.root_path / data_root_dict["train"])
        dataset_adapter: VisualPromptingDatasetAdapter = VisualPromptingDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=train_data_roots,
            use_mask=use_mask,
        )

        _ = dataset_adapter.get_label_schema()
        datasets = dataset_adapter.get_otx_dataset()

        assert len(datasets) > 0
        for _, dataset in datasets.items():
            for item in dataset:
                assert isinstance(item.media, Image)
                assert isinstance(item.media.data, np.ndarray)
                for annotation in item.annotations:
                    assert isinstance(annotation, expected_shape)
