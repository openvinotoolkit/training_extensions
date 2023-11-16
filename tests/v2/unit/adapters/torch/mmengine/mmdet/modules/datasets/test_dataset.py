"""Unit tests of otx/v2/adapters/torch/mmengine/mmdet/modules/datasets/dataset.py."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.v2.adapters.torch.mmengine.mmdet.modules.datasets.dataset import OTXDetDataset
from otx.api.entities.model_template import TaskType
from tests.v2.unit.adapters.torch.mmengine.mmdet.test_helpers import (
    generate_det_dataset,
)


class TestOTXDetDataset:
    """
    Test OTXDetDataset class.
    1. Test _DataInfoProxy
    2. Test prepare_train_img
    3. Test prepare_test_img
    4. Test get_ann_info
    5. Test evaluate
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.dataset = dict()
        for task_type in [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            self.dataset[task_type] = generate_det_dataset(task_type=task_type)
        self.pipeline = []

    @pytest.mark.parametrize("task_type", [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION])
    def test_DataInfoProxy(self, task_type):
        """Test _DataInfoProxy Class."""
        otx_dataset, labels = self.dataset[task_type]
        proxy = OTXDetDataset._DataInfoProxy(otx_dataset, labels)
        sample = proxy[0]
        assert len(otx_dataset) == len(proxy)
        assert "dataset_item" in sample
        assert "index" in sample
        assert "ann_info" in sample
        assert "ignored_labels" in sample

    @pytest.mark.parametrize("task_type", [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION])
    def test_init(self, mocker, task_type):
        """Test initialization of OTXDetDataset."""
        otx_dataset, labels = self.dataset[task_type]
        mock_pipeline = [{"type": "MockPipeline"}]
        mocker.patch(
            "otx.v2.adapters.torch.mmengine.mmdet.modules.datasets.dataset.Compose",
            return_value=True,
        )

        dataset = OTXDetDataset(otx_dataset, labels, mock_pipeline)
        assert len(dataset) == len(otx_dataset)
