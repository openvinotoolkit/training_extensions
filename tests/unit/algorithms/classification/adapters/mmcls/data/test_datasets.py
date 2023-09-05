# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.classification.adapters.mmcls.datasets import (
    OTXClsDataset,
    OTXHierarchicalClsDataset,
    OTXMultilabelClsDataset,
    SelfSLDataset,
)
from otx.algorithms.classification.utils import get_multihead_class_info
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    DEFAULT_CLS_TEMPLATE,
    init_environment,
    setup_configurable_parameters,
)
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


def create_cls_dataset():
    image = Image(data=np.random.randint(low=0, high=255, size=(8, 8, 3)).astype(np.uint8))
    annotation = Annotation(
        shape=Rectangle.generate_full_box(),
        labels=[ScoredLabel(LabelEntity(name="test_selfsl_dataset", domain=Domain.CLASSIFICATION))],
    )
    annotation_scene = AnnotationSceneEntity(annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION)
    dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene)

    dataset = DatasetEntity(items=[dataset_item])
    return dataset, dataset.get_labels()


class TestSelfSLDataset:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.otx_dataset, _ = create_cls_dataset()
        self.pipeline = {
            "view0": [dict(type="ImageToTensor", keys=["img"])],
            "view1": [dict(type="ImageToTensor", keys=["img"])],
        }

    @e2e_pytest_unit
    def test_getitem(self):
        """Test __getitem__ method."""
        dataset = SelfSLDataset(otx_dataset=self.otx_dataset, pipeline=self.pipeline)

        data_item = dataset[0]
        for i in range(1, 3):
            assert f"dataset_item{i}" in data_item
            assert f"width{i}" in data_item
            assert f"height{i}" in data_item
            assert f"index{i}" in data_item
            assert f"filename{i}" in data_item
            assert f"ori_filename{i}" in data_item
            assert f"img{i}" in data_item
            assert f"img_shape{i}" in data_item
            assert f"ori_shape{i}" in data_item
            assert f"pad_shape{i}" in data_item
            assert f"img_norm_cfg{i}" in data_item
            assert f"img_fields{i}" in data_item


class TestOTXClsDataset:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.hyper_parameters, self.model_template = setup_configurable_parameters(DEFAULT_CLS_TEMPLATE)
        self.dataset_len = 20

    @pytest.mark.parametrize(
        "adapter_type",
        [OTXClsDataset, OTXMultilabelClsDataset],
    )
    @e2e_pytest_unit
    def test_create_dataset_adapter(self, adapter_type):
        multilabel = adapter_type == OTXClsDataset
        self.task_environment, self.dataset = init_environment(
            self.hyper_parameters, self.model_template, multilabel, False, self.dataset_len
        )
        dataset = adapter_type(self.dataset, labels=self.dataset.get_labels())
        assert dataset.num_classes == len(self.dataset.get_labels())
        assert len(dataset) == self.dataset_len * dataset.num_classes
        for d in dataset:
            assert d is not None

    @pytest.mark.parametrize(
        "adapter_type",
        [OTXClsDataset, OTXMultilabelClsDataset],
    )
    @e2e_pytest_unit
    def test_metric_dataset_adapter(self, adapter_type):
        multilabel = adapter_type == OTXMultilabelClsDataset
        self.task_environment, self.dataset = init_environment(
            self.hyper_parameters, self.model_template, multilabel, False, self.dataset_len
        )
        dataset = adapter_type(self.dataset, labels=self.dataset.get_labels())
        results = np.ones((len(dataset), dataset.num_classes))
        metrics = dataset.evaluate(results)

        assert len(metrics) > 0
        if multilabel:
            metrics["mAP"] > 0
        else:
            metrics["accuracy"] > 0

    @e2e_pytest_unit
    def test_create_hierarchical_adapter(self):
        self.task_environment, self.dataset = init_environment(
            self.hyper_parameters, self.model_template, False, True, self.dataset_len
        )
        class_info = get_multihead_class_info(self.task_environment.label_schema)
        dataset = OTXHierarchicalClsDataset(
            otx_dataset=self.dataset, labels=self.dataset.get_labels(), hierarchical_info=class_info
        )
        assert dataset.num_classes == len(self.dataset.get_labels())
        assert len(dataset) == self.dataset_len * dataset.num_classes
        for d in dataset:
            assert d is not None

    @e2e_pytest_unit
    def test_metric_hierarchical_adapter(self):
        self.task_environment, self.dataset = init_environment(
            self.hyper_parameters, self.model_template, False, True, self.dataset_len
        )
        class_info = get_multihead_class_info(self.task_environment.label_schema)
        dataset = OTXHierarchicalClsDataset(
            otx_dataset=self.dataset, labels=self.dataset.get_labels(), hierarchical_info=class_info
        )
        results = np.zeros((len(dataset), dataset.num_classes))
        metrics = dataset.evaluate(results)

        assert len(metrics) > 0
        assert metrics["accuracy"] > 0

    @e2e_pytest_unit
    def test_hierarchical_with_empty_heads(self):
        self.task_environment, self.dataset = init_environment(
            self.hyper_parameters, self.model_template, False, True, self.dataset_len
        )
        class_info = get_multihead_class_info(self.task_environment.label_schema)
        dataset = OTXHierarchicalClsDataset(
            otx_dataset=self.dataset, labels=self.dataset.get_labels(), hierarchical_info=class_info
        )
        pseudo_gt_labels = []
        pseudo_head_idx = 0
        for label in dataset.gt_labels:
            pseudo_gt_label = label
            pseudo_gt_label[pseudo_head_idx] = -1
            pseudo_gt_labels.append(pseudo_gt_label)
        pseudo_gt_labels = np.array(pseudo_gt_labels)

        from copy import deepcopy

        pseudo_dataset = deepcopy(dataset)
        pseudo_dataset.gt_labels = pseudo_gt_labels
        pseudo_dataset._update_heads_information()
        assert pseudo_dataset.hierarchical_info["empty_multiclass_head_indices"][pseudo_head_idx] == 0
