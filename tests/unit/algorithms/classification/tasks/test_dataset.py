# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.classification.adapters.mmcls.data import (
    MPAClsDataset,
    MPAHierarchicalClsDataset,
    MPAMultilabelClsDataset,
)
from otx.algorithms.classification.utils import get_multihead_class_info
from tests.integration.api.classification.test_api_classification import (
    DEFAULT_CLS_TEMPLATE_DIR,
    ClassificationTaskAPIBase,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOTXClsDataset:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.hyper_parameters, self.model_template = ClassificationTaskAPIBase.setup_configurable_parameters(
            DEFAULT_CLS_TEMPLATE_DIR
        )
        self.dataset_len = 20

    @pytest.mark.parametrize(
        "adapter_type",
        [MPAClsDataset, MPAMultilabelClsDataset],
    )
    @e2e_pytest_unit
    def test_create_dataset_adapter(self, adapter_type):
        multilabel = adapter_type == MPAClsDataset
        self.task_environment, self.dataset = ClassificationTaskAPIBase.init_environment(
            self.hyper_parameters, self.model_template, multilabel, False, self.dataset_len
        )
        dataset = adapter_type(self.dataset, labels=self.dataset.get_labels())
        assert dataset.num_classes == len(self.dataset.get_labels())
        assert len(dataset) == self.dataset_len * dataset.num_classes
        for d in dataset:
            assert d is not None

    @pytest.mark.parametrize(
        "adapter_type",
        [MPAClsDataset, MPAMultilabelClsDataset],
    )
    @e2e_pytest_unit
    def test_metric_dataset_adapter(self, adapter_type):
        multilabel = adapter_type == MPAMultilabelClsDataset
        self.task_environment, self.dataset = ClassificationTaskAPIBase.init_environment(
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
        self.task_environment, self.dataset = ClassificationTaskAPIBase.init_environment(
            self.hyper_parameters, self.model_template, False, True, self.dataset_len
        )
        class_info = get_multihead_class_info(self.task_environment.label_schema)
        dataset = MPAHierarchicalClsDataset(
            otx_dataset=self.dataset, labels=self.dataset.get_labels(), hierarchical_info=class_info
        )
        assert dataset.num_classes == len(self.dataset.get_labels())
        assert len(dataset) == self.dataset_len * dataset.num_classes
        for d in dataset:
            assert d is not None

    @e2e_pytest_unit
    def test_metric_hierarchical_adapter(self):
        self.task_environment, self.dataset = ClassificationTaskAPIBase.init_environment(
            self.hyper_parameters, self.model_template, False, True, self.dataset_len
        )
        class_info = get_multihead_class_info(self.task_environment.label_schema)
        dataset = MPAHierarchicalClsDataset(
            otx_dataset=self.dataset, labels=self.dataset.get_labels(), hierarchical_info=class_info
        )

        results = np.zeros((len(dataset), dataset.num_classes))
        metrics = dataset.evaluate(results)

        assert len(metrics) > 0
        assert metrics["accuracy"] > 0
