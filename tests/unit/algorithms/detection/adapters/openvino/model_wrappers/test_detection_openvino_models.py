"""Unit Test for otx.algorithms.detection.adapters.openvino.model_wrappers.openvino_models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict

import numpy as np
import pytest
from mmcv.utils import Config
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter

from otx.algorithms.detection.adapters.openvino.model_wrappers.openvino_models import (
    BatchBoxesLabelsParser,
    OTXMaskRCNNModel,
    OTXSSDModel,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockOpenvinoAdapter(OpenvinoAdapter):
    """Mock class for OpenvinoAdapter."""

    def __init__(self):
        pass


class MockBatchBoxesLabelsParser(BatchBoxesLabelsParser):
    """Mock class for BatchBoxesLabelsParser."""

    def __init__(self):
        self.labels_layer = "labels"
        self.bboxes_layer = "boxes"
        self.input_size = (10, 10)


class MockOTXMaskRCNNModel(OTXMaskRCNNModel):
    """Mock class for OTXOVActionCls."""

    def __init__(self, *args):
        self.inputs: Dict[str, np.ndarray] = {
            "image": np.ndarray([1, 3, 10, 10]),
        }

        self.outputs: Dict[str, Config] = {
            "boxes": Config({"names": "boxes", "shape": [1, 1, 5]}),
            "labels": Config({"names": "labels", "shape": [1, 1]}),
            "masks": Config({"names": "masks", "shape": [1, 0, 28, 28]}),
            "feature_vector": Config({"names": "feature_vector", "shape": [1, 1, 1, 1]}),
            "saliency_map": Config({"names": "saliency_map", "shape": [1, 1, 1]}),
        }
        self.is_segmentoly = len(self.inputs) == 2
        self.output_blob_name = self._get_outputs()
        self.confidence_threshold = 0.5
        super().__init__(MockOpenvinoAdapter)


class MockOTXSSDModel(OTXSSDModel):
    """Mock class for OTXOVActionCls."""

    def __init__(self, *args):
        self.inputs: Dict[str, np.ndarray] = {
            "image": np.ndarray([1, 3, 10, 10]),
        }

        self.outputs: Dict[str, Config] = {
            "boxes": Config({"names": "boxes", "shape": [1, 1, 5]}),
            "labels": Config({"names": "labels", "shape": [1, 1]}),
            "masks": Config({"names": "masks", "shape": [1, 0, 28, 28]}),
            "feature_vector": Config({"names": "feature_vector", "shape": [1, 1, 1, 1]}),
            "saliency_map": Config({"names": "saliency_map", "shape": [1, 1, 1]}),
        }
        # self.is_segmentoly = len(self.inputs) == 2
        # self.output_blob_name = self._get_outputs()
        self.confidence_threshold = 0.375
        self.resize_type = "standard"
        self.output_parser = MockBatchBoxesLabelsParser()
        super().__init__(MockOpenvinoAdapter)


class TestOTXMaskRCNNModel:
    """Test OTXOVActionCls class.

    1. Test __init__ function
    <Steps>
        1. Check model's input, output name
        2. Check model's input's dimension
    2. Test preprocess function
    <Steps>
        1. Generate sample items: List[DatasetItemEntity]
        2. Check pre-processed inputs
            1. Check inputs' dimension
            2. Check meta information
    3. Test postprocess function
    <Steps>
        1. Generate sample output
        2. Check postprocess function's output return's argmax
    """

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch(
            "openvino.model_zoo.model_api.models.MaskRCNNModel.__init__",
            return_value=True,
        )
        self.model = MockOTXMaskRCNNModel()

    @e2e_pytest_unit
    def test_postprocess(self) -> None:
        """Test postprocess function."""

        sample_output = {
            "boxes": np.random.rand(1, 1, 5),
            "labels": np.random.rand(1, 1),
            "masks": np.random.rand(1, 1, 28, 28),
            "feature_vector": np.random.rand(1, 1, 1, 1),
            "saliency_map": np.random.rand(1, 1, 21),
        }
        sample_meta = {"original_shape": (10, 10, 3), "resized_shape": (5, 5, 3)}
        out = self.model.postprocess(sample_output, meta=sample_meta)
        # (scores, classes, boxes, resized_masks)
        assert len(out) == 4


class TestOTXSSDModel:
    """Test OTXOVActionCls class.

    1. Test __init__ function
    <Steps>
        1. Check model's input, output name
        2. Check model's input's dimension
    2. Test preprocess function
    <Steps>
        1. Generate sample items: List[DatasetItemEntity]
        2. Check pre-processed inputs
            1. Check inputs' dimension
            2. Check meta information
    3. Test postprocess function
    <Steps>
        1. Generate sample output
        2. Check postprocess function's output return's argmax
    """

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch(
            "otx.algorithms.detection.adapters.openvino.model_wrappers.openvino_models.OTXSSDModel.__init__",
            return_value=True,
        )
        self.model = MockOTXSSDModel()

    @e2e_pytest_unit
    def test_postprocess(self) -> None:
        """Test postprocess function."""

        sample_output = {
            "boxes": np.random.rand(1, 1, 5),
            "labels": np.random.rand(1, 1),
            "feature_vector": np.random.rand(1, 1, 1, 1),
            "saliency_map": np.random.rand(1, 1, 21),
        }
        sample_meta = {"original_shape": (10, 10, 3), "resized_shape": (5, 5, 3)}
        self.model.postprocess(sample_output, meta=sample_meta)


class TestBatchBoxesLabelsParser:
    """Test OTXOVActionCls class.

    1. Test __init__ function
    <Steps>
        1. Check model's input, output name
        2. Check model's input's dimension
    2. Test preprocess function
    <Steps>
        1. Generate sample items: List[DatasetItemEntity]
        2. Check pre-processed inputs
            1. Check inputs' dimension
            2. Check meta information
    3. Test postprocess function
    <Steps>
        1. Generate sample output
        2. Check postprocess function's output return's argmax
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.layers: Dict[str, Config] = {
            "boxes": Config({"names": "boxes", "shape": [1, 1, 5]}),
            "labels": Config({"names": "labels", "shape": [1, 1]}),
            "masks": Config({"names": "masks", "shape": [1, 0, 28, 28]}),
            "feature_vector": Config({"names": "feature_vector", "shape": [1, 1, 1, 1]}),
            "saliency_map": Config({"names": "saliency_map", "shape": [1, 1, 1]}),
        }
        input_size = (10, 10)
        self.parser = BatchBoxesLabelsParser(self.layers, input_size)

    @e2e_pytest_unit
    def test_init(self) -> None:
        assert hasattr(self.parser, "bboxes_layer")
        assert hasattr(self.parser, "input_size")

    @e2e_pytest_unit
    def test_layer_bboxes_output(self) -> None:
        """Test postprocess function."""
        self.parser.find_layer_bboxes_output(self.layers)
