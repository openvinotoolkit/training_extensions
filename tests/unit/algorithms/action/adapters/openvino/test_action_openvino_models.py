"""Unit Test for otx.algorithms.action.adapters.openvino.model_wrappers.openvino_models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict

import numpy as np
import pytest
from mmcv.utils import Config
from openvino.model_api.adapters import OpenvinoAdapter

from otx.algorithms.action.adapters.openvino.model_wrappers.openvino_models import (
    OTXOVActionCls,
    OTXOVActionDet,
    get_multiclass_predictions,
    softmax_numpy,
)
from otx.api.entities.label import Domain
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import (
    generate_action_cls_otx_dataset,
    generate_labels,
)


class MockOpenvinoAdapter(OpenvinoAdapter):
    """Mock class for OpenvinoAdapter."""

    def __init__(self):
        pass


class MockOTXOVActionCls(OTXOVActionCls):
    """Mock class for OTXOVActionCls."""

    def __init__(self, *args):
        self.inputs: Dict[str, np.ndarray] = {
            "cls_data": np.ndarray([1, 1, 3, 8, 256, 256]),
            "det_data": np.ndarray([1, 3, 8, 256, 256]),
            "cls_info": np.ndarray([1, 1, 1, 1]),
            "dummy": np.ndarray([1, 1, 1]),
        }

        self.outputs: Dict[str, Config] = {
            "logits": Config({"names": "cls_layer"}),
            "gt_bboxes": Config({"names": "reg_layer"}),
            "gt_labels": Config({"names": "cls_layer"}),
        }
        super().__init__(MockOpenvinoAdapter)


class MockOTXOVActionDet(OTXOVActionDet):
    """Mock class for OTXOVActionDet."""

    def __init__(self, *args):
        self.inputs: Dict[str, np.ndarray] = {
            "cls_data": np.ndarray([1, 1, 3, 8, 256, 256]),
            "det_data": np.ndarray([1, 3, 8, 256, 256]),
            "cls_info": np.ndarray([1, 1, 1, 1]),
            "dummy": np.ndarray([1, 1, 1]),
        }

        self.outputs: Dict[str, Config] = {
            "logits": Config({"names": "cls_layer"}),
            "gt_bboxes": Config({"names": "reg_layer"}),
            "gt_labels": Config({"names": "cls_layer"}),
        }
        super().__init__(MockOpenvinoAdapter)


@e2e_pytest_unit
def test_softmax_numpy() -> None:
    """Test softmax_numpy function.

    It checks argmax of inputs and outputs
    """

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = softmax_numpy(x)
    assert np.argmax(x) == np.argmax(out)


@e2e_pytest_unit
def test_get_multiclass_predictions(mocker) -> None:
    """Test get_multiclass_predictions function.

    It checks argmax and max of inputs and outputs
    """

    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    outputs = get_multiclass_predictions(inputs, False)
    assert 4 == outputs.top_labels[0][0]
    assert 5.0 == outputs.top_labels[0][1]

    mocker.patch(
        "otx.algorithms.action.adapters.openvino.model_wrappers.openvino_models.softmax_numpy", return_value=inputs
    )
    outputs = get_multiclass_predictions(inputs, False)
    assert 4 == outputs.top_labels[0][0]
    assert 5.0 == outputs.top_labels[0][1]


class TestOTXOVActionCls:
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
            "otx.algorithms.action.adapters.openvino.model_wrappers.openvino_models.Model.__init__",
            return_value=True,
        )
        self.model = MockOTXOVActionCls()

    @e2e_pytest_unit
    def test_init(self, mocker) -> None:
        """Test __init__ function."""
        assert self.model.image_blob_names == ["cls_data"]
        assert self.model.image_info_blob_names == ["cls_info"]
        assert self.model.out_layer_name == "logits"
        assert self.model.n == 1
        assert self.model.c == 3
        assert self.model.t == 8
        assert self.model.h == 256
        assert self.model.w == 256

    @e2e_pytest_unit
    def test_preprocess(self) -> None:
        """Test preprocess function."""

        labels = generate_labels(1, Domain.ACTION_CLASSIFICATION)
        items = generate_action_cls_otx_dataset(1, 10, labels)._items

        dict_inputs, meta = self.model.preprocess(items)
        assert dict_inputs["cls_data"].shape == (1, 1, 3, 10, 256, 256)
        assert meta["original_shape"] == (256, 256, 3)
        assert meta["resized_shape"] == (1, 3, 10, 256, 256)

    @e2e_pytest_unit
    def test_postprocess(self) -> None:
        """Test postprocess function."""

        sample_output = {"logits": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        out = self.model.postprocess(sample_output, meta={"Any": "Any"})
        assert out.top_labels[0][0] == 4


class TestOTXOVActionDet:
    """Test OTXOVActionDet class.

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
        2. Check postprocess function output's id, score, bbox
    """

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch(
            "otx.algorithms.action.adapters.openvino.model_wrappers.openvino_models.Model.__init__",
            return_value=True,
        )
        self.model = MockOTXOVActionDet()

    @e2e_pytest_unit
    def test_init(self, mocker) -> None:
        """Test __init__ function."""
        assert self.model.image_blob_names == ["det_data"]
        assert self.model.out_layer_names == {"bboxes": "gt_bboxes", "labels": "gt_labels"}
        assert self.model.n == 1
        assert self.model.c == 3
        assert self.model.t == 8
        assert self.model.h == 256
        assert self.model.w == 256

    @e2e_pytest_unit
    def test_preprocess(self) -> None:
        """Test preprocess function."""

        labels = generate_labels(1, Domain.ACTION_CLASSIFICATION)
        items = generate_action_cls_otx_dataset(1, 10, labels)._items

        dict_inputs, meta = self.model.preprocess(items)
        assert dict_inputs["det_data"].shape == (1, 3, 10, 256, 256)
        assert meta["original_shape"] == (256, 256, 3)
        assert meta["resized_shape"] == (1, 3, 10, 256, 256)

    @e2e_pytest_unit
    def test_postprocess(self) -> None:
        """Test postprocess function."""

        sample_output = {"gt_bboxes": np.array([[0, 0, 1, 1]]), "gt_labels": np.array([[0.3, 0.2, 0.1, 0.7]])}
        out = self.model.postprocess(sample_output, meta={"original_shape": (256, 256, 3)})
        # argmax index is 2 because first index is for background
        assert out[0].id == 2
        assert out[0].score == 0.7
        assert (out[0].xmin, out[0].ymin, out[0].xmax, out[0].ymax) == (0, 0, 256, 256)
