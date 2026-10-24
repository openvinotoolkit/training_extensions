# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the OpenVINO base model."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import openvino as ov
import pytest
import torch
from model_api.adapters.inference_adapter import Metadata
from model_api.models.result import ClassificationResult

from getitune.backend.openvino.models import OVModel
from getitune.data.entity.sample import SampleBatch

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestOVModel:
    @pytest.fixture
    def input_batch(self) -> SampleBatch:
        image = [torch.rand(3, 10, 10) for _ in range(3)]
        return SampleBatch(images=torch.stack(image), labels=[])

    @pytest.fixture
    def model(self, get_dummy_ov_cls_model) -> OVModel:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ov.save_model(get_dummy_ov_cls_model, f"{tmp_dir}/model.xml")
            return OVModel(model_path=f"{tmp_dir}/model.xml", model_type="Classification")

    def test_create_model(self, model) -> None:
        pass

    def test_customize_inputs(self, model, input_batch) -> None:
        inputs = model._customize_inputs(input_batch)
        assert isinstance(inputs, dict)
        assert "inputs" in inputs
        assert inputs["inputs"][1].shape == np.transpose(input_batch.images[1].numpy(), (1, 2, 0)).shape

    def test_forward(self, model, input_batch, mocker: MockerFixture) -> None:
        model._customize_outputs = lambda x, _: x
        model.model.postprocess = mocker.Mock(return_value=ClassificationResult())
        outputs = model.forward(input_batch)
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert isinstance(outputs[2], ClassificationResult)

    def test_dummy_input(self, model: OVModel):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size


class TestResolveModelType:
    """Tests for OVModel.model_type metadata resolution."""

    @pytest.fixture
    def ov_model_instance(self, get_dummy_ov_cls_model) -> OVModel:
        """Create a minimal OVModel for testing model_type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ov.save_model(get_dummy_ov_cls_model, f"{tmp_dir}/model.xml")
            return OVModel(model_path=f"{tmp_dir}/model.xml", model_type="Classification")

    def test_returns_rt_info_model_type_when_present(self, ov_model_instance: OVModel) -> None:
        """When rt_info has model_type, model_type should return it."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = "YOLO11"
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "YOLO11"
        mock_adapter.model.has_rt_info.assert_called_once_with(["model_info", "model_type"])

    def test_returns_class_default_when_no_rt_info(self, ov_model_instance: OVModel) -> None:
        """When rt_info has no model_type, should fall back to constructor default."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = False

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "Classification"

    def test_returns_class_default_when_rt_info_empty(self, ov_model_instance: OVModel) -> None:
        """When rt_info model_type is empty string, should fall back to constructor default."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = ""
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "Classification"

    def test_returns_same_type_without_logging_when_matching(self, ov_model_instance: OVModel) -> None:
        """When rt_info model_type matches class default, should return it without override log."""
        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = "Classification"
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model_instance._model_adapter = mock_adapter
        result = ov_model_instance.model_type
        assert result == "Classification"

    def test_ssd_default_overridden_by_yolo11(self) -> None:
        """Simulates the OVDetectionModel case: default 'SSD' overridden by 'YOLO11' from rt_info."""
        ov_model = object.__new__(OVModel)
        ov_model._model_type = "SSD"
        ov_model.model_path = "model.xml"

        mock_adapter = MagicMock()
        mock_adapter.model.has_rt_info.return_value = True

        mock_rt_value = MagicMock()
        mock_rt_value.value = "YOLO11"
        mock_adapter.model.get_rt_info.return_value = mock_rt_value

        ov_model._model_adapter = mock_adapter
        result = ov_model.model_type
        assert result == "YOLO11"


class TestMapCompiledOutputKeys:
    """Tests for mapping externally compiled model outputs to ModelAPI output keys."""

    @staticmethod
    def _wrapper(outputs: dict) -> MagicMock:
        model = MagicMock()
        model.outputs = outputs
        return model

    @staticmethod
    def _compiled(names: list[set[str]]) -> MagicMock:
        compiled_model = MagicMock()
        compiled_model.outputs = []
        for output_names in names:
            output = MagicMock()
            output.get_names.return_value = output_names
            compiled_model.outputs.append(output)
        return compiled_model

    def test_maps_named_outputs_by_name(self) -> None:
        wrapper = self._wrapper(
            {"boxes": Metadata(names={"boxes"}), "labels": Metadata(names={"labels"})},
        )
        # NNCF may return the outputs in a different order than the wrapper.
        compiled_model = self._compiled([{"labels"}, {"boxes"}])

        assert OVModel._map_compiled_output_keys(wrapper, compiled_model) == ["labels", "boxes"]

    def test_maps_unnamed_outputs_positionally(self) -> None:
        """Ultralytics YOLO-seg exports have unnamed output tensors, keyed by port object."""
        det_port, proto_port = object(), object()
        wrapper = self._wrapper({det_port: Metadata(names=set()), proto_port: Metadata(names=set())})
        compiled_model = self._compiled([set(), set()])

        assert OVModel._map_compiled_output_keys(wrapper, compiled_model) == [det_port, proto_port]

    def test_maps_unknown_names_positionally(self) -> None:
        wrapper = self._wrapper({"boxes": Metadata(names={"boxes"}), "labels": Metadata(names={"labels"})})
        compiled_model = self._compiled([{"renamed_by_nncf"}, {"labels"}])

        assert OVModel._map_compiled_output_keys(wrapper, compiled_model) == ["boxes", "labels"]


class TestSelectPrimaryMetric:
    """Tests for choosing the accuracy indicator of accuracy-aware quantization."""

    def test_returns_first_scalar_tensor(self) -> None:
        results = {"map": torch.tensor(0.75), "map_50": torch.tensor(0.9)}

        assert OVModel._select_primary_metric(results) == pytest.approx(0.75)

    def test_returns_plain_number(self) -> None:
        assert OVModel._select_primary_metric({"Dice": 0.5}) == pytest.approx(0.5)

    def test_skips_leading_non_scalar_entries(self) -> None:
        """The multi-label classification metric emits confusion matrices first."""
        results = {
            "conf_matrix": [torch.zeros(2, 2), torch.zeros(2, 2)],
            "accuracy": torch.tensor(0.8),
            "map": torch.tensor(0.6),
        }

        assert OVModel._select_primary_metric(results) == pytest.approx(0.8)

    def test_skips_multi_element_tensors(self) -> None:
        results = {"map_per_class": torch.tensor([0.1, 0.2]), "accuracy": torch.tensor(0.4)}

        assert OVModel._select_primary_metric(results) == pytest.approx(0.4)

    def test_raises_when_no_scalar_metric(self) -> None:
        with pytest.raises(RuntimeError, match="No scalar metric"):
            OVModel._select_primary_metric({"conf_matrix": [torch.zeros(2, 2)]})
