# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import onnx
import openvino as ov
import pytest
from onnx import TensorProto, helper
from pytest_mock import MockerFixture

from getitune.backend.openvino.engine import OVEngine
from getitune.backend.openvino.models import OVModel, OVMultilabelClassificationModel
from getitune.types.label import NullLabelInfo


@pytest.fixture
def fxt_ov_model(tmp_path, get_dummy_ov_cls_model) -> OVModel:
    ov.save_model(get_dummy_ov_cls_model, f"{tmp_path}/model.xml")
    return OVMultilabelClassificationModel(model_path=f"{tmp_path}/model.xml")


@pytest.fixture
def fxt_onnx_model_path(tmp_path) -> str:
    """Create a minimal ONNX model with getitune metadata embedded."""
    # Create a minimal ONNX model (identity)
    x_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    y_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "test_graph", [x_input], [y_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    # Embed getitune metadata (same format as _embed_onnx_metadata)
    metadata = {
        ("model_info", "task_type"): "detection",
        ("model_info", "model_type"): "SSD",
        ("model_info", "model_name"): "test_model",
        ("model_info", "multilabel"): "False",
        ("model_info", "hierarchical"): "False",
    }
    for k, v in metadata.items():
        meta = model.metadata_props.add()
        meta.key = " ".join(map(str, k))
        meta.value = str(v)

    onnx_path = f"{tmp_path}/model.onnx"
    onnx.save(model, onnx_path)
    return onnx_path


@pytest.fixture
def fxt_onnx_classification_model_path(tmp_path) -> str:
    """Create a minimal ONNX model with classification metadata."""
    x_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    y_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "test_graph", [x_input], [y_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    metadata = {
        ("model_info", "task_type"): "classification",
        ("model_info", "model_type"): "Classification",
        ("model_info", "model_name"): "test_cls_model",
        ("model_info", "multilabel"): "True",
        ("model_info", "hierarchical"): "False",
    }
    for k, v in metadata.items():
        meta = model.metadata_props.add()
        meta.key = " ".join(map(str, k))
        meta.value = str(v)

    onnx_path = f"{tmp_path}/cls_model.onnx"
    onnx.save(model, onnx_path)
    return onnx_path


@pytest.fixture
def fxt_engine(tmp_path, fxt_ov_model) -> OVEngine:
    data_root = "tests/assets/multilabel_classification_coco/"

    return OVEngine(
        data=data_root,
        work_dir=tmp_path,
        model=fxt_ov_model,
    )


class TestEngine:
    def test_constructor(self, mocker, tmp_path, fxt_ov_model) -> None:
        data_root = "tests/assets/multilabel_classification_coco/"
        engine = OVEngine(data=data_root, model=fxt_ov_model, work_dir=tmp_path)
        assert engine.datamodule.task == "MULTI_LABEL_CLS"
        assert isinstance(engine.model, OVModel)

        # init with xml path, test automatic model creation
        mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=fxt_ov_model)
        engine = OVEngine(work_dir=tmp_path, data=data_root, model=f"{tmp_path}/model.xml")
        assert engine.model == fxt_ov_model

    def test_constructor_with_onnx_path(
        self, mocker, tmp_path, fxt_ov_model, fxt_onnx_classification_model_path
    ) -> None:
        """Test that OVEngine can be initialized with an ONNX model path."""
        data_root = "tests/assets/multilabel_classification_coco/"
        mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=fxt_ov_model)
        engine = OVEngine(work_dir=tmp_path, data=data_root, model=fxt_onnx_classification_model_path)
        assert engine.model == fxt_ov_model

    def test_constructor_invalid_format(self, tmp_path) -> None:
        """Test that OVEngine raises ValueError for unsupported model formats."""
        data_root = "tests/assets/multilabel_classification_coco/"
        with pytest.raises(ValueError, match="Please provide a valid OpenVINO model"):
            OVEngine(work_dir=tmp_path, data=data_root, model="path/to/model.pth")

    def test_test(self, fxt_engine, mocker: MockerFixture) -> None:
        mocker.patch(
            "getitune.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_get_ov_model = mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model")
        fxt_engine._derive_task_from_ir = MagicMock(return_value="MULTI_LABEL_CLS")
        mock_model = MagicMock()
        mock_get_ov_model.return_value = mock_model
        fxt_engine._model = fxt_engine._auto_configurator.get_ov_model("model.xml")

        # Mock the dataloader to avoid actual data processing
        mock_dataloader = MagicMock()
        mock_batch = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = MagicMock(return_value=1)
        mocker.patch.object(fxt_engine.datamodule, "test_dataloader", return_value=mock_dataloader)

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        mock_model.prepare_metric_inputs = mocker.MagicMock(return_value={"preds": [1], "target": [1]})
        mock_model.compute_metrics = mocker.MagicMock(return_value={})
        fxt_engine.test(metric=MagicMock())

        mock_model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a test pipeline, the label information should be same (.*)",
        ):
            fxt_engine.test()

    @pytest.mark.parametrize("explain", [True, False])
    def test_predict(self, fxt_engine, tmp_path, explain, mocker: MockerFixture) -> None:
        checkpoint = f"{tmp_path}/model.xml"
        _ = mocker.patch(
            "getitune.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_process_saliency_maps = mocker.patch(
            "getitune.backend.lightning.models.utils.xai_utils.process_saliency_maps_in_pred_entity",
        )
        fxt_engine._derive_task_from_ir = MagicMock(return_value="MULTI_LABEL_CLS")
        mock_model = MagicMock()
        mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=mock_model)
        fxt_engine._model = fxt_engine._auto_configurator.get_ov_model("model.xml")

        # Mock the dataloader to avoid actual data processing
        mock_dataloader = MagicMock()
        mock_batch = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = MagicMock(return_value=1)
        mocker.patch.object(fxt_engine.datamodule, "test_dataloader", return_value=mock_dataloader)

        # Correct label_info from the checkpoint
        fxt_engine._model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(explain=explain)
        assert mock_process_saliency_maps.called == explain

        fxt_engine._model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a predict pipeline, the label information should be same (.*)",
        ):
            fxt_engine.predict(checkpoint=checkpoint)

    def test_predict_propagates_tile_config(self, fxt_engine, mocker: MockerFixture) -> None:
        """predict() must propagate datamodule.tile_config to the model, like test() does.

        Without this, tiled predictions route through OVModel.forward_tiles using the model's
        default tile_config, diverging from evaluation (test) for tile-merge parameters.
        """
        from getitune.config.data import TileConfig

        mocker.patch(
            "getitune.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_model = MagicMock()
        mock_model.label_info = fxt_engine.datamodule.label_info
        mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=mock_model)
        fxt_engine._model = mock_model

        # Attach a tiling configuration to the datamodule.
        tile_config = TileConfig(enable_tiler=True, tile_size=(200, 200))
        fxt_engine.datamodule.tile_config = tile_config

        # Mock the dataloader to avoid actual data processing.
        mock_dataloader = MagicMock()
        mock_batch = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = MagicMock(return_value=1)
        mocker.patch.object(fxt_engine.datamodule, "test_dataloader", return_value=mock_dataloader)

        fxt_engine.predict()

        # The model's tile_config should now match the datamodule's tile_config.
        assert mock_model.tile_config == tile_config

    def test_predict_with_onnx_checkpoint(self, fxt_engine, mocker, fxt_onnx_model_path) -> None:
        """Test that predict accepts ONNX checkpoints."""
        mocker.patch(
            "getitune.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_model = MagicMock()
        mock_model.label_info = fxt_engine.datamodule.label_info
        mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=mock_model)

        # Mock the dataloader
        mock_dataloader = MagicMock()
        mock_batch = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = MagicMock(return_value=1)
        mocker.patch.object(fxt_engine.datamodule, "test_dataloader", return_value=mock_dataloader)

        fxt_engine.predict(checkpoint=fxt_onnx_model_path)

    def test_optimizing_model(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="OVEngine supports only"):
            fxt_engine.optimize(checkpoint="path/to/model.pth")

        mocker.patch(
            "getitune.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_ov_model = mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model")
        mock_model = MagicMock()
        mock_ov_model.return_value = mock_model
        fxt_engine._derive_task_from_ir = MagicMock(return_value="MULTI_LABEL_CLS")

        # Fetch Checkpoint
        fxt_engine.optimize(checkpoint="path/to/exported_model.xml")
        mock_ov_model.assert_called_once()
        mock_ov_model.return_value.optimize.assert_called_once()

        # With max_data_subset_size
        fxt_engine.optimize(max_data_subset_size=100, checkpoint="path/to/exported_model.xml")
        assert mock_ov_model.return_value.optimize.call_args[0][2]["subset_size"] == 100

        # With max_drop and max_num_iterations (accuracy-aware quantization)
        fxt_engine.optimize(max_drop=0.01, max_num_iterations=5, checkpoint="path/to/exported_model.xml")
        ptq_config = mock_ov_model.return_value.optimize.call_args[0][2]
        assert ptq_config["max_drop"] == 0.01
        assert ptq_config["max_num_iterations"] == 5

        # Without max_num_iterations it should not be part of the PTQ config
        fxt_engine.optimize(max_drop=0.01, checkpoint="path/to/exported_model.xml")
        assert "max_num_iterations" not in mock_ov_model.return_value.optimize.call_args[0][2]

    def test_optimize_rejects_onnx(self, fxt_engine, fxt_onnx_model_path) -> None:
        """Test that optimize() raises RuntimeError for ONNX models."""
        with pytest.raises(RuntimeError, match="does not support ONNX models"):
            fxt_engine.optimize(checkpoint=fxt_onnx_model_path)

    def test_optimize_rejects_loaded_onnx_model(self, fxt_engine, fxt_onnx_model_path) -> None:
        """Test that optimize() raises RuntimeError when engine holds an ONNX model."""
        mock_model = MagicMock()
        mock_model.model_path = fxt_onnx_model_path
        fxt_engine._model = mock_model
        with pytest.raises(RuntimeError, match="does not support ONNX models"):
            fxt_engine.optimize()


class TestONNXSupport:
    """Tests for ONNX model support in OVEngine."""

    def test_is_supported_onnx(self, tmp_path) -> None:
        """Test that is_supported returns True for .onnx path with valid data."""
        from getitune.data.module import DataModule

        mock_datamodule = MagicMock(spec=DataModule)
        nonexistent_data_dir = tmp_path / "nonexistent_data_dir"
        assert OVEngine.is_supported("path/to/model.onnx", mock_datamodule) is True
        assert OVEngine.is_supported("path/to/model.onnx", str(nonexistent_data_dir)) is False

    def test_is_supported_xml(self) -> None:
        """Test that is_supported still works for .xml paths."""
        from getitune.data.module import DataModule

        mock_datamodule = MagicMock(spec=DataModule)
        assert OVEngine.is_supported("path/to/model.xml", mock_datamodule) is True

    def test_is_supported_unsupported_format(self) -> None:
        """Test that is_supported returns False for unsupported formats."""
        from getitune.data.module import DataModule

        mock_datamodule = MagicMock(spec=DataModule)
        assert OVEngine.is_supported("path/to/model.pth", mock_datamodule) is False
        assert OVEngine.is_supported("path/to/model.bin", mock_datamodule) is False

    def test_derive_task_from_onnx_detection(self, fxt_onnx_model_path) -> None:
        """Test task derivation from ONNX model with detection metadata."""
        from getitune.types import TaskType

        engine = OVEngine.__new__(OVEngine)
        task = engine._derive_task_from_onnx(fxt_onnx_model_path)
        assert task == TaskType.DETECTION

    def test_derive_task_from_onnx_classification_multilabel(self, fxt_onnx_classification_model_path) -> None:
        """Test task derivation from ONNX model with multilabel classification metadata."""
        from getitune.types import TaskType

        engine = OVEngine.__new__(OVEngine)
        task = engine._derive_task_from_onnx(fxt_onnx_classification_model_path)
        assert task == TaskType.MULTI_LABEL_CLS

    def test_derive_task_from_onnx_missing_metadata(self, tmp_path) -> None:
        """Test that _derive_task_from_onnx raises ValueError for models without metadata."""
        # Create ONNX model without metadata
        x_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        y_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
        graph = helper.make_graph([node], "test_graph", [x_input], [y_output])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx_path = f"{tmp_path}/no_meta.onnx"
        onnx.save(model, onnx_path)

        engine = OVEngine.__new__(OVEngine)
        with pytest.raises(ValueError, match="No 'task_type' found in ONNX model metadata"):
            engine._derive_task_from_onnx(onnx_path)

    def test_derive_task_from_model_dispatches_correctly(self, tmp_path, fxt_onnx_model_path, mocker) -> None:
        """Test that _derive_task_from_model dispatches to correct method based on suffix."""
        from getitune.types import TaskType

        engine = OVEngine.__new__(OVEngine)
        OVEngine._SUPPORTED_MODEL_SUFFIXES = [".xml", ".onnx"]

        # ONNX dispatch
        task = engine._derive_task_from_model(fxt_onnx_model_path)
        assert task == TaskType.DETECTION

        # Unsupported format
        with pytest.raises(ValueError, match="Unsupported model format"):
            engine._derive_task_from_model("model.pth")

    def test_update_checkpoint_onnx(self, fxt_engine, mocker, fxt_onnx_model_path) -> None:
        """Test that _update_checkpoint accepts ONNX paths."""
        mock_model = MagicMock()
        mocker.patch("getitune.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=mock_model)

        result = fxt_engine._update_checkpoint(fxt_onnx_model_path)
        assert result == mock_model

    def test_update_checkpoint_rejects_unsupported(self, fxt_engine) -> None:
        """Test that _update_checkpoint rejects unsupported formats."""
        with pytest.raises(RuntimeError, match="OVEngine supports only"):
            fxt_engine._update_checkpoint("path/to/model.pth")
