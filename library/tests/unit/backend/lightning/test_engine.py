# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from getitune.backend.lightning.engine import LightningEngine
from getitune.backend.lightning.models.base import DataInputParams, LightningModel
from getitune.backend.lightning.models.classification.multiclass_models import EfficientNetMulticlassCls
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision


@pytest.fixture
def fxt_engine(tmp_path) -> LightningEngine:
    return LightningEngine(
        data="tests/assets/classification_cifar10",
        model="src/getitune/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml",
        work_dir=tmp_path,
        max_epochs=9,
    )


class TestEngine:
    def test_constructor(self, tmp_path) -> None:
        # Check auto-configuration
        data_root = "tests/assets/classification_cifar10"
        engine = LightningEngine(
            work_dir=tmp_path,
            data=data_root,
            model="src/getitune/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
        )
        assert engine.task == "MULTI_CLASS_CLS"
        assert engine.datamodule.task == "MULTI_CLASS_CLS"
        assert isinstance(engine.model, EfficientNetMulticlassCls)

        assert "default_root_dir" in engine.trainer_params
        assert engine.trainer_params["default_root_dir"] == tmp_path
        assert "accelerator" in engine.trainer_params
        assert engine.trainer_params["accelerator"] == "auto"
        assert "devices" in engine.trainer_params
        assert engine.trainer_params["devices"] == 1

    def test_model_init(self, tmp_path, mocker):
        data_root = "tests/assets/classification_cifar10"
        mock_datamodule = MagicMock()
        mock_datamodule.label_info = 4321
        mock_datamodule.input_size = (1234, 1234)
        mock_datamodule.input_mean = (0.0, 0.0, 0.0)
        mock_datamodule.input_std = (1.0, 1.0, 1.0)
        mock_datamodule.input_intensity_config = None
        mock_datamodule.task = "MULTI_CLASS_CLS"

        mocker.patch(
            "getitune.tools.auto_configurator.AutoConfigurator.get_datamodule",
            return_value=mock_datamodule,
        )
        engine = LightningEngine(
            work_dir=tmp_path,
            data=data_root,
            model="src/getitune/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
        )

        assert engine._model.data_input_params == DataInputParams((1234, 1234), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert engine._model.label_info.num_classes == 4321

    def test_training_with_override_args(self, fxt_engine, mocker) -> None:
        mocker.patch("getitune.backend.lightning.engine.shutil.copy2")
        mocker.patch("getitune.backend.lightning.engine.Trainer.fit")
        mock_seed_everything = mocker.patch("getitune.backend.lightning.engine.seed_everything")

        assert fxt_engine._cache.args["max_epochs"] == 9

        fxt_engine.train(max_epochs=5, seed=1234)
        assert fxt_engine._cache.args["max_epochs"] == 5
        mock_seed_everything.assert_called_once_with(1234, workers=True)

    @pytest.mark.parametrize("resume", [True, False])
    def test_training_with_checkpoint(self, fxt_engine, resume: bool, mocker: MockerFixture, tmpdir) -> None:
        checkpoint = "path/to/checkpoint.ckpt"

        mock_trainer = mocker.patch("getitune.backend.lightning.engine.Trainer")
        mock_trainer.return_value.default_root_dir = Path(tmpdir)
        mock_trainer_fit = mock_trainer.return_value.fit

        mock_chkpt_load = mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})

        trained_checkpoint = Path(tmpdir) / "best.ckpt"
        trained_checkpoint.touch()
        mock_trainer.return_value.checkpoint_callback.best_model_path = trained_checkpoint

        fxt_engine.train(resume=resume, checkpoint=checkpoint)

        if resume:
            assert mock_trainer_fit.call_args.kwargs.get("ckpt_path") == checkpoint
        else:
            assert "ckpt_path" not in mock_trainer_fit.call_args.kwargs

            mock_chkpt_load.assert_called_once()

    def test_test(self, fxt_engine, mocker: MockerFixture) -> None:
        checkpoint = "path/to/checkpoint.ckpt"
        mock_test = mocker.patch("getitune.backend.lightning.engine.Trainer.test")
        _ = mocker.patch("getitune.backend.lightning.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})

        mock_model = mocker.create_autospec(LightningModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)
        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.test(checkpoint=checkpoint)
        mock_test.assert_called_once()

    @pytest.mark.parametrize("explain", [True, False])
    def test_predict(self, fxt_engine, explain, mocker: MockerFixture) -> None:
        checkpoint = "path/to/checkpoint.ckpt"
        mock_predict = mocker.patch("getitune.backend.lightning.engine.Trainer.predict")
        _ = mocker.patch("getitune.backend.lightning.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mock_process_saliency_maps = mocker.patch(
            "getitune.backend.lightning.models.utils.xai_utils.process_saliency_maps_in_pred_entity",
        )

        mock_model = mocker.create_autospec(LightningModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(checkpoint=checkpoint, explain=explain)
        mock_predict.assert_called_once()
        assert mock_process_saliency_maps.called == explain

    def test_exporting(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="To make export, checkpoint must be specified."):
            fxt_engine.export()

        mock_export = mocker.patch("getitune.backend.lightning.engine.LightningModel.export")

        mock_load_from_checkpoint = mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=fxt_engine.model)

        # Fetch Checkpoint
        checkpoint = "path/to/checkpoint.ckpt"
        fxt_engine.checkpoint = checkpoint
        fxt_engine.export()
        mock_load_from_checkpoint.assert_called_once()
        mock_export.assert_called_once_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=ExportFormat.OPENVINO,
            precision=Precision.FP32,
        )

        fxt_engine.export(export_precision=Precision.FP16)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=ExportFormat.OPENVINO,
            precision=Precision.FP16,
        )

        fxt_engine.export(export_format=ExportFormat.ONNX)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=ExportFormat.ONNX,
            precision=Precision.FP32,
        )

        fxt_engine.export(export_format=ExportFormat.ONNX, export_demo_package=True)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=ExportFormat.ONNX,
            precision=Precision.FP32,
        )

    @pytest.mark.parametrize("export_nms", [False, True])
    def test_export_forwards_nms_option(self, fxt_engine, export_nms, mocker) -> None:
        fxt_engine.checkpoint = "path/to/checkpoint.ckpt"
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mocker.patch.object(fxt_engine.model, "load_state_dict")
        observed_export_nms = []

        def export_model(**_kwargs) -> Path:
            observed_export_nms.append(fxt_engine.model.export_nms)
            return Path(fxt_engine.work_dir) / "exported_model.xml"

        mocker.patch.object(fxt_engine.model, "export", side_effect=export_model)
        fxt_engine.model.export_nms = False

        fxt_engine.export(export_nms=export_nms)

        assert observed_export_nms == [export_nms]
        assert not fxt_engine.model.export_nms

    def test_export_defaults_to_nms_disabled(self, fxt_engine, mocker) -> None:
        fxt_engine.checkpoint = "path/to/checkpoint.ckpt"
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mocker.patch.object(fxt_engine.model, "load_state_dict")
        observed_export_nms = []
        fxt_engine.model.export_nms = True

        def export_model(**_kwargs) -> Path:
            observed_export_nms.append(fxt_engine.model.export_nms)
            return Path(fxt_engine.work_dir) / "exported_model.xml"

        mocker.patch.object(fxt_engine.model, "export", side_effect=export_model)

        fxt_engine.export()

        assert observed_export_nms == [False]
        assert fxt_engine.model.export_nms

    def test_export_restores_nms_option_when_export_fails(self, fxt_engine, mocker) -> None:
        fxt_engine.checkpoint = "path/to/checkpoint.ckpt"
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mocker.patch.object(fxt_engine.model, "load_state_dict")
        mocker.patch.object(fxt_engine.model, "export", side_effect=ValueError("export failed"))
        fxt_engine.model.export_nms = True

        with pytest.raises(ValueError, match="export failed"):
            fxt_engine.export(export_nms=False)

        assert fxt_engine.model.export_nms

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "path/to/checkpoint.ckpt",
            "path/to/checkpoint.xml",
        ],
    )
    def test_explain(self, fxt_engine, checkpoint, mocker) -> None:
        mock_predict = mocker.patch("getitune.backend.lightning.engine.Trainer.predict")
        _ = mocker.patch("getitune.backend.lightning.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mock_process_saliency_maps = mocker.patch(
            "getitune.backend.lightning.models.utils.xai_utils.process_saliency_maps_in_pred_entity",
        )

        mock_model = mocker.create_autospec(LightningModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(checkpoint=checkpoint, explain=True)
        mock_predict.assert_called_once()

        mock_process_saliency_maps.assert_called_once()

    def test_from_config_with_model_name(self, tmp_path) -> None:
        model_name = "efficientnet_b0"
        data_root = "tests/assets/classification_cifar10"
        task_type = "MULTI_CLASS_CLS"

        overriding = {
            "data.train_subset.batch_size": 3,
            "data.test_subset.subset_name": "TESTING",
        }

        engine = LightningEngine.from_model_name(
            model_name=model_name,
            data=data_root,
            task=task_type,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.train_subset.batch_size == 3
        assert engine.datamodule.test_subset.subset_name == "TESTING"

        with pytest.raises(FileNotFoundError):
            engine = LightningEngine.from_model_name(
                model_name="wrong_model",
                task=task_type,
                data=data_root,
                work_dir=tmp_path,
                **overriding,
            )

    def test_from_config(self, tmp_path, mocker) -> None:
        recipe_path = "src/getitune/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml"
        data_root = "tests/assets/classification_cifar10"
        mocker.patch("getitune.backend.lightning.engine.shutil.copy2")
        mocker.patch("getitune.backend.lightning.engine.Trainer.fit")

        overriding = {
            "data.train_subset.batch_size": 3,
            "data.test_subset.subset_name": "TESTING",
            "max_epochs": 50,
        }

        engine = LightningEngine.from_config(
            config_path=recipe_path,
            data=data_root,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.train_subset.batch_size == 3
        assert engine.datamodule.test_subset.subset_name == "TESTING"
        # test overriding train_kwargs with config
        engine.train()
        assert engine._cache.args["max_epochs"] == 50
        assert engine.trainer.max_epochs == 50
        assert not engine._cache.args["deterministic"]
        engine.train(max_epochs=100, deterministic=True)
        assert engine._cache.args["max_epochs"] == 100
        assert engine.trainer.max_epochs == 100
        assert engine._cache.args["deterministic"]

    def test_benchmark(self, fxt_engine, mocker: MockerFixture) -> None:
        checkpoint = "path/to/checkpoint.ckpt"
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})

        mock_model = mocker.create_autospec(LightningModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        result = fxt_engine.benchmark(checkpoint=checkpoint)
        assert "latency" in result

    def test_num_devices(self, fxt_engine, tmp_path) -> None:
        assert fxt_engine.num_devices == 1
        assert fxt_engine._cache.args.get("devices") == 1

        fxt_engine.num_devices = 2
        assert fxt_engine.num_devices == 2
        assert fxt_engine._cache.args.get("devices") == 2

        data_root = "tests/assets/classification_cifar10"
        engine = LightningEngine(
            work_dir=tmp_path,
            data=data_root,
            num_devices=3,
            model="src/getitune/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
        )
        assert engine.num_devices == 3
        assert engine._cache.args.get("devices") == 3
