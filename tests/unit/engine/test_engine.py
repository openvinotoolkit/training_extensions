# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import create_autospec

import pytest
from otx.algo.classification.efficientnet_b0 import EfficientNetB0ForMulticlassCls
from otx.algo.classification.torchvision_model import OTXTVModel
from otx.core.config.device import DeviceConfig
from otx.core.data.dataset.tile import OTXTileDataset
from otx.core.model.entity.base import OVModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.engine import Engine


@pytest.fixture()
def fxt_engine(tmp_path) -> Engine:
    recipe_path = "src/otx/recipe/classification/multi_class_cls/tv_resnet_50.yaml"
    data_root = "tests/assets/classification_dataset"
    task_type = "MULTI_CLASS_CLS"

    return Engine.from_config(
        config_path=recipe_path,
        data_root=data_root,
        task=task_type,
        work_dir=tmp_path,
    )


class TestEngine:
    def test_constructor(self, tmp_path) -> None:
        with pytest.raises(RuntimeError):
            Engine(work_dir=tmp_path)

        # Check auto-configuration
        data_root = "tests/assets/classification_dataset"
        engine = Engine(work_dir=tmp_path, data_root=data_root)
        assert engine.task == "MULTI_CLASS_CLS"
        assert engine.datamodule.task == "MULTI_CLASS_CLS"
        assert isinstance(engine.model, EfficientNetB0ForMulticlassCls)

        # Create engine with task
        engine = Engine(work_dir=tmp_path, task="MULTI_CLASS_CLS")
        assert engine.task == "MULTI_CLASS_CLS"
        assert isinstance(engine.model, EfficientNetB0ForMulticlassCls)
        assert engine.device == DeviceConfig(accelerator="auto", devices=1)
        with pytest.raises(RuntimeError, match="Please include the `data_root` or `datamodule`"):
            _ = engine.datamodule
        with pytest.raises(RuntimeError, match="Please run train"):
            _ = engine.trainer

        assert "default_root_dir" in engine.trainer_params
        assert engine.trainer_params["default_root_dir"] == tmp_path
        assert "accelerator" in engine.trainer_params
        assert engine.trainer_params["accelerator"] == "auto"
        assert "devices" in engine.trainer_params
        assert engine.trainer_params["devices"] == 1

    def test_model_setter(self, fxt_engine, mocker) -> None:
        assert isinstance(fxt_engine.model, OTXTVModel)
        fxt_engine.model = "efficientnet_b0_light"
        assert isinstance(fxt_engine.model, EfficientNetB0ForMulticlassCls)

    def test_training_with_override_args(self, fxt_engine, mocker) -> None:
        mocker.patch("otx.engine.engine.Trainer.fit")
        mock_seed_everything = mocker.patch("otx.engine.engine.seed_everything")

        assert fxt_engine._cache.args["max_epochs"] == 90

        fxt_engine.train(max_epochs=100, seed=1234)
        assert fxt_engine._cache.args["max_epochs"] == 100
        mock_seed_everything.assert_called_once_with(1234, workers=True)

    def test_training_with_checkpoint(self, fxt_engine, mocker) -> None:
        mock_torch_load = mocker.patch("torch.load")
        mocker.patch("otx.engine.engine.OTXLitModule.load_state_dict")

        fxt_engine.checkpoint = "path/to/checkpoint"
        fxt_engine.train()
        mock_torch_load.assert_called_once_with("path/to/checkpoint")

    def test_training_with_run_hpo(self, fxt_engine, mocker) -> None:
        mock_fit = mocker.patch("otx.engine.engine.Trainer.fit")
        mock_execute_hpo = mocker.patch("otx.engine.engine.execute_hpo")
        mock_update_hyper_parameter = mocker.patch("otx.engine.engine.update_hyper_parameter")
        mock_execute_hpo.return_value = {}, "hpo/best/checkpoint"

        fxt_engine.train(run_hpo=True)
        mock_execute_hpo.assert_called_once()
        mock_update_hyper_parameter.assert_called_once_with(fxt_engine, {})
        assert mock_fit.call_args[1]["ckpt_path"] == "hpo/best/checkpoint"

    def test_training_with_resume(self, fxt_engine, mocker) -> None:
        mock_fit = mocker.patch("otx.engine.engine.Trainer.fit")

        fxt_engine.checkpoint = "path/to/checkpoint"
        fxt_engine.train(resume=True)
        assert mock_fit.call_args[1]["ckpt_path"] == "path/to/checkpoint"

    def test_testing_after_training(self, fxt_engine, mocker) -> None:
        mocker.patch("otx.engine.engine.OTXLitModule.load_state_dict")
        mock_test = mocker.patch("otx.engine.engine.Trainer.test")
        mock_torch_load = mocker.patch("torch.load")

        # Fetch Checkpoint
        fxt_engine.checkpoint = "path/to/checkpoint"
        fxt_engine.test()
        mock_torch_load.assert_called_once_with("path/to/checkpoint")
        mock_test.assert_called_once()

        fxt_engine.test(checkpoint="path/to/new/checkpoint")
        mock_torch_load.assert_called_with("path/to/new/checkpoint")

    def test_testing_with_ov_model(self, fxt_engine, mocker) -> None:
        mock_test = mocker.patch("otx.engine.engine.Trainer.test")
        mock_torch_load = mocker.patch("torch.load")
        mocker.patch("otx.engine.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")

        fxt_engine.test(checkpoint="path/to/model.xml")
        mock_test.assert_called_once()
        mock_torch_load.assert_not_called()

        fxt_engine.model = create_autospec(OVModel)
        fxt_engine.test(checkpoint="path/to/model.xml")

    def test_prediction_after_training(self, fxt_engine, mocker) -> None:
        mocker.patch("otx.engine.engine.OTXLitModule.load_state_dict")
        mock_predict = mocker.patch("otx.engine.engine.Trainer.predict")
        mock_torch_load = mocker.patch("torch.load")

        # Fetch Checkpoint
        fxt_engine.checkpoint = "path/to/checkpoint"
        fxt_engine.predict()
        mock_torch_load.assert_called_once_with("path/to/checkpoint")
        mock_predict.assert_called_once()

        fxt_engine.predict(checkpoint="path/to/new/checkpoint")
        mock_torch_load.assert_called_with("path/to/new/checkpoint")

    def test_prediction_with_ov_model(self, fxt_engine, mocker) -> None:
        mock_predict = mocker.patch("otx.engine.engine.Trainer.predict")
        mock_torch_load = mocker.patch("torch.load")
        mocker.patch("otx.engine.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")

        fxt_engine.predict(checkpoint="path/to/model.xml")
        mock_predict.assert_called_once()
        mock_torch_load.assert_not_called()

        fxt_engine.model = create_autospec(OVModel)
        fxt_engine.predict(checkpoint="path/to/model.xml")

    def test_prediction_explain_mode(self, fxt_engine, mocker) -> None:
        mocker.patch("otx.engine.engine.OTXLitModule.load_state_dict")
        mock_explain = mocker.patch("otx.algo.utils.xai_utils.process_saliency_maps_in_pred_entity")
        mock_predict = mocker.patch("otx.engine.engine.Trainer.predict")
        mock_torch_load = mocker.patch("torch.load")

        # Fetch Checkpoint
        fxt_engine.checkpoint = "path/to/checkpoint"
        fxt_engine.predict(explain=True)
        mock_torch_load.assert_called_once_with("path/to/checkpoint")
        mock_explain.assert_called_once()
        mock_predict.assert_called_once()

    def test_exporting(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="To make export, checkpoint must be specified."):
            fxt_engine.export()

        mocker.patch("otx.engine.engine.OTXLitModule.load_state_dict")
        mocker.patch("otx.engine.engine.OTXModel.label_info")
        mock_export = mocker.patch("otx.engine.engine.OTXLitModule.export")
        mock_torch_load = mocker.patch("torch.load")

        # Fetch Checkpoint
        fxt_engine.checkpoint = "path/to/checkpoint"
        fxt_engine.export()
        mock_torch_load.assert_called_once_with("path/to/checkpoint")
        mock_export.assert_called_once_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP32,
        )

        fxt_engine.export(export_precision=OTXPrecisionType.FP16)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP16,
        )

        fxt_engine.export(export_format=OTXExportFormatType.ONNX)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.ONNX,
            precision=OTXPrecisionType.FP32,
        )

    def test_optimizing_model(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="supports only OV IR or ONNX checkpoints"):
            fxt_engine.optimize()

        mocker.patch("otx.engine.engine.OTXLitModule.load_state_dict")
        mock_ov_model = mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")

        # Fetch Checkpoint
        fxt_engine.checkpoint = "path/to/exported_model.xml"
        fxt_engine.optimize()
        mock_ov_model.assert_called_once()
        mock_ov_model.return_value.optimize.assert_called_once()

        # With max_data_subset_size
        fxt_engine.optimize(max_data_subset_size=100)
        assert mock_ov_model.return_value.optimize.call_args[0][2]["subset_size"] == 100

    def test_explain(self, fxt_engine, mocker) -> None:
        mocker.patch("otx.engine.engine.OTXLitModule.load_state_dict")
        mock_process_explain = mocker.patch("otx.algo.utils.xai_utils.process_saliency_maps_in_pred_entity")

        mock_torch_load = mocker.patch("torch.load")
        mock_predict = mocker.patch("otx.engine.engine.Trainer.predict")

        fxt_engine.explain(checkpoint="path/to/checkpoint")
        mock_torch_load.assert_called_once_with("path/to/checkpoint")
        mock_predict.assert_called_once()
        mock_process_explain.assert_called_once()

        mock_dump_saliency_maps = mocker.patch("otx.algo.utils.xai_utils.dump_saliency_maps")
        fxt_engine.explain(checkpoint="path/to/checkpoint", dump=True)
        mock_torch_load.assert_called_with("path/to/checkpoint")
        mock_predict.assert_called()
        mock_process_explain.assert_called()
        mock_dump_saliency_maps.assert_called_once()

        mock_ov_pipeline = mocker.patch("otx.engine.engine.AutoConfigurator.update_ov_subset_pipeline")
        mock_ov_model = mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")
        fxt_engine.explain(checkpoint="path/to/model.xml")
        mock_predict.assert_called()
        mock_process_explain.assert_called()
        mock_ov_model.assert_called_once()
        mock_ov_pipeline.assert_called_once()

    def test_from_config_with_model_name(self, tmp_path) -> None:
        model_name = "efficientnet_b0_light"
        data_root = "tests/assets/classification_dataset"
        task_type = "MULTI_CLASS_CLS"

        overriding = {
            "data.config.train_subset.batch_size": 3,
            "data.config.test_subset.subset_name": "TESTING",
        }

        engine = Engine.from_model_name(
            model_name=model_name,
            data_root=data_root,
            task=task_type,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.config.train_subset.batch_size == 3
        assert engine.datamodule.config.test_subset.subset_name == "TESTING"

        with pytest.raises(FileNotFoundError):
            engine = Engine.from_model_name(
                model_name="wrong_model",
                data_root=data_root,
                task=task_type,
                work_dir=tmp_path,
                **overriding,
            )

    def test_from_config(self, tmp_path) -> None:
        recipe_path = "src/otx/recipe/classification/multi_class_cls/tv_resnet_50.yaml"
        data_root = "tests/assets/classification_dataset"
        task_type = "MULTI_CLASS_CLS"

        overriding = {
            "data.config.train_subset.batch_size": 3,
            "data.config.test_subset.subset_name": "TESTING",
        }

        engine = Engine.from_config(
            config_path=recipe_path,
            data_root=data_root,
            task=task_type,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.config.train_subset.batch_size == 3
        assert engine.datamodule.config.test_subset.subset_name == "TESTING"
