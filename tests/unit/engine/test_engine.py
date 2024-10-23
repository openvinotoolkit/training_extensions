# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from otx.algo.classification.efficientnet import EfficientNetForMulticlassCls
from otx.algo.classification.torchvision_model import TVModelForMulticlassCls
from otx.core.model.base import OTXModel, OVModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.label import NullLabelInfo
from otx.core.types.precision import OTXPrecisionType
from otx.engine import Engine
from pytest_mock import MockerFixture


@pytest.fixture()
def fxt_engine(tmp_path) -> Engine:
    recipe_path = "src/otx/recipe/classification/multi_class_cls/tv_mobilenet_v3_small.yaml"
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
        assert isinstance(engine.model, EfficientNetForMulticlassCls)

        assert "default_root_dir" in engine.trainer_params
        assert engine.trainer_params["default_root_dir"] == tmp_path
        assert "accelerator" in engine.trainer_params
        assert engine.trainer_params["accelerator"] == "auto"
        assert "devices" in engine.trainer_params
        assert engine.trainer_params["devices"] == 1

        # Create engine with no data_root
        with pytest.raises(ValueError, match="Given model class (.*) requires a valid label_info to instantiate."):
            _ = Engine(work_dir=tmp_path, task="MULTI_CLASS_CLS")

    @pytest.fixture()
    def mock_datamodule(self, mocker):
        input_size = (1234, 1234)
        label_info = 4321
        mock_datamodule = MagicMock()
        mock_datamodule.label_info = label_info
        mock_datamodule.input_size = input_size

        return mocker.patch(
            "otx.engine.utils.auto_configurator.AutoConfigurator.get_datamodule",
            return_value=mock_datamodule,
        )

    def test_model_init(self, tmp_path, mock_datamodule):
        data_root = "tests/assets/classification_dataset"
        engine = Engine(work_dir=tmp_path, data_root=data_root)

        assert engine._model.input_size == (1234, 1234)
        assert engine._model.label_info.num_classes == 4321

    def test_model_init_datamodule_ipt_size_int(self, tmp_path, mock_datamodule):
        mock_datamodule.input_size = 1234
        data_root = "tests/assets/classification_dataset"
        engine = Engine(work_dir=tmp_path, data_root=data_root)

        assert engine._model.input_size == (1234, 1234)
        assert engine._model.label_info.num_classes == 4321

    def test_model_setter(self, fxt_engine, mocker) -> None:
        assert isinstance(fxt_engine.model, TVModelForMulticlassCls)
        fxt_engine.model = "efficientnet_b0"
        assert isinstance(fxt_engine.model, EfficientNetForMulticlassCls)

    def test_training_with_override_args(self, fxt_engine, mocker) -> None:
        mocker.patch("pathlib.Path.symlink_to")
        mocker.patch("otx.engine.engine.Trainer.fit")
        mock_seed_everything = mocker.patch("otx.engine.engine.seed_everything")

        assert fxt_engine._cache.args["max_epochs"] == 90

        fxt_engine.train(max_epochs=100, seed=1234)
        assert fxt_engine._cache.args["max_epochs"] == 100
        mock_seed_everything.assert_called_once_with(1234, workers=True)

    @pytest.mark.parametrize("resume", [True, False])
    def test_training_with_checkpoint(self, fxt_engine, resume: bool, mocker: MockerFixture, tmpdir) -> None:
        checkpoint = "path/to/checkpoint.ckpt"

        mock_trainer = mocker.patch("otx.engine.engine.Trainer")
        mock_trainer.return_value.default_root_dir = Path(tmpdir)
        mock_trainer_fit = mock_trainer.return_value.fit

        mock_torch_load = mocker.patch("otx.engine.engine.torch.load")
        mock_load_state_dict_incrementally = mocker.patch.object(fxt_engine.model, "load_state_dict_incrementally")

        trained_checkpoint = Path(tmpdir) / "best.ckpt"
        trained_checkpoint.touch()
        mock_trainer.return_value.checkpoint_callback.best_model_path = trained_checkpoint

        fxt_engine.train(resume=resume, checkpoint=checkpoint)

        if resume:
            assert mock_trainer_fit.call_args.kwargs.get("ckpt_path") == checkpoint
        else:
            assert "ckpt_path" not in mock_trainer_fit.call_args.kwargs

            mock_torch_load.assert_called_once()
            mock_load_state_dict_incrementally.assert_called_once()

    def test_training_with_run_hpo(self, fxt_engine, mocker) -> None:
        mocker.patch("pathlib.Path.symlink_to")
        mock_fit = mocker.patch("otx.engine.engine.Trainer.fit")
        mock_execute_hpo = mocker.patch("otx.engine.engine.execute_hpo")
        mock_update_hyper_parameter = mocker.patch("otx.engine.engine.update_hyper_parameter")
        mock_execute_hpo.return_value = {}, "hpo/best/checkpoint"

        fxt_engine.train(run_hpo=True)
        mock_execute_hpo.assert_called_once()
        mock_update_hyper_parameter.assert_called_once_with(fxt_engine, {})
        assert mock_fit.call_args[1]["ckpt_path"] == "hpo/best/checkpoint"

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "path/to/checkpoint.ckpt",
            "path/to/checkpoint.xml",
        ],
    )
    def test_test(self, fxt_engine, checkpoint, mocker: MockerFixture) -> None:
        mock_test = mocker.patch("otx.engine.engine.Trainer.test")
        _ = mocker.patch("otx.engine.engine.AutoConfigurator.update_ov_subset_pipeline")
        mock_get_ov_model = mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")
        mock_load_from_checkpoint = mocker.patch.object(fxt_engine.model.__class__, "load_from_checkpoint")

        ext = Path(checkpoint).suffix

        if ext == ".ckpt":
            mock_model = mocker.create_autospec(OTXModel)

            mock_load_from_checkpoint.return_value = mock_model
        else:
            mock_model = mocker.create_autospec(OVModel)

            mock_get_ov_model.return_value = mock_model

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.test(checkpoint=checkpoint)
        mock_test.assert_called_once()

        mock_model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a test pipeline, the label information should be same (.*)",
        ):
            fxt_engine.test(checkpoint=checkpoint)

    @pytest.mark.parametrize("explain", [True, False])
    @pytest.mark.parametrize(
        "checkpoint",
        [
            "path/to/checkpoint.ckpt",
            "path/to/checkpoint.xml",
        ],
    )
    def test_predict(self, fxt_engine, checkpoint, explain, mocker: MockerFixture) -> None:
        mock_predict = mocker.patch("otx.engine.engine.Trainer.predict")
        _ = mocker.patch("otx.engine.engine.AutoConfigurator.update_ov_subset_pipeline")
        mock_get_ov_model = mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")
        mock_load_from_checkpoint = mocker.patch.object(fxt_engine.model.__class__, "load_from_checkpoint")
        mock_process_saliency_maps = mocker.patch("otx.algo.utils.xai_utils.process_saliency_maps_in_pred_entity")

        ext = Path(checkpoint).suffix

        if ext == ".ckpt":
            mock_model = mocker.create_autospec(OTXModel)

            mock_load_from_checkpoint.return_value = mock_model
        else:
            mock_model = mocker.create_autospec(OVModel)

            mock_get_ov_model.return_value = mock_model

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(checkpoint=checkpoint, explain=explain)
        mock_predict.assert_called_once()
        assert mock_process_saliency_maps.called == explain

        mock_model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a predict pipeline, the label information should be same (.*)",
        ):
            fxt_engine.predict(checkpoint=checkpoint)

    def test_exporting(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="To make export, checkpoint must be specified."):
            fxt_engine.export()

        mock_export = mocker.patch("otx.engine.engine.OTXModel.export")

        mock_load_from_checkpoint = mocker.patch.object(fxt_engine.model.__class__, "load_from_checkpoint")
        mock_load_from_checkpoint.return_value = fxt_engine.model

        # Fetch Checkpoint
        checkpoint = "path/to/checkpoint.ckpt"
        fxt_engine.checkpoint = checkpoint
        fxt_engine.export()
        mock_load_from_checkpoint.assert_called_once_with(checkpoint_path=checkpoint, map_location="cpu")
        mock_export.assert_called_once_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP32,
            to_exportable_code=False,
        )

        fxt_engine.export(export_precision=OTXPrecisionType.FP16)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP16,
            to_exportable_code=False,
        )

        fxt_engine.export(export_format=OTXExportFormatType.ONNX)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.ONNX,
            precision=OTXPrecisionType.FP32,
            to_exportable_code=False,
        )

        fxt_engine.export(export_format=OTXExportFormatType.ONNX, export_demo_package=True)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.ONNX,
            precision=OTXPrecisionType.FP32,
            to_exportable_code=False,
        )

        # check exportable code with IR OpenVINO model
        mock_export = mocker.patch("otx.engine.engine.OVModel.export")
        fxt_engine.checkpoint = "path/to/checkpoint.xml"
        mock_get_ov_model = mocker.patch(
            "otx.engine.engine.AutoConfigurator.get_ov_model",
            return_value=OVModel(model_name="efficientnet-b0-pytorch", model_type="classification"),
        )
        fxt_engine.export(checkpoint="path/to/checkpoint.xml", export_demo_package=True)
        mock_get_ov_model.assert_called_once()
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP32,
            to_exportable_code=True,
        )

    def test_optimizing_model(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="supports only OV IR or ONNX checkpoints"):
            fxt_engine.optimize()

        mocker.patch("otx.engine.engine.OTXModel.load_state_dict")
        mock_ov_model = mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")

        # Fetch Checkpoint
        fxt_engine.checkpoint = "path/to/exported_model.xml"
        fxt_engine.optimize()
        mock_ov_model.assert_called_once()
        mock_ov_model.return_value.optimize.assert_called_once()

        # With max_data_subset_size
        fxt_engine.optimize(max_data_subset_size=100)
        assert mock_ov_model.return_value.optimize.call_args[0][2]["subset_size"] == 100

        # Optimize and export with exportable code
        mocker_export = mocker.patch.object(fxt_engine, "export")
        fxt_engine.optimize(export_demo_package=True)
        mocker_export.assert_called_once()

    @pytest.mark.parametrize("dump", [True, False])
    @pytest.mark.parametrize(
        "checkpoint",
        [
            "path/to/checkpoint.ckpt",
            "path/to/checkpoint.xml",
        ],
    )
    def test_explain(self, fxt_engine, checkpoint, dump, mocker) -> None:
        mock_predict = mocker.patch("otx.engine.engine.Trainer.predict")
        _ = mocker.patch("otx.engine.engine.AutoConfigurator.update_ov_subset_pipeline")
        mock_get_ov_model = mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")
        mock_load_from_checkpoint = mocker.patch.object(fxt_engine.model.__class__, "load_from_checkpoint")
        mock_process_saliency_maps = mocker.patch("otx.algo.utils.xai_utils.process_saliency_maps_in_pred_entity")
        mock_dump_saliency_maps = mocker.patch("otx.algo.utils.xai_utils.dump_saliency_maps")

        ext = Path(checkpoint).suffix

        if ext == ".ckpt":
            mock_model = mocker.create_autospec(OTXModel)

            mock_load_from_checkpoint.return_value = mock_model
        else:
            mock_model = mocker.create_autospec(OVModel)

            mock_get_ov_model.return_value = mock_model

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.explain(checkpoint=checkpoint, dump=dump)
        mock_predict.assert_called_once()
        mock_process_saliency_maps.assert_called_once()
        assert mock_dump_saliency_maps.called == dump

        mock_model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a explain pipeline, the label information should be same (.*)",
        ):
            fxt_engine.explain(checkpoint=checkpoint)

    def test_from_config_with_model_name(self, tmp_path) -> None:
        model_name = "efficientnet_b0"
        data_root = "tests/assets/classification_dataset"
        task_type = "MULTI_CLASS_CLS"

        overriding = {
            "data.train_subset.batch_size": 3,
            "data.test_subset.subset_name": "TESTING",
        }

        engine = Engine.from_model_name(
            model_name=model_name,
            data_root=data_root,
            task=task_type,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.train_subset.batch_size == 3
        assert engine.datamodule.test_subset.subset_name == "TESTING"

        with pytest.raises(FileNotFoundError):
            engine = Engine.from_model_name(
                model_name="wrong_model",
                data_root=data_root,
                task=task_type,
                work_dir=tmp_path,
                **overriding,
            )

    def test_from_config(self, tmp_path) -> None:
        recipe_path = "src/otx/recipe/classification/multi_class_cls/tv_mobilenet_v3_small.yaml"
        data_root = "tests/assets/classification_dataset"
        task_type = "MULTI_CLASS_CLS"

        overriding = {
            "data.train_subset.batch_size": 3,
            "data.test_subset.subset_name": "TESTING",
        }

        engine = Engine.from_config(
            config_path=recipe_path,
            data_root=data_root,
            task=task_type,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.train_subset.batch_size == 3
        assert engine.datamodule.test_subset.subset_name == "TESTING"

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "path/to/checkpoint.ckpt",
            "path/to/checkpoint.xml",
        ],
    )
    def test_benchmark(self, fxt_engine, checkpoint, mocker: MockerFixture) -> None:
        _ = mocker.patch("otx.engine.engine.AutoConfigurator.update_ov_subset_pipeline")
        mock_get_ov_model = mocker.patch("otx.engine.engine.AutoConfigurator.get_ov_model")
        mock_load_from_checkpoint = mocker.patch.object(fxt_engine.model.__class__, "load_from_checkpoint")

        ext = Path(checkpoint).suffix

        if ext == ".ckpt":
            mock_model = mocker.create_autospec(OTXModel)

            mock_load_from_checkpoint.return_value = mock_model
        else:
            mock_model = mocker.create_autospec(OVModel)

            mock_get_ov_model.return_value = mock_model

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

        data_root = "tests/assets/classification_dataset"
        engine = Engine(work_dir=tmp_path, data_root=data_root, num_devices=3)
        assert engine.num_devices == 3
        assert engine._cache.args.get("devices") == 3
