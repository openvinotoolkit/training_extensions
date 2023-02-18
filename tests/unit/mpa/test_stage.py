import os

import mmcv
import pytest

from otx.mpa.stage import Stage, get_available_types
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_available_types():
    return_value = get_available_types()
    assert isinstance(return_value, list)


class TestStage:
    @e2e_pytest_unit
    def test_init(self, mocker):
        fake_cfg = {
            "work_dir": "test_init",
            "total_epochs": 5,
            "runner": {"max_epochs": 10},
            "checkpoint_config": {"interval": 20},
            "seed": 0,
        }
        fake_common_cfg = {"output_path": "/path/output"}
        mocker.patch.object(mmcv, "mkdir_or_exist")
        stage = Stage("mpa_test", "", fake_cfg, fake_common_cfg, 0, fake_kwargs=None)

        assert stage.cfg.get("runner", False)
        assert stage.cfg.runner.get("max_epochs", 5)
        assert stage.cfg.get("checkpoint_config", False)
        assert stage.cfg.checkpoint_config.get("interval", 5)

        work_dir = str(stage.cfg.get("work_dir", ""))

        assert work_dir.find("test_init")
        assert work_dir.find("mpa_test")

    @e2e_pytest_unit
    def test_init_with_distributed_enabled(self, mocker):
        fake_cfg = {"work_dir": "test_init_distributed"}
        fake_common_cfg = {"output_path": "/path/output"}
        mocker.patch.object(mmcv, "mkdir_or_exist")
        mocker.patch("torch.distributed.is_initialized", returned_value=True)
        os.environ["LOCAL_RANK"] = "2"
        stage = Stage("mpa_test", "", fake_cfg, fake_common_cfg, 0)

        assert stage.distributed is True

    @e2e_pytest_unit
    def test_configure_data(self):
        _base_pipeline = [dict(type="LoadImageFromFile"), dict(type="MultiScaleFlipAug")]
        _transform = [
            dict(type="Resize", keep_ratio=False),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ]
        data_cfg = mmcv.ConfigDict(
            data=dict(
                train=dict(dataset=dict(pipeline=_base_pipeline)),
                val=dict(pipeline=_base_pipeline),
                pipeline_options=dict(
                    MultiScaleFlipAug=dict(img_scale=(224, 224), transforms=_transform),
                ),
            )
        )
        Stage.configure_data(data_cfg, True)

        assert data_cfg.data.train.dataset.pipeline[1].img_scale == (224, 224)
        assert data_cfg.data.train.dataset.pipeline[1].transforms == _transform
        assert data_cfg.data.val.pipeline[1].img_scale == (224, 224)
        assert data_cfg.data.val.pipeline[1].transforms == _transform

    @e2e_pytest_unit
    def test_configure_ckpt(self, mocker):
        cfg = mmcv.ConfigDict(load_from=None)
        mocker.patch.object(mmcv, "mkdir_or_exist")
        stage = Stage("mpa_test", "", {}, {}, 0)

        mocker.patch.object(stage, "get_model_ckpt", return_value="/path/to/load/ckpt")
        stage.configure_ckpt(cfg, "/foo/bar")

        assert cfg.load_from == "/path/to/load/ckpt"

    @e2e_pytest_unit
    def test_configure_hook(self):
        cfg = mmcv.ConfigDict(
            custom_hook_options=dict(MockHook=dict(opt1="MockOpt1", opt2="MockOpt2")),
            custom_hooks=[dict(type="MockHook")],
        )
        Stage.configure_hook(cfg)

        assert cfg.custom_hooks[0].opt1 == "MockOpt1"
        assert cfg.custom_hooks[0].opt2 == "MockOpt2"

    @e2e_pytest_unit
    def test_configure_samples_per_gpu(self, mocker):
        cfg = mmcv.ConfigDict(data=dict(train_dataloader=dict(samples_per_gpu=2)))
        mock_otx_dataset = mocker.MagicMock()
        mock_otx_dataset.__len__.return_value = 1

        mocker.patch("otx.mpa.stage.get_data_cfg", return_value=mmcv.ConfigDict(otx_dataset=mock_otx_dataset))
        Stage.configure_samples_per_gpu(cfg, "train", False)

        assert "train_dataloader" in cfg.data
        assert cfg.data["train_dataloader"]["samples_per_gpu"] == 1

    @e2e_pytest_unit
    def test_configure_compat_cfg(self):
        cfg = mmcv.ConfigDict(
            data=dict(
                train=None,
                val=None,
                test=None,
                unlabeled=None,
            )
        )
        with pytest.raises(AttributeError):
            Stage.configure_compat_cfg(cfg)

    @e2e_pytest_unit
    def test_configure_fp16_optimizer(self):
        cfg = mmcv.ConfigDict(fp16=dict(loss_scale=512.0), optimizer_config=dict(type="OptimizerHook"))

        Stage.configure_fp16_optimizer(cfg)
        assert cfg.optimizer_config.type == "Fp16OptimizerHook"

    @e2e_pytest_unit
    def test_configure_unlabeled_dataloader(self, mocker):
        cfg = mmcv.ConfigDict(
            data=dict(
                unlabeled=dict(),
            ),
            model_task="classification",
        )

        mocker.patch("importlib.import_module")
        mock_build_ul_dataset = mocker.patch("otx.mpa.stage.build_dataset")
        mock_build_ul_dataloader = mocker.patch("otx.mpa.stage.build_dataloader")

        Stage.configure_unlabeled_dataloader(cfg)

        mock_build_ul_dataset.assert_called_once()
        mock_build_ul_dataloader.assert_called_once()
        assert cfg.custom_hooks[0].type == "ComposedDataLoadersHook"

    @e2e_pytest_unit
    def test_get_model_meta(self, mocker):
        cfg = dict(load_from="/foo/bar")
        from mmcv.runner import CheckpointLoader

        mock_load_ckpt = mocker.patch.object(
            CheckpointLoader, "load_checkpoint", return_value={"meta": {"model_meta": None}}
        )

        returned_value = Stage.get_model_meta(cfg)

        mock_load_ckpt.assert_called_once()
        assert returned_value == {"model_meta": None}

    @e2e_pytest_unit
    def test_get_data_cfg(self, mocker):
        cfg = mmcv.ConfigDict(data=dict(train=dict(dataset=dict(dataset="config"))))
        returned_value = Stage.get_data_cfg(cfg, "train")

        assert returned_value == "config"

    @e2e_pytest_unit
    def test_get_data_classes(self, mocker):
        cfg = mmcv.ConfigDict()
        mocker.patch.object(Stage, "get_data_cfg", return_value={"data_classes": ["foo"]})

        returned_value = Stage.get_data_classes(cfg)

        assert returned_value == ["foo"]

    @e2e_pytest_unit
    def test_get_model_classes(self, mocker):
        cfg = mmcv.ConfigDict(model=dict())
        mocker.patch.object(Stage, "get_model_meta", return_value={"CLASSES": ["foo", "bar"]})
        mocker.patch.object(Stage, "read_label_schema")
        returned_value = Stage.get_model_classes(cfg)

        assert returned_value == ["foo", "bar"]

    @e2e_pytest_unit
    def test_get_model_ckpt(self, mocker):
        from mmcv.runner import CheckpointLoader

        mock_load_ckpt = mocker.patch.object(
            CheckpointLoader, "load_checkpoint", return_value=dict(model=mocker.MagicMock())
        )
        mock_save = mocker.patch("torch.save")

        Stage.get_model_ckpt("/ckpt/path/weights.pth")

        mock_load_ckpt.assert_called_once()
        mock_save.assert_called_once()

    @e2e_pytest_unit
    def test_read_label_schema(self, mocker):
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("builtins.open")
        mocker.patch("json.load", return_value=dict(all_labels=dict(label=dict(name="foo"))))
        returned_value = Stage.read_label_schema("/ckpt/path/weights.pth")

        assert returned_value == ["foo"]

    @e2e_pytest_unit
    def test_set_inference_progress_callback(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.register_forward_pre_hook()
        mock_model.register_forward_hook()
        cfg = dict(custom_hooks=[dict(name="OTXProgressHook", time_monitor=mocker.MagicMock())])

        Stage.set_inference_progress_callback(cfg, mock_model)

        mock_model.register_forward_pre_hook.assert_called_once()
        mock_model.register_forward_hook.assert_called_once()

    @e2e_pytest_unit
    def test_build_model(self, mocker):
        mock_model_builder = mocker.patch.object(Stage, "MODEL_BUILDER")
        Stage.build_model(cfg=mmcv.Config())

        mock_model_builder.assert_called_once()
