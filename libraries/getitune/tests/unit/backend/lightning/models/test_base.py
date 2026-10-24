# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import warnings
from dataclasses import asdict

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.utilities.types import LRSchedulerConfig
from pytest_mock import MockerFixture

from getitune.backend.lightning.models import ATSS
from getitune.backend.lightning.models.base import DataInputParams, LightningModel
from getitune.backend.lightning.models.classification.multiclass_models.base import LightningMulticlassClsModel
from getitune.backend.lightning.models.segmentation.base import LightningSegmentationModel
from getitune.backend.lightning.schedulers.warmup_schedulers import LinearWarmupScheduler
from getitune.config.data import TileConfig
from getitune.types.label import LabelInfo, SegLabelInfo


class MockNNModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(1, num_classes)
        self.head.weight.data = torch.arange(num_classes, dtype=torch.float32).reshape(num_classes, 1)
        self.head.bias.data = torch.arange(num_classes, dtype=torch.float32)


class TestLightningModel:
    def test_init(self, monkeypatch):
        monkeypatch.setattr(LightningModel, "input_size_multiplier", 10, raising=False)
        with pytest.raises(ValueError, match="Input size should be a multiple"):
            LightningModel(
                label_info=2, data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            )

    def test_training_step_none_loss(self, mocker: MockerFixture) -> None:
        mock_trainer = mocker.create_autospec(spec=Trainer)
        mock_trainer.world_size = 1
        with mocker.patch.object(LightningModel, "_create_model", return_value=MockNNModule(3)) and mocker.patch.object(
            LightningModel,
            "forward",
            return_value=None,
        ):
            current_model = LightningModel(
                label_info=3,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )
            current_model.trainer = mock_trainer

        batch = {"input": torch.randn(2, 3)}
        batch_idx = 0

        with pytest.raises(ValueError, match="Loss is None."):
            current_model.training_step(batch, batch_idx)

    def test_smart_weight_loading(self, mocker) -> None:
        with mocker.patch.object(LightningModel, "_create_model", return_value=MockNNModule(2)):
            prev_model = LightningModel(
                label_info=2,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )
            prev_model.label_info = ["car", "truck"]
            prev_state_dict = prev_model.state_dict()

        with mocker.patch.object(LightningModel, "_create_model", return_value=MockNNModule(3)):
            current_model = LightningModel(
                label_info=3,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )
            current_model.label_info = ["car", "bus", "truck"]
            mocker.patch.object(
                current_model,
                "_identify_classification_layers",
                return_value=["model.head.weight", "model.head.bias"],
            )
            current_model.load_state_dict_incrementally(
                {"state_dict": prev_state_dict, "hyper_parameters": {"label_info": prev_model.label_info}},
            )
            curr_state_dict = current_model.state_dict()

        indices = torch.Tensor([0, 2]).to(torch.int32)

        assert torch.allclose(curr_state_dict["model.backbone.weight"], prev_state_dict["model.backbone.weight"])
        assert torch.allclose(curr_state_dict["model.backbone.bias"], prev_state_dict["model.backbone.bias"])
        assert torch.allclose(
            curr_state_dict["model.head.weight"].index_select(0, indices),
            prev_state_dict["model.head.weight"],
        )
        assert torch.allclose(
            curr_state_dict["model.head.bias"].index_select(0, indices),
            prev_state_dict["model.head.bias"],
        )

    def test_label_info_dispatch(self, mocker):
        with mocker.patch.object(LightningModel, "_create_model", return_value=MockNNModule(3)):
            with pytest.raises(TypeError, match="invalid_label_info"):
                LightningModel(
                    label_info="invalid_label_info",
                    data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                )

            # Test with LabelInfo
            label_info = LightningModel(
                label_info=LabelInfo(
                    ["label_1", "label_2"],
                    label_ids=["1", "2"],
                    label_groups=[["label_1", "label_2"]],
                ),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                pretrained=False,
            )
            assert isinstance(label_info.label_info, LabelInfo)

            # Test with SegLabelInfo
            seg_label_info = LightningModel(
                label_info=SegLabelInfo.from_num_classes(3),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
            )
            assert isinstance(seg_label_info.label_info, SegLabelInfo)

        with mocker.patch.object(LightningMulticlassClsModel, "_create_model", return_value=MockNNModule(3)):
            # Test simple Classfication model loading checkpoint
            cls_model = LightningMulticlassClsModel(
                label_info=LabelInfo(
                    ["label_1", "label_2"],
                    label_ids=["1", "2"],
                    label_groups=[["label_1", "label_2"]],
                ),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
            )
            label_info_dict = {
                "label_ids": ["1", "2"],
                "label_names": ["label_1", "label_2"],
                "label_groups": [["label_1", "label_2"]],
            }
            cls_model.load_state_dict_incrementally(
                {"state_dict": cls_model.state_dict(), "hyper_parameters": {"label_info": label_info_dict}},
            )
            assert isinstance(cls_model.label_info, LabelInfo)
            # test if ignore_index is not set
            label_info_dict["ignore_index"] = 255
            with pytest.raises(TypeError, match=r"unexpected keyword argument.*ignore_index"):
                cls_model.load_state_dict_incrementally(
                    {"state_dict": cls_model.state_dict(), "hyper_parameters": {"label_info": label_info_dict}},
                )

        with mocker.patch.object(LightningSegmentationModel, "_create_model", return_value=MockNNModule(3)):
            # test segmentation model loading checkpoint with SegLabelInfo
            segmentation_model = LightningSegmentationModel(
                label_info=SegLabelInfo.from_num_classes(3),
                data_input_params={"input_size": (224, 224), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                model_name="segmentation_model",
                pretrained=False,
            )
            segmentation_model.load_state_dict_incrementally(
                {"state_dict": segmentation_model.state_dict(), "hyper_parameters": {"label_info": label_info_dict}},
            )
            assert isinstance(segmentation_model.label_info, SegLabelInfo)
            assert hasattr(segmentation_model.label_info, "ignore_index")
            assert segmentation_model.label_info.ignore_index == 255

    def test_lr_scheduler_step(self, mocker: MockerFixture) -> None:
        mock_linear_warmup_scheduler = mocker.create_autospec(spec=LinearWarmupScheduler)
        mock_main_scheduler = mocker.create_autospec(spec=torch.optim.lr_scheduler.LRScheduler)

        with mocker.patch.object(LightningModel, "_create_model", return_value=MockNNModule(3)):
            current_model = LightningModel(
                label_info=3,
                data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            )

        mock_trainer = mocker.create_autospec(spec=Trainer)
        mock_trainer.lr_scheduler_configs = [
            LRSchedulerConfig(mock_linear_warmup_scheduler),
            LRSchedulerConfig(mock_main_scheduler),
        ]
        current_model.trainer = mock_trainer

        # Assume that LinearWarmupScheduler is activated
        mock_linear_warmup_scheduler.activated = True
        for scheduler in [mock_linear_warmup_scheduler, mock_main_scheduler]:
            current_model.lr_scheduler_step(scheduler=scheduler, metric=None)

        # Assert mock_main_scheduler's step() is not called
        mock_main_scheduler.step.assert_not_called()

        mock_main_scheduler.reset_mock()

        # Assume that LinearWarmupScheduler is not activated
        mock_linear_warmup_scheduler.activated = False

        for scheduler in [mock_linear_warmup_scheduler, mock_main_scheduler]:
            current_model.lr_scheduler_step(scheduler=scheduler, metric=None)

        # Assert mock_main_scheduler's step() is called
        mock_main_scheduler.step.assert_called()

        # Regardless of the activation status, LinearWarmupScheduler can be called
        assert mock_linear_warmup_scheduler.step.call_count == 2

    def test_on_load_checkpoint_restores_tile_config_without_warning(self) -> None:
        model = ATSS(
            label_info=1,
            pretrained=False,
        )
        original = model.tile_config
        restored = TileConfig(enable_tiler=not original.enable_tiler)

        checkpoint = {
            "hyper_parameters": {
                "tile_config": asdict(restored),
            },
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.on_load_checkpoint(checkpoint)

        assert model.tile_config == restored
        assert not caught


class TestDataInputParams:
    def test_as_dict(self):
        params = DataInputParams(input_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        params_dict = params.as_dict()
        assert params_dict == {
            "input_size": (224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }

    def test_as_ncwh(self):
        params = DataInputParams(input_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ncwh = params.as_ncwh(batch_size=4)
        assert ncwh == (4, 3, 224, 224)


class TestConfigurePreprocessingParams:
    """Regression tests for LightningModel._configure_preprocessing_params.

    Guards against reintroducing the 0% mAP bug where callers (e.g.
    AutoConfigurator, getitune_trainer) substitute a hardcoded (0.0, 0.0, 0.0) /
    (1.0, 1.0, 1.0) whenever the DataModule cannot derive mean/std from its CPU
    augmentation pipeline (the common case when normalization instead lives in
    augmentations_gpu). Such a substitution is a *truthy* tuple that must not
    permanently shadow a model's own `_default_preprocessing_params` fallback
    (e.g. YOLOX-S/L/X's std=(1/255, 1/255, 1/255), needed to undo the CPU
    pipeline's [0, 1] scaling before OV/ONNX export).
    """

    @pytest.fixture
    def model_default_std_1_over_255(self, mocker: MockerFixture) -> LightningModel:
        """A LightningModel whose model-specific default mimics YOLOX-S/L/X."""
        with mocker.patch.object(LightningModel, "_create_model", return_value=MockNNModule(3)):
            model = LightningModel(
                label_info=3,
                data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)),
            )
        mocker.patch.object(
            LightningModel,
            "_default_preprocessing_params",
            new_callable=mocker.PropertyMock,
            return_value=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)),
        )
        return model

    def test_none_mean_std_falls_back_to_model_default(self, model_default_std_1_over_255: LightningModel) -> None:
        """mean=None/std=None (e.g. datamodule couldn't derive them) must use the model default."""
        result = model_default_std_1_over_255._configure_preprocessing_params(
            {"input_size": (640, 640), "mean": None, "std": None},
        )
        assert result.mean == (0.0, 0.0, 0.0)
        assert result.std == (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)

    def test_missing_mean_std_keys_fall_back_to_model_default(
        self,
        model_default_std_1_over_255: LightningModel,
    ) -> None:
        """A partial dict (mean/std keys absent entirely) must also use the model default."""
        result = model_default_std_1_over_255._configure_preprocessing_params({"input_size": (640, 640)})
        assert result.std == (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)

    def test_hardcoded_identity_fallback_does_not_shadow_model_default(
        self,
        model_default_std_1_over_255: LightningModel,
    ) -> None:
        """Regression guard: callers must NOT substitute (1.0, 1.0, 1.0) for a missing std.

        A caller that (incorrectly) substitutes the generic identity std=(1.0, 1.0, 1.0)
        for a missing value defeats the model-specific default, since a non-empty tuple
        is truthy and therefore looks "provided". This test documents/pins the (bad)
        outcome of doing that, to make the anti-pattern obvious if it's reintroduced
        upstream: callers must pass None through, not (1.0, 1.0, 1.0).
        """
        result = model_default_std_1_over_255._configure_preprocessing_params(
            {"input_size": (640, 640), "mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        )
        assert result.std == (1.0, 1.0, 1.0)
        assert result.std != (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)

    def test_explicit_zero_mean_is_respected(self, model_default_std_1_over_255: LightningModel) -> None:
        """A genuinely-intentional, explicit mean/std must still be honored (not overridden)."""
        result = model_default_std_1_over_255._configure_preprocessing_params(
            {"input_size": (640, 640), "mean": (0.1, 0.2, 0.3), "std": (0.5, 0.5, 0.5)},
        )
        assert result.mean == (0.1, 0.2, 0.3)
        assert result.std == (0.5, 0.5, 0.5)
