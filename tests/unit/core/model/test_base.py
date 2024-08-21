import numpy as np
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.utilities.types import LRSchedulerConfig
from model_api.models.utils import ClassificationResult
from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.model.base import OTXModel, OVModel
from otx.core.schedulers.warmup_schedulers import LinearWarmupScheduler
from pytest_mock import MockerFixture


class MockNNModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(1, num_classes)
        self.head.weight.data = torch.arange(num_classes, dtype=torch.float32).reshape(num_classes, 1)
        self.head.bias.data = torch.arange(num_classes, dtype=torch.float32)


class TestOTXModel:
    def test_init(self, monkeypatch):
        monkeypatch.setattr(OTXModel, "input_size_multiplier", 10, raising=False)
        with pytest.raises(ValueError, match="Input size should be a multiple"):
            OTXModel(label_info=2, input_size=(1024, 1024))

    def test_smart_weight_loading(self, mocker) -> None:
        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(2)):
            prev_model = OTXModel(label_info=2)
            prev_model.label_info = ["car", "truck"]
            prev_state_dict = prev_model.state_dict()

        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3)):
            current_model = OTXModel(label_info=3)
            current_model.classification_layers = ["model.head.weight", "model.head.bias"]
            current_model.classification_layers = {
                "model.head.weight": {"stride": 1, "num_extra_classes": 0},
                "model.head.bias": {"stride": 1, "num_extra_classes": 0},
            }
            current_model.label_info = ["car", "bus", "truck"]
            current_model.load_state_dict_incrementally(
                {"state_dict": prev_state_dict, "label_info": prev_model.label_info},
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

    def test_lr_scheduler_step(self, mocker: MockerFixture) -> None:
        mock_linear_warmup_scheduler = mocker.create_autospec(spec=LinearWarmupScheduler)
        mock_main_scheduler = mocker.create_autospec(spec=torch.optim.lr_scheduler.LRScheduler)

        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3)):
            current_model = OTXModel(label_info=3)

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

    def test_v1_checkpoint_loading(self, mocker):
        model = OTXModel(label_info=3)
        mocker.patch.object(model, "load_from_otx_v1_ckpt", return_value={})
        v1_ckpt = {
            "model": {"state_dict": {"backbone": torch.randn(2, 2)}},
            "labels": {"label_0": (), "label_1": (), "label_2": ()},
            "VERSION": 1,
        }
        assert model.load_state_dict_incrementally(v1_ckpt) is None


class TestOVModel:
    @pytest.fixture()
    def input_batch(self) -> OTXBatchDataEntity:
        image = [torch.rand(3, 10, 10) for _ in range(3)]
        return OTXBatchDataEntity(3, image, [])

    @pytest.fixture()
    def model(self) -> OVModel:
        return OVModel(model_name="efficientnet-b0-pytorch", model_type="Classification")

    def test_create_model(self) -> None:
        OVModel(model_name="efficientnet-b0-pytorch", model_type="Classification", force_cpu=False)

    def test_customize_inputs(self, model, input_batch) -> None:
        inputs = model._customize_inputs(input_batch)
        assert isinstance(inputs, dict)
        assert "inputs" in inputs
        assert inputs["inputs"][1].shape == np.transpose(input_batch.images[1].numpy(), (1, 2, 0)).shape

    def test_forward(self, model, input_batch) -> None:
        model._customize_outputs = lambda x, _: x
        outputs = model.forward(input_batch)
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert isinstance(outputs[2], ClassificationResult)

    def test_dummy_input(self, model: OVModel):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
