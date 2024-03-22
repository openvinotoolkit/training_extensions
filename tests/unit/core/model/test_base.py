import numpy as np
import pytest
import torch
from openvino.model_api.models.utils import ClassificationResult
from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.model.base import OTXModel, OVModel


class MockNNModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(1, num_classes)
        self.head.weight.data = torch.arange(num_classes, dtype=torch.float32).reshape(num_classes, 1)
        self.head.bias.data = torch.arange(num_classes, dtype=torch.float32)


class TestOTXModel:
    def test_smart_weight_loading(self, mocker) -> None:
        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(2)):
            prev_model = OTXModel(num_classes=2)
            prev_model.label_info = ["car", "truck"]
            prev_state_dict = prev_model.state_dict()

        with mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3)):
            current_model = OTXModel(num_classes=3)
            current_model.classification_layers = ["model.head.weight", "model.head.bias"]
            current_model.classification_layers = {
                "model.head.weight": {"stride": 1, "num_extra_classes": 0},
                "model.head.bias": {"stride": 1, "num_extra_classes": 0},
            }
            current_model.label_info = ["car", "bus", "truck"]
            current_model.load_state_dict(prev_state_dict)
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


class TestOVModel:
    @pytest.fixture()
    def input_batch(self) -> OTXBatchDataEntity:
        image = [torch.rand(3, 10, 10) for _ in range(3)]
        return OTXBatchDataEntity(3, image, [])

    @pytest.fixture()
    def model(self) -> OVModel:
        return OVModel(num_classes=2, model_name="efficientnet-b0-pytorch", model_type="Classification")

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
