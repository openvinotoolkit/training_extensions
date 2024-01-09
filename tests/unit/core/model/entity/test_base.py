import torch
from otx.core.model.entity.base import OTXModel


class MockNNModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 1024)
        self.head = torch.nn.Linear(3, num_classes)


class TestOTXModel:
    def test_smart_weight_loading(self, mocker) -> None:
        mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(2))
        prev_model = OTXModel()

        mocker.patch.object(OTXModel, "_create_model", return_value=MockNNModule(3))
        current_model = OTXModel()
        current_model.classification_layers = ["model.head.weight", "model.head.bias"]

        prev_classes = ["car", "truck"]
        current_classes = ["car", "bus", "truck"]
        indices = torch.Tensor([0, 2]).to(torch.int32)

        current_model.register_load_state_dict_pre_hook(current_classes, prev_classes)
        current_model.load_state_dict(prev_model.state_dict())

        assert torch.all(
            current_model.state_dict()["model.backbone.weight"] == prev_model.state_dict()["model.backbone.weight"],
        )
        assert torch.all(
            current_model.state_dict()["model.backbone.bias"] == prev_model.state_dict()["model.backbone.bias"],
        )
        assert torch.all(
            current_model.state_dict()["model.head.weight"].index_select(0, indices)
            == prev_model.state_dict()["model.head.weight"],
        )
        assert torch.all(
            current_model.state_dict()["model.head.bias"].index_select(0, indices)
            == prev_model.state_dict()["model.head.bias"],
        )
