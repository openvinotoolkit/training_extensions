from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from otx.algo.segmentation.backbones import mscan as target_file
from otx.algo.segmentation.backbones.mscan import NNMSCAN, DropPath, drop_path


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_drop_path(dim: int):
    size = [10] + [2] * dim
    x = torch.ones(size)
    out = drop_path(x, 0.5, True)

    assert out.size() == x.size()
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_drop_path_not_train():
    x = torch.ones(2, 2, 2, 2)
    out = drop_path(x, 0.5, False)

    assert (x == out).all()
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_drop_path_zero_prob():
    x = torch.ones(2, 2, 2, 2)
    out = drop_path(x, 0.0, True)

    assert (x == out).all()
    assert out.dtype == x.dtype
    assert out.device == x.device


class TestDropPath:
    def test_init(self):
        drop_prob = 0.3
        drop_path = DropPath(drop_prob)

        assert drop_path.drop_prob == drop_prob

    def test_forward(self):
        drop_prob = 0.5
        drop_path = DropPath(drop_prob)
        drop_path.train()
        x = torch.ones(2, 2, 2, 2)

        out = drop_path.forward(x)

        assert out.size() == x.size()
        assert out.dtype == x.dtype
        assert out.device == x.device


class TestMSCABlock:
    def test_init(self):
        num_stages = 4
        mscan = NNMSCAN(num_stages=num_stages)

        for i in range(num_stages):
            assert hasattr(mscan, f"patch_embed{i + 1}")
            assert hasattr(mscan, f"block{i + 1}")
            assert hasattr(mscan, f"norm{i + 1}")

    def test_forward(self):
        num_stages = 4
        mscan = NNMSCAN(num_stages=num_stages)
        x = torch.rand(8, 3, 3, 3)
        out = mscan.forward(x)

        assert len(out) == num_stages

    @pytest.fixture()
    def mock_load_from_http(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "load_from_http")

    @pytest.fixture()
    def mock_load_checkpoint_to_model(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "load_checkpoint_to_model")

    @pytest.fixture()
    def pretrained_weight(self, tmp_path) -> str:
        weight = tmp_path / "pretrained.pth"
        weight.touch()
        return str(weight)

    @pytest.fixture()
    def mock_torch_load(self, mocker) -> MagicMock:
        return mocker.patch("otx.algo.segmentation.backbones.mscan.torch.load")

    def test_load_pretrained_weights(self, pretrained_weight, mock_torch_load, mock_load_checkpoint_to_model):
        NNMSCAN(pretrained_weights=pretrained_weight)

        mock_torch_load.assert_called_once_with(pretrained_weight, "cpu")
        mock_load_checkpoint_to_model.assert_called_once()

    def test_load_pretrained_weights_from_url(self, mock_load_from_http, mock_load_checkpoint_to_model):
        pretrained_weight = "www.fake.com/fake.pth"
        NNMSCAN(pretrained_weights=pretrained_weight)

        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        mock_load_from_http.assert_called_once_with(filename=pretrained_weight, map_location="cpu", model_dir=cache_dir)
        mock_load_checkpoint_to_model.assert_called_once()
