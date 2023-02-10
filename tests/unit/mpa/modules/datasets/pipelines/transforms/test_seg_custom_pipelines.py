import numpy as np
import pytest
import torch
from mmcv.parallel import DataContainer

from otx.mpa.modules.datasets.pipelines.transforms.seg_custom_pipelines import (
    BranchImage,
    DefaultFormatBundle,
    Normalize,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestNormalize:
    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "mean,std,to_rgb,expected",
        [
            (1.0, 1.0, True, np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)),
            (1.0, 1.0, False, np.array([[[-1.0, 0.0, 0.0]]], dtype=np.float32)),
        ],
    )
    def test_call(self, mean: float, std: float, to_rgb: bool, expected: np.array) -> None:
        """Test __call__."""
        normalize = Normalize(mean=mean, std=std, to_rgb=to_rgb)
        inputs = dict(img=np.arange(3).reshape(1, 1, 3))

        results = normalize(inputs.copy())

        assert "img" in results
        assert "img_norm_cfg" in results
        assert np.all(results["img"] == expected)

    @e2e_pytest_unit
    @pytest.mark.parametrize("mean,std,to_rgb", [(1.0, 1.0, True)])
    def test_repr(self, mean: float, std: float, to_rgb: bool) -> None:
        """Test __repr__."""
        normalize = Normalize(mean=mean, std=std, to_rgb=to_rgb)

        assert repr(normalize) == normalize.__class__.__name__ + f"(mean={mean}, std={std}, to_rgb=" f"{to_rgb})"


class TestDefaultFormatBundle:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.default_format_bundle = DefaultFormatBundle()

    @e2e_pytest_unit
    @pytest.mark.parametrize("img", [np.ones((1, 1)), np.ones((1, 1, 1)), np.ones((1, 1, 1, 1))])
    @pytest.mark.parametrize("gt_semantic_seg,pixel_weights", [(np.ones((1, 1)), np.ones((1, 1)))])
    def test_call(self, img: np.array, gt_semantic_seg: np.array, pixel_weights: np.array) -> None:
        """Test __call__."""
        inputs = dict(img=img, gt_semantic_seg=gt_semantic_seg, pixel_weights=pixel_weights)

        results = self.default_format_bundle(inputs.copy())

        assert isinstance(results, dict)
        assert "img" in results
        assert isinstance(results["img"], DataContainer)
        assert len(results["img"].data.shape) >= 3
        assert results["img"].data.dtype == torch.float32
        assert "gt_semantic_seg" in results
        assert len(results["gt_semantic_seg"].data.shape) == len(inputs["gt_semantic_seg"].shape) + 1
        assert results["gt_semantic_seg"].data.dtype == torch.int64
        assert "pixel_weights" in results
        assert len(results["pixel_weights"].data.shape) == len(inputs["pixel_weights"].shape) + 1
        assert results["pixel_weights"].data.dtype == torch.float32

    @e2e_pytest_unit
    @pytest.mark.parametrize("img", [np.ones((1,))])
    def test_call_invalid_shape(self, img: np.array):
        inputs = dict(img=img)

        with pytest.raises(ValueError):
            self.default_format_bundle(inputs.copy())

    @e2e_pytest_unit
    def test_repr(self) -> None:
        """Test __repr__."""
        assert repr(self.default_format_bundle) == self.default_format_bundle.__class__.__name__


class TestBranchImage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.branch_image = BranchImage(key_map={"key1": "key2"})

    @e2e_pytest_unit
    def test_call(self) -> None:
        """Test __call__."""
        inputs = dict(key1="key1", img_fields=["key1"])

        results = self.branch_image(inputs.copy())

        assert isinstance(results, dict)
        assert "key2" in results
        assert results["key1"] == results["key2"]
        assert "key2" in results["img_fields"]

    @e2e_pytest_unit
    def test_repr(self) -> None:
        """Test __repr__."""
        assert repr(self.branch_image) == self.branch_image.__class__.__name__
