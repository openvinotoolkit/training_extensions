from typing import Any, Dict

import numpy as np
import pytest
import torch
from mmcv.parallel import DataContainer
from PIL import Image

from otx.algorithms.segmentation.adapters.mmseg.datasets.pipelines.transforms import (
    BranchImage,
    DefaultFormatBundle,
    NDArrayToPILImage,
    Normalize,
    PILImageToNDArray,
    RandomColorJitter,
    RandomGaussianBlur,
    RandomGrayscale,
    RandomResizedCrop,
    RandomSolarization,
    TwoCropTransform,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@pytest.fixture(scope="module")
def inputs_np():
    return {
        "img": np.random.randint(0, 10, (16, 16, 3), dtype=np.uint8),
        "gt_semantic_seg": np.random.rand(16, 16),
        "flip": True,
    }


@pytest.fixture(scope="module")
def inputs_PIL():
    return {
        "img": Image.fromarray(np.random.randint(0, 10, (16, 16, 3), dtype=np.uint8)),
        "gt_semantic_seg": np.random.randint(0, 5, (16, 16), dtype=np.uint8),
        "seg_fields": ["gt_semantic_seg"],
        "ori_shape": (16, 16, 3),
    }


class TestNDArrayToPILImage:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)}
        self.nd_array_to_pil_image: NDArrayToPILImage = NDArrayToPILImage(keys=["img"])

    @e2e_pytest_unit
    def test_call(self) -> None:
        converted_img: dict = self.nd_array_to_pil_image(self.results)
        assert "img" in converted_img
        assert isinstance(converted_img["img"], Image.Image)

    @e2e_pytest_unit
    def test_repr(self) -> None:
        assert str(self.nd_array_to_pil_image) == "NDArrayToPILImage"


class TestPILImageToNDArray:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": Image.new("RGB", (3, 3))}
        self.pil_image_to_nd_array: PILImageToNDArray = PILImageToNDArray(keys=["img"])

    @e2e_pytest_unit
    def test_call(self) -> None:
        converted_array: dict = self.pil_image_to_nd_array(self.results)
        assert "img" in converted_array
        assert isinstance(converted_array["img"], np.ndarray)

    @e2e_pytest_unit
    def test_repr(self) -> None:
        assert str(self.pil_image_to_nd_array) == "PILImageToNDArray"


class TestRandomResizedCrop:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": Image.new("RGB", (10, 16)), "img_shape": (10, 16), "ori_shape": (10, 16)}
        self.random_resized_crop: RandomResizedCrop = RandomResizedCrop((5, 5), (0.5, 1.0))

    @e2e_pytest_unit
    def test_call(self) -> None:
        cropped_img: dict = self.random_resized_crop(self.results)
        assert cropped_img["img_shape"] == (5, 5)
        assert cropped_img["ori_shape"] == (10, 16)


class TestRandomSolarization:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.results: dict = {"img": np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)}
        self.random_solarization: RandomSolarization = RandomSolarization(p=1.0)

    @e2e_pytest_unit
    def test_call(self) -> None:
        solarized: dict = self.random_solarization(self.results)
        assert "img" in solarized
        assert isinstance(solarized["img"], np.ndarray)

    @e2e_pytest_unit
    def test_repr(self) -> None:
        assert str(self.random_solarization) == "RandomSolarization"


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


class TestTwoCropTransform:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch(
            "otx.algorithms.segmentation.adapters.mmseg.datasets.pipelines.transforms.build_from_cfg",
            return_value=lambda x: x,
        )
        self.two_crop_transform = TwoCropTransform(view0=[], view1=[])

    @e2e_pytest_unit
    def test_call(self, mocker, inputs_np: Dict[str, Any]) -> None:
        """Test __call__."""
        results = self.two_crop_transform(inputs_np)

        assert isinstance(results, dict)
        assert "img" in results and results["img"].ndim == 4
        assert "gt_semantic_seg" in results and results["gt_semantic_seg"].ndim == 3
        assert "flip" in results and isinstance(results["flip"], list)

    @e2e_pytest_unit
    def test_call_with_single_pipeline(self, mocker, inputs_np: Dict[str, Any]) -> None:
        """Test __call__ with single pipeline."""
        self.two_crop_transform.is_both = False

        results = self.two_crop_transform(inputs_np)

        assert isinstance(results, dict)
        assert "img" in results and results["img"].ndim == 3
        assert "gt_semantic_seg" in results and results["gt_semantic_seg"].ndim == 2
        assert "flip" in results and isinstance(results["flip"], bool)


@e2e_pytest_unit
def test_random_resized_crop(inputs_PIL: Dict[str, Any]) -> None:
    """Test RandomResizedCrop."""
    random_resized_crop = RandomResizedCrop(size=(8, 8))

    results = random_resized_crop(inputs_PIL)

    assert isinstance(results, dict)
    assert "img" in results and results["img"].size == (8, 8)
    assert "gt_semantic_seg" in results and results["gt_semantic_seg"].shape == (8, 8)
    assert "img_shape" in results
    assert "ori_shape" in results
    assert "scale_factor" in results


@e2e_pytest_unit
def test_random_color_jitter(inputs_PIL: Dict[str, Any]) -> None:
    """Test RandomColorJitter."""
    random_color_jitter = RandomColorJitter(p=1.0)

    results = random_color_jitter(inputs_PIL)

    assert isinstance(results, dict)
    assert "img" in results


@e2e_pytest_unit
def test_random_grayscale(inputs_PIL: Dict[str, Any]) -> None:
    """Test RandomGrayscale."""
    random_grayscale = RandomGrayscale()

    results = random_grayscale(inputs_PIL)

    assert isinstance(results, dict)
    assert "img" in results


@e2e_pytest_unit
def test_random_gaussian_blur(inputs_PIL: Dict[str, Any]) -> None:
    """Test RandomGaussianBlur."""
    random_gaussian_blur = RandomGaussianBlur(p=1.0, kernel_size=3)

    results = random_gaussian_blur(inputs_PIL)

    assert isinstance(results, dict)
    assert "img" in results


@e2e_pytest_unit
def test_random_solarization(inputs_np: Dict[str, Any]) -> None:
    """Test RandomSolarization."""
    random_solarization = RandomSolarization(p=1.0)

    results = random_solarization(inputs_np)

    assert isinstance(results, dict)
    assert "img" in results
    assert repr(random_solarization) == "RandomSolarization"


@e2e_pytest_unit
def test_nd_array_to_pil_image(inputs_np: Dict[str, Any]) -> None:
    """Test NDArrayToPILImage."""
    nd_array_to_pil_image = NDArrayToPILImage(keys=["img"])

    results = nd_array_to_pil_image(inputs_np)

    assert "img" in results
    assert isinstance(results["img"], Image.Image)
    assert repr(nd_array_to_pil_image) == "NDArrayToPILImage"


@e2e_pytest_unit
def test_pil_image_to_nd_array(inputs_PIL: Dict[str, Any]) -> None:
    """Test PILImageToNDArray."""
    pil_image_to_nd_array = PILImageToNDArray(keys=["img"])

    results = pil_image_to_nd_array(inputs_PIL)

    assert "img" in results
    assert isinstance(results["img"], np.ndarray)
    assert repr(pil_image_to_nd_array) == "PILImageToNDArray"
