import numpy as np
import pytest
from PIL import Image

from otx.algorithms.segmentation.adapters.mmseg.data.pipelines import (
    NDArrayToPILImage,
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
        "flip": True
    }


@pytest.fixture(scope="module")
def inputs_PIL():
    return {
        "img": Image.fromarray(np.random.randint(0, 10, (16, 16, 3), dtype=np.uint8)),
        "gt_semantic_seg": np.random.randint(0, 5, (16, 16), dtype=np.uint8),
        "seg_fields": ["gt_semantic_seg"],
        "ori_shape": (16, 16, 3)
    }


@e2e_pytest_unit
def test_two_crop_transform(mocker, inputs_np) -> None:
    """Test TwoCropTransform."""
    mocker.patch(
        "otx.algorithms.segmentation.adapters.mmseg.data.pipelines.build_from_cfg",
        return_value=lambda x: x)

    two_crop_transform = TwoCropTransform(view0=[], view1=[])

    results = two_crop_transform(inputs_np)

    assert isinstance(results, dict)
    assert "img" in results and results["img"].ndim == 4
    assert "gt_semantic_seg" in results and results["gt_semantic_seg"].ndim == 3
    assert "flip" in results and isinstance(results["flip"], list)


@e2e_pytest_unit
def test_random_resized_crop(inputs_PIL) -> None:
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
def test_random_color_jitter(inputs_PIL) -> None:
    """Test RandomColorJitter."""
    random_color_jitter = RandomColorJitter(p=1.)

    results = random_color_jitter(inputs_PIL)

    assert isinstance(results, dict)
    assert "img" in results


@e2e_pytest_unit
def test_random_grayscale(inputs_PIL) -> None:
    """Test RandomGrayscale."""
    random_grayscale = RandomGrayscale()

    results = random_grayscale(inputs_PIL)

    assert isinstance(results, dict)
    assert "img" in results


@e2e_pytest_unit
def test_random_gaussian_blur(inputs_PIL) -> None:
    """Test RandomGaussianBlur."""
    random_gaussian_blur = RandomGaussianBlur(p=1., kernel_size=3)

    results = random_gaussian_blur(inputs_PIL)

    assert isinstance(results, dict)
    assert "img" in results


@e2e_pytest_unit
def test_random_solarization(inputs_np) -> None:
    """Test RandomSolarization."""
    random_solarization = RandomSolarization(p=1.)

    results = random_solarization(inputs_np)

    assert isinstance(results, dict)
    assert "img" in results
    assert repr(random_solarization) == "RandomSolarization"


@e2e_pytest_unit
def test_nd_array_to_pil_image(inputs_np) -> None:
    """Test NDArrayToPILImage."""
    nd_array_to_pil_image = NDArrayToPILImage(keys=["img"])
    
    results = nd_array_to_pil_image(inputs_np)

    assert "img" in results
    assert isinstance(results["img"], Image.Image)
    assert repr(nd_array_to_pil_image) == "NDArrayToPILImage"


@e2e_pytest_unit
def test_pil_image_to_nd_array(inputs_PIL) -> None:
    """Test PILImageToNDArray."""
    pil_image_to_nd_array = PILImageToNDArray(keys=["img"])

    results = pil_image_to_nd_array(inputs_PIL)

    assert "img" in results
    assert isinstance(results["img"], np.ndarray)
    assert repr(pil_image_to_nd_array) == "PILImageToNDArray"
