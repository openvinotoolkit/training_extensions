import numpy as np
import pytest
from PIL import Image

from otx.algorithms.classification.adapters.mmcls.datasets.pipelines.otx_pipelines import (
    GaussianBlur,
    LoadImageFromOTXDataset,
    LoadResizeDataFromOTXDataset,
    ResizeTo,
    OTXColorJitter,
    PILImageToNDArray,
    PostAug,
    RandomAppliedTrans,
)
from otx.core.data.caching import MemCacheHandlerSingleton
from tests.test_suite.e2e_test_system import e2e_pytest_unit

from .test_datasets import create_cls_dataset


@pytest.fixture(scope="module")
def inputs_np():
    return {"img": np.random.randint(0, 10, (16, 16, 3), dtype=np.uint8), "img_fields": ["img"]}


@pytest.fixture(scope="module")
def inputs_PIL():
    return {
        "img": Image.fromarray(np.random.randint(0, 10, (16, 16, 3), dtype=np.uint8)),
    }


@e2e_pytest_unit
@pytest.mark.parametrize("to_float32", [False, True])
def test_load_image_from_otx_dataset_call(to_float32):
    """Test LoadImageFromOTXDataset."""
    otx_dataset, labels = create_cls_dataset()
    load_image_from_otx_dataset = LoadImageFromOTXDataset(to_float32)
    results = dict(
        dataset_item=otx_dataset[0],
        width=otx_dataset[0].width,
        height=otx_dataset[0].height,
        index=0,
        ann_info=dict(label_list=labels),
    )

    results = load_image_from_otx_dataset(results)

    assert "filename" in results
    assert "ori_filename" in results
    assert "img" in results
    assert "img_shape" in results
    assert "ori_shape" in results
    assert "pad_shape" in results
    assert "img_norm_cfg" in results
    assert "img_fields" in results
    assert isinstance(results["img"], np.ndarray)


@e2e_pytest_unit
def test_load_resize_data_from_otx_dataset_call(mocker):
    """Test LoadResizeDataFromOTXDataset."""
    otx_dataset, labels = create_cls_dataset()
    MemCacheHandlerSingleton.create("singleprocessing", otx_dataset[0].numpy.size)
    op = LoadResizeDataFromOTXDataset(
        resize_cfg=dict(type="Resize", size=(4, 4)),  # 8x8 -> 4x4
    )
    src_dict = dict(
        dataset_item=otx_dataset[0],
        width=otx_dataset[0].width,
        height=otx_dataset[0].height,
        index=0,
        ann_info=dict(label_list=labels),
    )
    dst_dict = op(src_dict)
    assert dst_dict["ori_shape"][0] == 8
    assert dst_dict["img_shape"][0] == 4
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    op._load_img_op = mocker.MagicMock()
    dst_dict_from_cache = op(src_dict)
    assert op._load_img_op.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert dst_dict["ann_info"] == dst_dict_from_cache["ann_info"]


@e2e_pytest_unit
def test_load_resize_data_from_otx_dataset_downscale_only(mocker):
    """Test LoadResizeDataFromOTXDataset."""
    otx_dataset, labels = create_cls_dataset()
    MemCacheHandlerSingleton.create("singleprocessing", otx_dataset[0].numpy.size)
    op = LoadResizeDataFromOTXDataset(
        resize_cfg=dict(type="Resize", size=(12, 12), downscale_only=True),  # 8x8 -> 12x12
    )
    src_dict = dict(
        dataset_item=otx_dataset[0],
        width=otx_dataset[0].width,
        height=otx_dataset[0].height,
        index=0,
        ann_info=dict(label_list=labels),
    )
    dst_dict = op(src_dict)
    assert dst_dict["ori_shape"][0] == 8
    assert dst_dict["img_shape"][0] == 8  # Skipped upscale
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    op._load_img_op = mocker.MagicMock()
    dst_dict_from_cache = op(src_dict)
    assert op._load_img_op.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert dst_dict["ann_info"] == dst_dict_from_cache["ann_info"]


@e2e_pytest_unit
def test_resize_to(mocker, inputs_np):
    """Test LoadResizeDataFromOTXDataset."""
    otx_dataset, labels = create_cls_dataset()
    src_dict = dict(
        **inputs_np,
        ori_shape=(16, 16),
        img_shape=(16, 16),
    )
    # Test downscale
    op = ResizeTo(size=(4, 4))
    dst_dict = op(src_dict)
    assert dst_dict["ori_shape"][0] == 16
    assert dst_dict["img_shape"][0] == 4
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    # Test upscale from output
    op = ResizeTo(size=(8, 8))
    dst_dict = op(dst_dict)
    assert dst_dict["ori_shape"][0] == 16
    assert dst_dict["img_shape"][0] == 8
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    # Test same size from output
    op = ResizeTo(size=(8, 8))
    op._resize_img = mocker.MagicMock()
    dst_dict = op(dst_dict)
    assert dst_dict["ori_shape"][0] == 16
    assert dst_dict["img_shape"][0] == 8
    assert op._resize_img.call_count == 0  # _resize_img() should not be called


@e2e_pytest_unit
def test_random_applied_transforms(mocker, inputs_np):
    """Test RandomAppliedTrans."""
    mocker.patch(
        "otx.algorithms.classification.adapters.mmcls.datasets.pipelines.otx_pipelines.build_from_cfg",
        return_value=lambda x: x,
    )

    random_applied_transforms = RandomAppliedTrans(transforms=[dict()])

    results = random_applied_transforms(inputs_np)

    assert isinstance(results, dict)
    assert "img" in results
    assert repr(random_applied_transforms) == "RandomAppliedTrans"


@e2e_pytest_unit
def test_otx_color_jitter(inputs_np):
    """Test OTXColorJitter."""
    otx_color_jitter = OTXColorJitter()

    results = otx_color_jitter(inputs_np)

    assert isinstance(results, dict)
    assert "img" in results


@e2e_pytest_unit
def test_gaussian_blur(inputs_np):
    """Test GaussianBlur."""
    gaussian_blur = GaussianBlur(sigma_min=0.1, sigma_max=0.2)

    results = gaussian_blur(inputs_np)

    assert isinstance(results, dict)
    assert "img" in results
    assert repr(gaussian_blur) == "GaussianBlur"


@e2e_pytest_unit
def test_pil_image_to_nd_array(inputs_PIL) -> None:
    """Test PILImageToNDArray."""
    pil_image_to_nd_array = PILImageToNDArray(keys=["img"])

    results = pil_image_to_nd_array(inputs_PIL)

    assert "img" in results
    assert isinstance(results["img"], np.ndarray)
    assert repr(pil_image_to_nd_array) == "PILImageToNDArray"


@e2e_pytest_unit
def test_post_aug(mocker, inputs_np):
    """Test PostAug."""
    mocker.patch(
        "otx.algorithms.classification.adapters.mmcls.datasets.pipelines.otx_pipelines.Compose",
        return_value=lambda x: x,
    )

    post_aug = PostAug(keys=dict(orig=lambda x: x))

    results = post_aug(inputs_np)

    assert isinstance(results, dict)
    assert "img" in results and "img" in results["img_fields"]
    assert "orig" in results and "orig" in results["img_fields"]
    assert repr(post_aug) == "PostAug"
