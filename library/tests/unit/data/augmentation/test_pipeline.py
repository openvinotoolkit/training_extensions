# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CPU and GPU augmentation pipelines and the hybrid approach."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import kornia.augmentation as kornia_aug
import pytest
import torch
import torchvision.transforms.v2 as tvt_v2
from torch import nn
from torchvision import tv_tensors

from getitune.config.data import IntensityConfig, SubsetConfig
from getitune.data.augmentation.pipeline import (
    CPUAugmentationPipeline,
    GPUAugmentationPipeline,
    _configure_input_size,
    _eval_input_size_str,
    _IntensityAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(h: int = 64, w: int = 64, c: int = 3, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Create a random image tensor (C, H, W)."""
    if dtype == torch.uint8:
        return torch.randint(0, 256, (c, h, w), dtype=torch.uint8)
    if dtype == torch.float32:
        return torch.rand(c, h, w)
    if dtype == torch.int32:
        return torch.randint(0, 65536, (c, h, w), dtype=torch.int32)
    return torch.rand(c, h, w)


def _make_batched_images(batch_size: int = 2, c: int = 3, h: int = 32, w: int = 32) -> torch.Tensor:
    """Create batched float images (B, C, H, W) in [0, 1]."""
    return torch.rand(batch_size, c, h, w)


@dataclass
class _SimpleSample:
    """Minimal sample for testing CPU pipeline forward."""

    image: torch.Tensor
    bboxes: tv_tensors.BoundingBoxes | None = None
    masks: torch.Tensor | None = None
    label: torch.Tensor | None = None
    img_info: Any = None


def _make_sample(
    h: int = 64,
    w: int = 64,
    dtype: torch.dtype = torch.uint8,
    with_bboxes: bool = False,
) -> _SimpleSample:
    """Build a minimal sample for CPU pipeline tests."""
    image = _make_image(h, w, dtype=dtype)
    bboxes = None
    label = None
    if with_bboxes:
        bboxes = tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
            torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )
        label = torch.tensor([0])
    return _SimpleSample(image=image, bboxes=bboxes, label=label)


# ===================================================================
# CPUAugmentationPipeline tests
# ===================================================================


class TestCPUAugmentationPipelineInit:
    """Tests for CPUAugmentationPipeline construction."""

    def test_empty_pipeline(self):
        pipeline = CPUAugmentationPipeline()
        assert len(pipeline.augmentations) == 0

    def test_pipeline_with_transforms(self):
        transforms: list[nn.Module] = [tvt_v2.RandomHorizontalFlip(p=0.5), tvt_v2.ToDtype(torch.float32, scale=True)]
        pipeline = CPUAugmentationPipeline(transforms)
        assert len(pipeline.augmentations) == 2

    def test_pipeline_is_nn_module(self):
        pipeline = CPUAugmentationPipeline()
        assert isinstance(pipeline, nn.Module)

    def test_augmentations_are_module_list(self):
        transforms: list[nn.Module] = [tvt_v2.RandomHorizontalFlip(p=0.5)]
        pipeline = CPUAugmentationPipeline(transforms)
        assert isinstance(pipeline.augmentations, nn.ModuleList)


class TestCPUAugmentationPipelineListAvailable:
    """Tests for list_available_transforms."""

    def test_returns_list(self):
        result = CPUAugmentationPipeline.list_available_transforms()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_are_transform_subclasses(self):
        result = CPUAugmentationPipeline.list_available_transforms()
        for cls in result:
            assert issubclass(cls, tvt_v2.Transform)


class TestCPUAugmentationPipelineDispatchTransform:
    """Tests for _dispatch_transform."""

    def test_dict_config(self):
        cfg = {
            "class_path": "torchvision.transforms.v2.RandomHorizontalFlip",
            "init_args": {"p": 0.5},
        }
        result = CPUAugmentationPipeline._dispatch_transform(cfg)
        assert isinstance(result, tvt_v2.RandomHorizontalFlip)

    def test_already_instantiated(self):
        transform = tvt_v2.RandomHorizontalFlip(p=0.5)
        result = CPUAugmentationPipeline._dispatch_transform(transform)
        assert result is transform

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="CPUAugmentationPipeline accepts only"):
            CPUAugmentationPipeline._dispatch_transform("bad_value")  # type: ignore[arg-type]


class TestCPUAugmentationPipelineFromConfig:
    """Tests for from_config."""

    def test_empty_config(self):
        """With default IntensityConfig, an empty augmentations_cpu still gets the intensity transform."""
        config = SubsetConfig(augmentations_cpu=[], input_size=None)
        pipeline = CPUAugmentationPipeline.from_config(config)
        # Default IntensityConfig(mode='scale_to_unit') always prepends an intensity transform
        assert len(pipeline.augmentations) == 1
        assert isinstance(pipeline.augmentations[0], _IntensityAdapter)

    def test_empty_config_no_intensity(self):
        """With intensity=None, no transforms are created."""
        config = SubsetConfig(augmentations_cpu=[], input_size=None)
        config.intensity = None  # type: ignore[assignment]
        pipeline = CPUAugmentationPipeline.from_config(config)
        assert len(pipeline.augmentations) == 0

    def test_with_augmentations(self):
        config = SubsetConfig(
            augmentations_cpu=[
                {
                    "class_path": "torchvision.transforms.v2.RandomHorizontalFlip",
                    "init_args": {"p": 0.5},
                },
            ],
            input_size=None,
        )
        pipeline = CPUAugmentationPipeline.from_config(config)
        # 1 intensity (default) + 1 user augmentation
        assert len(pipeline.augmentations) == 2
        assert isinstance(pipeline.augmentations[0], _IntensityAdapter)
        assert isinstance(pipeline.augmentations[1], tvt_v2.RandomHorizontalFlip)

    def test_nn_module_passthrough(self):
        """Pre-instantiated nn.Module should be passed through directly."""
        flip = tvt_v2.RandomHorizontalFlip(p=1.0)
        config = SubsetConfig(augmentations_cpu=[flip], input_size=None)  # type: ignore[arg-type]
        pipeline = CPUAugmentationPipeline.from_config(config)
        # 1 intensity (default) + 1 nn.Module passthrough
        assert len(pipeline.augmentations) == 2
        assert isinstance(pipeline.augmentations[1], tvt_v2.RandomHorizontalFlip)

    def test_unsupported_config_type_raises(self):
        config = SubsetConfig(augmentations_cpu=["bad_value"], input_size=None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported augmentation config type"):
            CPUAugmentationPipeline.from_config(config)

    def test_intensity_config_prepended(self):
        """IntensityConfig should add an intensity transform as the first augmentation."""
        config = SubsetConfig(
            augmentations_cpu=[
                {
                    "class_path": "torchvision.transforms.v2.RandomHorizontalFlip",
                    "init_args": {"p": 0.5},
                },
            ],
            intensity=IntensityConfig(storage_dtype="uint8", mode="scale_to_unit"),
            input_size=None,
        )
        pipeline = CPUAugmentationPipeline.from_config(config)
        # intensity transform (wrapped) + 1 user augmentation
        assert len(pipeline.augmentations) == 2
        # First transform should be an _IntensityAdapter wrapping ScaleToUnit
        from getitune.data.augmentation.intensity import ScaleToUnit

        assert isinstance(pipeline.augmentations[0], _IntensityAdapter)
        # The inner nn.Sequential should contain ScaleToUnit
        inner = pipeline.augmentations[0].transform
        assert isinstance(inner[0], ScaleToUnit)  # type: ignore[bad-index]

    def test_intensity_config_none_no_prepend(self):
        """No intensity config → no prepended transform."""
        config = SubsetConfig(
            augmentations_cpu=[
                {
                    "class_path": "torchvision.transforms.v2.RandomHorizontalFlip",
                    "init_args": {"p": 0.5},
                },
            ],
            input_size=None,
        )
        # Manually set intensity to None to simulate legacy config
        config.intensity = None  # type: ignore[assignment]
        pipeline = CPUAugmentationPipeline.from_config(config)
        assert len(pipeline.augmentations) == 1

    def test_intensity_uint16_range_scale(self):
        """IntensityConfig with range_scale mode should prepend RangeScale."""
        config = SubsetConfig(
            augmentations_cpu=[],
            intensity=IntensityConfig(
                storage_dtype="uint16",
                mode="range_scale",
                scale_factor=0.4,
                min_value=295.15,
                max_value=360.15,
            ),
            input_size=None,
        )
        pipeline = CPUAugmentationPipeline.from_config(config)
        from getitune.data.augmentation.intensity import RangeScale

        assert len(pipeline.augmentations) == 1
        assert isinstance(pipeline.augmentations[0], _IntensityAdapter)
        assert isinstance(pipeline.augmentations[0].transform[0], RangeScale)  # type: ignore[bad-index]


class TestCPUAugmentationPipelineInputSize:
    """Tests for _configure_input_size and _eval_input_size_str."""

    def test_eval_simple_tuple(self):
        result = _eval_input_size_str("(224, 224)")
        assert result == (224, 224)

    def test_eval_multiplication(self):
        result = _eval_input_size_str("(224, 224) * 2")
        assert result == (448, 448)

    def test_eval_division(self):
        result = _eval_input_size_str("(400, 400) / 2")
        assert result == (200, 200)

    def test_eval_int(self):
        result = _eval_input_size_str("224")
        assert result == 224

    def test_eval_negative(self):
        result = _eval_input_size_str("-1")
        assert result == -1

    def test_eval_bad_syntax_raises(self):
        """Addition is not supported in the safe eval — should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="Bad syntax"):
            _eval_input_size_str("(224, 224) + 1")

    def test_configure_input_size_no_placeholder(self):
        cfg = {
            "class_path": "torchvision.transforms.v2.Resize",
            "init_args": {"size": [224, 224]},
        }
        result = _configure_input_size(cfg, (320, 320))
        assert result["init_args"]["size"] == [224, 224]

    def test_configure_input_size_missing_raises(self):
        cfg = {
            "class_path": "torchvision.transforms.v2.Resize",
            "init_args": {"size": "$(input_size)"},
        }
        with pytest.raises(RuntimeError, match="input_size is set to None"):
            _configure_input_size(cfg, None)

    def test_configure_input_size_no_init_args(self):
        cfg = {"class_path": "torchvision.transforms.v2.RandomHorizontalFlip"}
        result = _configure_input_size(cfg, (224, 224))
        assert result == cfg


class TestCPUAugmentationPipelineForward:
    """Tests for forward pass."""

    def test_empty_pipeline_passthrough(self):
        pipeline = CPUAugmentationPipeline()
        sample = _make_sample()
        result = pipeline(sample)
        assert result is sample

    def test_native_torchvision_transform_applied(self):
        """Native torchvision transform should modify the image."""
        pipeline = CPUAugmentationPipeline([tvt_v2.ToDtype(torch.float32, scale=True)])
        sample = _make_sample(dtype=torch.uint8)
        result = pipeline(sample)
        assert result.image.dtype == torch.float32
        assert result.image.max() <= 1.0

    def test_custom_transform_called(self):
        """getitune-style transform (non-torchvision) should be called with sample."""

        class _IdentityTransform(nn.Module):
            def forward(self, sample):  # noqa: ANN202
                sample.image = sample.image.float() / 255.0
                return sample

        pipeline = CPUAugmentationPipeline([_IdentityTransform()])
        sample = _make_sample(dtype=torch.uint8)
        result = pipeline(sample)
        assert result.image.dtype == torch.float32

    def test_none_return_propagates(self):
        """If a transform returns None, forward should return None."""

        class _NoneTransform(nn.Module):
            def forward(self, sample) -> None:
                return None

        pipeline = CPUAugmentationPipeline([_NoneTransform()])
        sample = _make_sample()
        result = pipeline(sample)
        assert result is None

    def test_native_transform_detection(self):
        """Check _is_native_torchvision_transform correctly identifies transforms."""
        pipeline = CPUAugmentationPipeline()
        flip = tvt_v2.RandomHorizontalFlip(p=0.5)
        assert pipeline._is_native_torchvision_transform(flip)

        custom = nn.Identity()
        assert not pipeline._is_native_torchvision_transform(custom)

    def test_apply_native_transform_image_only(self):
        """_apply_native_transform with image-only sample."""
        pipeline = CPUAugmentationPipeline()
        sample = _SimpleSample(image=_make_image(32, 32, dtype=torch.uint8))
        transform = tvt_v2.ToDtype(torch.float32, scale=True)
        result = pipeline._apply_native_transform(transform, sample)  # type: ignore[arg-type]
        assert result.image.dtype == torch.float32

    def test_apply_native_transform_with_bboxes(self):
        """_apply_native_transform preserves bboxes through spatial transforms."""
        pipeline = CPUAugmentationPipeline()
        sample = _make_sample(64, 64, with_bboxes=True)
        # Use a transform that should keep image size the same
        transform = tvt_v2.RandomHorizontalFlip(p=1.0)  # Always flip
        result = pipeline._apply_native_transform(transform, sample)  # type: ignore[arg-type]
        assert result.bboxes is not None  # type: ignore[union-attr]

    def test_apply_native_transform_empty_sample(self):
        """Sample with no transformable fields returns unchanged."""
        pipeline = CPUAugmentationPipeline()
        sample = _SimpleSample(image=None)  # type: ignore[arg-type]
        transform = tvt_v2.RandomHorizontalFlip(p=0.5)
        result = pipeline._apply_native_transform(transform, sample)  # type: ignore[arg-type]
        assert result is sample

    def test_repr(self):
        transforms: list[nn.Module] = [tvt_v2.RandomHorizontalFlip(p=0.5)]
        pipeline = CPUAugmentationPipeline(transforms)
        r = repr(pipeline)
        assert "CPUAugmentationPipeline" in r
        assert "RandomHorizontalFlip" in r


class TestCPUAugmentationPipelineIntensityIntegration:
    """Tests that intensity transforms integrate correctly into the CPU pipeline."""

    def test_scale_to_unit_in_pipeline(self):
        """uint8 scale_to_unit through a full pipeline."""
        config = SubsetConfig(
            augmentations_cpu=[],
            intensity=IntensityConfig(storage_dtype="uint8", mode="scale_to_unit"),
            input_size=None,
        )
        pipeline = CPUAugmentationPipeline.from_config(config)
        sample = _make_sample(32, 32, dtype=torch.uint8)
        result = pipeline(sample)
        assert result.image.dtype == torch.float32
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_uint16_scale_to_unit_in_pipeline(self):
        """uint16 data with scale_to_unit mode."""
        config = SubsetConfig(
            augmentations_cpu=[],
            intensity=IntensityConfig(
                storage_dtype="uint16",
                mode="scale_to_unit",
                max_value=65535.0,
            ),
            input_size=None,
        )
        pipeline = CPUAugmentationPipeline.from_config(config)
        sample = _SimpleSample(image=torch.randint(0, 65536, (3, 32, 32), dtype=torch.int32))
        result = pipeline(sample)
        assert result.image.dtype == torch.float32
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_intensity_then_augmentation(self):
        """Intensity transform followed by spatial augmentation."""
        config = SubsetConfig(
            augmentations_cpu=[
                {
                    "class_path": "torchvision.transforms.v2.Resize",
                    "init_args": {"size": [16, 16]},
                },
            ],
            intensity=IntensityConfig(storage_dtype="uint8", mode="scale_to_unit"),
            input_size=None,
        )
        pipeline = CPUAugmentationPipeline.from_config(config)
        sample = _make_sample(64, 64, dtype=torch.uint8)
        result = pipeline(sample)
        assert result.image.shape == (3, 16, 16)
        assert result.image.dtype == torch.float32

    def test_repeat_channels_in_pipeline(self):
        """Intensity config with repeat_channels should replicate single-channel to 3."""
        config = SubsetConfig(
            augmentations_cpu=[],
            intensity=IntensityConfig(
                storage_dtype="uint16",
                mode="scale_to_unit",
                max_value=65535.0,
                repeat_channels=3,
            ),
            input_size=None,
        )
        pipeline = CPUAugmentationPipeline.from_config(config)
        # Single-channel uint16 image
        sample = _SimpleSample(image=torch.randint(0, 65536, (1, 32, 32), dtype=torch.int32))
        result = pipeline(sample)
        assert result.image.shape == (3, 32, 32)
        assert result.image.dtype == torch.float32


# ===================================================================
# GPUAugmentationPipeline tests
# ===================================================================


class TestGPUAugmentationPipelineInit:
    """Tests for GPUAugmentationPipeline construction."""

    def test_empty_pipeline(self):
        pipeline = GPUAugmentationPipeline()
        assert pipeline.aug_sequential is None
        assert pipeline.mean is None
        assert pipeline.std is None

    def test_pipeline_with_augmentations(self):
        augs: list[nn.Module] = [kornia_aug.RandomHorizontalFlip(p=0.5)]
        pipeline = GPUAugmentationPipeline(augs)
        assert pipeline.aug_sequential is not None

    def test_pipeline_is_nn_module(self):
        pipeline = GPUAugmentationPipeline()
        assert isinstance(pipeline, nn.Module)

    def test_data_keys_default(self):
        pipeline = GPUAugmentationPipeline()
        assert pipeline.data_keys == ["input"]

    def test_custom_data_keys(self):
        pipeline = GPUAugmentationPipeline([], data_keys=["input", "bbox_xyxy", "mask"])
        assert pipeline.data_keys == ["input", "bbox_xyxy", "mask"]

    def test_has_geometric_augs_with_flip(self):
        """Pipeline with RandomHorizontalFlip should flag geometric augs."""
        pipeline = GPUAugmentationPipeline([kornia_aug.RandomHorizontalFlip(p=0.5)])
        assert pipeline._has_geometric_augs is True

    def test_has_geometric_augs_with_affine(self):
        """Pipeline with RandomAffine should flag geometric augs."""
        pipeline = GPUAugmentationPipeline([kornia_aug.RandomAffine(degrees=10)])
        assert pipeline._has_geometric_augs is True

    def test_no_geometric_augs_with_normalize_only(self):
        """Pipeline with only Normalize should NOT flag geometric augs."""
        pipeline = GPUAugmentationPipeline(
            [kornia_aug.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))]
        )
        assert pipeline._has_geometric_augs is False

    def test_no_geometric_augs_empty(self):
        """Empty pipeline should NOT flag geometric augs."""
        pipeline = GPUAugmentationPipeline([])
        assert pipeline._has_geometric_augs is False

    def test_has_geometric_augs_mixed(self):
        """Pipeline with geometric + intensity should flag geometric augs."""
        pipeline = GPUAugmentationPipeline(
            [
                kornia_aug.RandomHorizontalFlip(p=0.5),
                kornia_aug.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
            ]
        )
        assert pipeline._has_geometric_augs is True


class TestGPUAugmentationPipelineNormalization:
    """Tests for normalization parameter extraction."""

    def test_extract_norm_from_kornia_normalize(self):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        augs: list[nn.Module] = [kornia_aug.Normalize(mean=mean, std=std)]
        pipeline = GPUAugmentationPipeline(augs)
        assert pipeline.mean is not None
        assert pipeline.std is not None
        assert len(pipeline.mean) == 3
        assert len(pipeline.std) == 3
        assert abs(pipeline.mean[0] - 0.485) < 1e-4

    def test_no_normalize_returns_none(self):
        augs: list[nn.Module] = [kornia_aug.RandomHorizontalFlip(p=0.5)]
        pipeline = GPUAugmentationPipeline(augs)
        assert pipeline.mean is None
        assert pipeline.std is None

    def test_extract_norm_among_others(self):
        """Normalization params found even when mixed with other augs."""
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.25, 0.25, 0.25])
        augs: list[nn.Module] = [kornia_aug.RandomHorizontalFlip(p=0.5), kornia_aug.Normalize(mean=mean, std=std)]
        pipeline = GPUAugmentationPipeline(augs)
        assert pipeline.mean is not None
        assert abs(pipeline.mean[0] - 0.5) < 1e-4


class TestGPUAugmentationPipelineListAvailable:
    """Tests for list_available_transforms."""

    def test_returns_list(self):
        result = GPUAugmentationPipeline.list_available_transforms()
        assert isinstance(result, list)
        assert len(result) > 0


class TestGPUAugmentationPipelineDispatchTransform:
    """Tests for _dispatch_transform."""

    def test_dict_config(self):
        cfg = {
            "class_path": "kornia.augmentation.RandomHorizontalFlip",
            "init_args": {"p": 0.5},
        }
        result = GPUAugmentationPipeline._dispatch_transform(cfg)
        assert isinstance(result, kornia_aug.RandomHorizontalFlip)

    def test_already_instantiated(self):
        aug = kornia_aug.RandomHorizontalFlip(p=0.5)
        result = GPUAugmentationPipeline._dispatch_transform(aug)
        assert result is aug

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="GPUAugmentationPipeline accepts only"):
            GPUAugmentationPipeline._dispatch_transform("bad_value")  # type: ignore[arg-type]


class TestGPUAugmentationPipelineFromConfig:
    """Tests for from_config."""

    def test_empty_config(self):
        config = SubsetConfig(augmentations_gpu=[], input_size=None)
        pipeline = GPUAugmentationPipeline.from_config(config)
        assert pipeline.aug_sequential is None

    def test_with_augmentations(self):
        config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.RandomHorizontalFlip",
                    "init_args": {"p": 0.5},
                },
            ],
            input_size=None,
        )
        pipeline = GPUAugmentationPipeline.from_config(config)
        assert pipeline.aug_sequential is not None

    def test_nn_module_passthrough(self):
        aug = kornia_aug.RandomHorizontalFlip(p=1.0)
        config = SubsetConfig(augmentations_gpu=[aug], input_size=None)  # type: ignore[arg-type]
        pipeline = GPUAugmentationPipeline.from_config(config)
        assert pipeline.aug_sequential is not None

    def test_unsupported_config_type_raises(self):
        config = SubsetConfig(augmentations_gpu=["bad_value"], input_size=None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported augmentation config type"):
            GPUAugmentationPipeline.from_config(config)

    def test_custom_data_keys(self):
        config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.RandomHorizontalFlip",
                    "init_args": {"p": 0.5},
                },
            ],
            input_size=None,
        )
        pipeline = GPUAugmentationPipeline.from_config(config, data_keys=["input", "bbox_xyxy"])
        assert pipeline.data_keys == ["input", "bbox_xyxy"]


class TestGPUAugmentationPipelineForward:
    """Tests for GPU pipeline forward pass."""

    def test_empty_pipeline_passthrough(self):
        pipeline = GPUAugmentationPipeline()
        images = _make_batched_images(2)
        result = pipeline(images)
        assert result["images"] is images
        assert result["labels"] is None
        assert result["bboxes"] is None
        assert result["masks"] is None

    def test_image_only_augmentation(self):
        """Single-key augmentation modifying only images."""
        pipeline = GPUAugmentationPipeline(
            [kornia_aug.RandomHorizontalFlip(p=1.0)],
            data_keys=["input"],
        )
        images = _make_batched_images(2, h=32, w=32)
        result = pipeline(images)
        assert result["images"].shape == images.shape
        # p=1.0 means always flip → images should differ
        assert not torch.allclose(result["images"], images)

    def test_normalization(self):
        """Normalize transform changes pixel range."""
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.25, 0.25, 0.25])
        pipeline = GPUAugmentationPipeline(
            [kornia_aug.Normalize(mean=mean, std=std)],
            data_keys=["input"],
        )
        images = torch.full((2, 3, 8, 8), 0.5)  # All pixels = 0.5
        result = pipeline(images)
        # (0.5 - 0.5) / 0.25 = 0.0
        assert torch.allclose(result["images"], torch.zeros_like(images), atol=1e-5)

    def test_forward_with_masks(self):
        """Forward with mask data key."""
        pipeline = GPUAugmentationPipeline(
            [kornia_aug.RandomHorizontalFlip(p=1.0)],
            data_keys=["input", "mask"],
        )
        images = _make_batched_images(2, h=16, w=16)
        masks = [torch.randint(0, 2, (1, 16, 16), dtype=torch.float32) for _ in range(2)]
        result = pipeline(images, masks=masks)
        assert result["images"] is not None
        assert result["masks"] is not None

    def test_mask_safe_random_erasing_preserves_instance_masks(self):
        """MaskSafeRandomErasing must not modify instance segmentation masks.

        Regression test for kornia RandomErasing that erases mask regions with
        incorrectly-shaped batch params, causing dimension mismatch errors.
        """
        from getitune.data.augmentation.transforms import MaskSafeRandomErasing

        pipeline = GPUAugmentationPipeline(
            [MaskSafeRandomErasing(p=1.0)],
            data_keys=["input", "mask"],
        )
        images = _make_batched_images(2, h=32, w=32)
        # Instance seg: different N_instances per sample
        masks = [
            torch.randint(0, 2, (3, 32, 32), dtype=torch.float32),  # 3 instances
            torch.randint(0, 2, (5, 32, 32), dtype=torch.float32),  # 5 instances
        ]
        original_masks = [m.clone() for m in masks]
        result = pipeline(images, masks=masks)

        # Images should be changed (erasing was applied with p=1.0)
        assert result["images"] is not None
        assert not torch.equal(result["images"], images)
        # Masks must be unchanged
        for orig, out in zip(original_masks, result["masks"]):
            assert torch.equal(orig, out), "masks must not be modified by MaskSafeRandomErasing"

    def test_forward_preserves_batch_size(self):
        pipeline = GPUAugmentationPipeline(
            [kornia_aug.RandomHorizontalFlip(p=0.5)],
            data_keys=["input"],
        )
        images = _make_batched_images(4, h=16, w=16)
        result = pipeline(images)
        assert result["images"].shape[0] == 4

    def test_repr_empty(self):
        pipeline = GPUAugmentationPipeline()
        r = repr(pipeline)
        assert "GPUAugmentationPipeline" in r
        assert "empty" in r

    def test_repr_with_augs(self):
        pipeline = GPUAugmentationPipeline([kornia_aug.RandomHorizontalFlip(p=0.5)])
        r = repr(pipeline)
        assert "GPUAugmentationPipeline" in r

    def test_repr_with_normalization(self):
        pipeline = GPUAugmentationPipeline(
            [kornia_aug.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))]
        )
        r = repr(pipeline)
        assert "mean=" in r
        assert "std=" in r

    def test_normalize_only_preserves_out_of_bounds_bboxes(self):
        """Critical regression test: Normalize-only pipeline must not clip/filter bboxes.

        When resize_targets=False is used during validation, bboxes stay in original
        image coordinates (e.g., 2048x1365) while images are resized (e.g., 800x992).
        A Normalize-only GPU pipeline must preserve these bboxes untouched.
        """
        pipeline = GPUAugmentationPipeline(
            [kornia_aug.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))],
            data_keys=["input", "bbox_xyxy", "label"],
        )
        # Small resized images (800x992) but bboxes in original space (2048x1365)
        images = _make_batched_images(2, h=800, w=992)
        bboxes = [
            torch.tensor([[1000.0, 500.0, 1500.0, 800.0], [100.0, 100.0, 300.0, 300.0]]),  # x>992 for first box
            torch.tensor([[500.0, 1000.0, 900.0, 1365.0]]),  # y>800 for this box
        ]
        labels = [torch.tensor([0, 1]), torch.tensor([2])]

        result = pipeline(images, bboxes=bboxes, labels=labels)

        # ALL bboxes must survive (no clipping or filtering)
        assert result["bboxes"] is not None
        assert len(result["bboxes"][0]) == 2, "First sample lost bboxes during normalize-only pipeline"
        assert len(result["bboxes"][1]) == 1, "Second sample lost bboxes during normalize-only pipeline"
        # Bbox coordinates must be unchanged
        assert torch.allclose(result["bboxes"][0], bboxes[0])
        assert torch.allclose(result["bboxes"][1], bboxes[1])
        # Labels must also survive
        assert result["labels"] is not None
        assert len(result["labels"][0]) == 2
        assert len(result["labels"][1]) == 1

    def test_geometric_pipeline_does_sanitize_bboxes(self):
        """Geometric augmentations (flip) should still sanitize bboxes."""
        pipeline = GPUAugmentationPipeline(
            [
                kornia_aug.RandomHorizontalFlip(p=0.0),  # no actual flip, but it IS geometric
                kornia_aug.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
            ],
            data_keys=["input", "bbox_xyxy", "label"],
        )
        images = _make_batched_images(1, h=32, w=32)
        # Box fully outside image bounds → should be sanitized away
        bboxes = [torch.tensor([[100.0, 100.0, 200.0, 200.0]])]
        labels = [torch.tensor([0])]

        result = pipeline(images, bboxes=bboxes, labels=labels)
        # The degenerate box (after clipping to 32x32) should be filtered
        assert result["bboxes"] is not None
        assert len(result["bboxes"][0]) == 0

    def test_sanitize_annotations_false_preserves_out_of_bounds_bboxes(self):
        """sanitize_annotations=False must preserve bboxes even with geometric augmentations.

        This models the val/test pipeline case: the callback passes
        sanitize_annotations=False so that GT bboxes in original image coordinates
        are never clipped to the smaller network input dimensions.
        """
        pipeline = GPUAugmentationPipeline(
            [
                kornia_aug.RandomHorizontalFlip(p=0.0),  # geometric, but sanitize disabled
                kornia_aug.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
            ],
            data_keys=["input", "bbox_xyxy", "label"],
            sanitize_annotations=False,
        )
        images = _make_batched_images(1, h=32, w=32)
        # Bboxes in original (large) image space -- all beyond 32x32 bounds
        bboxes = [torch.tensor([[100.0, 100.0, 200.0, 200.0], [10.0, 10.0, 20.0, 20.0]])]
        labels = [torch.tensor([0, 1])]

        result = pipeline(images, bboxes=bboxes, labels=labels)
        # Both bboxes must survive unchanged
        assert result["bboxes"] is not None
        assert len(result["bboxes"][0]) == 2, "sanitize_annotations=False must not filter bboxes"

    def test_sanitize_annotations_default_is_true(self):
        """Default value of sanitize_annotations should be True."""
        pipeline = GPUAugmentationPipeline([kornia_aug.RandomHorizontalFlip(p=0.5)])
        assert pipeline._sanitize_annotations_enabled is True

    def test_sanitize_annotations_false_stored(self):
        """sanitize_annotations=False is stored correctly."""
        pipeline = GPUAugmentationPipeline([kornia_aug.RandomHorizontalFlip(p=0.5)], sanitize_annotations=False)
        assert pipeline._sanitize_annotations_enabled is False

    def test_from_config_sanitize_annotations_false(self):
        """from_config forwards sanitize_annotations=False to the instance."""
        config = SubsetConfig(
            augmentations_gpu=[
                {"class_path": "kornia.augmentation.RandomHorizontalFlip", "init_args": {"p": 0.5}},
            ],
            input_size=None,
        )
        pipeline = GPUAugmentationPipeline.from_config(config, sanitize_annotations=False)
        assert pipeline._sanitize_annotations_enabled is False

    def test_from_config_empty_sanitize_annotations_false(self):
        """from_config with empty augmentations also respects sanitize_annotations=False."""
        config = SubsetConfig(augmentations_gpu=[], input_size=None)
        pipeline = GPUAugmentationPipeline.from_config(config, sanitize_annotations=False)
        assert pipeline._sanitize_annotations_enabled is False


# ===================================================================
# CPU/GPU Hybrid Integration Tests
# ===================================================================


class TestHybridCPUGPUPipeline:
    """Integration tests for the CPU→collate→GPU augmentation flow."""

    def test_cpu_to_gpu_uint8_flow(self):
        """Standard uint8: CPU scales to float → GPU augments."""
        cpu_config = SubsetConfig(
            augmentations_cpu=[],
            intensity=IntensityConfig(storage_dtype="uint8", mode="scale_to_unit"),
            input_size=None,
        )
        gpu_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.RandomHorizontalFlip",
                    "init_args": {"p": 1.0},
                },
            ],
            input_size=None,
        )

        cpu_pipeline = CPUAugmentationPipeline.from_config(cpu_config)
        gpu_pipeline = GPUAugmentationPipeline.from_config(gpu_config)

        # Simulate per-sample CPU processing
        samples = []
        for _ in range(4):
            sample = _make_sample(32, 32, dtype=torch.uint8)
            result = cpu_pipeline(sample)
            samples.append(result)

        # Simulate collate → batch
        batch_images = torch.stack([s.image for s in samples])
        assert batch_images.dtype == torch.float32
        assert batch_images.min() >= 0.0
        assert batch_images.max() <= 1.0

        # GPU stage
        gpu_result = gpu_pipeline(batch_images)
        assert gpu_result["images"].shape == batch_images.shape
        assert gpu_result["images"].dtype == torch.float32

    def test_cpu_to_gpu_uint16_flow(self):
        """uint16 thermal: CPU intensity maps → GPU augments + normalizes."""
        cpu_config = SubsetConfig(
            augmentations_cpu=[],
            intensity=IntensityConfig(
                storage_dtype="uint16",
                mode="range_scale",
                scale_factor=0.4,
                min_value=295.15,
                max_value=360.15,
                repeat_channels=3,
            ),
            input_size=None,
        )
        gpu_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.Normalize",
                    "init_args": {
                        "mean": [0.5, 0.5, 0.5],
                        "std": [0.25, 0.25, 0.25],
                    },
                },
            ],
            input_size=None,
        )

        cpu_pipeline = CPUAugmentationPipeline.from_config(cpu_config)
        gpu_pipeline = GPUAugmentationPipeline.from_config(gpu_config)

        # Simulate thermal sensor data (raw uint16 range ~738-900)
        raw_data = torch.randint(738, 901, (1, 32, 32), dtype=torch.int32)

        sample = _SimpleSample(image=raw_data)
        result = cpu_pipeline(sample)

        # After intensity mapping: float32, 3-channel, [0, 1]
        assert result.image.dtype == torch.float32
        assert result.image.shape[0] == 3  # repeat_channels=3
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

        # Simulate batch
        batch_images = result.image.unsqueeze(0)  # (1, 3, 32, 32)
        gpu_result = gpu_pipeline(batch_images)
        assert gpu_result["images"].shape == (1, 3, 32, 32)
        assert gpu_result["images"].dtype == torch.float32

    def test_cpu_resize_gpu_normalize(self):
        """CPU resizes, GPU normalizes — common production pattern."""
        cpu_config = SubsetConfig(
            augmentations_cpu=[
                {
                    "class_path": "torchvision.transforms.v2.Resize",
                    "init_args": {"size": [64, 64]},
                },
            ],
            intensity=IntensityConfig(storage_dtype="uint8", mode="scale_to_unit"),
            input_size=None,
        )
        gpu_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.Normalize",
                    "init_args": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            ],
            input_size=None,
        )

        cpu_pipeline = CPUAugmentationPipeline.from_config(cpu_config)
        gpu_pipeline = GPUAugmentationPipeline.from_config(gpu_config)

        # Per-sample CPU processing
        sample = _make_sample(128, 128, dtype=torch.uint8)
        result = cpu_pipeline(sample)
        assert result.image.shape == (3, 64, 64)  # Resized
        assert result.image.dtype == torch.float32

        # Batch and GPU
        batch_images = result.image.unsqueeze(0)
        gpu_result = gpu_pipeline(batch_images)
        assert gpu_result["images"].shape == (1, 3, 64, 64)
        # Normalization should shift values away from [0, 1]
        assert gpu_result["images"].min() < 0.0 or gpu_result["images"].max() > 1.0

    def test_cpu_empty_gpu_only_flow(self):
        """No CPU augmentations, only GPU — image should still be processed."""
        gpu_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.RandomHorizontalFlip",
                    "init_args": {"p": 1.0},
                },
            ],
            input_size=None,
        )
        gpu_pipeline = GPUAugmentationPipeline.from_config(gpu_config)

        # Manually create float batch
        images = _make_batched_images(2, h=16, w=16)
        result = gpu_pipeline(images)
        assert result["images"].shape == images.shape
        # Flipped → should not equal original
        assert not torch.allclose(result["images"], images)

    def test_gpu_normalization_params_for_model(self):
        """GPU pipeline should expose mean/std for model export."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        gpu_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.Normalize",
                    "init_args": {"mean": mean, "std": std},
                },
            ],
            input_size=None,
        )
        gpu_pipeline = GPUAugmentationPipeline.from_config(gpu_config)
        assert gpu_pipeline.mean is not None
        assert gpu_pipeline.std is not None
        for i in range(3):
            assert abs(gpu_pipeline.mean[i] - mean[i]) < 1e-4
            assert abs(gpu_pipeline.std[i] - std[i]) < 1e-4

    def test_full_train_pipeline_simulation(self):
        """Simulate a full training step: CPU aug → collate → GPU aug."""
        # Build pipelines
        cpu_config = SubsetConfig(
            augmentations_cpu=[
                {
                    "class_path": "torchvision.transforms.v2.RandomResizedCrop",
                    "init_args": {"size": [32, 32]},
                },
            ],
            intensity=IntensityConfig(storage_dtype="uint8", mode="scale_to_unit"),
            input_size=None,
        )
        gpu_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.RandomHorizontalFlip",
                    "init_args": {"p": 0.5},
                },
                {
                    "class_path": "kornia.augmentation.Normalize",
                    "init_args": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            ],
            input_size=None,
        )

        cpu_pipeline = CPUAugmentationPipeline.from_config(cpu_config)
        gpu_pipeline = GPUAugmentationPipeline.from_config(gpu_config)

        # CPU: per-sample processing (simulates Dataset.__getitem__)
        batch_images = []
        for _ in range(4):
            sample = _make_sample(64, 64, dtype=torch.uint8)
            result = cpu_pipeline(sample)
            assert result.image.shape == (3, 32, 32)
            assert result.image.dtype == torch.float32
            batch_images.append(result.image)

        # Collate: stack into batch
        batch = torch.stack(batch_images)
        assert batch.shape == (4, 3, 32, 32)

        # GPU: batch-level augmentation
        gpu_result = gpu_pipeline(batch)
        assert gpu_result["images"].shape == (4, 3, 32, 32)
        assert gpu_result["images"].dtype == torch.float32


# ===================================================================
# GPUAugmentationCallback tests
# ===================================================================


class TestGPUAugmentationCallback:
    """Tests for the Lightning Callback that orchestrates GPU augmentations."""

    def _make_callback(self, train_augs=None, val_augs=None, test_augs=None):  # noqa: ANN202
        """Create a GPUAugmentationCallback with optional configs."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        train_config = SubsetConfig(augmentations_gpu=train_augs or [])
        val_config = SubsetConfig(augmentations_gpu=val_augs or [])
        test_config = SubsetConfig(augmentations_gpu=test_augs or []) if test_augs else None
        return GPUAugmentationCallback(
            train_config=train_config,
            val_config=val_config,
            test_config=test_config,
        )

    def test_init_defaults(self):
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        callback = GPUAugmentationCallback()
        assert callback.train_config is None
        assert callback.val_config is None
        assert callback.test_config is None
        assert callback._train_pipeline is None
        assert callback._val_pipeline is None
        assert callback._test_pipeline is None

    def test_setup_creates_pipelines(self):
        """setup() should create train and val pipelines."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback
        from getitune.types.task import TaskType

        train_config = SubsetConfig(
            augmentations_gpu=[
                {"class_path": "kornia.augmentation.RandomHorizontalFlip", "init_args": {"p": 0.5}},
            ],
        )
        val_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.Normalize",
                    "init_args": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                },
            ],
        )

        callback = GPUAugmentationCallback(train_config=train_config, val_config=val_config)

        # Create mock module with required attributes
        pl_module = MagicMock()
        pl_module.task = TaskType.DETECTION
        pl_module.data_input_params = MagicMock()
        pl_module.data_input_params.mean = None
        pl_module.data_input_params.std = None

        trainer = MagicMock()
        callback.setup(trainer, pl_module, stage="fit")

        assert callback._train_pipeline is not None
        assert callback._val_pipeline is not None

    def test_setup_updates_model_normalization(self):
        """setup() should update model's mean/std from GPU pipeline."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback
        from getitune.types.task import TaskType

        val_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.Normalize",
                    "init_args": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            ],
        )

        callback = GPUAugmentationCallback(val_config=val_config)

        pl_module = MagicMock()
        pl_module.task = TaskType.MULTI_CLASS_CLS
        pl_module.data_input_params = MagicMock()
        pl_module.data_input_params.mean = None
        pl_module.data_input_params.std = None

        trainer = MagicMock()
        callback.setup(trainer, pl_module, stage="fit")

        # Model's mean/std should have been updated
        assert pl_module.data_input_params.mean is not None
        assert pl_module.data_input_params.std is not None

    def test_on_train_batch_start_no_pipeline(self):
        """If no train pipeline, on_train_batch_start should be a no-op."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        callback = GPUAugmentationCallback()
        batch = MagicMock()
        # Should not raise
        callback.on_train_batch_start(MagicMock(), MagicMock(), batch, batch_idx=0)

    def test_on_val_batch_start_disabled(self):
        """If no val pipeline, validation batches should not be augmented."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        callback = GPUAugmentationCallback()
        # _val_pipeline is None by default
        assert callback._val_pipeline is None
        batch = MagicMock()
        callback.on_validation_batch_start(MagicMock(), MagicMock(), batch, batch_idx=0)
        # Should still be None (no pipeline was created)
        assert callback._val_pipeline is None

    def test_on_test_batch_start_disabled(self):
        """If no test pipeline, test batches should not be augmented."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        callback = GPUAugmentationCallback()
        # _test_pipeline is None by default
        batch = MagicMock()
        callback.on_test_batch_start(MagicMock(), MagicMock(), batch, batch_idx=0)
        # No error, batch not modified

    def test_test_config_fallback_to_val(self):
        """If test_config is None, it should fall back to val_config."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback

        val_config = SubsetConfig(augmentations_gpu=[])
        callback = GPUAugmentationCallback(val_config=val_config, test_config=None)
        assert callback.test_config is val_config

    def test_data_keys_per_task(self):
        """Verify correct data_keys are used for different task types."""
        from getitune.data.augmentation.task_keys import DATA_KEYS_BY_TASK
        from getitune.types.task import TaskType

        expected_keys = {
            TaskType.DETECTION: ["input", "bbox_xyxy", "label"],
            TaskType.INSTANCE_SEGMENTATION: ["input", "bbox_xyxy", "mask", "label"],
            TaskType.SEMANTIC_SEGMENTATION: ["input", "mask"],
            TaskType.MULTI_CLASS_CLS: ["input", "label"],
        }

        for task_type, expected in expected_keys.items():
            data_keys = ["input", *DATA_KEYS_BY_TASK.get(task_type, ())]
            assert data_keys == expected, f"Mismatch for {task_type}: {data_keys} != {expected}"

    def _make_detection_tile_batch(self):  # noqa: ANN202
        """Build a minimal TileBatchDetDataEntity with two samples of tiles."""
        from getitune.data.entity.tile import TileBatchDetDataEntity

        def _tile() -> tv_tensors.Image:
            return tv_tensors.Image(torch.ones(3, 8, 8))

        # Two original images, each split into two tiles.
        batch_tiles = [[_tile(), _tile()], [_tile(), _tile()]]
        return TileBatchDetDataEntity(
            batch_size=2,
            batch_tiles=batch_tiles,
            batch_tile_img_infos=[[], []],
            batch_tile_tile_infos=[[], []],
            imgs_info=[],
            bboxes=[],
            labels=[],
        )

    def test_apply_pipeline_normalizes_tile_batch(self):
        """A TileBatchData validation batch is normalized per-tile without error."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback
        from getitune.types.task import TaskType

        val_config = SubsetConfig(
            augmentations_gpu=[
                {
                    "class_path": "kornia.augmentation.Normalize",
                    "init_args": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                },
            ],
        )
        callback = GPUAugmentationCallback(val_config=val_config)

        pl_module = MagicMock()
        pl_module.task = TaskType.DETECTION
        pl_module.data_input_params = MagicMock()
        pl_module.data_input_params.mean = None
        pl_module.data_input_params.std = None
        callback.setup(MagicMock(), pl_module, stage="fit")

        tile_batch = self._make_detection_tile_batch()

        # Should not raise AttributeError on ``batch.images`` and should normalize
        # every tile in place: (1.0 - 0.5) / 0.5 == 1.0 stays 1.0, so use a value
        # that changes to prove the transform ran.
        for tiles in tile_batch.batch_tiles:
            for i in range(len(tiles)):
                tiles[i] = tv_tensors.Image(torch.full((3, 8, 8), 0.75))

        callback.on_validation_batch_start(MagicMock(), pl_module, tile_batch, batch_idx=0)

        # Normalizing 0.75 with mean 0.5 and std 0.5 yields 0.5 for every tile.
        for tiles in tile_batch.batch_tiles:
            for tile in tiles:
                assert isinstance(tile, tv_tensors.Image)
                torch.testing.assert_close(tile, tv_tensors.Image(torch.full((3, 8, 8), 0.5)))

    def test_apply_pipeline_tile_batch_no_pipeline_augs(self):
        """A TileBatchData routed through an empty pipeline is left unchanged (no crash)."""
        from getitune.backend.lightning.callbacks.gpu_augmentation import GPUAugmentationCallback
        from getitune.types.task import TaskType

        callback = GPUAugmentationCallback(val_config=SubsetConfig(augmentations_gpu=[]))
        pl_module = MagicMock()
        pl_module.task = TaskType.DETECTION
        pl_module.data_input_params = MagicMock()
        pl_module.data_input_params.mean = None
        pl_module.data_input_params.std = None
        callback.setup(MagicMock(), pl_module, stage="fit")

        tile_batch = self._make_detection_tile_batch()
        # Must not raise 'TileBatchDetDataEntity object has no attribute images'
        callback.on_validation_batch_start(MagicMock(), pl_module, tile_batch, batch_idx=0)

        for tiles in tile_batch.batch_tiles:
            for tile in tiles:
                torch.testing.assert_close(tile, tv_tensors.Image(torch.ones(3, 8, 8)))
