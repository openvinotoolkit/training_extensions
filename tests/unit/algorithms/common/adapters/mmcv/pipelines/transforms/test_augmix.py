"""Unit Tests for the MPA Dataset Pipelines Transforms Augments."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import random

import numpy as np
import pytest
from PIL import Image

from otx.algorithms.classification.adapters.mmcls.datasets.pipelines.transforms.augmix import (
    AugMixAugment,
    OpsFabric,
)
from otx.algorithms.common.adapters.mmcv.pipelines.transforms.augments import (
    CythonAugments,
)


@pytest.fixture
def ops_fabric() -> OpsFabric:
    return OpsFabric("Rotate", 5, {"img_mean": 128})


@pytest.mark.xfail(reason="random may not return the same value on different machines.")
class TestOpsFabric:
    def test_init(self, ops_fabric: OpsFabric) -> None:
        """Test OpsFabric initialization."""
        assert ops_fabric.prob == 1.0
        assert ops_fabric.hparams == {"img_mean": 128}
        assert ops_fabric.aug_kwargs == {
            "fillcolor": 128,
            "resample": (Image.BILINEAR, Image.BICUBIC),
        }
        assert ops_fabric.aug_factory.magnitude == 5
        assert ops_fabric.aug_factory.magnitude_std == float("inf")
        assert ops_fabric.aug_factory.level_fn == ops_fabric._rotate_level_to_arg
        assert ops_fabric.aug_factory.aug_fn == CythonAugments.rotate

    def test_randomly_negate(self) -> None:
        """Test randomly_negate function."""
        random.seed(1234)
        assert OpsFabric.randomly_negate(5) == -5
        assert OpsFabric.randomly_negate(5) == 5
        assert OpsFabric.randomly_negate(5) == -5

    def test_rotate_level_to_arg(self, ops_fabric: OpsFabric) -> None:
        """Test rotate_level_to_arg function."""
        assert ops_fabric._rotate_level_to_arg(0, ops_fabric.hparams) == (0,)
        assert ops_fabric._rotate_level_to_arg(5, ops_fabric.hparams) == (5 / 10 * 30,)

    def test_enhance_increasing_level_to_arg(self, ops_fabric: OpsFabric) -> None:
        """Test enhance_increasing_level_to_arg function."""
        assert ops_fabric._enhance_increasing_level_to_arg(0, ops_fabric.hparams) == (1.0,)
        assert ops_fabric._enhance_increasing_level_to_arg(5, ops_fabric.hparams) == (1.0 + 5 / 10 * 0.9,)

    def test_shear_level_to_arg(self, ops_fabric: OpsFabric) -> None:
        """Test shear_level_to_arg function."""
        assert ops_fabric._shear_level_to_arg(0, ops_fabric.hparams) == (0,)
        assert ops_fabric._shear_level_to_arg(5, ops_fabric.hparams) == (5 / 10 * 0.3,)

    def test_translate_rel_level_to_arg(self, ops_fabric: OpsFabric) -> None:
        """Test translate_rel_level_to_arg function."""
        assert ops_fabric._translate_rel_level_to_arg(0, ops_fabric.hparams) == (0,)
        assert ops_fabric._translate_rel_level_to_arg(5, {"translate_pct": 0.5}) == (5 / 10 * 0.5,)

    def test_posterize_increasing_level_to_arg(self, ops_fabric: OpsFabric) -> None:
        """Test posterize_increasing_level_to_arg function."""
        assert ops_fabric._posterize_increasing_level_to_arg(0, ops_fabric.hparams) == (4,)
        assert ops_fabric._posterize_increasing_level_to_arg(5, ops_fabric.hparams) == (4 - int(5 / 10 * 4),)

    def test_solarize_increasing_level_to_arg(self, ops_fabric: OpsFabric) -> None:
        """Test solarize_increasing_level_to_arg function."""
        assert ops_fabric._solarize_increasing_level_to_arg(0, ops_fabric.hparams) == (0,)
        assert ops_fabric._solarize_increasing_level_to_arg(5, ops_fabric.hparams) == (256 - int(5 / 10 * 256),)

    def test_call(self, ops_fabric: OpsFabric) -> None:
        """Test __call__ function."""
        img = Image.new("RGB", (256, 256))
        transformed_img = ops_fabric(img)
        assert transformed_img != img  # make sure the image was actually transformed


class TestAugMixAugment:
    def test_init(self) -> None:
        """Test AugMixAugment initialization."""
        aug_mix_augment = AugMixAugment(config_str="augmix-m5-w3")
        assert isinstance(aug_mix_augment, AugMixAugment)
        assert len(aug_mix_augment.ops) > 0

    def test_apply_basic(self) -> None:
        """Test _apply_basic function."""
        aug_mix_augment = AugMixAugment(config_str="augmix-m5-w3")

        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        mixing_weights = np.float32(np.random.dirichlet([aug_mix_augment.alpha] * aug_mix_augment.width))
        m = np.float32(np.random.beta(aug_mix_augment.alpha, aug_mix_augment.alpha))

        mixed_img = aug_mix_augment._apply_basic(img, mixing_weights, m)
        assert isinstance(mixed_img, Image.Image)

    def test_augmix_ops(self) -> None:
        """Test augmix_ops function."""
        aug_mix_augment = AugMixAugment(config_str="augmix-m5-w3")
        assert len(aug_mix_augment.ops) > 0
        assert isinstance(aug_mix_augment.alpha, float)
        assert isinstance(aug_mix_augment.width, int)
        assert isinstance(aug_mix_augment.depth, int)

    def test_call(self) -> None:
        """Test __call__ method."""
        aug_mix_augment = AugMixAugment(config_str="augmix-m5-w3")
        data = {"img": np.random.randint(0, 255, size=(224, 224, 3)).astype(np.uint8)}
        results = aug_mix_augment(data)
        assert "augmix" in results
        assert isinstance(results["img"], Image.Image)
