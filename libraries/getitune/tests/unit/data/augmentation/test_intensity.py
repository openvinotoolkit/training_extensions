# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for intensity mapping transforms."""

from __future__ import annotations

import pytest
import torch

from getitune.config.data import IntensityConfig
from getitune.data.augmentation.intensity import (
    PercentileClip,
    RangeScale,
    RepeatChannels,
    ScaleToUnit,
    WindowLevel,
    build_intensity_transform,
)


class TestScaleToUnit:
    def test_uint8_scale(self):
        t = ScaleToUnit(max_value=255.0)
        x = torch.tensor([0, 128, 255], dtype=torch.uint8)
        out = t(x)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.tensor([0.0, 128.0 / 255.0, 1.0]))

    def test_uint16_scale(self):
        t = ScaleToUnit(max_value=65535.0)
        x = torch.tensor([0, 32768, 65535], dtype=torch.int32)  # int32 for uint16 data
        out = t(x)
        assert out.dtype == torch.float32
        assert torch.isclose(out[0], torch.tensor(0.0))
        assert torch.isclose(out[2], torch.tensor(1.0))
        assert 0.0 <= out[1].item() <= 1.0

    def test_clamps_above_max(self):
        t = ScaleToUnit(max_value=100.0)
        x = torch.tensor([200.0])
        out = t(x)
        assert out.item() == 1.0

    def test_3d_image(self):
        t = ScaleToUnit(max_value=255.0)
        x = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        out = t(x)
        assert out.shape == (3, 64, 64)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestWindowLevel:
    def test_basic_window(self):
        t = WindowLevel(center=100.0, width=200.0)
        x = torch.tensor([0.0, 100.0, 200.0, -50.0, 300.0])
        out = t(x)
        assert torch.isclose(out[0], torch.tensor(0.0))
        assert torch.isclose(out[1], torch.tensor(0.5))
        assert torch.isclose(out[2], torch.tensor(1.0))
        assert out[3].item() == 0.0  # clamped
        assert out[4].item() == 1.0  # clamped

    def test_ct_window(self):
        """Typical CT brain window: center=40, width=80 → [0, 80]."""
        t = WindowLevel(center=40.0, width=80.0)
        x = torch.tensor([0.0, 40.0, 80.0])
        out = t(x)
        assert torch.isclose(out[0], torch.tensor(0.0))
        assert torch.isclose(out[1], torch.tensor(0.5))
        assert torch.isclose(out[2], torch.tensor(1.0))


class TestPercentileClip:
    def test_uniform_image(self):
        t = PercentileClip(low=1.0, high=99.0)
        x = torch.arange(0, 1000, dtype=torch.float32)
        out = t(x)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_constant_image_returns_zeros(self):
        t = PercentileClip(low=1.0, high=99.0)
        x = torch.full((3, 32, 32), 42.0)
        out = t(x)
        assert torch.all(out == 0.0)

    def test_invalid_percentiles(self):
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            PercentileClip(low=99.0, high=1.0)

    def test_3d_image(self):
        t = PercentileClip(low=5.0, high=95.0)
        x = torch.randint(0, 65536, (1, 64, 64), dtype=torch.int32).float()
        out = t(x)
        assert out.shape == (1, 64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestRangeScale:
    def test_thermal_conversion(self):
        """Reproduce process_raw_thermal.py: raw * 0.4, clip [295.15, 360.15], normalize."""
        t = RangeScale(scale_factor=0.4, min_value=295.15, max_value=360.15)
        # A pixel with raw value 900 → 900 * 0.4 = 360.0 → in range → (360.0-295.15)/(360.15-295.15)
        x = torch.tensor([900.0])
        out = t(x)
        expected = (900.0 * 0.4 - 295.15) / (360.15 - 295.15)
        assert torch.isclose(out, torch.tensor(expected))

    def test_thermal_clamping_low(self):
        t = RangeScale(scale_factor=0.4, min_value=295.15, max_value=360.15)
        # 100 * 0.4 = 40 < 295.15 → clipped to 295.15 → 0.0
        x = torch.tensor([100.0])
        out = t(x)
        assert torch.isclose(out, torch.tensor(0.0))

    def test_thermal_clamping_high(self):
        t = RangeScale(scale_factor=0.4, min_value=295.15, max_value=360.15)
        # 60000 * 0.4 = 24000 > 360.15 → clipped to 360.15 → 1.0
        x = torch.tensor([60000.0])
        out = t(x)
        assert torch.isclose(out, torch.tensor(1.0))

    def test_int32_input(self):
        """Simulates raw uint16 data stored as int32 tensor."""
        t = RangeScale(scale_factor=0.4, min_value=295.15, max_value=360.15)
        x = torch.tensor([900], dtype=torch.int32)
        out = t(x)
        assert out.dtype == torch.float32
        assert 0.0 <= out.item() <= 1.0

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="max_value must be > min_value"):
            RangeScale(scale_factor=1.0, min_value=100.0, max_value=50.0)

    def test_batch_3d(self):
        t = RangeScale(scale_factor=0.4, min_value=295.15, max_value=360.15)
        x = torch.randint(700, 1000, (1, 32, 32), dtype=torch.int32)
        out = t(x)
        assert out.shape == (1, 32, 32)
        assert out.dtype == torch.float32


class TestRepeatChannels:
    def test_single_to_three(self):
        t = RepeatChannels(num_channels=3)
        x = torch.randn(1, 64, 64)
        out = t(x)
        assert out.shape == (3, 64, 64)
        # All channels should be identical
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[0], out[2])

    def test_three_channel_passthrough(self):
        t = RepeatChannels(num_channels=3)
        x = torch.randn(3, 64, 64)
        out = t(x)
        assert torch.equal(x, out)

    def test_batched_4d(self):
        t = RepeatChannels(num_channels=3)
        x = torch.randn(4, 1, 64, 64)
        out = t(x)
        assert out.shape == (4, 3, 64, 64)

    def test_batched_3ch_passthrough(self):
        t = RepeatChannels(num_channels=3)
        x = torch.randn(4, 3, 64, 64)
        out = t(x)
        assert torch.equal(x, out)


class TestBuildIntensityTransform:
    def test_default_uint8(self):
        cfg = IntensityConfig()
        pipeline = build_intensity_transform(cfg)
        # Should contain ScaleToUnit(255)
        assert len(pipeline) == 1
        assert isinstance(pipeline[0], ScaleToUnit)
        assert pipeline[0].max_value == 255.0

    def test_uint16_scale_to_unit(self):
        cfg = IntensityConfig(storage_dtype="uint16")
        pipeline = build_intensity_transform(cfg)
        assert isinstance(pipeline[0], ScaleToUnit)
        assert pipeline[0].max_value == 65535.0

    def test_window_mode(self):
        cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="window",
            window_center=40.0,
            window_width=80.0,
        )
        pipeline = build_intensity_transform(cfg)
        assert isinstance(pipeline[0], WindowLevel)

    def test_window_mode_missing_params(self):
        cfg = IntensityConfig(mode="window")
        with pytest.raises(ValueError, match="window_center and window_width"):
            build_intensity_transform(cfg)

    def test_percentile_mode(self):
        cfg = IntensityConfig(mode="percentile", percentile_low=2.0, percentile_high=98.0)
        pipeline = build_intensity_transform(cfg)
        assert isinstance(pipeline[0], PercentileClip)
        assert pipeline[0].low == 2.0
        assert pipeline[0].high == 98.0

    def test_range_scale_mode(self):
        cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="range_scale",
            scale_factor=0.4,
            min_value=295.15,
            max_value=360.15,
        )
        pipeline = build_intensity_transform(cfg)
        assert isinstance(pipeline[0], RangeScale)

    def test_range_scale_missing_max(self):
        cfg = IntensityConfig(mode="range_scale")
        with pytest.raises(ValueError, match="requires max_value"):
            build_intensity_transform(cfg)

    def test_unknown_mode(self):
        cfg = IntensityConfig(mode="foobar")
        with pytest.raises(ValueError, match="Unknown IntensityConfig mode"):
            build_intensity_transform(cfg)

    def test_repeat_channels(self):
        cfg = IntensityConfig(storage_dtype="uint16", repeat_channels=3)
        pipeline = build_intensity_transform(cfg)
        assert len(pipeline) == 2
        assert isinstance(pipeline[0], ScaleToUnit)
        assert isinstance(pipeline[1], RepeatChannels)

    def test_thermal_e2e(self):
        """End-to-end test matching process_raw_thermal.py behavior."""
        cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="range_scale",
            scale_factor=0.4,
            min_value=295.15,
            max_value=360.15,
            repeat_channels=3,
        )
        pipeline = build_intensity_transform(cfg)

        # Simulate a 1-channel thermal image with raw uint16 values (as int32)
        raw_image = torch.randint(700, 1000, (1, 64, 64), dtype=torch.int32)
        out = pipeline(raw_image)

        assert out.dtype == torch.float32
        assert out.shape == (3, 64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_uint8_e2e_matches_old_behavior(self):
        """Default uint8 path should produce same results as to_dtype(float32, scale=True)."""
        cfg = IntensityConfig()  # defaults
        pipeline = build_intensity_transform(cfg)

        x = torch.tensor([0, 128, 255], dtype=torch.uint8)
        new_result = pipeline(x)

        # Old behavior: to_dtype divides by 255
        import torchvision.transforms.v2.functional as f

        old_result = f.to_dtype(x, dtype=torch.float32, scale=True)

        assert torch.allclose(new_result, old_result, atol=1e-6)
