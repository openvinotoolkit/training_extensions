# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Intensity mapping transforms for high-bit-depth image support.

Converts raw pixel values (uint8, uint16, int32, etc.) to float32 in [0, 1]
with domain-specific strategies:

- :class:`ScaleToUnit` — simple divide-by-max (default for uint8 and uint16).
- :class:`WindowLevel` — CT-style window/level mapping for medical imaging.
- :class:`PercentileClip` — per-image percentile normalization for microscopy.
- :class:`RangeScale` — multiply-by-factor + clip to physical range (thermal cameras).
- :class:`RepeatChannels` — expand single-channel to N-channel for pretrained backbones.
- :func:`build_intensity_transform` — factory that builds the right pipeline from
  :class:`~getitune.config.data.IntensityConfig`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from getitune.config.data import IntensityConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-detected max values per storage dtype
# ---------------------------------------------------------------------------
_AUTO_MAX: dict[str, float] = {
    "uint8": 255.0,
    "uint16": 65535.0,
    "int16": 32767.0,
    "float32": 1.0,
}


class ScaleToUnit(nn.Module):
    """Linearly scale raw pixel values to [0, 1].

    ``output = clamp(input.float() / max_value, 0, 1)``

    This is the default intensity mapping for both uint8 (max_value=255) and
    uint16 (max_value=65535).  It replaces the previously-hardcoded
    ``torchvision.transforms.v2.functional.to_dtype(…, scale=True)`` with an
    explicit, dtype-aware alternative.

    Args:
        max_value: Denominator for the division.  Use 255 for uint8, 65535 for
            uint16, or any custom value.
    """

    def __init__(self, max_value: float = 255.0) -> None:
        super().__init__()
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """Scale to [0, 1] float32."""
        return torch.clamp(x.float() / self.max_value, 0.0, 1.0)

    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"max_value={self.max_value}"


class WindowLevel(nn.Module):
    """Window / level intensity mapping for CT-style medical imaging.

    Maps the raw intensity window ``[center - width/2, center + width/2]``
    linearly onto ``[0, 1]``.  Values outside the window are clamped.

    Args:
        center: Centre of the intensity window (in raw pixel units).
        width: Width of the intensity window (in raw pixel units).
    """

    def __init__(self, center: float, width: float) -> None:
        super().__init__()
        self.center = center
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        """Apply window/level mapping."""
        low = self.center - self.width / 2.0
        high = self.center + self.width / 2.0
        out = (x.float() - low) / (high - low)
        return torch.clamp(out, 0.0, 1.0)

    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"center={self.center}, width={self.width}"


class PercentileClip(nn.Module):
    """Per-image percentile-based intensity normalization.

    For each image independently:
    1. Compute the ``low`` and ``high`` percentile values.
    2. Clip to ``[p_low, p_high]``.
    3. Normalize to ``[0, 1]``.

    Useful for microscopy, pathology, or any domain where the dynamic range
    varies strongly between images.

    Args:
        low: Lower percentile (0-100).  Default 1.0.
        high: Upper percentile (0-100).  Default 99.0.
    """

    def __init__(self, low: float = 1.0, high: float = 99.0) -> None:
        super().__init__()
        if not 0.0 <= low < high <= 100.0:
            msg = f"Percentiles must satisfy 0 <= low < high <= 100, got low={low}, high={high}"
            raise ValueError(msg)
        self.low = low
        self.high = high

    def forward(self, x: Tensor) -> Tensor:
        """Per-image percentile normalization."""
        x_float = x.float()
        # Flatten spatial dims for quantile computation, keep channel dim
        flat = x_float.reshape(-1)
        p_low = torch.quantile(flat, self.low / 100.0)
        p_high = torch.quantile(flat, self.high / 100.0)
        # Avoid division by zero when the image is constant
        denom = p_high - p_low
        if denom < 1e-8:
            return torch.zeros_like(x_float)
        out = (x_float - p_low) / denom
        return torch.clamp(out, 0.0, 1.0)

    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"low={self.low}, high={self.high}"


class RangeScale(nn.Module):
    """Multiply-then-clip intensity mapping for thermal / physical-range data.

    Reproduces the thermal pipeline from ``process_raw_thermal.py``::

        scaled = raw_pixels * scale_factor
        clipped = clip(scaled, min_value, max_value)
        normalized = (clipped - min_value) / (max_value - min_value)

    For a FLIR A65 camera with Kelvin conversion:
    ``RangeScale(scale_factor=0.4, min_value=295.15, max_value=360.15)``

    Args:
        scale_factor: Multiplicative factor applied to raw pixel values.
        min_value: Lower clip bound (physical units after scaling).
        max_value: Upper clip bound (physical units after scaling).
    """

    def __init__(self, scale_factor: float = 1.0, min_value: float = 0.0, max_value: float = 1.0) -> None:
        super().__init__()
        if max_value <= min_value:
            msg = f"max_value must be > min_value, got min_value={min_value}, max_value={max_value}"
            raise ValueError(msg)
        self.scale_factor = scale_factor
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """Apply scale, clip, normalize."""
        scaled = x.float() * self.scale_factor
        clipped = torch.clamp(scaled, self.min_value, self.max_value)
        return (clipped - self.min_value) / (self.max_value - self.min_value)

    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"scale_factor={self.scale_factor}, min_value={self.min_value}, max_value={self.max_value}"


class RepeatChannels(nn.Module):
    """Repeat single-channel images to N channels.

    Many pretrained backbones expect 3-channel (RGB) input.  Medical and thermal
    images are often single-channel (grayscale).  This transform repeats the
    channel dimension so the data is compatible.

    Operates on tensors of shape ``(C, H, W)`` (per-sample) or
    ``(B, C, H, W)`` (batched).  Only repeats when ``C == 1``; otherwise
    passes through unchanged.

    Args:
        num_channels: Target number of channels.  Default 3.
    """

    def __init__(self, num_channels: int = 3) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        """Repeat channel dim if C == 1."""
        if x.ndim == 3 and x.shape[0] == 1:
            return x.repeat(self.num_channels, 1, 1)
        if x.ndim == 4 and x.shape[1] == 1:
            return x.repeat(1, self.num_channels, 1, 1)
        return x

    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"num_channels={self.num_channels}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_intensity_transform(config: IntensityConfig) -> nn.Sequential:
    """Build an intensity-mapping transform pipeline from :class:`IntensityConfig`.

    The returned ``nn.Sequential`` converts a raw-dtype image tensor to
    ``float32`` in ``[0, 1]`` and optionally repeats channels.

    For ``storage_dtype="uint8"`` with ``mode="scale_to_unit"`` (the default),
    this produces a :class:`ScaleToUnit(255)` which is functionally equivalent
    to the old ``torchvision.transforms.v2.functional.to_dtype(…, scale=True)``.

    Args:
        config: An :class:`~getitune.config.data.IntensityConfig` instance.

    Returns:
        ``nn.Sequential`` of intensity transforms ready to prepend to the
        CPU augmentation pipeline.

    Raises:
        ValueError: If the ``mode`` is unknown or required fields are missing.

    Examples:
        Standard uint8 (no-op equivalent to prior behavior)::

            cfg = IntensityConfig()  # defaults
            t = build_intensity_transform(cfg)  # ScaleToUnit(255)

        Thermal (FLIR A65)::

            cfg = IntensityConfig(
                storage_dtype="uint16",
                mode="range_scale",
                scale_factor=0.4,
                min_value=295.15,
                max_value=360.15,
                repeat_channels=3,
            )
            t = build_intensity_transform(cfg)

        Medical CT::

            cfg = IntensityConfig(
                storage_dtype="uint16",
                mode="window",
                window_center=40.0,
                window_width=400.0,
                repeat_channels=3,
            )
            t = build_intensity_transform(cfg)
    """
    mode = config.mode
    storage_dtype = config.storage_dtype
    transforms: list[nn.Module] = []

    if mode == "scale_to_unit":
        max_value = config.max_value
        if max_value is None:
            max_value = _AUTO_MAX.get(storage_dtype)
            if max_value is None:
                msg = (
                    f"Cannot auto-detect max_value for storage_dtype={storage_dtype!r}. "
                    "Please set IntensityConfig.max_value explicitly."
                )
                raise ValueError(msg)
        transforms.append(ScaleToUnit(max_value=max_value))

    elif mode == "window":
        if config.window_center is None or config.window_width is None:
            msg = "IntensityConfig mode='window' requires both window_center and window_width."
            raise ValueError(msg)
        transforms.append(WindowLevel(center=config.window_center, width=config.window_width))

    elif mode == "percentile":
        transforms.append(PercentileClip(low=config.percentile_low, high=config.percentile_high))

    elif mode == "range_scale":
        if config.max_value is None:
            msg = "IntensityConfig mode='range_scale' requires max_value to be set."
            raise ValueError(msg)
        transforms.append(
            RangeScale(
                scale_factor=config.scale_factor,
                min_value=config.min_value,
                max_value=config.max_value,
            )
        )
    else:
        msg = (
            f"Unknown IntensityConfig mode: {mode!r}. "
            "Supported: 'scale_to_unit', 'window', 'percentile', 'range_scale'."
        )
        raise ValueError(msg)

    # Optional channel repetition (e.g. grayscale → 3ch for pretrained backbones)
    if config.repeat_channels > 0:
        transforms.append(RepeatChannels(num_channels=config.repeat_channels))

    logger.info("Built intensity transform pipeline: %s", transforms)
    return nn.Sequential(*transforms)
