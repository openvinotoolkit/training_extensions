# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Physical hardware discovery for performance benchmark metadata."""

from __future__ import annotations

import platform
from contextlib import suppress
from pathlib import Path


def get_training_device_name(accelerator: str) -> str:
    """Return the physical device used by the training backend.

    Raises:
        RuntimeError: If the requested accelerator is unavailable or its
            physical device name cannot be determined.
    """
    normalized = accelerator.lower()
    if normalized == "cpu":
        try:
            from cpuinfo import get_cpu_info

            name = str(get_cpu_info().get("brand_raw", "")).strip()
        except Exception:
            name = platform.processor().strip()
        if not name:
            with suppress(OSError, StopIteration):
                name = next(
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
                    if line.lower().startswith("model name")
                )
        if name:
            return name
    else:
        import torch

        if normalized in {"cuda", "gpu"} and torch.cuda.is_available():
            return str(torch.cuda.get_device_name(torch.cuda.current_device()))
        if normalized == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return str(torch.xpu.get_device_name(torch.xpu.current_device()))

    msg = f"Could not determine physical training device for accelerator '{accelerator}'."
    raise RuntimeError(msg)


def get_openvino_device_name(device: str) -> str:
    """Return the physical OpenVINO device selected by *device*.

    Raises:
        RuntimeError: If the target is unavailable or does not expose a full
            physical device name.
    """
    import openvino as ov

    core = ov.Core()
    available = core.available_devices
    requested = device.upper()
    matches = [candidate for candidate in available if candidate.upper() == requested]
    if not matches and "." not in requested:
        matches = [candidate for candidate in available if candidate.upper().startswith(f"{requested}.")]
    target = matches[0] if matches else device
    try:
        name = str(core.get_property(target, "FULL_DEVICE_NAME")).strip()
    except Exception as exc:
        msg = f"Could not determine physical OpenVINO device for target '{device}'. Available: {available}"
        raise RuntimeError(msg) from exc
    if not name:
        msg = f"OpenVINO target '{device}' returned an empty FULL_DEVICE_NAME."
        raise RuntimeError(msg)
    return name
