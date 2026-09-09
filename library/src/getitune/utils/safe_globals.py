# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Allowlists for PyTorch's restricted (``weights_only=True``) unpickler.

PyTorch's safe deserialization only reconstructs an internal allowlist of
globals (tensors, storages, primitive containers). Additional globals must be
registered explicitly via ``torch.serialization.safe_globals``. The lists here
are scoped per loader: each checkpoint-loading path uses only the set of
globals its own file format legitimately contains, so the surface a malicious
file can reach stays minimal.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path, PosixPath, WindowsPath
from typing import Callable

# Checkpoints saved by getitune <= 0.3.0 capture the model's `pretrained_weights`
# init argument into `hyper_parameters`, which pickles a `pathlib.Path` object.
# pathlib paths are inert data objects (their pickle reconstructs via a plain
# constructor call with a string), so they are safe for PyTorch's restricted
# unpickler. The tuple entries register the module paths used by different
# Python versions (`pathlib` up to 3.12, `pathlib._local` from 3.13).
CHECKPOINT_SAFE_GLOBALS: list[Callable | tuple[Callable, str]] = [
    (Path, "pathlib.Path"),
    (PosixPath, "pathlib.PosixPath"),
    (PosixPath, "pathlib._local.PosixPath"),
    (WindowsPath, "pathlib.WindowsPath"),
    (WindowsPath, "pathlib._local.WindowsPath"),
]

# Third-party pretrained checkpoints (e.g. mmsegmentation archives) store run
# metadata as `argparse.Namespace` in `meta`. Namespace is an inert attribute
# container that pickle reconstructs via a plain constructor call, so it is
# safe for PyTorch's restricted (weights_only=True) unpickler.
PRETRAINED_SAFE_GLOBALS: list[Callable | tuple[Callable, str]] = [Namespace]
