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

import numpy as np
from numpy import dtypes as np_dtypes
from numpy.core.multiarray import _reconstruct  # noqa: SLF001 - private numpy API; numpy.core legacy shim for 1.x pickles

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

# Third-party pretrained checkpoints (mmsegmentation, mmpose, ...) embed run
# metadata as `argparse.Namespace`, and arrays/scalars as pickled numpy objects,
# when they were saved with NumPy 1.x (the pickle references the legacy
# `numpy.core.*` module paths, which NumPy 2.x shims keep importable). All of
# these are inert data containers reconstructed via plain constructor calls, so
# they are safe for PyTorch's restricted (weights_only=True) unpickler. Tuple
# entries register the module paths used by different NumPy versions.
_dtype_classes = [
    getattr(np_dtypes, name)
    for name in dir(np_dtypes)
    if not name.startswith("_") and isinstance(getattr(np_dtypes, name), type)
]
PRETRAINED_SAFE_GLOBALS: list[Callable | tuple[Callable, str]] = [
    Namespace,
    np.ndarray,
    (_reconstruct, "numpy.core.multiarray._reconstruct"),
    (_reconstruct, "numpy._core.multiarray._reconstruct"),
    (np.dtype, "numpy.dtype"),
    (np.dtype, "numpy._core.dtype"),
    *_dtype_classes,
    *[(cls, f"numpy.{cls.__name__}") for cls in _dtype_classes],
    *[(cls, f"numpy._core.{cls.__name__}") for cls in _dtype_classes],
]
