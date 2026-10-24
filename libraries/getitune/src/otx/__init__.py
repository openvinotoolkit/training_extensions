# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility shim: 'otx' has been renamed to 'getitune'.

This stub re-exports everything from the new ``getitune`` package so that
existing scripts and PRs that still ``import otx`` continue to work for a
transition period.  A ``DeprecationWarning`` is emitted on first import to
nudge callers toward the new name.
"""

import warnings

warnings.warn(
    "'otx' has been renamed to 'getitune'. "
    "Please update your imports: use 'from getitune import ...' instead of 'from otx import ...'. "
    "The 'otx' compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from getitune import *  # noqa: E402, F403
from getitune import __version__  # noqa: E402, F401
