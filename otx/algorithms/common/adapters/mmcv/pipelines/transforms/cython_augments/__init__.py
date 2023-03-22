"""Module to init cython augments."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ignore mypy attr-defined error by cython modules
# pylint: disable=import-self

from . import pil_augment  # type: ignore[attr-defined]

__all__ = ["pil_augment"]
