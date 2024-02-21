# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI Utils."""

from __future__ import annotations

from pathlib import Path


def absolute_path(path: str | Path | None) -> str | None:
    """Returns the absolute path of the given path.

    Args:
        path (str | Path | None): The path to be resolved.

    Returns:
        str | None: The absolute path of the given path, or None if the path is None.

    """
    return str(Path(path).resolve()) if path is not None else None
