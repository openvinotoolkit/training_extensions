"""Utility functions collection."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import Union


def add_suffix_to_filename(file_path: Union[str, Path], suffix: str) -> Path:
    """Add suffix to file name.

    Args:
        file_path (Union[str, Path]): File path to add suffix to.
        suffix (str): Suffix to add.

    Returns:
        Path: Suffix added path.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.parent / f"{file_path.stem}{suffix}{file_path.suffix}"
