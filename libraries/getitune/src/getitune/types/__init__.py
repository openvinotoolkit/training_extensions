# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module reserved for definitions used in getitune."""

import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

from getitune.types.label import LabelInfo, NullLabelInfo, SegLabelInfo
from getitune.types.task import TaskType

__all__ = ["LabelInfo", "NullLabelInfo", "PathLike", "SegLabelInfo", "TaskType"]

PathLike: TypeAlias = Union[str, Path, os.PathLike]
