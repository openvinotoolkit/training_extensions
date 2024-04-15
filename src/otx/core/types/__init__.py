# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module reserved for definitions used in OTX."""

import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

from otx.core.types.label import HLabelInfo, LabelInfo, NullLabelInfo, SegLabelInfo
from otx.core.types.task import OTXTaskType

__all__ = [
    # label_info
    "LabelInfo",
    "HLabelInfo",
    "SegLabelInfo",
    "NullLabelInfo",
    # task_type
    "OTXTaskType",
]

PathLike: TypeAlias = Union[str, Path, os.PathLike]
