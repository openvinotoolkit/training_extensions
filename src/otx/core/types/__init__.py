# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module reserved for definitions used in OTX."""

import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

PathLike: TypeAlias = Union[str, Path, os.PathLike]
