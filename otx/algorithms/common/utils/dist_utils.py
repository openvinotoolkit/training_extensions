"""Module for defining distance utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from pathlib import Path
from typing import Union

import torch.distributed as dist


def get_dist_info():  # pylint: disable=inconsistent-return-statements
    """A function that retrieves information about the current distributed training environment."""
    if dist.is_available():
        # data distributed parallel
        try:
            return dist.get_rank(), dist.get_world_size(), True
        except RuntimeError:
            return 0, 1, False


def append_dist_rank_suffix(file_name: Union[str, Path]) -> str:
    """Append distributed training rank suffix to the file name."""
    if "LOCAL_RANK" in os.environ:
        file_name = Path(file_name)
        dist_suffix = f"_proc{os.environ['LOCAL_RANK']}"
        file_name = file_name.parent / f"{file_name.stem}{dist_suffix}{file_name.suffix}"

    return str(file_name)
