"""Module for defining distance utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Union


def get_dist_info():  # pylint: disable=inconsistent-return-statements
    """A function that retrieves information about the current distributed training environment."""
    if dist.is_available():
        # data distributed parallel
        try:
            return dist.get_rank(), dist.get_world_size(), True
        except RuntimeError:
            return 0, 1, False


def save_file_considering_dist_train(obj, file_name: Union[str, Path]):
    """Wrapper function of 'torch.save'.
    
       Save a file with rank suffix if training is distributed.
       Use this function in the case where multiple processes save the weight and will use it in distributed training.
    """
    if "LOCAL_RANK" in os.environ:
        file_name = Path(file_name)
        dist_suffix = f"_proc{os.environ['LOCAL_RANK']}"
        file_name = file_name.parent / f"{file_name.stem}{dist_suffix}{file_name.suffix}"
    torch.save(obj, file_name)
