"""Module for defining distance utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch.distributed as dist


def get_dist_info():  # pylint: disable=inconsistent-return-statements
    """A function that retrieves information about the current distributed training environment."""
    if dist.is_available():
        # data distributed parallel
        try:
            return dist.get_rank(), dist.get_world_size(), True
        except RuntimeError:
            return 0, 1, False
