# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.distributed as dist
import torch.utils.data.distributed


def get_dist_info():
    try:
        # data distributed parallel
        return dist.get_rank(), dist.get_world_size(), True
    except Exception:
        return 0, 1, False
