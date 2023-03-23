# Copyright (c) 2019 MendelXu
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn


class PSPModule(nn.Module):
    """Reference: https://github.com/MendelXu/ANN"""

    methods = {"max": nn.AdaptiveMaxPool2d, "avg": nn.AdaptiveAvgPool2d}

    def __init__(self, sizes=(1, 3, 6, 8), method="max"):
        super().__init__()

        assert method in self.methods
        pool_block = self.methods[method]

        self.stages = nn.ModuleList([pool_block(output_size=(size, size)) for size in sizes])

    def forward(self, feats):
        n, c, _, _ = feats.size()

        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        out = torch.cat(priors, -1)

        return out
