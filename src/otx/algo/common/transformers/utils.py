# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX Transformers Utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def gen_encoder_output_proposals(
    memory: Tensor,
    memory_padding_mask: Tensor,
    spatial_shapes: Tensor,
) -> tuple[Tensor, Tensor]:
    """Generate encoder output and proposals.

    Args:
        memory (Tensor): Memory tensor of shape (N, S, C).
        memory_padding_mask (Tensor): Memory padding mask tensor of shape (N, S).
        spatial_shapes (List[Tuple[int, int]]): List of spatial shapes.

    Returns:
        Tuple[Tensor, Tensor]: Encoder output tensor of shape (N, S, C) and proposals tensor of shape (N, L, 6).
    """
    batch_size = memory.shape[0]
    proposals = []
    _cur = 0
    for lvl, (height, width) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)
        valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, device=memory.device),
            torch.linspace(0, width - 1, width, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((grid, wh), -1).view(batch_size, -1, 4)
        proposals.append(proposal)
        _cur += height * width
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals
