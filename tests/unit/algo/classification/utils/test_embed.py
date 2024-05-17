# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from otx.algo.classification.utils.embed import resize_pos_embed


def test_resize_pos_embed():
    pos_embed = torch.randn(1, 32 * 32 + 1, 256)
    src_shape = (32, 32)
    dst_shape = (64, 64)
    mode = "bicubic"
    num_extra_tokens = 1

    resized_pos_embed = resize_pos_embed(pos_embed, src_shape, dst_shape, mode, num_extra_tokens)

    assert resized_pos_embed.shape == (1, 4097, 256)
    assert resized_pos_embed.dtype == pos_embed.dtype
    assert resized_pos_embed[:, :num_extra_tokens].equal(pos_embed[:, :num_extra_tokens])
