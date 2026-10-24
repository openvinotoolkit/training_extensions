# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from timm.models.vision_transformer import Attention

from getitune.backend.lightning.models.classification.utils.peft import AttentionWithDoRA, AttentionWithLoRA


def test_attention_with_lora_forward():
    x = torch.randn(2, 256, 384)
    attention = Attention(dim=384, num_heads=6, qkv_bias=True)
    attn = AttentionWithLoRA(qkv=attention.qkv, rank=4, alpha=1.0)
    out = attn(x)
    assert out.shape == (2, 256, 384 * 3)


def test_attention_with_dora_forward():
    x = torch.randn(2, 256, 384)
    attention = Attention(dim=384, num_heads=6, qkv_bias=True)
    attn = AttentionWithDoRA(qkv=attention.qkv, rank=4, alpha=1.0)
    out = attn(x)
    assert out.shape == (2, 256, 384 * 3)
