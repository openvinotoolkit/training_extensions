# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Transformers-backed DINOv3 wrapper for DEIMv2."""

from __future__ import annotations

from typing import Any

import torch
from transformers import DINOv3ViTBackbone, DINOv3ViTConfig


def _map_legacy_dinov3_state_dict(state_dict: dict[str, Any], prefix: str) -> None:
    source_prefix = f"{prefix}dinov3."
    top_level_names = (
        "_model.",
        "patch_embed.",
        "cls_token",
        "storage_tokens",
        "mask_token",
        "rope_embed.",
        "blocks.",
        "norm.",
    )

    for source_key in tuple(state_dict):
        if not source_key.startswith(source_prefix):
            continue

        suffix = source_key.removeprefix(source_prefix)
        if suffix.startswith("_model."):
            suffix = suffix.removeprefix("_model.")
        if not suffix.startswith(top_level_names):
            continue

        if suffix == "rope_embed.periods" or suffix.endswith(".attn.qkv.bias_mask"):
            del state_dict[source_key]
            continue

        if ".attn.qkv." in suffix:
            block_prefix, parameter = suffix.split(".attn.qkv.", maxsplit=1)
            block_index = block_prefix.removeprefix("blocks.")
            if parameter == "bias":
                mask = state_dict.get(source_key.removesuffix("bias") + "bias_mask")
                effective_bias = state_dict[source_key] if mask is None else state_dict[source_key] * mask
                if torch.count_nonzero(effective_bias).item():
                    msg = f"Cannot map active DINOv3 QKV bias to the bias-free HF backbone: {source_key}"
                    raise ValueError(msg)
                del state_dict[source_key]
                continue
            projections = state_dict[source_key].chunk(3, dim=0)
            if len(projections) != 3:
                msg = f"Invalid DINOv3 fused QKV tensor: {source_key}"
                raise ValueError(msg)
            for projection_name, tensor in zip(("q", "k", "v"), projections, strict=True):
                target_key = f"{source_prefix}model.layer.{block_index}.attention.{projection_name}_proj.{parameter}"
                if target_key in state_dict and not torch.equal(state_dict[target_key], tensor):
                    msg = f"Conflicting DINOv3 checkpoint tensors for {target_key}"
                    raise ValueError(msg)
                state_dict[target_key] = tensor
            del state_dict[source_key]
            continue

        target_suffix = suffix
        for old, new in (
            ("patch_embed.proj.", "embeddings.patch_embeddings."),
            ("cls_token", "embeddings.cls_token"),
            ("storage_tokens", "embeddings.register_tokens"),
            ("mask_token", "embeddings.mask_token"),
            ("blocks.", "model.layer."),
            (".attn.proj.", ".attention.o_proj."),
            (".ls1.gamma", ".layer_scale1.lambda1"),
            (".ls2.gamma", ".layer_scale2.lambda1"),
            (".mlp.fc1.", ".mlp.up_proj."),
            (".mlp.fc2.", ".mlp.down_proj."),
            (".mlp.w1.", ".mlp.gate_proj."),
            (".mlp.w2.", ".mlp.up_proj."),
            (".mlp.w3.", ".mlp.down_proj."),
        ):
            if (old.startswith(".") and old in target_suffix) or (
                not old.startswith(".") and target_suffix.startswith(old)
            ):
                target_suffix = target_suffix.replace(old, new, 1)

        target_key = f"{source_prefix}{target_suffix}"
        if target_key == source_key:
            continue
        if suffix == "mask_token" and state_dict[source_key].ndim == 2:
            state_dict[source_key] = state_dict[source_key].unsqueeze(1)
        if target_key in state_dict and not torch.equal(state_dict[target_key], state_dict[source_key]):
            msg = f"Conflicting DINOv3 checkpoint tensors for {target_key}"
            raise ValueError(msg)
        state_dict[target_key] = state_dict.pop(source_key)


class DinoVisionTransformer(DINOv3ViTBackbone):
    """Transformers DINOv3 backbone configured for DEIMv2 variants."""

    def __init__(self, name: str, interaction_indexes: list[int] | None = None) -> None:
        if name not in {"dinov3_vits16", "dinov3_vits16plus"}:
            msg = f"Unsupported DINOv3 backbone: {name}"
            raise ValueError(msg)

        interaction_indexes = interaction_indexes or []
        use_gated_mlp = name.endswith("plus")
        config = DINOv3ViTConfig(
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            patch_size=16,
            image_size=224,
            hidden_act="silu" if use_gated_mlp else "gelu",
            query_bias=False,
            key_bias=False,
            value_bias=False,
            proj_bias=True,
            mlp_bias=True,
            layerscale_value=1e-5,
            num_register_tokens=4,
            use_gated_mlp=use_gated_mlp,
            rope_theta=100.0,
            pos_embed_rescale=2.0,
        )
        config.set_output_features_output_indices(
            out_indices=[index + 1 for index in interaction_indexes] or [12], out_features=None
        )
        super().__init__(config)

    @property
    def embed_dim(self) -> int:
        """Return the hidden feature width."""
        return self.config.hidden_size

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        _map_legacy_dinov3_state_dict(state_dict, prefix.removesuffix("dinov3."))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
