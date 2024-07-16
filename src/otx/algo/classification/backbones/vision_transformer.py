# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from mmpretrain/models/backbones/vision_transformer.py."""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
from timm.layers import (
    LayerType,
    Mlp,
    PatchDropout,
    PatchEmbed,
    SwiGLUPacked,
    get_act_layer,
    get_norm_layer,
    resample_abs_pos_embed,
    resample_patch_embed,
    trunc_normal_,
)
from timm.models._manipulate import adapt_input_conv
from timm.models.vision_transformer import Attention, Block
from torch import nn

from otx.algo.modules.base_module import BaseModule

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


VIT_ARCH_TYPE = Literal[
    "vit-t",
    "vit-tiny",
    "vit-s",
    "vit-small",
    "vit-b",
    "vit-base",
    "vit-l",
    "vit-large",
    "vit-h",
    "vit-huge",
    "dinov2-s",
    "dinov2-small",
    "dinov2-b",
    "dinov2-base",
    "dinov2-l",
    "dinov2-large",
    "dinov2-g",
    "dinov2-giant",
]


class VisionTransformer(BaseModule):
    """Implementation of Vision Transformer from Timm.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
        - https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

    Args:
        arch: Vision Transformer architecture.
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of image input channels.
        num_classes: Mumber of classes for classification head.
        embed_dim: Transformer embedding dimension.
        depth: Depth of transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Enable bias for qkv projections if True.
        init_values: Layer-scale init values (layer-scale enabled if not None).
        class_token: Use class token.
        no_embed_class: Don't include position embeddings for class (or reg) tokens.
        reg_tokens: Number of register tokens.
        drop_rate: Head dropout rate.
        pos_drop_rate: Position embedding dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        weight_init: Weight initialization scheme.
        fix_init: Apply weight initialization fix (scaling w/ layer index).
        embed_layer: Patch embedding layer.
        norm_layer: Normalization layer.
        act_layer: MLP activation layer.
        block_fn: Transformer block layer.
        lora: Enable LoRA training.
    """

    arch_zoo: dict[str, dict] = {  # noqa: RUF012
        **dict.fromkeys(
            ["vit-t", "vit-tiny"],
            {
                "patch_size": 16,
                "embed_dim": 192,
                "depth": 12,
                "num_heads": 3,
            },
        ),
        **dict.fromkeys(
            ["vit-s", "vit-small"],
            {
                "patch_size": 16,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
            },
        ),
        **dict.fromkeys(
            ["vit-b", "vit-base"],
            {
                "patch_size": 16,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
            },
        ),
        **dict.fromkeys(
            ["vit-l", "vit-large"],
            {
                "patch_size": 16,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
            },
        ),
        **dict.fromkeys(
            ["vit-h", "vit-huge"],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                "patch_size": 16,
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
            },
        ),
        **dict.fromkeys(
            ["dinov2-s", "dinov2-small"],
            {
                "patch_size": 14,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-5,
            },
        ),
        **dict.fromkeys(
            ["dinov2-b", "dinov2-base"],
            {
                "patch_size": 14,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-5,
            },
        ),
        **dict.fromkeys(
            ["dinov2-l", "dinov2-large"],
            {
                "patch_size": 14,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-5,
            },
        ),
        **dict.fromkeys(
            ["dinov2-g", "dinov2-giant"],
            {
                "patch_size": 14,
                "embed_dim": 1536,
                "depth": 40,
                "num_heads": 24,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-5,
                "mlp_ratio": 2.66667 * 2,
                "mlp_layer": SwiGLUPacked,
                "act_layer": nn.SiLU,
            },
        ),
    }

    def __init__(  # noqa: PLR0913
        self,
        arch: VIT_ARCH_TYPE = "vit-base",
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] | None = None,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float | None = None,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool | None = None,
        reg_tokens: int | None = None,
        pre_norm: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Callable = PatchEmbed,
        block_fn: nn.Module = Block,
        mlp_layer: nn.Module | None = None,
        act_layer: LayerType | None = None,
        norm_layer: LayerType | None = None,
        lora: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(arch, str):
            if arch not in set(self.arch_zoo):
                msg = f"Arch {arch} is not in default archs {set(self.arch_zoo)}"
                raise ValueError(msg)
            arch_settings: dict[str, Any] = self.arch_zoo[arch]

        self.img_size: int | tuple[int, int] = img_size
        self.patch_size: int | tuple[int, int] = patch_size or arch_settings.get("patch_size", 16)
        self.embed_dim = embed_dim or arch_settings.get("embed_dim", 768)
        depth = depth or arch_settings.get("depth", 12)
        num_heads = num_heads or arch_settings.get("num_heads", 12)
        no_embed_class = no_embed_class or arch_settings.get("no_embed_class", False)
        reg_tokens = reg_tokens or arch_settings.get("reg_tokens", 0)
        init_values = init_values or arch_settings.get("init_values", None)
        mlp_layer = mlp_layer or arch_settings.get("mlp_layer", Mlp)
        mlp_ratio = mlp_ratio or arch_settings.get("mlp_ratio", 4.0)
        norm_layer = get_norm_layer(norm_layer) or arch_settings.get("norm_layer", partial(nn.LayerNorm, eps=1e-6))
        act_layer = get_act_layer(act_layer) or arch_settings.get("act_layer", nn.GELU)

        self.num_classes = num_classes
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update({"strict_img_size": False, "output_fmt": "NHWC"})
        self.patch_embed = embed_layer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, self.embed_dim)) if reg_tokens else None

        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_len, self.embed_dim))

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(self.embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ],
        )

        self.norm = norm_layer(self.embed_dim)

        self.lora = lora
        if self.lora:
            lora_rank = 8
            lora_alpha = 1.0
            assign_lora = partial(AttentionWithLoRA, rank=lora_rank, alpha=lora_alpha)
            for block in self.blocks:
                block.attn.qkv = assign_lora(block.attn.qkv)

            # Freeze all params
            for param in self.parameters():
                param.requires_grad = False

            # Unfreeze LoRA layers
            for block in self.blocks:
                for param in block.attn.qkv.lora_q.parameters():
                    param.requires_grad = True
                for param in block.attn.qkv.lora_v.parameters():
                    param.requires_grad = True

    def init_weights(self) -> None:
        """Initializes the weights of the VisionTransformer."""
        super().init_weights()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: Path, prefix: str = "") -> None:
        """Loads the pretrained weight to the VisionTransformer."""
        checkpoint_ext = checkpoint_path.suffix
        if checkpoint_ext == ".npz":  # deit models
            self._load_npz_weights(self, checkpoint_path, prefix)
        elif checkpoint_ext == ".pth":  # dinov2 models

            def resize_positional_embeddings(pos_embed: torch.Tensor, new_shape: tuple[int, int]) -> torch.Tensor:
                # Resize the embeddings using bilinear interpolation.
                pos_embed = pos_embed.permute(0, 2, 1).reshape(1, -1, 37, 37)  # 560 (img_size) / 14 (patch_size) = 37
                pos_embed_resized = nn.functional.interpolate(
                    pos_embed,
                    size=(new_shape[0], new_shape[1]),
                    mode="bilinear",
                )
                return pos_embed_resized.reshape(1, -1, new_shape[0] * new_shape[1]).permute(0, 2, 1)

            # convert dinov2 pretrained weights
            state_dict = torch.load(checkpoint_path)
            state_dict.pop("mask_token", None)
            state_dict["reg_token"] = state_dict.pop("register_tokens")
            state_dict["cls_token"] = state_dict.pop("cls_token") + state_dict["pos_embed"][:, 0]

            img_size = (self.img_size, self.img_size) if isinstance(self.img_size, int) else self.img_size
            patch_size = (self.patch_size, self.patch_size) if isinstance(self.patch_size, int) else self.patch_size
            state_dict["pos_embed"] = resize_positional_embeddings(
                state_dict.pop("pos_embed")[:, 1:],
                (img_size[0] // patch_size[0], img_size[1] // patch_size[1]),
            )
            self.load_state_dict(state_dict, strict=False)
        else:
            msg = f"Unsupported `checkpoint_extension` {checkpoint_ext}, please choose from 'npz' or 'pth'."
            raise ValueError(msg)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Implements positional embedding."""
        if self.dynamic_img_size:
            b, h, w, c = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (h, w),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(b, -1, c)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005
            x = x + pos_embed

        return self.pos_drop(x)

    def forward(
        self,
        x: torch.Tensor,
        out_type: Literal["raw", "cls_token", "featmap", "avg_featmap"] = "cls_token",
    ) -> tuple:
        """Forward pass of the VisionTransformer model."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.blocks(x)
        x = self.norm(x)

        if out_type == "raw":
            return (x,)
        if out_type == "cls_token":
            return (x[:, 0],)
        msg = f"Unsupported `out_type` {out_type}, please choose from {self.OUT_TYPES}"
        raise ValueError(msg)

    @torch.no_grad()
    def _load_npz_weights(  # noqa: C901
        self,
        model: VisionTransformer,
        checkpoint_path: str,
        prefix: str = "",
    ) -> None:
        """Load weights from .npz checkpoints for official Google Brain Flax implementation."""
        import numpy as np

        def _n2p(w: np.ndarray, t: bool = True, idx: int | None = None) -> torch.Tensor:
            if idx is not None:
                w = w[idx]
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)

        w = np.load(checkpoint_path)
        interpolation = "bilinear"
        antialias = False
        big_vision = False
        if not prefix:
            if "opt/target/embedding/kernel" in w:
                prefix = "opt/target/"
            elif "params/embedding/kernel" in w:
                prefix = "params/"
                big_vision = True
            elif "params/img/embedding/kernel" in w:
                prefix = "params/img/"
                big_vision = True

        if hasattr(model.patch_embed, "backbone"):
            # hybrid
            backbone = model.patch_embed.backbone
            stem_only = not hasattr(backbone, "stem")
            stem = backbone if stem_only else backbone.stem
            stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])))
            stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
            stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
            if not stem_only:
                for i, stage in enumerate(backbone.stages):
                    for j, block in enumerate(stage.blocks):
                        bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                        for r in range(3):
                            getattr(block, f"conv{r + 1}").weight.copy_(_n2p(w[f"{bp}conv{r + 1}/kernel"]))
                            getattr(block, f"norm{r + 1}").weight.copy_(_n2p(w[f"{bp}gn{r + 1}/scale"]))
                            getattr(block, f"norm{r + 1}").bias.copy_(_n2p(w[f"{bp}gn{r + 1}/bias"]))
                        if block.downsample is not None:
                            block.downsample.conv.weight.copy_(_n2p(w[f"{bp}conv_proj/kernel"]))
                            block.downsample.norm.weight.copy_(_n2p(w[f"{bp}gn_proj/scale"]))
                            block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
            embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
        else:
            embed_conv_w = adapt_input_conv(
                model.patch_embed.proj.weight.shape[1],
                _n2p(w[f"{prefix}embedding/kernel"]),
            )
        if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
            embed_conv_w = resample_patch_embed(
                embed_conv_w,
                model.patch_embed.proj.weight.shape[-2:],
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        model.patch_embed.proj.weight.copy_(embed_conv_w)
        model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
        if model.cls_token is not None:
            model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
        if big_vision:
            pos_embed_w = _n2p(w[f"{prefix}pos_embedding"], t=False)
        else:
            pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
        if pos_embed_w.shape != model.pos_embed.shape:
            num_prefix_tokens = 0 if getattr(model, "no_embed_class", False) else getattr(model, "num_prefix_tokens", 1)
            pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        model.pos_embed.copy_(pos_embed_w)
        model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
        model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))

        mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
        idx: int | None = None
        for i, block in enumerate(model.blocks.children()):
            if f"{prefix}Transformer/encoderblock/LayerNorm_0/scale" in w:
                block_prefix = f"{prefix}Transformer/encoderblock/"
                idx = i
            else:
                embed_conv_w = adapt_input_conv(
                    model.patch_embed.proj.weight.shape[1],
                    _n2p(w[f"{prefix}embedding/kernel"]),
                )
            if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
                embed_conv_w = resample_patch_embed(
                    embed_conv_w,
                    model.patch_embed.proj.weight.shape[-2:],
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )

            model.patch_embed.proj.weight.copy_(embed_conv_w)
            model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
            if model.cls_token is not None:
                model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
            if big_vision:
                pos_embed_w = _n2p(w[f"{prefix}pos_embedding"], t=False)
            else:
                pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
            if pos_embed_w.shape != model.pos_embed.shape:
                num_prefix_tokens = (
                    0 if getattr(model, "no_embed_class", False) else getattr(model, "num_prefix_tokens", 1)
                )
                pos_embed_w = (
                    resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                        pos_embed_w,
                        new_size=model.patch_embed.grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        interpolation=interpolation,
                        antialias=antialias,
                        verbose=True,
                    )
                )
            model.pos_embed.copy_(pos_embed_w)
            model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
            model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))

            mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
            for i, block in enumerate(model.blocks.children()):  # noqa: PLW2901
                if f"{prefix}Transformer/encoderblock/LayerNorm_0/scale" in w:
                    block_prefix = f"{prefix}Transformer/encoderblock/"
                    idx = i
                else:
                    block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
                    idx = None
                mha_prefix = block_prefix + f"MultiHeadDotProductAttention_{mha_sub}/"
                block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"], idx=idx))
                block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"], idx=idx))
                if not self.lora:
                    block.attn.qkv.weight.copy_(
                        torch.cat(
                            [
                                _n2p(w[f"{mha_prefix}{n}/kernel"], t=False, idx=idx).flatten(1).T
                                for n in ("query", "key", "value")
                            ],
                        ),
                    )
                    block.attn.qkv.bias.copy_(
                        torch.cat(
                            [
                                _n2p(w[f"{mha_prefix}{n}/bias"], t=False, idx=idx).reshape(-1)
                                for n in ("query", "key", "value")
                            ],
                        ),
                    )
                else:
                    block.attn.qkv.qkv.weight.copy_(
                        torch.cat(
                            [
                                _n2p(w[f"{mha_prefix}{n}/kernel"], t=False, idx=idx).flatten(1).T
                                for n in ("query", "key", "value")
                            ],
                        ),
                    )
                    block.attn.qkv.qkv.bias.copy_(
                        torch.cat(
                            [
                                _n2p(w[f"{mha_prefix}{n}/bias"], t=False, idx=idx).reshape(-1)
                                for n in ("query", "key", "value")
                            ],
                        ),
                    )
                block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"], idx=idx).flatten(1))
                block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"], idx=idx))
                block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/scale"], idx=idx))
                block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/bias"], idx=idx))
                for r in range(2):
                    getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                        _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel"], idx=idx),
                    )
                    getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                        _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias"], idx=idx),
                    )


class LoRALayer(torch.nn.Module):
    """LoRA layer implementation for computing A, B composition."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LoRA layer."""
        return self.alpha * (x @ self.A @ self.B)


class AttentionWithLoRA(torch.nn.Module):
    """Add LoRA layer into QKV attention layer in VisionTransformer."""

    def __init__(self, qkv: Attention, rank: int, alpha: float):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AttentionWithLoRA."""
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim :] += self.lora_v(x)
        return qkv
