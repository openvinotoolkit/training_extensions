# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from mmpretrain/models/backbones/vision_transformer.py."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal

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
from timm.models.vision_transformer import Block
from torch import nn

from getitune.backend.lightning.models.classification.utils.peft import (
    AttentionWithDoRA,
    AttentionWithLoRA,
)
from getitune.backend.lightning.models.modules.base_module import BaseModule

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class VisionTransformerBackbone(BaseModule):
    """Implementation of Vision Transformer from Timm.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
        - https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

    Args:
        model_name: Vision Transformer architecture.
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of image input channels.
        num_classes: Number of classes for classification head.
        embed_dim: Transformer embedding dimension.
        depth: Depth of transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Enable bias for qkv projections if True.
        init_values: Layer-scale init values (layer-scale enabled if not None).
        class_token: Use class token.
        no_embed_class: Don't include position embeddings for class (or reg) tokens.
        reg_tokens: Number of register tokens.
        pos_drop_rate: Position embedding dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        embed_layer: Patch embedding layer.
        norm_layer: Normalization layer.
        act_layer: MLP activation layer.
        block_fn: Transformer block layer.
        interpolate_offset: work-around offset to apply when interpolating positional embeddings
        peft: Selects PEFT method ("lora" or "dora")
    """

    model_zoo: ClassVar[dict[str, dict[str, Any]]] = {
        **{key: {"patch_size": 16, "embed_dim": 192, "depth": 12, "num_heads": 3} for key in ["vit-t", "vit-tiny"]},
        **{key: {"patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6} for key in ["vit-s", "vit-small"]},
        **{key: {"patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12} for key in ["vit-b", "vit-base"]},
        **{key: {"patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16} for key in ["vit-l", "vit-large"]},
        **{key: {"patch_size": 16, "embed_dim": 1280, "depth": 32, "num_heads": 16} for key in ["vit-h", "vit-huge"]},
        **{
            key: {
                "patch_size": 14,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-05,
            }
            for key in ["dinov2-s", "dinov2-small"]
        },
        **{
            key: {
                "patch_size": 14,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "reg_tokens": 0,
                "no_embed_class": False,
                "init_values": 1e-05,
            }
            for key in ["dinov2-small-seg"]
        },
        **{
            key: {
                "patch_size": 14,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-05,
            }
            for key in ["dinov2-b", "dinov2-base"]
        },
        **{
            key: {
                "patch_size": 14,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-05,
            }
            for key in ["dinov2-l", "dinov2-large"]
        },
        **{
            key: {
                "patch_size": 14,
                "embed_dim": 1536,
                "depth": 40,
                "num_heads": 24,
                "reg_tokens": 4,
                "no_embed_class": True,
                "init_values": 1e-05,
                "mlp_ratio": 2.66667 * 2,
                "mlp_layer": SwiGLUPacked,
                "act_layer": nn.SiLU,
            }
            for key in ["dinov2-g", "dinov2-giant"]
        },
    }

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = "vit-base",
        img_size: int | tuple[int, int] = 224,
        patch_size: int | None = None,
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
        interpolate_offset: float = 0.1,
        peft: Literal["lora", "dora"] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(model_name, str):
            if model_name not in set(self.model_zoo):
                msg = f"Arch {model_name} is not in default archs {set(self.model_zoo)}"
                raise ValueError(msg)
            arch_settings: dict[str, Any] = self.model_zoo[model_name]

        self.img_size: int | tuple[int, int] = img_size
        self.patch_size: int = patch_size or arch_settings.get("patch_size", 16)
        self.embed_dim = embed_dim or arch_settings.get("embed_dim", 768)
        depth = depth or arch_settings.get("depth", 12)
        num_heads = num_heads or arch_settings.get("num_heads", 12)
        no_embed_class = no_embed_class or arch_settings.get("no_embed_class", False)
        reg_tokens = reg_tokens or arch_settings.get("reg_tokens", 0)
        init_values = init_values or arch_settings.get("init_values")
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
        self.interpolate_offset = interpolate_offset

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

        self.peft = peft
        if self.peft is not None:
            if self.peft not in ["lora", "dora"]:
                msg = f"Unsupported `peft` value '{self.peft}', please choose from 'lora' or 'dora'."
                raise ValueError(msg)

            attention_func = AttentionWithLoRA if self.peft == "lora" else AttentionWithDoRA
            target_attrs = ["lora_q", "lora_v"] if self.peft == "lora" else ["dora_q", "dora_v"]

            lora_rank = 8
            lora_alpha = 1.0

            # Apply PEFT to all blocks
            for block in self.blocks:
                block.attn.qkv = attention_func(block.attn.qkv, rank=lora_rank, alpha=lora_alpha)

            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False

            # Unfreeze PEFT layers
            for block in self.blocks:
                for attr_name in target_attrs:
                    peft_layer = getattr(block.attn.qkv, attr_name)
                    for param in peft_layer.parameters():
                        param.requires_grad = True

    def init_weights(self) -> None:
        """Initializes the weights of the VisionTransformer."""
        super().init_weights()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    @torch.jit.ignore()
    def load_checkpoint(self, checkpoint_path: Path, prefix: str = "") -> None:
        """Loads the pretrained weight to the VisionTransformer."""
        checkpoint_ext = checkpoint_path.suffix
        if checkpoint_ext == ".npz":  # ViT models (JAX format)
            self._load_npz_weights(self, str(checkpoint_path), prefix)
        elif checkpoint_ext == ".pth":  # dinov2 models

            def resize_positional_embeddings(pos_embed: torch.Tensor, new_shape: tuple[int, int]) -> torch.Tensor:
                # Resize the embeddings using bilinear interpolation.
                old_num_patches = pos_embed.shape[1]
                old_size = int(math.sqrt(old_num_patches))
                if old_size * old_size != old_num_patches:
                    msg = f"Expected square positional embeddings, but got {old_num_patches} patches."
                    raise ValueError(msg)

                pos_embed = pos_embed.permute(0, 2, 1).reshape(1, -1, old_size, old_size)
                pos_embed_resized = nn.functional.interpolate(
                    pos_embed,
                    size=(new_shape[0], new_shape[1]),
                    mode="bilinear",
                )
                return pos_embed_resized.reshape(1, -1, new_shape[0] * new_shape[1]).permute(0, 2, 1)

            # convert dinov2 pretrained weights
            from getitune.utils.safe_globals import PRETRAINED_SAFE_GLOBALS

            with torch.serialization.safe_globals(PRETRAINED_SAFE_GLOBALS):
                state_dict = torch.load(checkpoint_path)
            if prefix:
                state_dict = {
                    key.removeprefix(f"{prefix}."): value
                    for key, value in state_dict.items()
                    if key.startswith(f"{prefix}.")
                }
            state_dict.pop("mask_token", None)
            if "register_tokens" in state_dict:
                state_dict["reg_token"] = state_dict.pop("register_tokens")
            state_dict["cls_token"] = state_dict.pop("cls_token") + state_dict["pos_embed"][:, 0]

            img_size = (self.img_size, self.img_size) if isinstance(self.img_size, int) else self.img_size
            patch_size = (self.patch_size, self.patch_size)

            if state_dict["pos_embed"].shape != self.pos_embed.shape:
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
            # ViT with no_embed_class, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005
        else:
            # original timm, JAX, and ViT impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005
            x = x + pos_embed

        return self.pos_drop(x)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolates the positional encoding to match the input dimensions.

        Args:
            x (torch.Tensor): Input tensor.
            w (int): Width of the input image.
            h (int): Height of the input image.

        Returns:
            torch.Tensor: Tensor with interpolated positional encoding.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1]
        n = self.pos_embed.shape[1]
        if npatch == n and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        m = int(math.sqrt(n))  # Recover the number of patches in each dimension
        if m * m != n:
            msg = f"Expected m * m to equal n, but got m={m}, n={n}"
            raise ValueError(msg)
        kwargs = {}
        if self.interpolate_offset:
            # fix float error by introducing small offset
            sx = float(w0 + self.interpolate_offset) / m
            sy = float(h0 + self.interpolate_offset) / m
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = pos_embed[:, 1:]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, m, m, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            **kwargs,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim).to(previous_dtype)
        class_pos_embed = pos_embed[:, 0].unsqueeze(0).to(previous_dtype)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens_with_masks(self, x: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare tokens with optional masks.

        Args:
            x (torch.Tensor): Input tensor.
            masks (torch.Tensor | None): Optional masks tensor.

        Returns:
            torch.Tensor: Tensor with prepared tokens.
        """
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # pyrefly: ignore[missing-attribute]
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.reg_token is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.reg_token.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def _get_intermediate_layers_not_chunked(self, x: torch.Tensor, n: int = 1) -> list[torch.Tensor]:
        """Get intermediate layers without chunking.

        Args:
            x (torch.Tensor): Input tensor.
            n (int): Number of last blocks to take. If it's a list, take the specified blocks.

        Returns:
            list[torch.Tensor]: List of intermediate layer outputs.
        """
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        if len(output) != len(blocks_to_take):
            msg = f"only {len(output)} / {len(blocks_to_take)} blocks found"
            raise RuntimeError(msg)
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple:
        """Get intermediate layers of the VisionTransformer.

        Args:
            x (torch.Tensor): Input tensor.
            n (int): Number of last blocks to take. If it's a list, take the specified blocks.
            reshape (bool): Whether to reshape the output feature maps.
            return_class_token (bool): Whether to return the class token.
            norm (bool): Whether to apply normalization to the outputs.

        Returns:
            tuple: A tuple containing the intermediate layer outputs.
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_reg_tokens :] for out in outputs]
        if reshape:
            b, _, w, h = x.shape
            outputs = [
                out.reshape(b, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

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
        model: VisionTransformerBackbone,
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
                if self.peft != "lora":
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
