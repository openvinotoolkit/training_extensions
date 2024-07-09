# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from mmpretrain/models/backbones/vision_transformer.py."""
from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from timm.layers import (
    LayerType,
    Mlp,
    PatchDropout,
    get_act_layer,
    get_norm_layer,
    resample_abs_pos_embed,
    resample_patch_embed,
    trunc_normal_,
)
from timm.layers import PatchEmbed as TimmPatchEmbed
from timm.models._manipulate import adapt_input_conv
from timm.models.vision_transformer import Block
from torch import nn

from otx.algo.classification.utils.attention import MultiheadAttention
from otx.algo.classification.utils.embed import resize_pos_embed
from otx.algo.classification.utils.swiglu_ffn import SwiGLUFFNFused
from otx.algo.modules.base_module import BaseModule, ModuleList
from otx.algo.modules.norm import build_norm_layer
from otx.algo.modules.transformer import FFN, PatchEmbed


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_fcs: int = 2,
        qkv_bias: bool = True,
        ffn_type: str = "origin",
        act_cfg: dict = {"type": "GELU"},  # noqa: B006
        norm_cfg: dict = {"type": "LN"},  # noqa: B006
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg=init_cfg)

        act_cfg = act_cfg if act_cfg else {"type": "GELU"}
        self.embed_dims = embed_dims

        _, self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer={"type": "DropPath", "drop_prob": drop_path_rate},
            qkv_bias=qkv_bias,
        )

        _, self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        if ffn_type == "origin":
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                act_cfg=act_cfg,
                ffn_drop=drop_rate,
                dropout_layer={"type": "DropPath", "drop_prob": drop_path_rate},
            )
        elif ffn_type == "swiglu_fused":
            self.ffn = SwiGLUFFNFused(embed_dims=embed_dims, feedforward_channels=feedforward_channels)
        else:
            raise NotImplementedError

    @property
    def norm1(self) -> nn.Module:
        """Returns the normalization layer used in the Vision Transformer backbone.

        Returns:
            nn.Module: The normalization layer.
        """
        return self.ln1

    @property
    def norm2(self) -> nn.Module:
        """Returns the second normalization layer of the VisionTransformer backbone.

        Returns:
            nn.Module: The second normalization layer.
        """
        return self.ln2

    def init_weights(self) -> None:
        """Initializes the weights of the TransformerEncoderLayer.

        This method overrides the `init_weights` method of the base class and initializes the weights
        of the linear layers in the feed-forward network (ffn) using Xavier uniform initialization
        for the weights and normal distribution with a standard deviation of 1e-6 for the biases.
        """
        super().init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the VisionTransformer model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x + self.attn(self.ln1(x))
        return self.ffn(self.ln2(x), identity=x)


VIT_ARCH_TYPE = Literal[
    "small",
    "base",
    "large",
    "huge",
    "eva-g",
    "eva-giant",
    "deit-t",
    "deit-tiny",
    "deit-s",
    "deit-small",
    "dinov2-s",
    "dinov2-small",
    "deit-b",
    "deit-base",
    "dinov2-g",
    "dinov2-giant",
]


class VisionTransformer(BaseModule):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"cls_token"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    arch_zoo = {  # noqa: RUF012
        **dict.fromkeys(
            ["s", "small"],
            {
                "embed_dims": 768,
                "num_layers": 8,
                "num_heads": 8,
                "feedforward_channels": 768 * 3,
            },
        ),
        **dict.fromkeys(
            ["b", "base"],
            {
                "embed_dims": 768,
                "num_layers": 12,
                "num_heads": 12,
                "feedforward_channels": 3072,
            },
        ),
        **dict.fromkeys(
            ["l", "large"],
            {
                "embed_dims": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "feedforward_channels": 4096,
            },
        ),
        **dict.fromkeys(
            ["h", "huge"],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                "embed_dims": 1280,
                "num_layers": 32,
                "num_heads": 16,
                "feedforward_channels": 5120,
            },
        ),
        **dict.fromkeys(
            ["eva-g", "eva-giant"],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                "embed_dims": 1408,
                "num_layers": 40,
                "num_heads": 16,
                "feedforward_channels": 6144,
            },
        ),
        **dict.fromkeys(
            ["deit-t", "deit-tiny"],
            {
                "embed_dims": 192,
                "num_layers": 12,
                "num_heads": 3,
                "feedforward_channels": 192 * 4,
            },
        ),
        **dict.fromkeys(
            ["deit-s", "deit-small", "dinov2-s", "dinov2-small"],
            {
                "embed_dims": 384,
                "num_layers": 12,
                "num_heads": 6,
                "feedforward_channels": 384 * 4,
            },
        ),
        **dict.fromkeys(
            ["deit-b", "deit-base"],
            {
                "embed_dims": 768,
                "num_layers": 12,
                "num_heads": 12,
                "feedforward_channels": 768 * 4,
            },
        ),
        **dict.fromkeys(
            ["dinov2-g", "dinov2-giant"],
            {
                "embed_dims": 1536,
                "num_layers": 40,
                "num_heads": 24,
                "feedforward_channels": 6144,
            },
        ),
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {"raw", "cls_token", "featmap", "avg_featmap"}  # noqa: RUF012

    def __init__(
        self,
        arch: VIT_ARCH_TYPE = "base",
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        out_indices: int | list[int] = -1,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        norm_cfg: dict = {"type": "LN", "eps": 1e-6},  # noqa: B006
        final_norm: bool = True,
        out_type: str = "cls_token",
        with_cls_token: bool = True,
        frozen_stages: int = -1,
        interpolate_mode: str = "bicubic",
        patch_cfg: dict = {},  # noqa: B006
        layer_cfgs: dict | list[dict] = {},  # noqa: B006
        pre_norm: bool = False,
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg)

        self.arch = arch
        if isinstance(arch, str):
            if arch not in set(self.arch_zoo):
                msg = f"Arch {arch} is not in default archs {set(self.arch_zoo)}"
                raise ValueError(msg)
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                "embed_dims",
                "num_layers",
                "num_heads",
                "feedforward_channels",
            }
            if not (isinstance(arch, dict) and essential_keys <= set(arch)):
                msg = f"Custom arch needs a dict with keys {essential_keys}"
                raise ValueError(msg)
            self.arch_settings = arch

        self.embed_dims = self.arch_settings["embed_dims"]
        self.num_layers = self.arch_settings["num_layers"]
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        # Set patch embedding
        _patch_cfg = {
            "in_channels": in_channels,
            "input_size": img_size,
            "embed_dims": self.embed_dims,
            "conv_type": "Conv2d",
            "kernel_size": patch_size,
            "stride": patch_size,
            "bias": not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
        }
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)  # type: ignore [arg-type]
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            msg = f"Unsupported `out_type` {out_type}, please choose from {self.OUT_TYPES}"
            raise ValueError(msg)
        self.out_type = out_type

        # Set cls token
        self.with_cls_token = with_cls_token
        self.cls_token: nn.Parameter | None
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != "cls_token":
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            msg = 'with_cls_token must be True when `out_type="cls_token"`.'
            raise ValueError(msg)

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        if not isinstance(out_indices, Sequence):
            msg = f"'out_indices' must by a sequence or int, get {type(out_indices)} instead."
            raise TypeError(msg)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            if not (0 <= out_indices[i] <= self.num_layers):
                msg = f"Invalid out_indices {index}"
                raise AssertionError(msg)
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = {
                "embed_dims": self.embed_dims,
                "num_heads": self.arch_settings["num_heads"],
                "feedforward_channels": self.arch_settings["feedforward_channels"],
                "drop_rate": drop_rate,
                "drop_path_rate": dpr[i],
                "qkv_bias": qkv_bias,
                "norm_cfg": norm_cfg,
            }
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))  # type: ignore [arg-type]

        self.frozen_stages = frozen_stages
        self.pre_norm: nn.Module
        if pre_norm:
            _, self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = final_norm
        if final_norm:
            _, self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        if self.out_type == "avg_featmap":
            _, self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self) -> nn.Module:
        """Returns the normalization layer used in the Vision Transformer backbone.

        Returns:
            nn.Module: The normalization layer.
        """
        return self.ln1

    @property
    def norm2(self) -> nn.Module:
        """Returns the normalization layer used in the Vision Transformer backbone.

        Returns:
            nn.Module: The normalization layer.
        """
        return self.ln2

    def init_weights(self) -> None:
        """Initializes the weights of the VisionTransformer."""
        super().init_weights()

        if not (
            isinstance(self.init_cfg, dict) and self.init_cfg["type"] == "Pretrained"
        ) and self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict: dict, prefix: str, *args, **kwargs) -> None:
        name = prefix + "pos_embed"
        if name not in state_dict:
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if not self.with_cls_token and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1:
            # Remove cls token from state dict if it's not used.
            state_dict[name] = state_dict[name][:, 1:]
            ckpt_pos_embed_shape = state_dict[name].shape

        if self.pos_embed.shape != ckpt_pos_embed_shape:
            embed_size = int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens))
            # ckpt_pos_embed_shape = to_2tuple(
            #     int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            ckpt_pos_embed_shape = (embed_size, embed_size)
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(
                state_dict[name],
                ckpt_pos_embed_shape,
                pos_embed_shape,
                self.interpolate_mode,
                self.num_extra_tokens,
            )

    @staticmethod
    def resize_pos_embed(*args, **kwargs) -> torch.Tensor:
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self) -> None:
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == "avg_featmap":
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass of the VisionTransformer model."""
        b = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens,
        )
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x: torch.Tensor, hw: nn.Module) -> torch.Tensor:
        if self.out_type == "raw":
            return x
        if self.out_type == "cls_token":
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens :]
        if self.out_type == "featmap":
            b = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(b, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == "avg_featmap":
            return self.ln2(patch_token.mean(dim=1))
        raise NotImplementedError

    def get_layer_depth(self, param_name: str, prefix: str = "") -> tuple[int, int]:
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed") or param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("layers"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers


class TimmVisionTransformer(BaseModule):
    """Implementation of Vision Transformer from Timm.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
        - https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of image input channels.
        num_classes: Mumber of classes for classification head.
        global_pool: Type of global pooling for final sequence (default: 'token').
        embed_dim: Transformer embedding dimension.
        depth: Depth of transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Enable bias for qkv projections if True.
        init_values: Layer-scale init values (layer-scale enabled if not None).
        class_token: Use class token.
        no_embed_class: Don't include position embeddings for class (or reg) tokens.
        reg_tokens: Number of register tokens.
        fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
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
    """

    def __init__(  # noqa: PLR0913
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: bool | None = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Callable = TimmPatchEmbed,
        norm_layer: LayerType | None = None,
        act_layer: LayerType | None = None,
        block_fn: nn.Module = Block,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")  # noqa: S101
        assert class_token or global_pool != "token"  # noqa: S101
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
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
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.with_cls_token = class_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
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

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

    def init_weights(self) -> None:
        """Initializes the weights of the VisionTransformer."""
        super().init_weights()

        if not (
            isinstance(self.init_cfg, dict) and self.init_cfg["type"] == "Pretrained"
        ) and self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = "") -> None:
        """Loads the pretrained weight to the VisionTransformer."""
        _load_weights(self, checkpoint_path, prefix)

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

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass of the VisionTransformer model."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.blocks(x)
        x = self.norm(x)[:, 0]  # extract cls_token only
        return (x,)


@torch.no_grad()
def _load_weights(  # noqa: C901
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
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"]))
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
    for i, block in enumerate(model.blocks.children()):
        if f"{prefix}Transformer/encoderblock/LayerNorm_0/scale" in w:
            block_prefix = f"{prefix}Transformer/encoderblock/"
            idx = i
        else:
            block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
            idx = None
        mha_prefix = block_prefix + f"MultiHeadDotProductAttention_{mha_sub}/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"], idx=idx))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"], idx=idx))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [_n2p(w[f"{mha_prefix}{n}/kernel"], t=False, idx=idx).flatten(1).T for n in ("query", "key", "value")],
            ),
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [_n2p(w[f"{mha_prefix}{n}/bias"], t=False, idx=idx).reshape(-1) for n in ("query", "key", "value")],
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
