"""Module for defining CLIP Vision Transformer in mmcls."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
from mmcls.models import VisionTransformer
from mmcls.models.builder import BACKBONES
from mmcv.cnn import build_norm_layer


@BACKBONES.register_module()
class CLIPVisionTransformer(VisionTransformer):
    """CLIPVisionTransformer class."""

    def __init__(self, *args, **kwargs):
        """Initializes the CLIPVisionTransformer module.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None.
        """
        super().__init__(*args, **kwargs)
        _, norm_layer = build_norm_layer(kwargs["norm_cfg"], self.embed_dims, postfix=1)
        self.add_module("pre_norm", norm_layer)
        # self.proj = nn.Parameter((self.embed_dims ** -0.5) * torch.randn(self.embed_dims, 768))

    def forward(self, x):
        """Forward pass of the CLIPVisionTransformer module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple of output tensors.
        """
        b = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + VisionTransformer.resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens,
        )
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)
        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                b, _, c = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(b, *patch_resolution, c)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(b, *patch_resolution, c)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

        return tuple(outs)
