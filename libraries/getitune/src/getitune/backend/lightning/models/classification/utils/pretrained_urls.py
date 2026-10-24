# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pretrained weight URLs for classification backbones.

Central registry of remote checkpoint URLs used to initialize classification
backbones with pretrained weights. Each mapping is keyed by ``model_name`` (the
identifier passed to a model/backbone) and resolves to a downloadable checkpoint
hosted on Intel's weight storage.

These constants are consumed by the classification loader mixins (see
:mod:`getitune.backend.lightning.models.classification.utils.pretrained_weights`), which
resolve the URL for a given ``model_name``, download the file into
``PRETRAINED_WEIGHTS_CACHE_DIR``, and load it into the backbone.

Note:
    URLs point to external resources and require network access at load time unless the
    checkpoint is already cached.
"""

from __future__ import annotations

_AUGREG_BASE_URL = "https://storage.googleapis.com/vit_models/augreg"
_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
_MOBILENETV3_BASE_URL = "https://raw.githubusercontent.com/d-li14/mobilenetv3.pytorch/master/pretrained"


# ViT / DINOv2 ImageNet-21k checkpoints shared by classification models.
VIT_PRETRAINED_URLS: dict[str, str] = {
    "vit-tiny": (
        f"{_AUGREG_BASE_URL}/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit-small": (
        f"{_AUGREG_BASE_URL}/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit-base": (
        f"{_AUGREG_BASE_URL}/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit-large": (
        f"{_AUGREG_BASE_URL}/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "dinov2-small": f"{_DINOV2_BASE_URL}/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2-base": f"{_DINOV2_BASE_URL}/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2-large": f"{_DINOV2_BASE_URL}/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2-giant": f"{_DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
}

# MobileNetV3 ImageNet checkpoints.
MOBILENETV3_PRETRAINED_URLS: dict[str, str] = {
    "mobilenetv3_small": f"{_MOBILENETV3_BASE_URL}/mobilenetv3-small-55df8e1f.pth",
    "mobilenetv3_large": f"{_MOBILENETV3_BASE_URL}/mobilenetv3-large-1cd25616.pth",
    "mobilenetv3_large_075": f"{_MOBILENETV3_BASE_URL}/mobilenetv3-large-0.75-9632d2a8.pth",
}
