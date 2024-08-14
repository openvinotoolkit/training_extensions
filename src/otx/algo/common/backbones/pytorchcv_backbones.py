# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone of pytorchcv for mmdetection backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from otx.algo.modules.norm import build_norm_layer
from otx.algo.utils.mmengine_utils import get_dist_info
from pytorchcv.model_provider import _models
from pytorchcv.models.model_store import download_model
from torch import distributed, nn
from torch.nn.modules.batchnorm import _BatchNorm

# ruff: noqa: SLF001


def replace_activation(model: nn.Module, activation: Callable[..., nn.Module]) -> nn.Module:
    """Replace activation funtion."""
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activation(module, activation)
        if "activ" in name:
            model._modules[name] = activation()
    return model


def replace_norm(model: nn.Module, normalization: Callable[..., nn.Module]) -> nn.Module:
    """Replace norm funtion."""
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_norm(module, normalization)
        if "bn" in name:
            model._modules[name] = build_norm_layer(normalization, num_features=module.num_features)[1]
    return model


def multioutput_forward(self: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Multioutput forward function for new model (copy from mmdet older)."""
    outputs: list[torch.Tensor] = []

    last_stage = max(self.out_indices)
    for i, stage in enumerate(self.features):
        x = stage(x)
        s_verbose = str(i) + " " + str(x.shape)
        if i in self.out_indices:
            outputs.append(x)
            s_verbose += "*"
        if self.verbose:
            print(s_verbose)
        if i == last_stage:
            break

    return outputs


def train(self: nn.Module, mode: bool = True) -> None:
    """Train forward function for new model (copy from mmdet older)."""
    super(self.__class__, self).train(mode)

    for i in range(self.frozen_stages + 1):
        feature = self.features[i]
        feature.eval()
        for param in feature.parameters():
            param.requires_grad = False

    if mode and self.norm_eval:
        for module in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(module, _BatchNorm):
                module.eval()


def init_weights(self: nn.Module, pretrained: bool = True) -> None:
    """Init weights function for new model (copy from mmdet)."""
    if pretrained:
        rank, world_size = get_dist_info()
        if rank == 0:
            # Make sure that model is fetched to the local storage.
            download_model(net=self, model_name=self.model_name, local_model_store_dir_path=self.models_cache_root)
            if world_size > 1:
                distributed.barrier()
        else:
            # Wait for model to be in the local storage, then load it.
            distributed.barrier()
            download_model(net=self, model_name=self.model_name, local_model_store_dir_path=self.models_cache_root)


def _pytorchcv_model_reduce(self) -> nn.Module:  # noqa: ANN001
    return (build_model_including_pytorchcv, (self.otx_cfg,))


def build_model_including_pytorchcv(
    cfg: dict,
    default_args: dict | None = None,
) -> nn.Module:
    """Build model from pytorchcv."""
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    model = _build_pytorchcv_model(**args)

    # support pickle
    model.otx_cfg = args
    model.__class__.__reduce__ = _pytorchcv_model_reduce.__get__(model, model.__class__)

    return model


def _build_pytorchcv_model(
    type: str,  # noqa: A002
    out_indices: list[int],
    frozen_stages: int = 0,
    norm_eval: bool = False,
    verbose: bool = False,
    activation: Callable[..., nn.Module] | None = None,
    normalization: Callable[..., nn.Module] | None = None,
    **kwargs,
) -> nn.Module:
    """Build pytorchcv model."""
    models_cache_root = kwargs.pop("root", Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    pretrained = kwargs.pop("pretrained", False)
    print(f"Init model {type}, pretrained={pretrained}, models cache {models_cache_root}")
    model = _models[type](root=models_cache_root, pretrained=pretrained, **kwargs)
    if activation:
        model = replace_activation(model, activation)
    if normalization:
        model = replace_norm(model, normalization)
    model.out_indices = out_indices
    model.frozen_stages = frozen_stages
    model.norm_eval = norm_eval
    model.verbose = verbose
    model.model_name = type
    model.models_cache_root = models_cache_root
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        # Save original forward, just in case.
        model.forward_single_output = model.forward
        model.forward = multioutput_forward.__get__(model)
        model.init_weights = init_weights.__get__(model)
        model.train = train.__get__(model)

        model.output = None
        for i, _ in enumerate(model.features):
            if i > max(out_indices):
                model.features[i] = None
    else:
        print(
            "Failed to automatically wrap backbone network. "
            f"Object of type {model.__class__} has no valid attribute called "
            "'features'.",
        )

    return model
