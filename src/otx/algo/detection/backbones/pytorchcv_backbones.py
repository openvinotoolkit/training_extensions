# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone of pytorchcv for mmdetection backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdet.registry import MODELS
from mmengine.dist import get_dist_info
from pytorchcv.model_provider import _models
from pytorchcv.models.model_store import download_model
from torch import distributed, nn
from torch.nn.modules.batchnorm import _BatchNorm

if TYPE_CHECKING:
    from mmengine.config import Config, ConfigDict
    from mmdet.registry import Registry

# ruff: noqa: SLF001


def replace_activation(model: nn.Module, activation_cfg: dict) -> nn.Module:
    """Replace activate funtion."""
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activation(module, activation_cfg)
        if "activ" in name:
            if activation_cfg["type"] == "torch_swish":
                model._modules[name] = nn.SiLU()
            else:
                model._modules[name] = build_activation_layer(activation_cfg)
    return model


def replace_norm(model: nn.Module, cfg: dict) -> nn.Module:
    """Replace norm funtion."""
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_norm(module, cfg)
        if "bn" in name:
            model._modules[name] = build_norm_layer(cfg, num_features=module.num_features)[1]
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

ori_build_func = MODELS.build_func

def torchcv_model_reduce(self) -> Any:
    return (build_model_including_torchcv, (self.otx_cfg,))

def build_model_including_torchcv(cfg: dict | ConfigDict | Config, registry: Registry = MODELS, *args, **kwargs) -> Any:
    try:
        model = ori_build_func(cfg, registry, *args, **kwargs)
    except KeyError:  # build from torchcv
        model_name = cfg.get('type')
        models_cache_root = kwargs.get("root", Path.home() / ".torch" / "models")
        is_pretrained = kwargs.get("pretrained", False)

        print(f"Init model {model_name}, pretrained={is_pretrained}, models cache {models_cache_root}")

        model = _models[model_name](*args, **kwargs)

        if activation_cfg := cfg.get("activation_cfg"):
            model = replace_activation(model, activation_cfg)
        if norm_cfg := cfg.get("norm_cfg"):
            model = replace_norm(model, norm_cfg)

        model.out_indices = cfg["out_indices"]
        model.frozen_stages = cfg.get("frozen_stages", 0)
        model.norm_eval = cfg.get("norm_eval", False)
        model.verbose = cfg.get("verbose", False)
        model.model_name = model_name
        model.models_cache_root = models_cache_root
        model.otx_cfg = cfg

        if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
            # Save original forward, just in case.
            model.forward_single_output = model.forward
            model.forward = multioutput_forward.__get__(model)
            model.init_weights = init_weights.__get__(model)
            model.train = train.__get__(model)

            model.output = None
            for i, _ in enumerate(model.features):
                if i > max(model.out_indices):
                    model.features[i] = None
        else:
            print(
                "Failed to automatically wrap backbone network. "
                f"Object of type {model.__class__} has no valid attribute called "
                "'features'.",
            )
        model.__class__.__reduce__ = torchcv_model_reduce.__get__(model)

    return model

MODELS.build_func = build_model_including_torchcv
