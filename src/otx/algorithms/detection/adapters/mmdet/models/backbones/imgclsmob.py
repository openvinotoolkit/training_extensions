"""Backbone of pytorchcv for mmdetection backbones."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import get_dist_info
from mmdet.models.builder import BACKBONES
from mmdet.utils.logger import get_root_logger
from pytorchcv.model_provider import _models
from pytorchcv.models.model_store import download_model
from torch import distributed, nn
from torch.nn.modules.batchnorm import _BatchNorm

# TODO: Need to fix pylint issues
# pylint: disable=protected-access, abstract-method, no-value-for-parameter, assignment-from-no-return


def replace_activation(model, activation_cfg):
    """Replace activate funtion."""
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activation(module, activation_cfg)
        if name == "activ":
            if activation_cfg["type"] == "torch_swish":
                model._modules[name] = nn.SiLU()
            else:
                model._modules[name] = build_activation_layer(activation_cfg)
    return model


def replace_norm(model, cfg):
    """Replace norm funtion."""
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_norm(module, cfg)
        if name == "bn":
            model._modules[name] = build_norm_layer(cfg, num_features=module.num_features)[1]
    return model


def multioutput_forward(self, x):
    """Multioutput forward function for new model (copy from mmdet older)."""
    outputs = []
    y = x

    last_stage = max(self.out_indices)
    for i, stage in enumerate(self.features):
        y = stage(y)
        s_verbose = str(i) + " " + str(y.shape)
        if i in self.out_indices:
            outputs.append(y)
            s_verbose += "*"
        if self.verbose:
            print(s_verbose)
        if i == last_stage:
            break

    return outputs


def train(self, mode=True):
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


def init_weights(self, pretrained=True):
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


def generate_backbones():
    """Generate backbones of pytorchcv funtion."""
    logger = get_root_logger()

    for model_name, model_getter in _models.items():

        def closure(model_name, model_getter):
            """Get Model builder for mmcv (copy from mmdet old version)."""

            class CustomModelGetter(nn.Module):
                """Custom Model getter class."""

                def __init__(
                    self,
                    *args,
                    out_indices=None,
                    frozen_stages=0,
                    norm_eval=False,
                    verbose=False,
                    activation_cfg=None,
                    norm_cfg=None,
                    **kwargs,
                ):
                    super().__init__()
                    models_cache_root = kwargs.get("root", os.path.join("~", ".torch", "models"))
                    is_pretrained = kwargs.get("pretrained", False)
                    logger.warning(
                        f"Init model {model_name}, pretrained={is_pretrained}, models cache {models_cache_root}"
                    )
                    model = model_getter(*args, **kwargs)
                    if activation_cfg:
                        model = replace_activation(model, activation_cfg)
                    if norm_cfg:
                        model = replace_norm(model, norm_cfg)
                    model.out_indices = out_indices
                    model.frozen_stages = frozen_stages
                    model.norm_eval = norm_eval
                    model.verbose = verbose
                    model.model_name = model_name
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
                        raise ValueError(
                            "Failed to automatically wrap backbone network. "
                            f"Object of type {model.__class__} has no valid attribute called "
                            "'features'."
                        )
                    self.__dict__.update(model.__dict__)

            CustomModelGetter.__name__ = model_name
            return CustomModelGetter

        BACKBONES.register_module(name=model_name, module=closure(model_name, model_getter))


generate_backbones()
