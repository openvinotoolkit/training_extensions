"""Torchvision Model Backbone Class Generation.

For torchvision backbone support for OTX models,
this is a code that converts torchvision backbone classes to match
the mmcv backbone class format and registers them in the mmcv registry.
This copied the format of "mmdet/models/backbones/imgclsmob.py"
as it is and made some modifications & code cleaning.
"""
# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import inspect
import os

from mmcv.cnn import build_activation_layer, build_norm_layer
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import models

from ..builder import TORCHVISION_BACKBONES

# pylint: disable=protected-access, assignment-from-no-return, no-value-for-parameter, too-many-statements


def get_torchvision_models():
    """Get torchvision backbones of current version."""
    torchvision_urls = {}
    torchvision_models = {}
    for model_key, model_value in models.__dict__.items():
        if callable(model_value) and model_key[0].islower() and model_key[0] != "_":
            torchvision_models[model_key] = model_value
        elif inspect.ismodule(model_value) and hasattr(model_value, "model_urls"):
            torchvision_urls.update(model_value.model_urls)
    return torchvision_models, torchvision_urls


TORCHVISION_MODELS, TORCHVISION_MODEL_URLS = get_torchvision_models()


def replace_activation(model, activation_cfg):
    """Replace Activate function (copy from mmdet)."""
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
    """Replace Norm function (copy from mmdet)."""
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_norm(module, cfg)
        if name == "bn":
            model._modules[name] = build_norm_layer(cfg, num_features=module.num_features)[1]
    return model


def resnet_forward(self, x):
    """Resnet forward function for wrapping model (refer to torchvision)."""
    outputs = []
    y = x
    stages = [self.layer1, self.layer2, self.layer3, self.layer4]
    last_stage = max(self.out_indices)
    y = self.conv1(y)
    y = self.bn1(y)
    y = self.relu(y)
    y = self.maxpool(y)
    for i, stage in enumerate(stages):
        y = stage(y)
        if i in self.out_indices:
            outputs.append(y)
        if i == last_stage:
            break
    return tuple(outputs)


def shufflenet_forward(self, x):
    """Shufflenet forward function for wrapping model (refer to torchvision)."""
    outputs = []
    y = x
    y = self.conv1(y)
    y = self.maxpool(y)
    stages = [self.stage2, self.stage3, self.stage4, self.conv5]
    last_stage = max(self.out_indices)
    for i, stage in enumerate(stages):
        y = stage(y)
        if i in self.out_indices:
            outputs.append(y)
        if i == last_stage:
            break
    return tuple(outputs)


def multioutput_forward(self, x):
    """Multioutput forward function for new model (copy from mmdet)."""
    outputs = []
    y = x

    last_stage = max(self.out_indices)
    if hasattr(self, "features"):
        stages = self.features
    elif hasattr(self, "layers"):
        stages = self.layers
    else:
        raise ValueError(f"Not supported multioutput forward: {self}")

    for i, stage in enumerate(stages):
        y = stage(y)
        temp_s = str(i) + " " + str(y.shape)
        if i in self.out_indices:
            outputs.append(y)
            temp_s += "*"
        if self.verbose:
            print(temp_s)
        if i == last_stage:
            break
    return tuple(outputs)


def train(self, mode=True):
    """Train forward function for new model (copy from mmdet)."""
    super(self.__class__, self).train(mode)

    if hasattr(self, "features"):
        stages = self.features
    elif hasattr(self, "layers"):
        stages = self.layers
    else:
        raise ValueError(f"Not supported multioutput forward: {self}")

    for i in range(self.frozen_stages + 1):
        temp_m = stages[i]
        temp_m.eval()
        for param in temp_m.parameters():
            param.requires_grad = False

    if mode and self.norm_eval:
        for mmodule in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(mmodule, _BatchNorm):
                mmodule.eval()


def init_weights(self):
    """Init weights function for new model (copy from mmdet)."""
    if self.init_cfg.get("Pretrained", False) and self.model_urls:
        state_dict = load_state_dict_from_url(self.model_urls)
        self.load_state_dict(state_dict)


def generate_torchvision_backbones():
    """Regist Torchvision Backbone into mmX Registry (copy from mmdet)."""
    for model_name, model_builder in TORCHVISION_MODELS.items():

        def closure(model_name, model_builder):
            """Get Model builder for mmcv (copy from mmdet)."""

            class TorchvisionModelWrapper(nn.Module):  # pylint: disable=abstract-method
                """Torchvision Model to MMX.model Wrapper (copy from mmdet)."""

                def __init__(
                    self,
                    *args,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=0,
                    norm_eval=False,
                    verbose=False,
                    activation_cfg=None,
                    norm_cfg=None,
                    init_cfg=None,
                    **kwargs,
                ):
                    super().__init__()
                    models_cache_root = kwargs.get("root", os.path.join("~", ".torch", "models"))
                    model = model_builder(*args, **kwargs)
                    if activation_cfg:
                        model = replace_activation(model, activation_cfg)
                    if norm_cfg:
                        model = replace_norm(model, norm_cfg)
                    model.out_indices = out_indices
                    model.frozen_stages = frozen_stages
                    model.norm_eval = norm_eval
                    model.verbose = verbose
                    model_name = str(self).strip("()")
                    model.model_urls = (
                        TORCHVISION_MODEL_URLS[model_name] if model_name in TORCHVISION_MODEL_URLS else None
                    )
                    model.models_cache_root = models_cache_root
                    model.init_cfg = init_cfg
                    model.init_weights = init_weights.__get__(model)
                    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
                        # Save original forward, just in case.
                        model.forward_single_output = model.forward
                        model.forward = multioutput_forward.__get__(model)
                        model.train = train.__get__(model)

                        model.output = None
                        for i, _ in enumerate(model.features):
                            if out_indices is not None and i > max(out_indices):
                                model.features[i] = None
                    elif hasattr(model, "layers") and isinstance(model.layers, nn.Sequential):
                        # Save original forward, just in case.
                        model.forward_single_output = model.forward
                        model.forward = multioutput_forward.__get__(model)
                        model.train = train.__get__(model)

                        model.classifier = None
                        for i, _ in enumerate(model.layers):
                            if out_indices is not None and i > max(out_indices):
                                model.layers[i] = None
                    elif model_name.startswith(("resne", "wide_resne")):
                        # torchvision.resne* -> resnet_forward
                        model.forward = resnet_forward.__get__(model)
                        model.fc = None
                    elif model_name.startswith("shufflenet"):
                        model.forward = shufflenet_forward.__get__(model)
                        model.fc = None
                    else:
                        raise ValueError(
                            "Failed to automatically wrap backbone network. "
                            f"Object of type {model.__class__} has no valid attribute called "
                            '"features".'
                        )
                    self.__dict__.update(model.__dict__)

            TorchvisionModelWrapper.__name__ = model_name
            return TorchvisionModelWrapper

        TORCHVISION_BACKBONES.register_module(name=model_name, module=closure(model_name, model_builder))


generate_torchvision_backbones()
