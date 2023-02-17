# Copyright (C) 2020-2021 Intel Corporation
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

import logging
import os
import tempfile

import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import get_dist_info

# from ..builder import BACKBONES
from mmdet.models.builder import BACKBONES
from mmdet.utils.logger import get_root_logger
from pytorchcv.model_provider import _models
from pytorchcv.models.model_store import download_model
from torch import distributed
from torch.nn.modules.batchnorm import _BatchNorm

# import types


def replace_activation(model, activation_cfg):
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
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            model._modules[name] = replace_norm(module, cfg)
        if name == "bn":
            model._modules[name] = build_norm_layer(cfg, num_features=module.num_features)[1]
    return model


def generate_backbones():
    logger = get_root_logger()

    for model_name, model_getter in _models.items():

        def closure(model_name, model_getter):
            def multioutput_forward(self, x):
                outputs = []
                y = x

                last_stage = max(self.out_indices)
                for i, stage in enumerate(self.features):
                    y = stage(y)
                    s = str(i) + " " + str(y.shape)
                    if i in self.out_indices:
                        outputs.append(y)
                        s += "*"
                    if self.verbose:
                        print(s)
                    if i == last_stage:
                        break

                return outputs

            def init_weights(self, pretrained=True):
                if pretrained:
                    rank, world_size = get_dist_info()
                    logger.warning(f"imgclsmob::loading weights proc rank {rank}")
                    if rank == 0:
                        # Make sure that model is fetched to the local storage.
                        logger.warning(f"imgclsmob::downloading {rank}")
                        download_model(
                            net=self, model_name=model_name, local_model_store_dir_path=self.models_cache_root
                        )
                        if world_size > 1:
                            logger.warning(f"imgclsmob::barrier {rank}")
                            distributed.barrier()
                    else:
                        # Wait for model to be in the local storage, then load it.
                        logger.warning(f"imgclsmob::barrier {rank}")
                        distributed.barrier()
                        logger.warning(f"imgclsmob::loading {rank}")
                        download_model(
                            net=self, model_name=model_name, local_model_store_dir_path=self.models_cache_root
                        )
                    logger.warning(f"imgclsmob::done {rank}")

            def train(self, mode=True):
                super(self.__class__, self).train(mode)

                for i in range(self.frozen_stages + 1):
                    m = self.features[i]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

                if mode and self.norm_eval:
                    for m in self.modules():
                        # trick: eval have effect on BatchNorm only
                        if isinstance(m, _BatchNorm):
                            m.eval()

            class custom_model_getter(nn.Module):
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
                    # if 'pretrained' in kwargs and kwargs['pretrained']:
                    #     rank, _ = get_dist_info()
                    #     if rank > 0:
                    #         if 'root' not in kwargs:
                    #             kwargs['root'] = tempfile.mkdtemp()
                    #         kwargs['root'] = tempfile.mkdtemp(dir=kwargs['root'])
                    #         logger.info('Rank: {}, Setting {} as a target location of pretrained models'.format(rank, kwargs['root']))
                    model = model_getter(*args, **kwargs)
                    if activation_cfg:
                        model = replace_activation(model, activation_cfg)
                    if norm_cfg:
                        model = replace_norm(model, norm_cfg)
                    model.out_indices = out_indices
                    model.frozen_stages = frozen_stages
                    model.norm_eval = norm_eval
                    model.verbose = verbose
                    model.models_cache_root = models_cache_root
                    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
                        # Save original forward, just in case.
                        model.forward_single_output = model.forward
                        model.forward = multioutput_forward.__get__(model)
                        model.init_weights = init_weights.__get__(model)
                        model.train = train.__get__(model)

                        # model.forward = types.MethodType(multioutput_forward, model)
                        # model.init_weights = types.MethodType(init_weights, model)
                        # model.train = types.MethodType(train, model)

                        model.output = None
                        for i, _ in enumerate(model.features):
                            if i > max(out_indices):
                                model.features[i] = None
                    else:
                        raise ValueError(
                            "Failed to automatically wrap backbone network. "
                            "Object of type {} has no valid attribute called "
                            '"features".'.format(model.__class__)
                        )
                    self.__dict__.update(model.__dict__)

            custom_model_getter.__name__ = model_name
            return custom_model_getter

        BACKBONES.register_module(name=model_name, module=closure(model_name, model_getter))


generate_backbones()
