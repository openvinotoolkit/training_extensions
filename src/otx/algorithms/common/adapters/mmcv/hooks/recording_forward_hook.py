"""Recording forward hooks for explain mode."""
# Copyright (C) 2023 Intel Corporation
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

from __future__ import annotations

from abc import ABC
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn import LayerNorm

from otx.algorithms.classification import MMCLS_AVAILABLE
from otx.algorithms.common.utils.utils import cast_bf16_to_fp32

if MMCLS_AVAILABLE:
    from mmcls.models.necks.gap import GlobalAveragePooling


class BaseRecordingForwardHook(ABC):
    """While registered with the designated PyTorch module, this class caches feature vector during forward pass.

    Example::
        with BaseRecordingForwardHook(model.module.backbone) as hook:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            print(hook.records)

    Args:
        module (torch.nn.Module): The PyTorch module to be registered in forward pass
        fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                  Defaults to 0 which uses the largest feature map from FPN.
        normalize (bool): Whether to normalize the resulting saliency maps.
    """

    def __init__(self, module: torch.nn.Module, fpn_idx: int = -1, normalize: bool = True) -> None:
        self._module = module
        self._handle = None
        self._records: List[torch.Tensor] = []
        self._fpn_idx = fpn_idx
        self._norm_saliency_maps = normalize

    @property
    def records(self):
        """Return records."""
        return self._records

    def func(self, feature_map: torch.Tensor, fpn_idx: int = -1) -> torch.Tensor:
        """This method get the feature vector or saliency map from the output of the module.

        Args:
            feature_map (torch.Tensor): Feature map from the backbone module
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                    Defaults to 0 which uses the largest feature map from FPN.

        Returns:
            torch.Tensor (torch.Tensor): Saliency map for feature vector
        """
        raise NotImplementedError

    def _recording_forward(
        self, _: torch.nn.Module, x: torch.Tensor, output: torch.Tensor
    ):  # pylint: disable=unused-argument
        tensors = self.func(output)
        if isinstance(tensors, torch.Tensor):
            tensors_np = cast_bf16_to_fp32(tensors).detach().cpu().numpy()
        elif isinstance(tensors, np.ndarray):
            tensors_np = tensors
        else:
            self._torch_to_numpy_from_list(tensors)
            tensors_np = tensors

        for tensor in tensors_np:
            self._records.append(tensor)

    def _torch_to_numpy_from_list(self, tensor_list: List[Optional[torch.Tensor]]):
        for i in range(len(tensor_list)):
            if isinstance(tensor_list[i], list):
                self._torch_to_numpy_from_list(tensor_list[i])
            elif isinstance(tensor_list[i], torch.Tensor):
                tensor_list[i] = tensor_list[i].detach().cpu().numpy()

    def __enter__(self) -> BaseRecordingForwardHook:
        """Enter."""
        self._handle = self._module.backbone.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit."""
        self._handle.remove()

    def _normalize_map(self, saliency_maps: torch.Tensor) -> torch.Tensor:
        """Normalize saliency maps."""
        max_values, _ = torch.max(saliency_maps, -1)
        min_values, _ = torch.min(saliency_maps, -1)
        if len(saliency_maps.shape) == 2:
            saliency_maps = 255 * (saliency_maps - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        else:
            saliency_maps = (
                255 * (saliency_maps - min_values[:, :, None]) / (max_values - min_values + 1e-12)[:, :, None]
            )
        return saliency_maps.to(torch.uint8)


class EigenCamHook(BaseRecordingForwardHook):
    """EigenCamHook."""

    def func(self, feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the saliency map."""
        if isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[fpn_idx]

        x = feature_map.type(torch.float)
        batch_size, channel, h, w = x.size()
        reshaped_fmap = x.reshape((batch_size, channel, h * w)).transpose(1, 2)
        reshaped_fmap = reshaped_fmap - reshaped_fmap.mean(1)[:, None, :]
        _, _, vh = torch.linalg.svd(reshaped_fmap, full_matrices=True)  # pylint: disable=invalid-name

        if self._norm_saliency_maps:
            saliency_map = (reshaped_fmap @ vh[:, 0][:, :, None]).squeeze(-1)
            self._normalize_map(saliency_map)

        saliency_map = saliency_map.reshape((batch_size, h, w))
        return saliency_map


class ActivationMapHook(BaseRecordingForwardHook):
    """ActivationMapHook."""

    def func(self, feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the saliency map by average feature maps then normalizing to (0, 255)."""
        if isinstance(feature_map, (list, tuple)):
            assert fpn_idx < len(
                feature_map
            ), f"fpn_idx: {fpn_idx} is out of scope of feature_map length {len(feature_map)}!"
            feature_map = feature_map[fpn_idx]

        batch_size, _, h, w = feature_map.size()
        activation_map = torch.mean(feature_map, dim=1)

        if self._norm_saliency_maps:
            activation_map = activation_map.reshape((batch_size, h * w))
            activation_map = self._normalize_map(activation_map)

        activation_map = activation_map.reshape((batch_size, h, w))
        return activation_map


class FeatureVectorHook(BaseRecordingForwardHook):
    """FeatureVectorHook."""

    @staticmethod
    def func(feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the feature vector by average pooling feature maps."""
        if isinstance(feature_map, (list, tuple)):
            # aggregate feature maps from Feature Pyramid Network
            feature_vector = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)) for f in feature_map]
            feature_vector = torch.cat(feature_vector, 1)
        else:
            feature_vector = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
        return feature_vector


class ViTFeatureVectorHook(BaseRecordingForwardHook):
    """FeatureVectorHook for transformer-based classifiers."""

    @staticmethod
    def func(features: Tuple[List[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the feature vector for transformer-based classifiers by returning the cls token."""
        _, cls_token = features[0]
        return cls_token


class ReciproCAMHook(BaseRecordingForwardHook):
    """Implementation of recipro-cam for class-wise saliency map.

    recipro-cam: gradient-free reciprocal class activation map (https://arxiv.org/pdf/2209.14074.pdf)
    """

    def __init__(self, module: torch.nn.Module, fpn_idx: int = -1) -> None:
        super().__init__(module, fpn_idx)
        self._neck = module.neck if module.with_neck else None
        self._head = module.head
        self._num_classes = module.head.num_classes

    def func(self, feature_map: Union[torch.Tensor, Sequence[torch.Tensor]], fpn_idx: int = -1) -> torch.Tensor:
        """Generate the class-wise saliency maps using Recipro-CAM and then normalizing to (0, 255).

        Args:
            feature_map (Union[torch.Tensor, List[torch.Tensor]]): feature maps from backbone or list of feature maps
                                                                    from FPN.
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                      Defaults to 0 which uses the largest feature map from FPN.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        if isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[fpn_idx]

        batch_size, channel, h, w = feature_map.size()
        saliency_maps = torch.empty(batch_size, self._num_classes, h, w)
        for f in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[f], channel, h, w)
            mosaic_prediction = self._predict_from_feature_map(mosaic_feature_map)
            saliency_maps[f] = mosaic_prediction.transpose(0, 1).reshape((self._num_classes, h, w))

        if self._norm_saliency_maps:
            saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h * w))
            saliency_maps = self._normalize_map(saliency_maps)

        saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h, w))
        return saliency_maps

    def _predict_from_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._neck is not None:
                x = self._neck(x)
            logits = self._head.simple_test(x)
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits)
        return logits

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
        if MMCLS_AVAILABLE and self._neck is not None and isinstance(self._neck, GlobalAveragePooling):
            # Optimization workaround for the GAP case (simulate GAP with more simple compute graph)
            # Possible due to static sparsity of mosaic_feature_map
            # Makes the downstream GAP operation to be dummy
            feature_map_transposed = torch.flatten(feature_map, start_dim=1).transpose(0, 1)[:, :, None, None]
            mosaic_feature_map = feature_map_transposed / (h * w)
        else:
            feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
            mosaic_feature_map_mask = torch.zeros(h * w, c, h, w).to(feature_map.device)
            spacial_order = torch.arange(h * w).reshape(h, w)
            for i in range(h):
                for j in range(w):
                    k = spacial_order[i, j]
                    mosaic_feature_map_mask[k, :, i, j] = torch.ones(c).to(feature_map.device)
            mosaic_feature_map = feature_map_repeated * mosaic_feature_map_mask
        return mosaic_feature_map


class ViTReciproCAMHook(BaseRecordingForwardHook):
    """Implementation of ViTRecipro-CAM for class-wise saliency map for transformer-based classifiers.

    Args:
    module (torch.nn.Module): The PyTorch module.
    layer_index (int): Index of the target transformer_encoder layer.
    use_gaussian (bool): Defines kernel type for mosaic feature map generation.
    If True, use gaussian 3x3 kernel. If False, use 1x1 kernel.
    cls_token (bool): If True, includes classification token into the mosaic feature map.
    """

    def __init__(
        self, module: torch.nn.Module, layer_index: int = -1, use_gaussian: bool = True, cls_token: bool = True
    ):
        super().__init__(module)
        self._layer_index = layer_index
        self._target_layernorm = self._get_target_layernorm()
        self._final_norm = module.backbone.norm1 if module.backbone.final_norm else None
        self._neck = module.neck if module.with_neck else None
        self._num_classes = module.head.num_classes
        self._use_gaussian = use_gaussian
        self._cls_token = cls_token

    def _get_target_layernorm(self) -> torch.nn.Module:
        """Returns the first (out of two) layernorm layer from the layer_index backbone layer."""
        assert self._layer_index < 0, "negative index expected, e.g. -1 for the last layer."
        layernorm_layers = []
        for module in self._module.backbone.modules():
            if isinstance(module, LayerNorm):
                layernorm_layers.append(module)
        assert len(layernorm_layers) == self._module.backbone.num_layers * 2 + int(self._module.backbone.final_norm)
        target_layernorm_index = self._layer_index * 2 - int(self._module.backbone.final_norm)
        return layernorm_layers[target_layernorm_index]

    def func(self, feature_map: torch.Tensor, _: int = -1) -> torch.Tensor:
        """Generate the class-wise saliency maps using ViTRecipro-CAM and then normalizing to (0, 255).

        Args:
            feature_map (torch.Tensor): feature maps from target layernorm layer.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        batch_size, token_number, _ = feature_map.size()
        h = w = int((token_number - 1) ** 0.5)
        saliency_maps = torch.empty(batch_size, self._num_classes, h, w)
        for i in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[i])
            mosaic_prediction = self._predict_from_feature_map(mosaic_feature_map)
            saliency_maps[i] = mosaic_prediction.transpose(1, 0).reshape((self._num_classes, h, w))

        if self._norm_saliency_maps:
            saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h * w))
            saliency_maps = self._normalize_map(saliency_maps)

        saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, h, w))
        return saliency_maps

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        token_number, dim = feature_map.size()
        mosaic_feature_map = torch.zeros(token_number - 1, token_number, dim).to(feature_map.device)
        h = w = int((token_number - 1) ** 0.5)

        if self._use_gaussian:
            if self._cls_token:
                mosaic_feature_map[:, 0, :] = feature_map[0, :]
            feature_map_spacial = feature_map[1:, :].reshape(1, h, w, dim)
            feature_map_spacial_repeated = feature_map_spacial.repeat(h * w, 1, 1, 1)  # 196, 14, 14, 192

            spacial_order = torch.arange(h * w).reshape(h, w)
            gaussian = torch.tensor(
                [[1 / 16.0, 1 / 8.0, 1 / 16.0], [1 / 8.0, 1 / 4.0, 1 / 8.0], [1 / 16.0, 1 / 8.0, 1 / 16.0]]
            ).to(feature_map.device)
            mosaic_feature_map_mask_padded = torch.zeros(h * w, h + 2, w + 2).to(feature_map.device)
            for i in range(h):
                for j in range(w):
                    k = spacial_order[i, j]
                    i_pad = i + 1
                    j_pad = j + 1
                    mosaic_feature_map_mask_padded[k, i_pad - 1 : i_pad + 2, j_pad - 1 : j_pad + 2] = gaussian
            mosaic_feature_map_mask = mosaic_feature_map_mask_padded[:, 1:-1, 1:-1]
            mosaic_feature_map_mask = torch.tensor(mosaic_feature_map_mask.unsqueeze(3).repeat(1, 1, 1, dim))

            mosaic_fm_wo_cls_token = feature_map_spacial_repeated * mosaic_feature_map_mask
            mosaic_feature_map[:, 1:, :] = mosaic_fm_wo_cls_token.reshape(h * w, h * w, dim)
        else:
            feature_map_repeated = feature_map.unsqueeze(0).repeat(h * w, 1, 1)
            mosaic_feature_map_mask = torch.zeros(h * w, token_number).to(feature_map.device)
            for i in range(h * w):
                mosaic_feature_map_mask[i, i + 1] = torch.ones(1).to(feature_map.device)
            if self._cls_token:
                mosaic_feature_map_mask[:, 0] = torch.ones(1).to(feature_map.device)
            mosaic_feature_map_mask = torch.tensor(mosaic_feature_map_mask.unsqueeze(2).repeat(1, 1, dim))
            mosaic_feature_map = feature_map_repeated * mosaic_feature_map_mask

        return mosaic_feature_map

    def _predict_from_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Part of the target transformer_encoder layer (except first LayerNorm)
            target_layer = self._module.backbone.layers[self._layer_index]
            x = x + target_layer.attn(x)
            x = target_layer.ffn(target_layer.norm2(x), identity=x)

            # Rest transformer_encoder layers, if not the last one picked as a target
            if self._layer_index < -1:
                for layer in self._module.backbone.layers[(self._layer_index + 1) :]:
                    x = layer(x)

            if self._final_norm:
                x = self._final_norm(x)
            if self._neck:
                x = self._neck(x)

            cls_token = x[:, 0]
            layer_output = [None, cls_token]
            logit = self._module.head.simple_test(layer_output)
            if isinstance(logit, list):
                logit = torch.from_numpy(np.array(logit))
        return logit

    def __enter__(self) -> BaseRecordingForwardHook:
        """Enter."""
        self._handle = self._target_layernorm.register_forward_hook(self._recording_forward)
        return self
