# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Algorithms for calculcalating XAI branch for Explainable AI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from otx.core.types.explain import FeatureMapType

if TYPE_CHECKING:
    import numpy as np

    from otx.algo.utils.mmengine_utils import InstanceData

HeadForwardFn = Callable[[FeatureMapType], torch.Tensor]
ExplainerForwardFn = HeadForwardFn


def feature_vector_fn(feature_map: FeatureMapType) -> torch.Tensor:
    """Generate the feature vector by average pooling feature maps."""
    if isinstance(feature_map, (list, tuple)):
        # aggregate feature maps from Feature Pyramid Network
        feature_vector = [
            # Spatially pooling and flatten, B x C x H x W => B x C'
            torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(start_dim=1)
            for f in feature_map
        ]
        return torch.cat(feature_vector, 1)

    return torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(start_dim=1)


class BaseExplainAlgo:
    """While registered with the designated PyTorch module, this class caches feature vector during forward pass.

    Args:
        normalize (bool): Whether to normalize the resulting saliency maps.
    """

    def __init__(self, head_forward_fn: HeadForwardFn | None = None, normalize: bool = True) -> None:
        self._head_forward_fn = head_forward_fn
        self._norm_saliency_maps = normalize

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

    def _predict_from_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._head_forward_fn:
                x = self._head_forward_fn(x)
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)
        return x

    @staticmethod
    def _normalize_map(saliency_map: torch.Tensor) -> torch.Tensor:
        """Normalize saliency maps."""
        max_values, _ = torch.max(saliency_map, -1)
        min_values, _ = torch.min(saliency_map, -1)
        if len(saliency_map.shape) == 2:
            saliency_map = 255 * (saliency_map - min_values[:, None]) / (max_values - min_values + 1e-12)[:, None]
        else:
            saliency_map = 255 * (saliency_map - min_values[:, :, None]) / (max_values - min_values + 1e-12)[:, :, None]
        return saliency_map.to(torch.uint8)


class ActivationMap(BaseExplainAlgo):
    """ActivationMap. Mean of the feature map along the channel dimension."""

    def func(self, feature_map: FeatureMapType, fpn_idx: int = -1) -> torch.Tensor:
        """Generate the saliency map by average feature maps then normalizing to (0, 255)."""
        if isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[fpn_idx]

        batch_size, _, h, w = feature_map.size()
        activation_map = torch.mean(feature_map, dim=1)

        if self._norm_saliency_maps:
            activation_map = activation_map.reshape((batch_size, h * w))
            activation_map = self._normalize_map(activation_map)

        return activation_map.reshape((batch_size, h, w))


class ReciproCAM(BaseExplainAlgo):
    """Implementation of Recipro-CAM for class-wise saliency map.

    Recipro-CAM: gradient-free reciprocal class activation map (https://arxiv.org/pdf/2209.14074.pdf)
    """

    def __init__(
        self,
        head_forward_fn: HeadForwardFn,
        num_classes: int,
        normalize: bool = True,
        optimize_gap: bool = False,
    ) -> None:
        super().__init__(head_forward_fn, normalize)
        self._num_classes = num_classes
        self._optimize_gap = optimize_gap

    def func(self, feature_map: FeatureMapType, fpn_idx: int = -1) -> torch.Tensor:
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
        saliency_map = torch.empty(batch_size, self._num_classes, h, w)
        for f in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[f], channel, h, w)
            mosaic_prediction = self._predict_from_feature_map(mosaic_feature_map)
            saliency_map[f] = mosaic_prediction.transpose(0, 1).reshape((self._num_classes, h, w))

        if self._norm_saliency_maps:
            saliency_map = saliency_map.reshape((batch_size, self._num_classes, h * w))
            saliency_map = self._normalize_map(saliency_map)

        return saliency_map.reshape((batch_size, self._num_classes, h, w))

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
        if self._optimize_gap:
            # if isinstance(model_neck, GlobalAveragePooling):
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


class ViTReciproCAM(BaseExplainAlgo):
    """Implementation of ViTRecipro-CAM for class-wise saliency map for transformer-based classifiers.

    Args:
        head_forward_fn (callable): Forward pass function for the top of the model.
        num_classes (int): Number of classes.
        use_gaussian (bool): Defines kernel type for mosaic feature map generation.
        If True, use gaussian 3x3 kernel. If False, use 1x1 kernel.
        cls_token (bool): If True, includes classification token into the mosaic feature map.
        normalize (bool): If True, Normalizes saliency maps.
    """

    def __init__(
        self,
        head_forward_fn: HeadForwardFn,
        num_classes: int,
        use_gaussian: bool = True,
        cls_token: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__(head_forward_fn, normalize)
        self._num_classes = num_classes
        self._use_gaussian = use_gaussian
        self._cls_token = cls_token

    def func(self, feature_map: torch.Tensor, _: int = -1) -> torch.Tensor:
        """Generate the class-wise saliency maps using ViTRecipro-CAM and then normalizing to (0, 255).

        Args:
            feature_map (torch.Tensor): feature maps from target layernorm layer.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        batch_size, token_number, _ = feature_map.size()
        h = w = int((token_number - 1) ** 0.5)
        saliency_map = torch.empty(batch_size, self._num_classes, h, w)
        for i in range(batch_size):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[i])
            mosaic_prediction = self._predict_from_feature_map(mosaic_feature_map)
            saliency_map[i] = mosaic_prediction.transpose(1, 0).reshape((self._num_classes, h, w))

        if self._norm_saliency_maps:
            saliency_map = saliency_map.reshape((batch_size, self._num_classes, h * w))
            saliency_map = self._normalize_map(saliency_map)
        return saliency_map.reshape((batch_size, self._num_classes, h, w))

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
                [[1 / 16.0, 1 / 8.0, 1 / 16.0], [1 / 8.0, 1 / 4.0, 1 / 8.0], [1 / 16.0, 1 / 8.0, 1 / 16.0]],
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


class DetClassProbabilityMap(BaseExplainAlgo):
    """Saliency map generation algo for object detection models."""

    def __init__(
        self,
        num_classes: int,
        num_anchors: list[int],
        normalize: bool = True,
        use_cls_softmax: bool = True,
    ) -> None:
        super().__init__(head_forward_fn=None, normalize=normalize)
        # SSD-like heads also have background class
        self._num_classes = num_classes
        self._num_anchors = num_anchors
        # Should be switched off for tiling
        self.use_cls_softmax = use_cls_softmax

    def func(
        self,
        cls_scores: FeatureMapType,
        _: int = -1,
    ) -> torch.Tensor:
        """Generate the saliency map from raw classification head output, then normalizing to (0, 255).

        Args:
            cls_scores (FeatureMapType): Classification scores from cls_head.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        middle_idx = len(cls_scores) // 2
        # Resize to the middle feature map
        batch_size, _, height, width = cls_scores[middle_idx].size()
        saliency_map = torch.empty(batch_size, self._num_classes, height, width)
        for batch_idx in range(batch_size):
            cls_scores_anchorless = []
            for scale_idx, cls_scores_per_scale in enumerate(cls_scores):
                cls_scores_anchor_grouped = cls_scores_per_scale[batch_idx].reshape(
                    self._num_anchors[scale_idx],
                    (self._num_classes),
                    *cls_scores_per_scale.shape[-2:],
                )
                cls_scores_out, _ = cls_scores_anchor_grouped.max(dim=0)
                cls_scores_anchorless.append(cls_scores_out.unsqueeze(0))

            cls_scores_anchorless_resized = [
                torch.nn.functional.interpolate(cls_scores_anchorless_per_level, (height, width), mode="bilinear")
                for cls_scores_anchorless_per_level in cls_scores_anchorless
            ]

            saliency_map[batch_idx] = torch.cat(cls_scores_anchorless_resized, dim=0).mean(dim=0)

        # Don't use softmax for tiles in tiling detection, if the tile doesn't contain objects,
        # it would highlight one of the class maps as a background class
        if self.use_cls_softmax:
            saliency_map = torch.stack([torch.softmax(b, dim=0) for b in saliency_map])

        if self._norm_saliency_maps:
            saliency_map = saliency_map.reshape((batch_size, self._num_classes, -1))
            saliency_map = self._normalize_map(saliency_map)

        return saliency_map.reshape((batch_size, self._num_classes, height, width))


class InstSegExplainAlgo(BaseExplainAlgo):
    """Dummy saliency map algo for Mask R-CNN and RTMDetInst model.

    Predicted masks are combined and aggregated per-class to generate the saliency maps.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def func(
        self,
        predictions: list[InstanceData],
        _: int = -1,
    ) -> list[np.array]:
        """Generate saliency maps from predicted masks by averaging and normalizing them per-class.

        Args:
            predictions (list[InstanceData]): Predictions of Instance Segmentation model.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        # TODO(gzalessk): Add unit tests
        batch_saliency_maps = []
        for prediction in predictions:
            class_averaged_masks = self.average_and_normalize(prediction, self.num_classes)
            batch_saliency_maps.append(class_averaged_masks)
        return torch.stack(batch_saliency_maps)

    @classmethod
    def average_and_normalize(
        cls,
        pred: InstanceData | dict[str, torch.Tensor],
        num_classes: int,
    ) -> np.array:
        """Average and normalize masks in prediction per-class.

        Args:
            preds (InstanceData | dict): Predictions of Instance Segmentation model.
            num_classes (int): Num classes that model can predict.

        Returns:
            np.array: Class-wise Saliency Maps. One saliency map per each class - [class_id, H, W]
        """
        if isinstance(pred, dict):
            masks, scores, labels = pred["masks"], pred["scores"], pred["labels"]
        else:
            masks, scores, labels = (pred.masks, pred.scores, pred.labels)  # type: ignore[attr-defined]
        _, height, width = masks.shape

        saliency_map = torch.zeros((num_classes, height, width), dtype=torch.float32, device=labels.device)
        class_objects = [0 for _ in range(num_classes)]

        for confidence, class_ind, raw_mask in zip(scores, labels, masks):
            weighted_mask = raw_mask * confidence
            saliency_map[class_ind] += weighted_mask
            class_objects[class_ind] += 1

        for class_ind in range(num_classes):
            # Normalize by number of objects of the certain class
            saliency_map[class_ind] /= max(class_objects[class_ind], 1)

        saliency_map = saliency_map.reshape((num_classes, -1))
        saliency_map = cls._normalize_map(saliency_map)

        return saliency_map.reshape(num_classes, height, width)
