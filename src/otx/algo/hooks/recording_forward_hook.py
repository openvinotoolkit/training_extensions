# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Hooks for recording/updating model internal activations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


class BaseRecordingForwardHook:
    """While registered with the designated PyTorch module, this class caches feature vector during forward pass.

    Args:
        normalize (bool): Whether to normalize the resulting saliency maps.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.handle: RemovableHandle | None = None
        self._records: list[torch.Tensor] = []
        self._norm_saliency_maps = normalize

    @property
    def records(self) -> list[torch.Tensor]:
        """Return records."""
        return self._records

    def reset(self) -> None:
        """Clear all history of records."""
        self._records.clear()

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

    def recording_forward(
        self,
        _: torch.nn.Module,
        x: torch.Tensor,
        output: torch.Tensor,
    ) -> None:  # pylint: disable=unused-argument
        """Record the XAI result during executing model forward function."""
        tensors = self.func(output)
        if isinstance(tensors, torch.Tensor):
            tensors_np = tensors.detach().cpu().numpy()
        elif isinstance(tensors, np.ndarray):
            tensors_np = tensors
        else:
            self._torch_to_numpy_from_list(tensors)
            tensors_np = tensors

        for tensor in tensors_np:
            self._records.append(tensor)

    def _torch_to_numpy_from_list(self, tensor_list: list[torch.Tensor | None]) -> None:
        for i in range(len(tensor_list)):
            tensor = tensor_list[i]
            if isinstance(tensor, list):
                self._torch_to_numpy_from_list(tensor)
            elif isinstance(tensor, torch.Tensor):
                tensor_list[i] = tensor.detach().cpu().numpy()

    @staticmethod
    def _normalize_map(saliency_maps: torch.Tensor) -> torch.Tensor:
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


class ReciproCAMHook(BaseRecordingForwardHook):
    """Implementation of Recipro-CAM for class-wise saliency map.

    Recipro-CAM: gradient-free reciprocal class activation map (https://arxiv.org/pdf/2209.14074.pdf)
    """

    def __init__(
        self,
        head_forward_fn: Callable,
        num_classes: int,
        normalize: bool = True,
        optimize_gap: bool = False,
    ) -> None:
        super().__init__(normalize)
        self._head_forward_fn = head_forward_fn
        self._num_classes = num_classes
        self._optimize_gap = optimize_gap

    @classmethod
    def create_and_register_hook(
        cls,
        backbone: torch.nn.Module,
        head_forward_fn: Callable,
        num_classes: int,
        optimize_gap: bool,
    ) -> BaseRecordingForwardHook:
        """Create this object and register it to the module forward hook."""
        hook = cls(
            head_forward_fn,
            num_classes=num_classes,
            optimize_gap=optimize_gap,
        )
        hook.handle = backbone.register_forward_hook(hook.recording_forward)
        return hook

    def func(self, feature_map: torch.Tensor | Sequence[torch.Tensor], fpn_idx: int = -1) -> torch.Tensor:
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

        return saliency_maps.reshape((batch_size, self._num_classes, h, w))

    def _predict_from_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self._head_forward_fn(x)
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits)
        return logits

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


class DetClassProbabilityMapHook(BaseRecordingForwardHook):
    """Saliency map hook for object detection models."""

    def __init__(
        self,
        cls_head_forward_fn: Callable,
        num_classes: int,
        num_anchors: list[int],
        normalize: bool = True,
        use_cls_softmax: bool = True,
    ) -> None:
        super().__init__(normalize)
        self._cls_head_forward_fn = cls_head_forward_fn
        # SSD-like heads also have background class
        self._num_classes = num_classes
        self._num_anchors = num_anchors
        # Should be switched off for tiling
        self.use_cls_softmax = use_cls_softmax

    @classmethod
    def create_and_register_hook(
        cls,
        backbone: torch.nn.Module,
        cls_head_forward_fn: Callable,
        num_classes: int,
        num_anchors: list[int],
    ) -> BaseRecordingForwardHook:
        """Create this object and register it to the module forward hook."""
        hook = cls(
            cls_head_forward_fn,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        hook.handle = backbone.register_forward_hook(hook.recording_forward)
        return hook

    def func(
        self,
        feature_map: torch.Tensor | Sequence[torch.Tensor],
        _: int = -1,
    ) -> torch.Tensor:
        """Generate the saliency map from raw classification head output, then normalizing to (0, 255).

        Args:
            feature_map (Union[torch.Tensor, List[torch.Tensor]]): Feature maps from backbone/FPN or
            classification scores from cls_head.

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        cls_scores = self._cls_head_forward_fn(feature_map)

        middle_idx = len(cls_scores) // 2
        # resize to the middle feature map
        batch_size, _, height, width = cls_scores[middle_idx].size()
        saliency_maps = torch.empty(batch_size, self._num_classes, height, width)
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

            saliency_maps[batch_idx] = torch.cat(cls_scores_anchorless_resized, dim=0).mean(dim=0)

        # Don't use softmax for tiles in tiling detection, if the tile doesn't contain objects,
        # it would highlight one of the class maps as a background class
        if self.use_cls_softmax:
            saliency_maps[0] = torch.stack([torch.softmax(t, dim=1) for t in saliency_maps[0]])

        if self._norm_saliency_maps:
            saliency_maps = saliency_maps.reshape((batch_size, self._num_classes, -1))
            saliency_maps = self._normalize_map(saliency_maps)

        return saliency_maps.reshape((batch_size, self._num_classes, height, width))


class MaskRCNNRecordingForwardHook(BaseRecordingForwardHook):
    """Saliency map hook for Mask R-CNN model."""

    # def __init__(
    #     self,
    #     module: torch.nn.Module,
    #     input_img_shape: Tuple[int, int],
    #     saliency_map_shape: Tuple[int, int] = (224, 224),
    #     max_detections_per_img: int = 300,
    #     normalize: bool = True,
    # ) -> None:
    #     super().__init__(module)
    #     self._neck = module.neck if module.with_neck else None
    #     self._input_img_shape = input_img_shape
    #     self._saliency_map_shape = saliency_map_shape
    #     self._max_detections_per_img = max_detections_per_img
    #     self._norm_saliency_maps = normalize


    def __init__(
        self,
        cls_head_forward_fn: Callable,
        num_classes: int,
        # num_anchors: list[int],
        normalize: bool = True,
        use_cls_softmax: bool = True,
        input_img_shape: Tuple[int, int]  = (224, 224),
        saliency_map_shape: Tuple[int, int] = (224, 224),
        max_detections_per_img: int = 300,
    ) -> None:
        super().__init__(normalize)
        self._cls_head_forward_fn = cls_head_forward_fn
        # SSD-like heads also have background class
        self._num_classes = num_classes
        # self._num_anchors = num_anchors
        # Should be switched off for tiling
        self.use_cls_softmax = use_cls_softmax


        input_img_shape = input_img_shape
        saliency_map_shape = saliency_map_shape
        max_detections_per_img = max_detections_per_img

    @classmethod
    def create_and_register_hook(
        cls,
        backbone: torch.nn.Module,
        head_forward_fn: Callable,
        num_classes: int,
        optimize_gap: bool,
    ) -> BaseRecordingForwardHook:
        """Create this object and register it to the module forward hook."""
        hook = cls(
            head_forward_fn,
            num_classes=num_classes,
            optimize_gap=optimize_gap,
        )
        hook.handle = backbone.register_forward_hook(hook.recording_forward)
        return hook

    def func(
        self,
        feature_map: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        _: int = -1,
    ) -> List[List[Optional[np.ndarray]]]:
        """Generate saliency maps by aggregating per-class soft predictions of mask head for all detected boxes.

        :param feature_map: Feature maps from backbone.
        :return: Class-wise Saliency Maps. One saliency map per each predicted class.
        """

        feature_map = self.cls_head_forward_fn(feature_map)
        # with torch.no_grad():
        #     if self._neck is not None:
        #         feature_map = self._module.neck(feature_map)

        #     det_bboxes, det_labels = self._get_detections(feature_map)
        #     saliency_maps = self._get_saliency_maps_from_mask_predictions(feature_map, det_bboxes, det_labels)
        #     if self._norm_saliency_maps:
        #         saliency_maps = self._normalize(saliency_maps)
        return feature_map

    # def _get_detections(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    #     batch_size = x[0].shape[0]
    #     img_metas = [
    #         {
    #             "scale_factor": [1, 1, 1, 1],  # dummy scale_factor, not used
    #             "img_shape": self._input_img_shape,
    #         }
    #     ]
    #     img_metas *= batch_size
    #     proposals = self._module.rpn_head.simple_test_rpn(x, img_metas)
    #     test_cfg = copy.deepcopy(self._module.roi_head.test_cfg)
    #     test_cfg["max_per_img"] = self._max_detections_per_img
    #     test_cfg["nms"]["iou_threshold"] = 1
    #     test_cfg["nms"]["max_num"] = self._max_detections_per_img
    #     det_bboxes, det_labels = self._module.roi_head.simple_test_bboxes(
    #         x, img_metas, proposals, test_cfg, rescale=False
    #     )
    #     return det_bboxes, det_labels

    # def _get_saliency_maps_from_mask_predictions(
    #     self, x: torch.Tensor, det_bboxes: List[torch.Tensor], det_labels: List[torch.Tensor]
    # ) -> List[List[Optional[np.ndarray]]]:
    #     _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
    #     mask_rois = bbox2roi(_bboxes)
    #     mask_results = self._module.roi_head._mask_forward(x, mask_rois)
    #     mask_pred = mask_results["mask_pred"]
    #     num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
    #     mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

    #     batch_size = x[0].shape[0]

    #     scale_x = self._input_img_shape[1] / self._saliency_map_shape[1]
    #     scale_y = self._input_img_shape[0] / self._saliency_map_shape[0]
    #     scale_factor = torch.FloatTensor((scale_x, scale_y, scale_x, scale_y))
    #     test_cfg = self._module.roi_head.test_cfg.copy()
    #     test_cfg["mask_thr_binary"] = -1

    #     saliency_maps = [[None for _ in range(self._module.roi_head.mask_head.num_classes)] for _ in range(batch_size)]

    #     for i in range(batch_size):
    #         if det_bboxes[i].shape[0] == 0:
    #             continue
    #         else:
    #             segm_result = self._module.roi_head.mask_head.get_seg_masks(
    #                 mask_preds[i],
    #                 _bboxes[i],
    #                 det_labels[i],
    #                 test_cfg,
    #                 self._saliency_map_shape,
    #                 scale_factor=scale_factor,
    #                 rescale=True,
    #             )
    #             for class_id, segm_res in enumerate(segm_result):
    #                 if segm_res:
    #                     saliency_maps[i][class_id] = np.mean(np.array(segm_res), axis=0)
    #     return saliency_maps

    # @staticmethod
    # def _normalize(saliency_maps: List[List[Optional[np.ndarray]]]) -> List[List[Optional[np.ndarray]]]:
    #     batch_size = len(saliency_maps)
    #     num_classes = len(saliency_maps[0])
    #     for i in range(batch_size):
    #         for class_id in range(num_classes):
    #             per_class_map = saliency_maps[i][class_id]
    #             if per_class_map is not None:
    #                 max_values = np.max(per_class_map)
    #                 per_class_map = 255 * (per_class_map) / (max_values + 1e-12)
    #                 per_class_map = per_class_map.astype(np.uint8)
    #                 saliency_maps[i][class_id] = per_class_map
    #     return saliency_maps
