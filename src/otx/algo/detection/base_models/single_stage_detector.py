# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""SSD object detector for the OTX detection.

Implementation modified from mmdet.models.detectors.single_stage.
Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/detectors/single_stage.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.detection import DetBatchDataEntity

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn


class SingleStageDetector(BaseModule):
    """Single stage detector implementation.

    Args:
        backbone (nn.Module): Backbone module.
        bbox_head (nn.Module): Bbox head module.
        neck (nn.Module | None, optional): Neck module. Defaults to None.
        train_cfg (dict | None, optional): Training config. Defaults to None.
        test_cfg (dict | None, optional): Test config. Defaults to None.
        init_cfg (dict | list[dict], optional): Initialization config. Defaults to None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        bbox_head: nn.Module,
        neck: nn.Module | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__()
        self._is_init = False
        self.backbone = backbone
        self.bbox_head = bbox_head
        self.neck = neck
        self.init_cfg = init_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str] | str,
        unexpected_keys: list[str] | str,
        error_msgs: list[str] | str,
    ) -> None:
        """Exchange bbox_head key to rpn_head key.

        When loading two-stage weights into single-stage model.
        """
        bbox_head_prefix = prefix + ".bbox_head" if prefix else "bbox_head"
        bbox_head_keys = [k for k in state_dict if k.startswith(bbox_head_prefix)]
        rpn_head_prefix = prefix + ".rpn_head" if prefix else "rpn_head"
        rpn_head_keys = [k for k in state_dict if k.startswith(rpn_head_prefix)]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + rpn_head_key[len(rpn_head_prefix) :]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        entity: DetBatchDataEntity,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor] | list[InstanceData] | tuple[torch.Tensor] | torch.Tensor:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`InstanceData`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`InstanceData`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`InstanceData`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == "loss":
            return self.loss(entity)
        if mode == "predict":
            return self.predict(entity)
        if mode == "tensor":
            return self._forward(entity)

        msg = f"Invalid mode {mode}. Only supports loss, predict and tensor mode"
        raise RuntimeError(msg)

    def loss(
        self,
        entity: DetBatchDataEntity,
    ) -> dict | list:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`InstanceData`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(entity.images)
        return self.bbox_head.loss(x, entity)

    def predict(
        self,
        entity: DetBatchDataEntity,
        rescale: bool = True,
    ) -> list[InstanceData]:
        """Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`InstanceData`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Detection results of the
            input images. Each InstanceData usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(entity.images)
        return self.bbox_head.predict(x, entity, rescale=rescale)

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
        rescale: bool = True,
        explain_mode: bool = False,
    ) -> list[InstanceData] | dict:
        """Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`InstanceData`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Detection results of the
            input images. Each InstanceData usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if explain_mode:
            backbone_feat = self.extract_feat(batch_inputs)
            bbox_head_feat = self.bbox_head.forward(backbone_feat)
            feature_vector = self.feature_vector_fn(backbone_feat)
            saliency_map = self.explain_fn(bbox_head_feat[0])
            bboxes, labels = self.bbox_head.export(backbone_feat, batch_img_metas, rescale=rescale)
            return {
                "bboxes": bboxes,
                "labels": labels,
                "feature_vector": feature_vector,
                "saliency_map": saliency_map,
            }
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.export(x, batch_img_metas, rescale=rescale)

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: list[InstanceData] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Network forward process.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`InstanceData`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.forward(x)

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head."""
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head."""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head."""
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )
