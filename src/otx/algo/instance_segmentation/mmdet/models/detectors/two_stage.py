# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet TwoStageDetector."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity

from .base import BaseDetector

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        rpn_head: nn.Module,
        roi_head: nn.Module,
        train_cfg: DictConfig | dict,
        test_cfg: DictConfig | dict,
        init_cfg: DictConfig | dict | list[DictConfig | dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head

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
        """Exchange bbox_head key to rpn_head key when loading single-stage weights into two-stage model."""
        bbox_head_prefix = prefix + ".bbox_head" if prefix else "bbox_head"
        bbox_head_keys = [k for k in state_dict if k.startswith(bbox_head_prefix)]
        rpn_head_prefix = prefix + ".rpn_head" if prefix else "rpn_head"
        rpn_head_keys = [k for k in state_dict if k.startswith(rpn_head_prefix)]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + bbox_head_key[len(bbox_head_prefix) :]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN."""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs: InstanceSegBatchDataEntity) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs.images)

        losses = {}

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg["rpn"])

        # Copy data entity and set gt_labels to 0 in RPN
        rpn_entity = InstanceSegBatchDataEntity(
            images=torch.empty(0),
            batch_size=batch_inputs.batch_size,
            imgs_info=batch_inputs.imgs_info,
            bboxes=batch_inputs.bboxes,
            masks=batch_inputs.masks,
            labels=[torch.zeros_like(labels) for labels in batch_inputs.labels],
            polygons=batch_inputs.polygons,
        )

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x,
            rpn_entity,
            proposal_cfg=proposal_cfg,
        )
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if "loss" in key and "rpn" not in key:
                rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_inputs)
        losses.update(roi_losses)

        return losses

    def predict(
        self,
        entity,
        rescale: bool = True,
    ) -> list[InstanceData]:
        """Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if not self.with_bbox:
            msg = "Bbox head is not implemented."
            raise NotImplementedError(msg)
        x = self.extract_feat(entity.images)

        rpn_results_list = self.rpn_head.predict(x, entity, rescale=False)

        return self.roi_head.predict(x, rpn_results_list, entity, rescale=rescale)
