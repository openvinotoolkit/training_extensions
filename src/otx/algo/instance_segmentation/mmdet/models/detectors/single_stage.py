"""Single stage detector model for instance segmentation."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseDetector

if TYPE_CHECKING:
    from mmdet.structures.det_data_sample import DetDataSample
    from torch import Tensor, nn


class InstSegSingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        bbox_head: nn.Module,
        data_preprocessor: nn.Module,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

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
        """Exchange bbox_head key to rpn_head key when loading two-stage weights into single-stage model."""
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

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: list[DetDataSample],
    ) -> dict | tuple:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(x, batch_data_samples)

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: list[DetDataSample],
        rescale: bool = True,
    ) -> list[DetDataSample]:
        """Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: list[DetDataSample] | None = None,
    ) -> tuple[list[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
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
        if self.with_neck:
            x = self.neck(x)
        return x
