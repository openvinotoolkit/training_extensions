# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet TwoStageDetector."""
from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from otx.algo.detection.backbones.pytorchcv_backbones import _build_pytorchcv_model
from otx.algo.instance_segmentation.mmdet.models.backbones import ResNet, SwinTransformer
from otx.algo.instance_segmentation.mmdet.models.custom_roi_head import CustomRoIHead
from otx.algo.instance_segmentation.mmdet.models.dense_heads import RPNHead
from otx.algo.instance_segmentation.mmdet.models.necks import FPN

from .base import BaseDetector

if TYPE_CHECKING:
    from mmdet.structures.det_data_sample import DetDataSample
    from mmengine.config import ConfigDict


class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(
        self,
        backbone: ConfigDict | dict,
        neck: ConfigDict | dict,
        rpn_head: ConfigDict | dict,
        roi_head: ConfigDict | dict,
        train_cfg: ConfigDict | dict,
        test_cfg: ConfigDict | dict,
        data_preprocessor: ConfigDict | dict | None = None,
        init_cfg: ConfigDict | dict | list[ConfigDict | dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        backbone_type = backbone.pop("type")
        if backbone_type == ResNet.__name__:
            self.backbone = ResNet(**backbone)
        elif backbone_type == SwinTransformer.__name__:
            self.backbone = SwinTransformer(**backbone)
        else:
            self.backbone = _build_pytorchcv_model(backbone_type, **backbone)

        if neck["type"] != FPN.__name__:
            msg = f"neck type must be {FPN.__name__}, but got {neck['type']}"
            raise ValueError(msg)
        # pop out type for FPN
        neck.pop("type")
        self.neck = FPN(**neck)

        rpn_train_cfg = train_cfg["rpn"]
        rpn_head_ = rpn_head.copy()
        rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg["rpn"])
        rpn_head_num_classes = rpn_head_.get("num_classes", None)
        if rpn_head_num_classes is None:
            rpn_head_.update(num_classes=1)
        elif rpn_head_num_classes != 1:
            warnings.warn(
                "The `num_classes` should be 1 in RPN, but get "
                f"{rpn_head_num_classes}, please set "
                "rpn_head.num_classes = 1 in your config file.",
                stacklevel=2,
            )
            rpn_head_.update(num_classes=1)
        if rpn_head_["type"] != RPNHead.__name__:
            msg = f"rpn_head type must be {RPNHead.__name__}, but got {rpn_head_['type']}"
            raise ValueError(msg)
        # pop out type for RPNHead
        rpn_head_.pop("type")
        self.rpn_head = RPNHead(**rpn_head_)

        # update train and test cfg here for now
        rcnn_train_cfg = train_cfg["rcnn"]
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg["rcnn"])
        if roi_head["type"] != CustomRoIHead.__name__:
            msg = f"roi_head type must be {CustomRoIHead.__name__}, but got {roi_head['type']}"
            raise ValueError(msg)
        # pop out type for RoIHead
        roi_head.pop("type")
        self.roi_head = CustomRoIHead(**roi_head)

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

    def _forward(self, batch_inputs: Tensor, batch_data_samples: list[DetDataSample]) -> tuple:
        """Network forward process. Usually includes backbone, neck and head forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            if batch_data_samples[0].get("proposals", None) is None:
                msg = "No 'proposals' in data samples."
                raise ValueError(msg)
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]
        roi_outs = self.roi_head.forward(x, rpn_results_list, batch_data_samples)
        return (*results, roi_outs)

    def loss(self, batch_inputs: Tensor, batch_data_samples: list[DetDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = {}

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg["rpn"])
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x,
                rpn_data_samples,
                proposal_cfg=proposal_cfg,
            )
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if "loss" in key and "rpn" not in key:
                    rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            if batch_data_samples[0].get("proposals", None) is None:
                msg = "No 'proposals' in data samples."
                raise ValueError(msg)
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses

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
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get("proposals", None) is None:
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

        results_list = self.roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=rescale)

        return self.add_pred_to_datasample(batch_data_samples, results_list)
