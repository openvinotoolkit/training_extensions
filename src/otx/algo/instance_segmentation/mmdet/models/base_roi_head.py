# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet BaseRoIHead."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from otx.algo.modules.base_module import BaseModule

if TYPE_CHECKING:
    from mmdet.structures import DetDataSample
    from mmengine import ConfigDict
    from mmengine.structures import InstanceData
    from torch import Tensor, nn


class BaseRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(
        self,
        bbox_roi_extractor: nn.Module,
        bbox_head: nn.Module,
        mask_roi_extractor: nn.Module,
        mask_head: nn.Module,
        train_cfg: ConfigDict | dict,
        test_cfg: ConfigDict | dict,
        init_cfg: ConfigDict | dict | list[ConfigDict | dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head

        self.mask_roi_extractor = mask_roi_extractor
        self.mask_head = mask_head

        self.init_assigner_sampler()

    @property
    def with_bbox(self) -> bool:
        """bool: whether the RoI head contains a `bbox_head`."""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask(self) -> bool:
        """bool: whether the RoI head contains a `mask_head`."""
        return hasattr(self, "mask_head") and self.mask_head is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the RoI head contains a `shared_head`."""
        return hasattr(self, "shared_head") and self.shared_head is not None

    @abstractmethod
    def init_assigner_sampler(self, *args, **kwargs) -> None:
        """Initialize assigner and sampler."""

    @abstractmethod
    def loss(
        self,
        x: tuple[Tensor],
        rpn_results_list: list[InstanceData],
        batch_data_samples: list[DetDataSample],
    ) -> dict:
        """Perform forward propagation and loss calculation of the roi head on the features of the upstream network."""

    def predict(
        self,
        x: tuple[Tensor],
        rpn_results_list: list[InstanceData],
        batch_data_samples: list[DetDataSample],
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Forward the roi head and predict detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if not self.with_bbox:
            msg = "Bbox head must be implemented."
            raise NotImplementedError(msg)
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale,
        )

        if self.with_mask:
            results_list = self.predict_mask(x, batch_img_metas, results_list, rescale=rescale)

        return results_list
