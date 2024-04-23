# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet BaseDetector."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, TypeAlias

import torch
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.model import BaseModel
from torch import Tensor

ForwardResults: TypeAlias = dict[str, torch.Tensor] | list[DetDataSample] | tuple[torch.Tensor] | torch.Tensor

if TYPE_CHECKING:
    from mmengine.config import ConfigDict
    from mmengine.structures import InstanceData


class BaseDetector(BaseModel, metaclass=ABCMeta):
    """Base class for detectors.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(
        self,
        data_preprocessor: ConfigDict | dict | None = None,
        init_cfg: ConfigDict | dict | list[ConfigDict | dict] | None = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head."""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    def forward(self, inputs: torch.Tensor, data_samples: list[DetDataSample], mode: str = "tensor") -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == "loss":
            return self.loss(inputs, data_samples)
        if mode == "predict":
            return self.predict(inputs, data_samples)
        msg = f"Invalid mode {mode}. Only supports loss and predict mode."
        raise RuntimeError(msg)

    @abstractmethod
    def loss(self, batch_inputs: Tensor, batch_data_samples: list[DetDataSample]) -> dict | tuple:
        """Calculate losses from a batch of inputs and data samples."""

    @abstractmethod
    def predict(self, batch_inputs: Tensor, batch_data_samples: list[DetDataSample]) -> list[DetDataSample]:
        """Predict results from a batch of inputs and data samples with post-processing."""

    @abstractmethod
    def _forward(self, batch_inputs: Tensor, batch_data_samples: list[DetDataSample]) -> tuple:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor) -> tuple:
        """Extract features from images."""

    def add_pred_to_datasample(
        self,
        data_samples: list[DetDataSample],
        results_list: list[InstanceData],
    ) -> list[DetDataSample]:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

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
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        return data_samples
