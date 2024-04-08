"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# TODO(Eugene): Revisit mypy errors after deprecation of mmlab
# https://github.com/openvinotoolkit/training_extensions/pull/3281
# mypy: ignore-errors
# ruff: noqa

# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Optional

from mmengine.structures import InstanceData


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: Optional[InstanceData] = None,
        **kwargs,
    ):
        """Assign boxes to either a ground truth boxes or a negative boxes."""
