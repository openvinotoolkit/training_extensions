# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

# mypy: disable-error-code="attr-defined"

"""Implementation of action data sample."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from otx.algo.utils.mmengine_utils import InstanceData

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence, dict]


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The formatted label tensor.
    """
    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        msg = f"Type {type(value)} is not an available label type."
        raise TypeError(msg)

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor | dict:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | dict):
            Score values or dict of scores values.

    Returns:
        :obj:`torch.Tensor` | dict: The formatted scores.
    """
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value).float()
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = format_score(v)
    elif not isinstance(value, torch.Tensor):
        msg = f"Type {type(value)} is not an available label type."
        raise TypeError(msg)

    return value


class ActionDataSample(InstanceData):
    """A data interface for action data that supports Tensor-like and dict-like operations."""

    def set_gt_label(self, value: LABEL_TYPE) -> ActionDataSample:
        """Set `gt_label``."""
        self.set_field(format_label(value), "gt_label", dtype=torch.Tensor)
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> ActionDataSample:
        """Set ``pred_label``."""
        self.set_field(format_label(value), "pred_label", dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE) -> ActionDataSample:
        """Set score of ``pred_label``."""
        score = format_score(value)
        self.set_field(score, "pred_score")
        if hasattr(self, "num_classes"):
            assert (  # noqa: S101
                len(score) == self.num_classes
            ), f"The length of score {len(score)} should be equal to the num_classes {self.num_classes}."
        else:
            self.set_field(name="num_classes", value=len(score), field_type="metainfo")
        return self

    @property
    def proposals(self) -> InstanceData:
        """Property of `proposals`."""
        return self._proposals

    @proposals.setter
    def proposals(self, value) -> None:  # noqa: ANN001
        """Setter of `proposals`."""
        self.set_field(value, "_proposals", dtype=InstanceData)

    @proposals.deleter
    def proposals(self) -> None:
        """Deleter of `proposals`."""
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        """Property of `gt_instances`."""
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value) -> None:  # noqa: ANN001
        """Setter of `gt_instances`."""
        self.set_field(value, "_gt_instances", dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self) -> None:
        """Deleter of `gt_instances`."""
        del self._gt_instances

    @property
    def features(self) -> InstanceData:
        """Setter of `features`."""
        return self._features

    @features.setter
    def features(self, value) -> None:  # noqa: ANN001
        """Setter of `features`."""
        self.set_field(value, "_features", dtype=InstanceData)

    @features.deleter
    def features(self) -> None:
        """Deleter of `features`."""
        del self._features
