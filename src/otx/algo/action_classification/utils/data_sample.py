# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Implementation of action data sample."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import copy
from typing import Sequence, Union

import numpy as np
import torch
from otx.algo.utils.mmengine_utils import InstanceData

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence, dict]


def is_str(x) -> bool:  # noqa:ANN001
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


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
    elif isinstance(value, Sequence) and not is_str(value):
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
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = format_score(v)
    elif not isinstance(value, torch.Tensor):
        msg = f"Type {type(value)} is not an available label type."
        raise TypeError(msg)

    return value


class ActionDataSample:
    """A base data interface that supports Tensor-like and dict-like operations."""

    def __init__(self, *, metainfo: dict | None = None, **kwargs) -> None:
        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(metainfo, dict), f"metainfo should be a ``dict`` but got {type(metainfo)}"  # noqa: S101
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type="metainfo", dtype=None)

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data, dict), f"data should be a `dict` but got {data}"  # noqa: S101
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def set_field(
        self,
        value,  # noqa: ANN001
        name: str,
        dtype: type | tuple[type, ...] | None = None,
        field_type: str = "data",
    ) -> None:
        """Special method for set union field, used as property.setter functions."""
        assert field_type in ["metainfo", "data"]  # noqa: S101
        if dtype is not None:
            assert isinstance(value, dtype), f"{value} should be a {dtype} but got {type(value)}"  # noqa: S101

        if field_type == "metainfo":
            if name in self._data_fields:
                msg = f"Cannot set {name} to be a field of metainfo because {name} is already a data field"
                raise AttributeError(msg)
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                msg = f"Cannot set {name} to be a field of data because {name} is already a metainfo field"
                raise AttributeError(msg)
            self._data_fields.add(name)
        super().__setattr__(name, value)

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
