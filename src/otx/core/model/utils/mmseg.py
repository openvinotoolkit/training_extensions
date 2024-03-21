# Copyright (C) 2024 Intel Corporation
# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Please note that parts of source code here are
# borrowed from https://github.com/open-mmlab/mmsegmentation
#
"""Utility files for the mmseg package."""
# ruff: noqa
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmseg.models.data_preprocessor import SegDataPreProcessor as _SegDataPreProcessor
from mmseg.registry import MODELS

from otx.core.utils.build import build_mm_model, get_classification_layers

if TYPE_CHECKING:
    from mmseg.utils.typing_utils import SampleList
    from omegaconf import DictConfig
    from torch import device, nn


def stack_batch(
    inputs: list[torch.Tensor] | torch.Tensor,
    data_samples: SampleList | None = None,
    size: tuple | None = None,
    size_divisor: int | None = None,
    pad_val: int | float = 0,
    seg_pad_val: int | float = 255,
) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor] | torch.Tensor): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list) or (
        isinstance(inputs, torch.Tensor) and inputs.ndim == 4
    ), f"Expected input type to be a list or 4D torch.Tensor, but got {type(inputs)}"
    assert (
        len({tensor.ndim for tensor in inputs}) == 1
    ), f"Expected the dimensions of all inputs must be the same, but got {[tensor.ndim for tensor in inputs]}"
    assert inputs[0].ndim == 3, f"Expected tensor dimension to be 3, but got {inputs[0].ndim}"
    assert (
        len({tensor.shape[0] for tensor in inputs}) == 1
    ), f"Expected the channels of all inputs must be the same, but got {[tensor.shape[0] for tensor in inputs]}"

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), "only one of size and size_divisor should be valid"

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = (0, 0, 0, 0)

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None
            if "gt_sem_seg" in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(gt_sem_seg, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_sem_seg.shape
            if "gt_edge_map" in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(gt_edge_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_edge_map.shape
            if "gt_depth_map" in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(gt_depth_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_depth_map.shape
            data_sample.set_metainfo(
                {"img_shape": tensor.shape[-2:], "pad_shape": pad_shape, "padding_size": padding_size},
            )
            padded_samples.append(data_sample)
        else:
            padded_samples.append(dict(img_padding_size=padding_size, pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples


# NOTE: For the history of this monkey patching, please see
# https://github.com/openvinotoolkit/training_extensions/issues/2743
@MMENGINE_MODELS.register_module(force=True)
class SegDataPreProcessor(_SegDataPreProcessor):
    """Class for monkey patching preprocessor.

    NOTE: For the history of this monkey patching, please see
    https://github.com/openvinotoolkit/training_extensions/issues/2743
    """

    @property
    def device(self) -> device:
        """Which device being used."""
        try:
            buf = next(self.buffers())
        except StopIteration:
            return super().device
        else:
            return buf.device

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data["inputs"]
        data_samples = data.get("data_samples", None)

        # NOTE: We slightly changed this code for small optimization.
        # The original code flow is
        # 1) Change image tensor's dtype from uint8 to float32 and normalize => 2) Pad and stack images and masks
        # However, we reverse it as 2) => 1) to perform pad operation on low bits (uint8) rather than high bits (float32)

        # Pad and stack images and masks
        if training:
            assert data_samples is not None, ("During training, ", "`data_samples` must be define.")
            # Call stack_batch only if the image is not 4D tensor and there is a mismatch in the ground truth masks
            if not (
                isinstance(inputs, torch.Tensor)
                and inputs.ndim == 4
                and all("gt_sem_seg" in data_sample for data_sample in data_samples)
                and len(set(data_sample.gt_sem_seg.shape for data_sample in data_samples)) == 1
            ):
                inputs, data_samples = stack_batch(
                    inputs=inputs,
                    data_samples=data_samples,
                    size=self.size,
                    size_divisor=self.size_divisor,
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val,
                )

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(
                input_.shape[1:] == img_size for input_ in inputs
            ), "The image size in a batch should be the same."
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get("size", None),
                    size_divisor=self.test_cfg.get("size_divisor", None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val,
                )
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        # Change image tensor's dtype from uint8 to float32 and normalize
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = (
                inputs[:, [2, 1, 0], ...]
                if isinstance(inputs, torch.Tensor)
                else [_input[[2, 1, 0], ...] for _input in inputs]
            )

        if self._enable_normalize:
            inputs = (
                (inputs.float() - self.mean) / self.std
                if isinstance(inputs, torch.Tensor)
                else [(_input.float() - self.mean) / self.std for _input in inputs]
            )

        return dict(inputs=inputs, data_samples=data_samples)


def create_model(config: DictConfig, load_from: str | None) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    """Create a model from mmseg Model registry.

    Args:
        config (DictConfig): Model configuration.
        load_from (str | None): Model weight file path.

    Returns:
        tuple[nn.Module, dict[str, dict[str, int]]]: Model instance and classification layers.
    """
    classification_layers = get_classification_layers(config, MODELS, "model.")
    return build_mm_model(config, MODELS, load_from), classification_layers
