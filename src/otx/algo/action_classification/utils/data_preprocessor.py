# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Implementation of action data preprocessor."""

# mypy: disable-error-code="arg-type,assignment,override,return-value"

from __future__ import annotations

from typing import Mapping, Sequence, Union

import torch
import torch.nn.functional as F  # noqa: N812
from otx.algo.action_classification.utils.data_sample import ActionDataSample
from torch import device, nn

CastData = Union[tuple, dict, ActionDataSample, torch.Tensor, list, bytes, str, None]


def stack_batch(
    tensor_list: list[torch.Tensor],
    pad_size_divisor: int = 1,
    pad_value: int | float = 0,
) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max shape.

    Use the right bottom padding mode in these images. If ``pad_size_divisor > 0``,
    add padding to ensure the shape of each dim is divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(tensor_list, list), f"Expected input type to be list, but got {type(tensor_list)}"  # noqa: S101
    assert tensor_list, "`tensor_list` could not be an empty list"  # noqa: S101
    assert len({tensor.ndim for tensor in tensor_list}) == 1, (  # noqa: S101
        f"Expected the dimensions of all tensors must be the same, "
        f"but got {[tensor.ndim for tensor in tensor_list]}"
    )

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


def is_seq_of(seq: list, expected_type: type | tuple, seq_type: type = None) -> bool:  # noqa: RUF013
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = list
    else:
        assert isinstance(seq_type, type)  # noqa: S101
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    if all(isinstance(item, expected_type) for item in seq):
        return True
    return False


class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for copying data to the target device.

    Subclasses inherit from ``BaseDataPreprocessor`` could override the
    forward method to implement custom data pre-processing, such as
    batch-resize, MixUp, or CutMix.

    Args:
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version 0.3.0.

    Note:
        Data dictionary returned by dataloader must be a dict and at least
        contain the ``inputs`` key.
    """

    def __init__(self, non_blocking: bool | None = False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device("cpu")

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        if isinstance(data, (str, bytes)) or data is None:
            return data
        if isinstance(data, tuple) and hasattr(data, "_fields"):
            return type(data)(*(self.cast_data(sample) for sample in data))
        if isinstance(data, list):
            return type(data)(self.cast_data(sample) for sample in data)
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=self._non_blocking)
        return data

    def forward(self, data: dict, training: bool = False) -> dict | list:
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`cast_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        return self.cast_data(data)

    @property
    def device(self) -> device:
        """Return device info."""
        return self._device

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`.

        Returns:
            nn.Module: The model itself.
        """
        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        if args and isinstance(args[0], str) and "npu" in args[0]:
            args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])  # noqa: RUF015, C409
        if kwargs and "npu" in str(kwargs.get("device", "")):
            kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]  # noqa: SLF001
        if device is not None:
            self._device = torch.device(device)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`.

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        return super().cuda()

    def musa(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`.

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.musa.current_device())
        return super().musa()

    def npu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`.

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.npu.current_device())
        return super().npu()

    def mlu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`.

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.mlu.current_device())
        return super().mlu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`.

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device("cpu")
        return super().cpu()


class ActionDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for action recognition tasks.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        to_float32 (bool): Whether to convert data to float32.
            Defaults to True.
        blending (dict, optional): Config for batch blending.
            Defaults to None.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(
        self,
        mean: Sequence[float | int] | None = None,
        std: Sequence[float | int] | None = None,
        to_rgb: bool = False,
        to_float32: bool = True,
        blending: dict | None = None,
        format_shape: str = "NCHW",
    ) -> None:
        super().__init__()
        self.to_rgb = to_rgb
        self.to_float32 = to_float32
        self.format_shape = format_shape

        if mean is not None:
            assert (  # noqa: S101
                std is not None
            ), "To enable the normalization in preprocessing, please specify both `mean` and `std`."
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == "NCHW":
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape in ["NCTHW", "MIX2d3d"]:
                normalizer_shape = (-1, 1, 1, 1)
            else:
                msg = f"Invalid format shape: {format_shape}"
                raise ValueError(msg)

            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(normalizer_shape), False)
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(normalizer_shape), False)
        else:
            self._enable_normalize = False

        self.blending = None

    def forward(self, data: dict | tuple[dict], training: bool = False) -> dict | tuple[dict]:
        """Perform normalization, padding, bgr2rgb conversion and batch augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict or Tuple[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or Tuple[dict]: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        if isinstance(data, dict):
            return self.forward_onesample(data, training=training)
        if isinstance(data, (tuple, list)):
            outputs = []
            for data_sample in data:
                output = self.forward_onesample(data_sample, training=training)
                outputs.append(output)
            return tuple(outputs)

        msg = f"Unsupported data type: {type(data)}!"
        raise TypeError(msg)

    def forward_onesample(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch augmentation on one data sample.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs, data_samples = data["inputs"], data["data_samples"]
        inputs, data_samples = self.preprocess(inputs, data_samples, training)
        data["inputs"] = inputs
        data["data_samples"] = data_samples
        return data

    def preprocess(  # noqa: D102
        self,
        inputs: list[torch.Tensor],
        data_samples: list[ActionDataSample],
        training: bool = False,
    ) -> tuple:
        # --- Pad and stack --
        batch_inputs = stack_batch(inputs)

        if self.format_shape == "MIX2d3d":
            if batch_inputs.ndim == 4:
                format_shape, view_shape = "NCHW", (-1, 1, 1)
            else:
                format_shape, view_shape = "NCTHW", None
        else:
            format_shape, view_shape = self.format_shape, None

        # ------ To RGB ------
        if self.to_rgb:
            if format_shape == "NCHW":
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
            elif format_shape == "NCTHW":
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
            else:
                msg = f"Invalid format shape: {format_shape}"
                raise ValueError(msg)

        # -- Normalization ---
        if self._enable_normalize:
            if view_shape is None:
                batch_inputs = (batch_inputs - self.mean) / self.std
            else:
                mean = self.mean.view(view_shape)
                std = self.std.view(view_shape)
                batch_inputs = (batch_inputs - mean) / std
        elif self.to_float32:
            batch_inputs = batch_inputs.to(torch.float32)

        # ----- Blending -----
        if training and self.blending is not None:
            batch_inputs, data_samples = self.blending(batch_inputs, data_samples)

        return batch_inputs, data_samples

    @property
    def device(self) -> device:
        """Which device being used."""
        try:
            buf = next(self.buffers())
        except StopIteration:
            return super().device
        else:
            return buf.device
