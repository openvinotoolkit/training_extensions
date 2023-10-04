"""Movement-related modules for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Union

import torch
from torch.nn import functional

from .builder import OPS
from .op import Attribute, Operation

# pylint: disable=too-many-branches


@dataclass
class PadV1Attribute(Attribute):
    """PadV1Attribute class."""

    pad_mode: str

    def __post_init__(self) -> None:
        """PadV1Attribute's post-init function."""
        super().__post_init__()
        valid_pad_mode = ["constant", "edge", "reflect", "symmetric"]
        if self.pad_mode not in valid_pad_mode:
            raise ValueError(f"Invalid pad_mode {self.pad_mode}. " f"It must be one of {valid_pad_mode}.")


@OPS.register()
class PadV1(Operation[PadV1Attribute]):
    """PadV1 class."""

    TYPE = "Pad"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = PadV1Attribute
    attrs: PadV1Attribute

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pad_mode = self.get_torch_pad_mode(self.attrs.pad_mode)

    @staticmethod
    def get_torch_pad_mode(pad_mode: str) -> str:
        """PadV1's get_torch_pad_mode function."""
        if pad_mode == "constant":
            return "constant"
        if pad_mode == "edge":
            return "replicate"
        if pad_mode == "reflect":
            return "reflect"
        raise NotImplementedError

    @staticmethod
    def get_torch_pad_dim(pads_begin: list, pads_end: list) -> list:
        """PadV1's get_torch_pad_dim function."""
        # reverse padding
        return [val for tup in zip(pads_begin[::-1], pads_end[::-1]) for val in tup]

    def forward(
        self,
        inputs: torch.Tensor,
        pads_begin: Union[list, torch.Tensor],
        pads_end: Union[list, torch.Tensor],
        pad_value: int = 0,
    ) -> torch.Tensor:
        """PadV1's forward function."""
        pads_begin = pads_begin if isinstance(pads_begin, list) else pads_begin.detach().cpu().tolist()
        pads_end = pads_end if isinstance(pads_end, list) else pads_end.detach().cpu().tolist()
        pad = self.get_torch_pad_dim(pads_begin, pads_end)
        pad = list(map(math.ceil, pad))
        return functional.pad(input=inputs, pad=pad, mode=self._pad_mode, value=pad_value)


@dataclass
class ConcatV0Attribute(Attribute):
    """ConcatV0Attribute class."""

    axis: int


@OPS.register()
class ConcatV0(Operation[ConcatV0Attribute]):
    """ConcatV0 class."""

    TYPE = "Concat"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ConcatV0Attribute
    attrs: ConcatV0Attribute

    def forward(self, *inputs) -> torch.Tensor:
        """ConcatV0's forward function."""
        return torch.cat(inputs, self.attrs.axis)


@dataclass
class TransposeV1Attribute(Attribute):
    """TransposeV1Attribute class."""

    # pylint: disable=unnecessary-pass


@OPS.register()
class TransposeV1(Operation[TransposeV1Attribute]):
    """TransposeV1 class."""

    TYPE = "Transpose"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = TransposeV1Attribute
    attrs: TransposeV1Attribute

    def forward(self, inputs: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """TransposeV1's forward function."""
        if order.numel() == 0:
            order = list(range(inputs.dim()))[::-1]
        elif isinstance(order, torch.Tensor):
            order = order.detach().cpu().tolist()
        return inputs.permute(order)


@dataclass
class GatherV0Attribute(Attribute):
    """GatherV0Attribute class."""

    batch_dims: int = field(default=0)


@OPS.register()
class GatherV0(Operation[GatherV0Attribute]):
    """GatherV0 class."""

    TYPE = "Gather"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = GatherV0Attribute
    attrs: GatherV0Attribute

    def forward(self, inputs: torch.Tensor, indices: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """GatherV0's forward function."""
        axis = axis.squeeze()
        squeeze_axis = indices.dim() == 0

        batch_dims = self.attrs.batch_dims
        if batch_dims < 0:
            batch_dims = indices.dim() + batch_dims

        indices_shape = torch.tensor(indices.shape)
        if batch_dims < axis:
            indices = indices.reshape(*indices_shape[:batch_dims], -1)
            indices_shape = indices_shape[batch_dims:]

        if indices.dim() != inputs.dim():
            if indices.dim() != 0:
                while indices.dim() - 1 < axis:
                    indices = indices.unsqueeze(batch_dims)
            while indices.dim() < inputs.dim():
                indices = indices.unsqueeze(-1)

            repeat = []
            for i, (j, k) in enumerate(zip(inputs.shape, indices.shape)):
                if i == axis:
                    repeat.append(1)
                else:
                    repeat.append(j // k)
            indices = indices.repeat(repeat)
        output = torch.gather(input=inputs, dim=axis, index=indices.type(torch.int64))

        if squeeze_axis:
            output = output.squeeze(axis)

        return output


@dataclass
class GatherV1Attribute(Attribute):
    """GatherV1Attribute class."""

    # pylint: disable=unnecessary-pass


@OPS.register()
class GatherV1(Operation[GatherV1Attribute]):
    """GatherV1 class."""

    TYPE = "Gather"
    VERSION = "opset2"
    ATTRIBUTE_FACTORY = GatherV1Attribute
    attrs: GatherV1Attribute

    def forward(self, inputs: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
        """GatherV1's forward function."""
        return torch.gather(input=inputs, dim=axis, index=indices)


@dataclass
class StridedSliceV1Attribute(Attribute):
    """StridedSliceV1Attribute class."""

    begin_mask: List[int]
    end_mask: List[int]
    new_axis_mask: List[int] = field(default_factory=lambda: [0])
    shrink_axis_mask: List[int] = field(default_factory=lambda: [0])
    ellipsis_mask: List[int] = field(default_factory=lambda: [0])


@OPS.register()
class StridedSliceV1(Operation[StridedSliceV1Attribute]):
    """StridedSliceV1 class."""

    TYPE = "StridedSlice"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = StridedSliceV1Attribute
    attrs: StridedSliceV1Attribute

    def forward(
        self,
        inputs: torch.Tensor,
        begin: torch.Tensor,
        end: list,
        stride: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """StridedSliceV1's forward function."""
        if sum(self.attrs.ellipsis_mask) > 0:
            raise NotImplementedError

        for i, mask in enumerate(self.attrs.begin_mask):
            if mask == 1:
                begin[i] = 0
        for i, mask in enumerate(self.attrs.end_mask):
            if mask == 1:
                end[i] = inputs.size(i)

        if stride is None:
            stride = torch.tensor([1 for _ in begin], dtype=begin.dtype)

        output = inputs
        for i, (b, e, stride_0) in enumerate(zip(begin, end, stride)):
            length = inputs.size(i)

            # begin index is inclusive
            _b = torch.clamp(b, -length, length - 1)
            # end index is exclusive
            _e = torch.clamp(e, -length - 1, length)

            if stride_0 > 0:
                _b = _b + length if _b < 0 else _b
                _e = _e + length if _e < 0 else _e
                indices = torch.arange(_b, _e, stride_0, device=inputs.device)
            else:
                _b = _b - length if _b >= 0 else _b
                _e = _e - length if _e >= 0 else _e
                indices = torch.arange(_b, _e, stride_0, device=inputs.device)
                indices += length

            output = torch.index_select(output, i, indices)

        for i, mask in enumerate(self.attrs.new_axis_mask[::-1]):
            if mask == 1:
                _i = abs(i - len(self.attrs.new_axis_mask) + 1)
                output = output.unsqueeze(_i)

        for i, mask in enumerate(self.attrs.shrink_axis_mask[::-1]):
            if mask == 1:
                _i = abs(i - len(self.attrs.new_axis_mask) + 1)
                if output.size(_i) != 1:
                    raise NotImplementedError
                output = output.squeeze(_i)

        return output


@dataclass
class SplitV1Attribute(Attribute):
    """SplitV1Attribute class."""

    num_splits: int


@OPS.register()
class SplitV1(Operation[SplitV1Attribute]):
    """SplitV1 class."""

    TYPE = "Split"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = SplitV1Attribute
    attrs: SplitV1Attribute

    def forward(self, inputs: torch.Tensor, axis: int) -> torch.Tensor:
        """SplitV1's forward function."""
        split_size = inputs.shape[axis] // self.attrs.num_splits
        return torch.split(tensor=inputs, split_size_or_sections=split_size, dim=axis)


@dataclass
class VariadicSplitV1Attribute(Attribute):
    """VariadicSplitV1Attribute class."""

    # pylint: disable=unnecessary-pass


@OPS.register()
class VariadicSplitV1(Operation[VariadicSplitV1Attribute]):
    """VariadicSplitV1 class."""

    TYPE = "VariadicSplit"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = VariadicSplitV1Attribute
    attrs: VariadicSplitV1Attribute

    def forward(self, inputs: torch.Tensor, axis: int, split_lengths: list) -> tuple:
        """VariadicSplitV1's forward function."""
        idx = [i for i, j in enumerate(split_lengths) if j == -1]
        if idx:
            _idx = idx[0]
            split_lengths[_idx] = inputs.size(axis) - sum(split_lengths) - 1
        outputs = []
        start_idx = 0
        for length in split_lengths:
            outputs.append(
                torch.index_select(
                    inputs,
                    axis,
                    torch.arange(start_idx, start_idx + length, device=inputs.device),
                ),
            )
            start_idx += length
        return tuple(outputs)


@dataclass
class ShuffleChannelsV0Attribute(Attribute):
    """ShuffleChannelsV0Attribute class."""

    axis: int = field(default=1)
    group: int = field(default=1)


@OPS.register()
class ShuffleChannelsV0(Operation[ShuffleChannelsV0Attribute]):
    """ShuffleChannelsV0 class."""

    TYPE = "ShuffleChannels"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ShuffleChannelsV0Attribute
    attrs: ShuffleChannelsV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ShuffleChannelsV0's forward function."""
        origin_shape = inputs.shape
        origin_dim = inputs.dim()

        axis = self.attrs.axis
        axis = axis if axis >= 0 else axis + inputs.dim()

        target_shape = [
            0,
            self.attrs.group,
            int(origin_shape[axis] / self.attrs.group),
            0,
        ]
        if axis == 0:
            target_shape[0] = 1
            target_shape[-1] = math.prod([origin_shape[i] for i in range(axis + 1, origin_dim)])
        elif axis == inputs.dim() - 1:
            target_shape[0] = math.prod([origin_shape[i] for i in range(0, axis)])
            target_shape[-1] = 1
        else:
            target_shape[0] = math.prod([origin_shape[i] for i in range(0, axis)])
            target_shape[-1] = math.prod([origin_shape[i] for i in range(axis + 1, origin_dim)])

        output = inputs.reshape(target_shape)
        output = output.permute([0, 2, 1, 3])
        output = output.reshape(origin_shape)
        return output


@dataclass
class BroadcastV3Attribute(Attribute):
    """BroadcastV3Attribute class."""

    mode: str = field(default="numpy")

    def __post_init__(self) -> None:
        """BroadcastV3Attribute's post-init function."""
        super().__post_init__()
        valid_mode = ["numpy", "explicit", "bidirectional"]
        if self.mode not in valid_mode:
            raise ValueError(f"Invalid mode {self.mode}. " f"It must be one of {valid_mode}.")


@OPS.register()
class BroadcastV3(Operation[BroadcastV3Attribute]):
    """BroadcastV3 class."""

    TYPE = "Broadcast"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = BroadcastV3Attribute
    attrs: BroadcastV3Attribute

    def forward(self, inputs: torch.Tensor, target_shape: list, axes_mapping: list = []) -> torch.Tensor:
        """BroadcastV3's forward function."""
        if self.attrs.mode == "numpy":
            return inputs.expand(*target_shape)
        if self.attrs.mode == "bidirectional":
            return torch.ones(*target_shape, device=inputs.device) * inputs
        prev = -1
        for axes in axes_mapping:
            prev += 1
            while axes - prev > 0:
                inputs = inputs.unsqueeze(axes - 1)
                prev += 1
        while inputs.dim() < len(target_shape):
            inputs = inputs.unsqueeze(-1)
        return inputs.expand(*target_shape)


@dataclass
class ScatterNDUpdateV3Attribute(Attribute):
    """ScatterNDUpdateV3Attribute class."""

    # pylint: disable=unnecessary-pass


@OPS.register()
class ScatterNDUpdateV3(Operation[ScatterNDUpdateV3Attribute]):
    """ScatterNDUpdateV3 class."""

    TYPE = "ScatterNDUpdate"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ScatterNDUpdateV3Attribute
    attrs: ScatterNDUpdateV3Attribute

    def forward(self, inputs: torch.Tensor, indicies: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """ScatterNDUpdateV3's forward function."""
        # TODO: need to verify
        if updates.numel() == 1:
            raise NotImplementedError

        # FIXME: hard-coded
        indicies.shape[-1]
        inputs.shape[indicies.shape[-1] :]  # pylint: disable=pointless-statement
        index = indicies[..., -1]
        for i in inputs.shape[indicies.shape[-1] :]:
            index = index.unsqueeze(-1).tile((i,))
        output = torch.scatter(inputs, 1, index, updates)

        return output


@dataclass
class ScatterUpdateV3Attribute(Attribute):
    """ScatterUpdateV3Attribute class."""

    # pylint: disable=unnecessary-pass


@OPS.register()
class ScatterUpdateV3(Operation[ScatterUpdateV3Attribute]):
    """ScatterUpdateV3 class."""

    TYPE = "ScatterUpdate"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ScatterUpdateV3Attribute
    attrs: ScatterUpdateV3Attribute

    def forward(self, inputs: torch.Tensor, indicies: torch.Tensor, updates: torch.Tensor, axis: int) -> torch.Tensor:
        """ScatterUpdateV3's forward function."""
        # TODO: need to verify

        if inputs.dtype != updates.dtype:
            updates = updates.type(inputs.dtype)

        if indicies.dim() == 0:
            output = inputs
            output[indicies] = updates

        output = torch.scatter(inputs, axis, indicies, updates)

        return output


@dataclass
class TileV0Attribute(Attribute):
    """TileV0Attribute class."""

    # pylint: disable=unnecessary-pass


@OPS.register()
class TileV0(Operation[TileV0Attribute]):
    """TileV0 class."""

    TYPE = "Tile"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = TileV0Attribute
    attrs: TileV0Attribute

    def forward(self, inputs: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
        """TileV0's forward function."""
        return torch.tile(inputs, repeats.tolist())


def get_torch_padding(
    pads_begin: list,
    pads_end: list,
    auto_pad: str,
    input_size: list,
    weight_size: list,
    stride: list,
    dilation: Optional[list] = None,
) -> Union[Callable, int]:
    """Getter function for torch padding."""
    if dilation is None:
        dilation = [1 for _ in input_size]

    if auto_pad == "valid":
        return 0
    if auto_pad in ("same_upper", "same_lower"):
        pads_begin = []
        pads_end = []
        for input_size_, weight_size_, stride_, _ in zip(input_size, weight_size, stride, dilation):
            out_size = math.ceil(input_size_ / stride_)
            padding_needed = max(0, (out_size - 1) * stride_ + weight_size_ - input_size_)
            padding_lhs = int(padding_needed / 2)
            padding_rhs = padding_needed - padding_lhs

            pads_begin.append(padding_lhs if auto_pad == "same_upper" else padding_rhs)
            pads_end.append(padding_rhs if auto_pad == "same_upper" else padding_lhs)
        pad = PadV1.get_torch_pad_dim(pads_begin, pads_end)
        return partial(functional.pad, pad=pad, mode="constant", value=0)
    if auto_pad == "explicit":
        pad = PadV1.get_torch_pad_dim(pads_begin, pads_end)
        return partial(functional.pad, pad=pad, mode="constant", value=0)
    raise NotImplementedError
