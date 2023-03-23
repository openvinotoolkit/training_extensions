"""Movement-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass, field
from functools import partial
from typing import List

import torch
from torch.nn import functional as F

from .builder import OPS
from .op import Attribute, Operation

# pylint: disable=too-many-branches


@dataclass
class PadV1Attribute(Attribute):
    """PadV1Attribute class."""

    pad_mode: str

    def __post_init__(self):
        """PadV1Attribute's post-init function."""
        super().__post_init__()
        valid_pad_mode = ["constant", "edge", "reflect", "symmetric"]
        if self.pad_mode not in valid_pad_mode:
            raise ValueError(f"Invalid pad_mode {self.pad_mode}. " f"It must be one of {valid_pad_mode}.")


@OPS.register()
class PadV1(Operation[PadV1Attribute]):
    """PadV1 class."""

    TYPE = "Pad"
    VERSION = 1
    ATTRIBUTE_FACTORY = PadV1Attribute

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_mode = self.get_torch_pad_mode(self.attrs.pad_mode)

    @staticmethod
    def get_torch_pad_mode(pad_mode):
        """PadV1's get_torch_pad_mode function."""
        if pad_mode == "constant":
            return "constant"
        if pad_mode == "edge":
            return "replicate"
        if pad_mode == "reflect":
            return "reflect"
        raise NotImplementedError

    @staticmethod
    def get_torch_pad_dim(pads_begin, pads_end):
        """PadV1's get_torch_pad_dim function."""
        # reverse padding
        return [val for tup in zip(pads_begin[::-1], pads_end[::-1]) for val in tup]

    def forward(self, inputs, pads_begin, pads_end, pad_value=0):
        """PadV1's forward function."""
        pads_begin = pads_begin if isinstance(pads_begin, list) else pads_begin.detach().cpu().tolist()
        pads_end = pads_end if isinstance(pads_end, list) else pads_end.detach().cpu().tolist()
        pad = self.get_torch_pad_dim(pads_begin, pads_end)
        pad = list(map(math.ceil, pad))
        return F.pad(input=inputs, pad=pad, mode=self._pad_mode, value=pad_value)


@dataclass
class ConcatV0Attribute(Attribute):
    """ConcatV0Attribute class."""

    axis: int


@OPS.register()
class ConcatV0(Operation[ConcatV0Attribute]):
    """ConcatV0 class."""

    TYPE = "Concat"
    VERSION = 0
    ATTRIBUTE_FACTORY = ConcatV0Attribute

    def forward(self, *inputs):
        """ConcatV0's forward function."""
        return torch.cat(inputs, self.attrs.axis)


@dataclass
class TransposeV1Attribute(Attribute):
    """TransposeV1Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class TransposeV1(Operation[TransposeV1Attribute]):
    """TransposeV1 class."""

    TYPE = "Transpose"
    VERSION = 1
    ATTRIBUTE_FACTORY = TransposeV1Attribute

    def forward(self, inputs, order):
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
    VERSION = 0
    ATTRIBUTE_FACTORY = GatherV0Attribute

    def forward(self, inputs, indices, axis):
        """GatherV0's forward function."""
        assert axis.numel() == 1
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
                    assert j % k == 0
                    repeat.append(j // k)
            indices = indices.repeat(repeat)
        output = torch.gather(input=inputs, dim=axis, index=indices.type(torch.int64))

        if squeeze_axis:
            output = output.squeeze(axis)

        return output


@dataclass
class GatherV1Attribute(Attribute):
    """GatherV1Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class GatherV1(Operation[GatherV1Attribute]):
    """GatherV1 class."""

    TYPE = "Gather"
    VERSION = 1
    ATTRIBUTE_FACTORY = GatherV1Attribute

    def forward(self, inputs, indices, axis):
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
    VERSION = 1
    ATTRIBUTE_FACTORY = StridedSliceV1Attribute

    def forward(self, inputs, begin, end, stride=None):
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
            b = torch.clamp(b, -length, length - 1)
            # end index is exclusive
            e = torch.clamp(e, -length - 1, length)

            if stride_0 > 0:
                b = b + length if b < 0 else b
                e = e + length if e < 0 else e
                indices = torch.arange(b, e, stride_0, device=inputs.device)
            else:
                b = b - length if b >= 0 else b
                e = e - length if e >= 0 else e
                indices = torch.arange(b, e, stride_0, device=inputs.device)
                indices += length

            output = torch.index_select(output, i, indices)

        for i, mask in enumerate(self.attrs.new_axis_mask[::-1]):
            if mask == 1:
                i = abs(i - len(self.attrs.new_axis_mask) + 1)
                output = output.unsqueeze(i)

        for i, mask in enumerate(self.attrs.shrink_axis_mask[::-1]):
            if mask == 1:
                i = abs(i - len(self.attrs.new_axis_mask) + 1)
                if output.size(i) != 1:
                    raise NotImplementedError
                output = output.squeeze(i)

        return output


@dataclass
class SplitV1Attribute(Attribute):
    """SplitV1Attribute class."""

    num_splits: int


@OPS.register()
class SplitV1(Operation[SplitV1Attribute]):
    """SplitV1 class."""

    TYPE = "Split"
    VERSION = 1
    ATTRIBUTE_FACTORY = SplitV1Attribute

    def forward(self, inputs, axis):
        """SplitV1's forward function."""
        split_size = inputs.shape[axis] // self.attrs.num_splits
        return torch.split(tensor=inputs, split_size_or_sections=split_size, dim=axis)


@dataclass
class VariadicSplitV1Attribute(Attribute):
    """VariadicSplitV1Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class VariadicSplitV1(Operation[VariadicSplitV1Attribute]):
    """VariadicSplitV1 class."""

    TYPE = "VariadicSplit"
    VERSION = 1
    ATTRIBUTE_FACTORY = VariadicSplitV1Attribute

    def forward(self, inputs, axis, split_lengths):
        """VariadicSplitV1's forward function."""
        idx = [i for i, j in enumerate(split_lengths) if j == -1]
        if idx:
            assert len(idx) == 1
            idx = idx[0]
            split_lengths[idx] = inputs.size(axis) - sum(split_lengths) - 1
        assert inputs.size(axis) == sum(split_lengths)
        outputs = []
        start_idx = 0
        for length in split_lengths:
            outputs.append(
                torch.index_select(
                    inputs,
                    axis,
                    torch.arange(start_idx, start_idx + length, device=inputs.device),
                )
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
    VERSION = 0
    ATTRIBUTE_FACTORY = ShuffleChannelsV0Attribute

    def forward(self, inputs):
        """ShuffleChannelsV0's forward function."""
        #  n, c, h, w = input.shape
        assert inputs.dim() == 4
        origin_shape = inputs.shape
        origin_dim = inputs.dim()
        assert origin_shape[self.attrs.axis] % self.attrs.group == 0

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

    def __post_init__(self):
        """BroadcastV3Attribute's post-init function."""
        super().__post_init__()
        valid_mode = ["numpy", "explicit", "bidirectional"]
        if self.mode not in valid_mode:
            raise ValueError(f"Invalid mode {self.mode}. " f"It must be one of {valid_mode}.")


@OPS.register()
class BroadcastV3(Operation[BroadcastV3Attribute]):
    """BroadcastV3 class."""

    TYPE = "Broadcast"
    VERSION = 3
    ATTRIBUTE_FACTORY = BroadcastV3Attribute

    def forward(self, inputs, target_shape, axes_mapping=None):
        """BroadcastV3's forward function."""
        if self.attrs.mode == "numpy":
            return inputs.expand(*target_shape)
        if self.attrs.mode == "bidirectional":
            return torch.ones(*target_shape, device=inputs.device) * inputs
        assert axes_mapping is not None
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

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class ScatterNDUpdateV3(Operation[ScatterNDUpdateV3Attribute]):
    """ScatterNDUpdateV3 class."""

    TYPE = "ScatterNDUpdate"
    VERSION = 3
    ATTRIBUTE_FACTORY = ScatterNDUpdateV3Attribute

    def forward(self, inputs, indicies, updates):
        """ScatterNDUpdateV3's forward function."""
        # TODO: need to verify
        if updates.numel() == 1:
            raise NotImplementedError

        # FIXME: hard-coded
        last_dim = indicies.shape[-1]
        assert last_dim == 2
        assert indicies[..., -2].sum() == 0
        inputs.shape[indicies.shape[-1] :]  # pylint: disable=pointless-statement
        index = indicies[..., -1]
        for i in inputs.shape[indicies.shape[-1] :]:
            index = index.unsqueeze(-1).tile((i,))
        output = torch.scatter(inputs, 1, index, updates)

        return output


@dataclass
class ScatterUpdateV3Attribute(Attribute):
    """ScatterUpdateV3Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class ScatterUpdateV3(Operation[ScatterUpdateV3Attribute]):
    """ScatterUpdateV3 class."""

    TYPE = "ScatterUpdate"
    VERSION = 3
    ATTRIBUTE_FACTORY = ScatterUpdateV3Attribute

    def forward(self, inputs, indicies, updates, axis):
        """ScatterUpdateV3's forward function."""
        # TODO: need to verify
        axis = axis.item()

        if inputs.dtype != updates.dtype:
            updates = updates.type(inputs.dtype)

        if indicies.dim() == 0:
            assert axis == 0
            output = inputs
            output[indicies] = updates

        output = torch.scatter(inputs, axis, indicies, updates)

        return output


@dataclass
class TileV0Attribute(Attribute):
    """TileV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class TileV0(Operation[TileV0Attribute]):
    """TileV0 class."""

    TYPE = "Tile"
    VERSION = 0
    ATTRIBUTE_FACTORY = TileV0Attribute

    def forward(self, inputs, repeats):
        """TileV0's forward function."""
        return torch.tile(inputs, repeats.tolist())


def get_torch_padding(pads_begin, pads_end, auto_pad, input_size, weight_size, stride, dilation=None):
    """Getter function for torch padding."""
    if dilation is None:
        dilation = [1 for _ in input_size]

    if auto_pad == "valid":
        return 0
    if auto_pad in ("same_upper", "same_lower"):
        assert len(set(dilation)) == 1 and dilation[0] == 1
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
        return partial(F.pad, pad=pad, mode="constant", value=0)
    if auto_pad == "explicit":
        pad = PadV1.get_torch_pad_dim(pads_begin, pads_end)
        return partial(F.pad, pad=pad, mode="constant", value=0)
    raise NotImplementedError
