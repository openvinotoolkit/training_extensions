# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from dataclasses import dataclass, field
from typing import List

import torch
from torch.nn import functional as F

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class PadV1Attribute(Attribute):
    pad_mode: str

    def __post_init__(self):
        super().__post_init__()
        valid_pad_mode = ["constant", "edge", "reflect", "symmetric"]
        if self.pad_mode not in valid_pad_mode:
            raise ValueError(f"Invalid pad_mode {self.pad_mode}. " f"It must be one of {valid_pad_mode}.")


@OPS.register()
class PadV1(Operation[PadV1Attribute]):
    TYPE = "Pad"
    VERSION = 1
    ATTRIBUTE_FACTORY = PadV1Attribute

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_mode = self.get_torch_pad_mode(self.attrs.pad_mode)

    @staticmethod
    def get_torch_pad_mode(pad_mode):
        if pad_mode == "constant":
            return "constant"
        elif pad_mode == "edge":
            return "replicate"
        elif pad_mode == "reflect":
            return "reflect"
        elif pad_mode == "symmetric":
            raise NotImplementedError
        else:
            raise NotImplementedError

    @staticmethod
    def get_torch_pad_dim(pads_begin, pads_end):
        # reverse padding
        return [val for tup in zip(pads_begin[::-1], pads_end[::-1]) for val in tup]

    def forward(self, input, pads_begin, pads_end, pad_value=0):
        pads_begin = pads_begin if isinstance(pads_begin, list) else pads_begin.detach().cpu().tolist()
        pads_end = pads_end if isinstance(pads_end, list) else pads_end.detach().cpu().tolist()
        pad = self.get_torch_pad_dim(pads_begin, pads_end)
        pad = list(map(math.ceil, pad))
        return F.pad(input=input, pad=pad, mode=self._pad_mode, value=pad_value)


@dataclass
class ConcatV0Attribute(Attribute):
    axis: int


@OPS.register()
class ConcatV0(Operation[ConcatV0Attribute]):
    TYPE = "Concat"
    VERSION = 0
    ATTRIBUTE_FACTORY = ConcatV0Attribute

    def forward(self, *inputs):
        return torch.cat(inputs, self.attrs.axis)


@dataclass
class TransposeV1Attribute(Attribute):
    pass


@OPS.register()
class TransposeV1(Operation[TransposeV1Attribute]):
    TYPE = "Transpose"
    VERSION = 1
    ATTRIBUTE_FACTORY = TransposeV1Attribute

    def forward(self, input, order):
        if order.numel() == 0:
            order = list(range(input.dim()))[::-1]
        elif isinstance(order, torch.Tensor):
            order = order.detach().cpu().tolist()
        return input.permute(order)


@dataclass
class GatherV0Attribute(Attribute):
    batch_dims: int = field(default=0)


@OPS.register()
class GatherV0(Operation[GatherV0Attribute]):
    TYPE = "Gather"
    VERSION = 0
    ATTRIBUTE_FACTORY = GatherV0Attribute

    def forward(self, input, indices, axis):
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

        if indices.dim() != input.dim():
            if indices.dim() != 0:
                while indices.dim() - 1 < axis:
                    indices = indices.unsqueeze(batch_dims)
            while indices.dim() < input.dim():
                indices = indices.unsqueeze(-1)

            repeat = []
            for i, (j, k) in enumerate(zip(input.shape, indices.shape)):
                if i == axis:
                    repeat.append(1)
                else:
                    assert j % k == 0
                    repeat.append(j // k)
            indices = indices.repeat(repeat)
        output = torch.gather(input=input, dim=axis, index=indices.type(torch.int64))

        if squeeze_axis:
            output = output.squeeze(axis)

        return output


@dataclass
class GatherV1Attribute(Attribute):
    pass


@OPS.register()
class GatherV1(Operation[GatherV1Attribute]):
    TYPE = "Gather"
    VERSION = 1
    ATTRIBUTE_FACTORY = GatherV1Attribute

    def forward(self, input, indices, axis):
        return torch.gather(input=input, dim=axis, index=indices)


@dataclass
class StridedSliceV1Attribute(Attribute):
    begin_mask: List[int]
    end_mask: List[int]
    new_axis_mask: List[int] = field(default_factory=lambda: [0])
    shrink_axis_mask: List[int] = field(default_factory=lambda: [0])
    ellipsis_mask: List[int] = field(default_factory=lambda: [0])


@OPS.register()
class StridedSliceV1(Operation[StridedSliceV1Attribute]):
    TYPE = "StridedSlice"
    VERSION = 1
    ATTRIBUTE_FACTORY = StridedSliceV1Attribute

    def forward(self, input, begin, end, stride=None):
        if sum(self.attrs.ellipsis_mask) > 0:
            raise NotImplementedError

        for i, mask in enumerate(self.attrs.begin_mask):
            if mask == 1:
                begin[i] = 0
        for i, mask in enumerate(self.attrs.end_mask):
            if mask == 1:
                end[i] = input.size(i)

        if stride is None:
            stride = torch.tensor([1 for _ in begin], dtype=begin.dtype)

        output = input
        for i, (b, e, s) in enumerate(zip(begin, end, stride)):
            length = input.size(i)

            # begin index is inclusive
            b = torch.clamp(b, -length, length - 1)
            # end index is exclusive
            e = torch.clamp(e, -length - 1, length)

            if s > 0:
                b = b + length if b < 0 else b
                e = e + length if e < 0 else e
                indices = torch.arange(b, e, s, device=input.device)
            else:
                b = b - length if b >= 0 else b
                e = e - length if e >= 0 else e
                indices = torch.arange(b, e, s, device=input.device)
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
    num_splits: int


@OPS.register()
class SplitV1(Operation[SplitV1Attribute]):
    TYPE = "Split"
    VERSION = 1
    ATTRIBUTE_FACTORY = SplitV1Attribute

    def forward(self, input, axis):
        split_size = input.shape[axis] // self.attrs.num_splits
        return torch.split(tensor=input, split_size_or_sections=split_size, dim=axis)


@dataclass
class VariadicSplitV1Attribute(Attribute):
    pass


@OPS.register()
class VariadicSplitV1(Operation[VariadicSplitV1Attribute]):
    TYPE = "VariadicSplit"
    VERSION = 1
    ATTRIBUTE_FACTORY = VariadicSplitV1Attribute

    def forward(self, input, axis, split_lengths):
        idx = [i for i, j in enumerate(split_lengths) if j == -1]
        if idx:
            assert len(idx) == 1
            idx = idx[0]
            split_lengths[idx] = input.size(axis) - sum(split_lengths) - 1
        assert input.size(axis) == sum(split_lengths)
        outputs = []
        start_idx = 0
        for length in split_lengths:
            outputs.append(
                torch.index_select(
                    input,
                    axis,
                    torch.arange(start_idx, start_idx + length, device=input.device),
                )
            )
            start_idx += length
        return tuple(outputs)


@dataclass
class ShuffleChannelsV0Attribute(Attribute):
    axis: int = field(default=1)
    group: int = field(default=1)


@OPS.register()
class ShuffleChannelsV0(Operation[ShuffleChannelsV0Attribute]):
    TYPE = "ShuffleChannels"
    VERSION = 0
    ATTRIBUTE_FACTORY = ShuffleChannelsV0Attribute

    def forward(self, input):
        #  n, c, h, w = input.shape
        assert input.dim() == 4
        origin_shape = input.shape
        origin_dim = input.dim()
        assert origin_shape[self.attrs.axis] % self.attrs.group == 0

        axis = self.attrs.axis
        axis = axis if axis >= 0 else axis + input.dim()

        target_shape = [
            0,
            self.attrs.group,
            int(origin_shape[axis] / self.attrs.group),
            0,
        ]
        if axis == 0:
            target_shape[0] = 1
            target_shape[-1] = math.prod([origin_shape[i] for i in range(axis + 1, origin_dim)])
        elif axis == input.dim() - 1:
            target_shape[0] = math.prod([origin_shape[i] for i in range(0, axis)])
            target_shape[-1] = 1
        else:
            target_shape[0] = math.prod([origin_shape[i] for i in range(0, axis)])
            target_shape[-1] = math.prod([origin_shape[i] for i in range(axis + 1, origin_dim)])

        output = input.reshape(target_shape)
        output = output.permute([0, 2, 1, 3])
        output = output.reshape(origin_shape)
        return output


@dataclass
class BroadcastV3Attribute(Attribute):
    mode: str = field(default="numpy")

    def __post_init__(self):
        super().__post_init__()
        valid_mode = ["numpy", "explicit", "bidirectional"]
        if self.mode not in valid_mode:
            raise ValueError(f"Invalid mode {self.mode}. " f"It must be one of {valid_mode}.")


@OPS.register()
class BroadcastV3(Operation[BroadcastV3Attribute]):
    TYPE = "Broadcast"
    VERSION = 3
    ATTRIBUTE_FACTORY = BroadcastV3Attribute

    def forward(self, input, target_shape, axes_mapping=None):
        if self.attrs.mode == "numpy":
            return input.expand(*target_shape)
        if self.attrs.mode == "bidirectional":
            return torch.ones(*target_shape, device=input.device) * input
        else:
            assert axes_mapping is not None
            prev = -1
            for axes in axes_mapping:
                prev += 1
                while axes - prev > 0:
                    input = input.unsqueeze(axes - 1)
                    prev += 1
            while input.dim() < len(target_shape):
                input = input.unsqueeze(-1)
            return input.expand(*target_shape)


@dataclass
class ScatterNDUpdateV3Attribute(Attribute):
    pass


@OPS.register()
class ScatterNDUpdateV3(Operation[ScatterNDUpdateV3Attribute]):
    TYPE = "ScatterNDUpdate"
    VERSION = 3
    ATTRIBUTE_FACTORY = ScatterNDUpdateV3Attribute

    def forward(self, input, indicies, updates):
        # TODO: need to verify
        if updates.numel() == 1:
            raise NotImplementedError

        # FIXME: hard-coded
        last_dim = indicies.shape[-1]
        assert last_dim == 2
        assert indicies[..., -2].sum() == 0
        input.shape[indicies.shape[-1] :]
        index = indicies[..., -1]
        for i in input.shape[indicies.shape[-1] :]:
            index = index.unsqueeze(-1).tile((i,))
        output = torch.scatter(input, 1, index, updates)

        return output


@dataclass
class ScatterUpdateV3Attribute(Attribute):
    pass


@OPS.register()
class ScatterUpdateV3(Operation[ScatterUpdateV3Attribute]):
    TYPE = "ScatterUpdate"
    VERSION = 3
    ATTRIBUTE_FACTORY = ScatterUpdateV3Attribute

    def forward(self, input, indicies, updates, axis):
        # TODO: need to verify
        axis = axis.item()

        if input.dtype != updates.dtype:
            updates = updates.type(input.dtype)

        if indicies.dim() == 0:
            assert axis == 0
            output = input
            output[indicies] = updates

        output = torch.scatter(input, axis, indicies, updates)

        return output


@dataclass
class TileV0Attribute(Attribute):
    pass


@OPS.register()
class TileV0(Operation[TileV0Attribute]):
    TYPE = "Tile"
    VERSION = 0
    ATTRIBUTE_FACTORY = TileV0Attribute

    def forward(self, input, repeats):
        return torch.tile(input, repeats.tolist())
