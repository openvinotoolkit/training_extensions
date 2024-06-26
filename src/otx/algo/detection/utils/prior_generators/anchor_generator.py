# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.task_modules.prior_generators.anchor_generator.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/prior_generators/anchor_generator.py
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair


class AnchorGenerator:
    """Standard anchor generator for 2D anchor-based detectors.

    # TODO (sungchul): change strides format from (w, h) to (h, w)

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int], Optional): Anchor scales for anchors
            in a single level. It cannot be set at the same time
            if `octave_base_scale` and `scales_per_octave` are set.
        base_sizes (list[int], Optional): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int, Optional): The base scale of octave.
        scales_per_octave (int, Optional): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float]], Optional): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.
    """

    def __init__(
        self,
        strides: list[int] | list[tuple[int, int]],
        ratios: list[float],
        scales: list[int] | None = None,
        base_sizes: list[int] | None = None,
        scale_major: bool = True,
        octave_base_scale: int | None = None,
        scales_per_octave: int | None = None,
        centers: list[tuple[float, float]] | None = None,
        center_offset: float = 0.0,
    ) -> None:
        # check center and center_offset
        if center_offset != 0 and centers is None:
            msg = f"center cannot be set when center_offset != 0, {centers} is given."
            raise ValueError(msg)
        if not (0 <= center_offset <= 1):
            msg = f"center_offset should be in range [0, 1], {center_offset} is given."
            raise ValueError(msg)
        if centers is not None and len(centers) != len(strides):
            msg = f"The number of strides should be the same as centers, got {strides} and {centers}"
            raise ValueError(msg)

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides] if base_sizes is None else base_sizes

        if scales is not None:
            self.scales = Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array([2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = Tensor(scales)
        else:
            msg = "Either scales or octave_base_scale with scales_per_octave should be set"
            raise ValueError(msg)

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self) -> list[int]:
        """list[int]: total number of base anchors in a feature grid."""
        return self.num_base_priors

    @property
    def num_base_priors(self) -> list[int]:
        """Return the number of priors (anchors) at a point on the feature grid."""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied."""
        return len(self.strides)

    def gen_base_anchors(self) -> list[Tensor]:
        """Generate base anchors.

        Returns:
            list(Tensor): Base anchors of a feature grid in multiple feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_size, scales=self.scales, ratios=self.ratios, center=center),
            )
        return multi_level_base_anchors

    def gen_single_level_base_anchors(
        self,
        base_size: int | float,
        scales: Tensor,
        ratios: Tensor,
        center: tuple[float, float] | None = None,
    ) -> Tensor:
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (Tensor): Scales of the anchor.
            ratios (Tensor): The ratio between the height
                and width of anchors in a single level.
            center (tuple[float, float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws,
            y_center - 0.5 * hs,
            x_center + 0.5 * ws,
            y_center + 0.5 * hs,
        ]
        return torch.stack(base_anchors, dim=-1)

    def _meshgrid(self, x: Tensor, y: Tensor, row_major: bool = True) -> tuple[Tensor, ...]:
        """Generate mesh grid of x and y.

        Args:
            x (Tensor): Grids of x dimension.
            y (Tensor): Grids of y dimension.
            row_major (bool): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        return yy, xx

    def grid_priors(
        self,
        featmap_sizes: list[tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> list[Tensor]:
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): list of feature map sizes in
                multiple feature levels.
            dtype (torch.dtype): Dtype of priors.
                Defaults to torch.float32.
            device (str | torch.device): The device where the anchors
                will be put on.

        Return:
            list[Tensor]: Anchors in multiple feature levels.
                The sizes of each tensor should be [N, 4], where
                N = width * height * num_base_anchors, width and height
                are the sizes of the corresponding feature level,
                num_base_anchors is the number of anchors for that level.
        """
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(
        self,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> Tensor:
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int, int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (torch.dtype): Date type of points. Defaults to ``torch.float32``.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            Tensor: Anchors in the overall feature maps.
        """
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors.view(-1, 4)

    def sparse_priors(
        self,
        prior_idxs: Tensor,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> Tensor:
        """Generate sparse anchors according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int, int]): feature map size arrange as (h, w).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (torch.dtype): Date type of points. Defaults to ``torch.float32``.
            device (str | torch.device): The device where the points is
                located.

        Returns:
            Tensor: Anchor with shape (N, 4), N should be equal to
                the length of ``prior_idxs``.
        """
        height, width = featmap_size
        num_base_anchors = self.num_base_anchors[level_idx]
        base_anchor_id = prior_idxs % num_base_anchors
        x = (prior_idxs // num_base_anchors) % width * self.strides[level_idx][0]
        y = (prior_idxs // width // num_base_anchors) % height * self.strides[level_idx][1]
        return torch.stack([x, y, x, y], 1).to(dtype).to(device) + self.base_anchors[level_idx][base_anchor_id, :].to(
            device,
        )

    def grid_anchors(
        self,
        featmap_sizes: list[tuple[int, int]],
        device: str | torch.device = "cuda",
    ) -> list[Tensor]:
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): list of feature map sizes in
                multiple feature levels.
            device (str | torch.device): Device where the anchors will be
                put on.

        Return:
            list[Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        warnings.warn("``grid_anchors`` would be deprecated soon. Please use ``grid_priors`` ", stacklevel=2)

        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device,
            )
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(
        self,
        base_anchors: Tensor,
        featmap_size: tuple[int, int],
        stride: tuple[int, int] = (16, 16),
        device: str | torch.device = "cuda",
    ) -> Tensor:
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int, int]): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str | torch.device): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            Tensor: Anchors in the overall feature maps.
        """
        warnings.warn(
            "``single_level_grid_anchors`` would be deprecated soon. Please use ``single_level_grid_priors`` ",
            stacklevel=2,
        )

        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors.view(-1, 4)

    def valid_flags(
        self,
        featmap_sizes: list[tuple[int, int]],
        pad_shape: tuple,
        device: str | torch.device = "cuda",
    ) -> list[Tensor]:
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple[int, int])): list of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str | torch.device): Device where the anchors will be
                put on.

        Return:
            list(Tensor): Valid flags of anchors in multiple levels.
        """
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w),
                (valid_feat_h, valid_feat_w),
                self.num_base_anchors[i],
                device=device,
            )
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(
        self,
        featmap_size: tuple[int, int],
        valid_size: tuple[int, int],
        num_base_anchors: int,
        device: str | torch.device = "cuda",
    ) -> Tensor:
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str | torch.device): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid[:, None].expand(valid.size(0), num_base_anchors).contiguous().view(-1)

    def __repr__(self) -> str:
        """str: a string that describes the module."""
        indent_str = "    "
        repr_str = self.__class__.__name__ + "(\n"
        repr_str += f"{indent_str}strides={self.strides},\n"
        repr_str += f"{indent_str}ratios={self.ratios},\n"
        repr_str += f"{indent_str}scales={self.scales},\n"
        repr_str += f"{indent_str}base_sizes={self.base_sizes},\n"
        repr_str += f"{indent_str}scale_major={self.scale_major},\n"
        repr_str += f"{indent_str}octave_base_scale="
        repr_str += f"{self.octave_base_scale},\n"
        repr_str += f"{indent_str}scales_per_octave="
        repr_str += f"{self.scales_per_octave},\n"
        repr_str += f"{indent_str}num_levels={self.num_levels}\n"
        repr_str += f"{indent_str}centers={self.centers},\n"
        repr_str += f"{indent_str}center_offset={self.center_offset})"
        return repr_str


class SSDAnchorGeneratorClustered(AnchorGenerator):
    """Custom Anchor Generator for SSD."""

    def __init__(
        self,
        strides: list[int],
        widths: list[list[float]],
        heights: list[list[float]],
    ) -> None:
        """Initialize SSDAnchorGeneratorClustered.

        Args:
            strides (tuple[int]): Anchor's strides.
            widths (list[list[int]]): Anchor's widths.
            heights (list[list[int]]): Anchor's height.
        """
        self.strides = [_pair(stride) for stride in strides]
        self.widths = widths
        self.heights = heights
        self.centers: list[tuple[float, float]] = [(stride / 2.0, stride / 2.0) for stride in strides]

        self.center_offset = 0
        self.gen_base_anchors()

    def gen_base_anchors(self) -> None:  # type: ignore[override]
        """Generate base anchor for SSD."""
        multi_level_base_anchors = []
        for widths, heights, centers in zip(self.widths, self.heights, self.centers):
            base_anchors = self.gen_single_level_base_anchors(
                widths=Tensor(widths),
                heights=Tensor(heights),
                center=Tensor(centers),
            )
            multi_level_base_anchors.append(base_anchors)
        self.base_anchors = multi_level_base_anchors

    def gen_single_level_base_anchors(  # type: ignore[override]
        self,
        widths: Tensor,
        heights: Tensor,
        center: Tensor,
    ) -> Tensor:
        """Generate single_level_base_anchors for SSD.

        Args:
            widths (Tensor): Widths of base anchors.
            heights (Tensor): Heights of base anchors.
            center (Tensor): Centers of base anchors.
        """
        x_center, y_center = center

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * widths,
            y_center - 0.5 * heights,
            x_center + 0.5 * widths,
            y_center + 0.5 * heights,
        ]
        return torch.stack(base_anchors, dim=-1)

    def __repr__(self) -> str:
        """Str: a string that describes the module."""
        indent_str = "    "
        repr_str = self.__class__.__name__ + "(\n"
        repr_str += f"{indent_str}strides={self.strides},\n"
        repr_str += f"{indent_str}widths={self.widths},\n"
        repr_str += f"{indent_str}heights={self.heights},\n"
        repr_str += f"{indent_str}num_levels={self.num_levels}\n"
        repr_str += f"{indent_str}centers={self.centers},\n"
        repr_str += f"{indent_str}center_offset={self.center_offset})"
        return repr_str
