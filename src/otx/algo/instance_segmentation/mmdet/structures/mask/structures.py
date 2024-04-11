"""The original source code is from mmdet.mask.structures. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import itertools
from abc import ABCMeta
from typing import Sequence

import cv2

# TODO(Eugene): remove mmcv
# https://github.com/openvinotoolkit/training_extensions/pull/3281
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch

# TODO(Eugene): remove mmcv
# https://github.com/openvinotoolkit/training_extensions/pull/3281
from mmcv.ops.roi_align import roi_align
from shapely import geometry


class BaseInstanceMasks(metaclass=ABCMeta):
    """Base class for instance masks."""


class BitmapMasks(BaseInstanceMasks):
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.data_elements.mask.structures import *  # NOQA
        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(np.int64)
        >>> self = BitmapMasks(masks, height=H, width=W)

        >>> # demo crop_and_resize
        >>> num_boxes = 5
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (14, 14)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks: np.array | list, height: int, width: int) -> None:
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation="nearest"):
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack([mmcv.imrescale(mask, scale, interpolation=interpolation) for mask in self.masks])
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape, interpolation="nearest"):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack(
                [mmcv.imresize(mask, out_shape[::-1], interpolation=interpolation) for mask in self.masks],
            )
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction="horizontal"):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ("horizontal", "vertical", "diagonal")

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([mmcv.imflip(mask, direction=flip_direction) for mask in self.masks])
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape, pad_val=0):
        """See :func:`BaseInstanceMasks.pad`."""
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack([mmcv.impad(mask, shape=out_shape, pad_val=pad_val) for mask in self.masks])
        return BitmapMasks(padded_masks, *out_shape)

    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`."""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1 : y1 + h, x1 : x1 + w]
        return BitmapMasks(cropped_masks, h, w)

    def crop_and_resize(self, bboxes, out_shape, inds, device="cpu", interpolation="bilinear", binarize=True):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = torch.from_numpy(self.masks).to(device).index_select(0, inds).to(dtype=rois.dtype)
            targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape, 1.0, 0, "avg", True).squeeze(1)
            if binarize:
                resized_masks = (targets >= 0.5).cpu().numpy()
            else:
                resized_masks = targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h, expanded_w, top, left):
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w), dtype=np.uint8)
        else:
            expanded_mask = np.zeros((len(self), expanded_h, expanded_w), dtype=np.uint8)
            expanded_mask[:, top : top + self.height, left : left + self.width] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(self, out_shape, offset, direction="horizontal", border_value=0, interpolation="bilinear"):
        """Translate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            border_value (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            BitmapMasks: Translated BitmapMasks.

        Example:
            >>> from mmdet.data_elements.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random(dtype=np.uint8)
            >>> out_shape = (32, 32)
            >>> offset = 4
            >>> direction = 'horizontal'
            >>> border_value = 0
            >>> interpolation = 'bilinear'
            >>> # Note, There seem to be issues when:
            >>> # * the mask dtype is not supported by cv2.AffineWarp
            >>> new = self.translate(out_shape, offset, direction,
            >>>                      border_value, interpolation)
            >>> assert len(new) == len(self)
            >>> assert new.height, new.width == out_shape
        """
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            masks = self.masks
            if masks.shape[-2:] != out_shape:
                empty_masks = np.zeros((masks.shape[0], *out_shape), dtype=masks.dtype)
                min_h = min(out_shape[0], masks.shape[1])
                min_w = min(out_shape[1], masks.shape[2])
                empty_masks[:, :min_h, :min_w] = masks[:, :min_h, :min_w]
                masks = empty_masks
            translated_masks = mmcv.imtranslate(
                masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=border_value,
                interpolation=interpolation,
            )
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(translated_masks, *out_shape)

    def shear(self, out_shape, magnitude, direction="horizontal", border_value=0, interpolation="bilinear"):
        """Shear the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            BitmapMasks: The sheared masks.
        """
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            sheared_masks = mmcv.imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation,
            )
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(sheared_masks, *out_shape)

    def rotate(self, out_shape, angle, center=None, scale=1.0, border_value=0, interpolation="bilinear"):
        """Rotate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            border_value (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as in :func:`mmcv.imrotate`.

        Returns:
            BitmapMasks: Rotated BitmapMasks.
        """
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = mmcv.imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=border_value,
                interpolation=interpolation,
            )
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(rotated_masks, *out_shape)

    @property
    def areas(self):
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return torch.tensor(self.masks, dtype=dtype, device=device)

    @classmethod
    def random(cls, num_masks=3, height=32, width=32, dtype=np.uint8, rng=None):
        """Generate random bitmap masks for demo / testing purposes.

        Example:
            >>> from mmdet.data_elements.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random()
            >>> print('self = {}'.format(self))
            self = BitmapMasks(num_masks=3, height=32, width=32)
        """
        from otx.algo.instance_segmentation.mmdet.models.utils.util_random import ensure_rng

        rng = ensure_rng(rng)
        masks = (rng.rand(num_masks, height, width) > 0.1).astype(dtype)
        self = cls(masks, height=height, width=width)
        return self

    @classmethod
    def cat(cls: type[BitmapMasks], masks: Sequence[BitmapMasks]) -> BitmapMasks:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[BitmapMasks]): A sequence of mask instances.

        Returns:
            BitmapMasks: Concatenated mask instance.
        """
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError("masks should not be an empty list.")
        assert all(isinstance(m, cls) for m in masks)

        mask_array = np.concatenate([m.masks for m in masks], axis=0)
        return cls(mask_array, *mask_array.shape[1:])


class PolygonMasks:
    """This class represents masks in the form of polygons.

    Polygons is a list of three levels. The first level of the list
    corresponds to objects, the second level to the polys that compose the
    object, the third level to the poly coordinates

    Args:
        masks (list[list[ndarray]]): The first level of the list
            corresponds to objects, the second level to the polys that
            compose the object, the third level to the poly coordinates
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.data_elements.mask.structures import *  # NOQA
        >>> masks = [
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 0, 0]) ]
        >>> ]
        >>> height, width = 16, 16
        >>> self = PolygonMasks(masks, height, width)

        >>> # demo translate
        >>> new = self.translate((16, 16), 4., direction='horizontal')
        >>> assert np.all(new.masks[0][0][1::2] == masks[0][0][1::2])
        >>> assert np.all(new.masks[0][0][0::2] == masks[0][0][0::2] + 4)

        >>> # demo crop_and_resize
        >>> num_boxes = 3
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (16, 16)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        assert isinstance(masks, list)
        if len(masks) > 0:
            assert isinstance(masks[0], list)
            assert isinstance(masks[0][0], np.ndarray)

        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index):
        """Index the polygon masks.

        Args:
            index (ndarray | List): The indices.

        Returns:
            :obj:`PolygonMasks`: The indexed polygon masks.
        """
        if isinstance(index, np.ndarray):
            if index.dtype == bool:
                index = np.where(index)[0].tolist()
            else:
                index = index.tolist()
        if isinstance(index, list):
            masks = [self.masks[i] for i in index]
        else:
            try:
                masks = self.masks[index]
            except Exception:
                raise ValueError(f"Unsupported input of type {type(index)} for indexing!")
        if len(masks) and isinstance(masks[0], np.ndarray):
            masks = [masks]  # ensure a list of three levels
        return PolygonMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation=None):
        """See :func:`BaseInstanceMasks.rescale`"""
        new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
        if len(self.masks) == 0:
            rescaled_masks = PolygonMasks([], new_h, new_w)
        else:
            rescaled_masks = self.resize((new_h, new_w))
        return rescaled_masks

    def resize(self, out_shape, interpolation=None):
        """See :func:`BaseInstanceMasks.resize`"""
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] = p[0::2] * w_scale
                    p[1::2] = p[1::2] * h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, *out_shape)
        return resized_masks

    def flip(self, flip_direction="horizontal"):
        """See :func:`BaseInstanceMasks.flip`"""
        assert flip_direction in ("horizontal", "vertical", "diagonal")
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            flipped_masks = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if flip_direction == "horizontal":
                        p[0::2] = self.width - p[0::2]
                    elif flip_direction == "vertical":
                        p[1::2] = self.height - p[1::2]
                    else:
                        p[0::2] = self.width - p[0::2]
                        p[1::2] = self.height - p[1::2]
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            flipped_masks = PolygonMasks(flipped_masks, self.height, self.width)
        return flipped_masks

    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`"""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = PolygonMasks([], h, w)
        else:
            # reference: https://github.com/facebookresearch/fvcore/blob/main/fvcore/transforms/transform.py
            crop_box = geometry.box(x1, y1, x2, y2).buffer(0.0)
            cropped_masks = []
            # suppress shapely warnings util it incorporates GEOS>=3.11.2
            # reference: https://github.com/shapely/shapely/issues/1345
            initial_settings = np.seterr()
            np.seterr(invalid="ignore")
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p = geometry.Polygon(p.reshape(-1, 2)).buffer(0.0)
                    # polygon must be valid to perform intersection.
                    if not p.is_valid:
                        continue
                    cropped = p.intersection(crop_box)
                    if cropped.is_empty:
                        continue
                    if isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                        cropped = cropped.geoms
                    else:
                        cropped = [cropped]
                    # one polygon may be cropped to multiple ones
                    for poly in cropped:
                        # ignore lines or points
                        if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                            continue
                        coords = np.asarray(poly.exterior.coords)
                        # remove an extra identical vertex at the end
                        coords = coords[:-1]
                        coords[:, 0] -= x1
                        coords[:, 1] -= y1
                        cropped_poly_per_obj.append(coords.reshape(-1))
                # a dummy polygon to avoid misalignment between masks and boxes
                if len(cropped_poly_per_obj) == 0:
                    cropped_poly_per_obj = [np.array([0, 0, 0, 0, 0, 0])]
                cropped_masks.append(cropped_poly_per_obj)
            np.seterr(**initial_settings)
            cropped_masks = PolygonMasks(cropped_masks, h, w)
        return cropped_masks

    def pad(self, out_shape, pad_val=0):
        """Padding has no effect on polygons`"""
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs):
        raise NotImplementedError

    def crop_and_resize(self, bboxes, out_shape, inds, device="cpu", interpolation="bilinear", binarize=True):
        """See :func:`BaseInstanceMasks.crop_and_resize`"""
        out_h, out_w = out_shape
        if len(self.masks) == 0:
            return PolygonMasks([], out_h, out_w)

        if not binarize:
            raise ValueError("Polygons are always binary, setting binarize=False is unsupported")

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[inds[i]]
            bbox = bboxes[i, :]
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1, 1)
            h = np.maximum(y2 - y1, 1)
            h_scale = out_h / max(h, 0.1)  # avoid too large scale
            w_scale = out_w / max(w, 0.1)

            resized_mask = []
            for p in mask:
                p = p.copy()
                # crop
                # pycocotools will clip the boundary
                p[0::2] = p[0::2] - bbox[0]
                p[1::2] = p[1::2] - bbox[1]

                # resize
                p[0::2] = p[0::2] * w_scale
                p[1::2] = p[1::2] * h_scale
                resized_mask.append(p)
            resized_masks.append(resized_mask)
        return PolygonMasks(resized_masks, *out_shape)

    def translate(self, out_shape, offset, direction="horizontal", border_value=None, interpolation=None):
        """Translate the PolygonMasks.

        Example:
            >>> self = PolygonMasks.random(dtype=np.int64)
            >>> out_shape = (self.height, self.width)
            >>> new = self.translate(out_shape, 4., direction='horizontal')
            >>> assert np.all(new.masks[0][0][1::2] == self.masks[0][0][1::2])
            >>> assert np.all(new.masks[0][0][0::2] == self.masks[0][0][0::2] + 4)  # noqa: E501
        """
        assert border_value is None or border_value == 0, (
            "Here border_value is not " f"used, and defaultly should be None or 0. got {border_value}."
        )
        if len(self.masks) == 0:
            translated_masks = PolygonMasks([], *out_shape)
        else:
            translated_masks = []
            for poly_per_obj in self.masks:
                translated_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if direction == "horizontal":
                        p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                    elif direction == "vertical":
                        p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                    translated_poly_per_obj.append(p)
                translated_masks.append(translated_poly_per_obj)
            translated_masks = PolygonMasks(translated_masks, *out_shape)
        return translated_masks

    def shear(self, out_shape, magnitude, direction="horizontal", border_value=0, interpolation="bilinear"):
        """See :func:`BaseInstanceMasks.shear`."""
        if len(self.masks) == 0:
            sheared_masks = PolygonMasks([], *out_shape)
        else:
            sheared_masks = []
            if direction == "horizontal":
                shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(np.float32)
            elif direction == "vertical":
                shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
            for poly_per_obj in self.masks:
                sheared_poly = []
                for p in poly_per_obj:
                    p = np.stack([p[0::2], p[1::2]], axis=0)  # [2, n]
                    new_coords = np.matmul(shear_matrix, p)  # [2, n]
                    new_coords[0, :] = np.clip(new_coords[0, :], 0, out_shape[1])
                    new_coords[1, :] = np.clip(new_coords[1, :], 0, out_shape[0])
                    sheared_poly.append(new_coords.transpose((1, 0)).reshape(-1))
                sheared_masks.append(sheared_poly)
            sheared_masks = PolygonMasks(sheared_masks, *out_shape)
        return sheared_masks

    def rotate(self, out_shape, angle, center=None, scale=1.0, border_value=0, interpolation="bilinear"):
        """See :func:`BaseInstanceMasks.rotate`."""
        if len(self.masks) == 0:
            rotated_masks = PolygonMasks([], *out_shape)
        else:
            rotated_masks = []
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            for poly_per_obj in self.masks:
                rotated_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    coords = np.stack([p[0::2], p[1::2]], axis=1)  # [n, 2]
                    # pad 1 to convert from format [x, y] to homogeneous
                    # coordinates format [x, y, 1]
                    coords = np.concatenate((coords, np.ones((coords.shape[0], 1), coords.dtype)), axis=1)  # [n, 3]
                    rotated_coords = np.matmul(rotate_matrix[None, :, :], coords[:, :, None])[
                        ...,
                        0,
                    ]  # [n, 2, 1] -> [n, 2]
                    rotated_coords[:, 0] = np.clip(rotated_coords[:, 0], 0, out_shape[1])
                    rotated_coords[:, 1] = np.clip(rotated_coords[:, 1], 0, out_shape[0])
                    rotated_poly.append(rotated_coords.reshape(-1))
                rotated_masks.append(rotated_poly)
            rotated_masks = PolygonMasks(rotated_masks, *out_shape)
        return rotated_masks

    def to_bitmap(self):
        """Convert polygon masks to bitmap masks."""
        bitmap_masks = self.to_ndarray()
        return BitmapMasks(bitmap_masks, self.height, self.width)

    @property
    def areas(self):
        """Compute areas of masks.

        This func is modified from `detectron2
        <https://github.com/facebookresearch/detectron2/blob/ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9/detectron2/structures/masks.py#L387>`_.
        The function only works with Polygons using the shoelace formula.

        Return:
            ndarray: areas of each instance
        """
        area = []
        for polygons_per_obj in self.masks:
            area_per_obj = 0
            for p in polygons_per_obj:
                area_per_obj += self._polygon_area(p[0::2], p[1::2])
            area.append(area_per_obj)
        return np.asarray(area)

    def _polygon_area(self, x, y):
        """Compute the area of a component of a polygon.

        Using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Args:
            x (ndarray): x coordinates of the component
            y (ndarray): y coordinates of the component

        Return:
            float: the are of the component
        """
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def to_ndarray(self):
        """Convert masks to the format of ndarray."""
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(polygon_to_bitmap(poly_per_obj, self.height, self.width))
        return np.stack(bitmap_masks)

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        if len(self.masks) == 0:
            return torch.empty((0, self.height, self.width), dtype=dtype, device=device)
        ndarray_masks = self.to_ndarray()
        return torch.tensor(ndarray_masks, dtype=dtype, device=device)

    @classmethod
    def random(cls, num_masks=3, height=32, width=32, n_verts=5, dtype=np.float32, rng=None):
        """Generate random polygon masks for demo / testing purposes.

        Adapted from [1]_

        References:
            .. [1] https://gitlab.kitware.com/computer-vision/kwimage/-/blob/928cae35ca8/kwimage/structs/polygon.py#L379  # noqa: E501

        Example:
            >>> from mmdet.data_elements.mask.structures import PolygonMasks
            >>> self = PolygonMasks.random()
            >>> print('self = {}'.format(self))
        """
        from otx.algo.instance_segmentation.mmdet.models.utils.util_random import ensure_rng

        rng = ensure_rng(rng)

        def _gen_polygon(n, irregularity, spikeyness):
            """Creates the polygon by sampling points on a circle around the
            centre.  Random noise is added by varying the angular spacing
            between sequential points, and by varying the radial distance of
            each point from the centre.

            Based on original code by Mike Ounsworth

            Args:
                n (int): number of vertices
                irregularity (float): [0,1] indicating how much variance there
                    is in the angular spacing of vertices. [0,1] will map to
                    [0, 2pi/numberOfVerts]
                spikeyness (float): [0,1] indicating how much variance there is
                    in each vertex from the circle of radius aveRadius. [0,1]
                    will map to [0, aveRadius]

            Returns:
                a list of vertices, in CCW order.
            """
            from scipy.stats import truncnorm

            # Generate around the unit circle
            cx, cy = (0.0, 0.0)
            radius = 1

            tau = np.pi * 2

            irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / n
            spikeyness = np.clip(spikeyness, 1e-9, 1)

            # generate n angle steps
            lower = (tau / n) - irregularity
            upper = (tau / n) + irregularity
            angle_steps = rng.uniform(lower, upper, n)

            # normalize the steps so that point 0 and point n+1 are the same
            k = angle_steps.sum() / (2 * np.pi)
            angles = (angle_steps / k).cumsum() + rng.uniform(0, tau)

            # Convert high and low values to be wrt the standard normal range
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            low = 0
            high = 2 * radius
            mean = radius
            std = spikeyness
            a = (low - mean) / std
            b = (high - mean) / std
            tnorm = truncnorm(a=a, b=b, loc=mean, scale=std)

            # now generate the points
            radii = tnorm.rvs(n, random_state=rng)
            x_pts = cx + radii * np.cos(angles)
            y_pts = cy + radii * np.sin(angles)

            points = np.hstack([x_pts[:, None], y_pts[:, None]])

            # Scale to 0-1 space
            points = points - points.min(axis=0)
            points = points / points.max(axis=0)

            # Randomly place within 0-1 space
            points = points * (rng.rand() * 0.8 + 0.2)
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)

            high = 1 - max_pt
            low = 0 - min_pt
            offset = (rng.rand(2) * (high - low)) + low
            points = points + offset
            return points

        def _order_vertices(verts):
            """References:
            https://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
            """
            mlat = verts.T[0].sum() / len(verts)
            mlng = verts.T[1].sum() / len(verts)

            tau = np.pi * 2
            angle = (np.arctan2(mlat - verts.T[0], verts.T[1] - mlng) + tau) % tau
            sortx = angle.argsort()
            verts = verts.take(sortx, axis=0)
            return verts

        # Generate a random exterior for each requested mask
        masks = []
        for _ in range(num_masks):
            exterior = _order_vertices(_gen_polygon(n_verts, 0.9, 0.9))
            exterior = (exterior * [(width, height)]).astype(dtype)
            masks.append([exterior.ravel()])

        self = cls(masks, height, width)
        return self

    @classmethod
    def cat(cls: type[PolygonMasks], masks: Sequence[PolygonMasks]) -> PolygonMasks:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[PolygonMasks]): A sequence of mask instances.

        Returns:
            PolygonMasks: Concatenated mask instance.
        """
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError("masks should not be an empty list.")
        assert all(isinstance(m, cls) for m in masks)

        mask_list = list(itertools.chain(*[m.masks for m in masks]))
        return cls(mask_list, masks[0].height, masks[0].width)


def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(bool)
    return bitmap_mask


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole
