"""The original source code is from mmdet.mask.structures. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import cv2

# TODO(Eugene): remove mmcv
# https://github.com/openvinotoolkit/training_extensions/pull/3281
import mmcv
import numpy as np
import pycocotools.mask as mask_utils
import torch

# TODO(Eugene): remove mmcv
# https://github.com/openvinotoolkit/training_extensions/pull/3281
from mmcv.ops.roi_align import roi_align
from shapely import geometry


class BitmapMasks:
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
            if not isinstance(masks, (list, np.ndarray)):
                msg = f"masks must be a list or ndarray, but got {type(masks)}"
                raise TypeError(msg)
            if isinstance(masks, list):
                if not isinstance(masks[0], np.ndarray):
                    msg = f"masks[0] must be a ndarray, but got {type(masks[0])}"
                    raise TypeError(msg)
                if masks[0].ndim != 2:
                    msg = f"Each mask should be a 2D array, but got {masks[0].ndim}"
                    raise ValueError(msg)
            elif masks.ndim != 3:
                msg = f"masks should be a 3D array, but got {masks.ndim}"
                raise ValueError(msg)

            self.masks = np.stack(masks).reshape(-1, height, width)
            if self.masks.shape[1] != self.height:
                msg = f"height mismatch: {self.masks.shape[1]} vs {self.height}"
                raise ValueError(msg)
            if self.masks.shape[2] != self.width:
                msg = f"width mismatch: {self.masks.shape[2]} vs {self.width}"
                raise ValueError(msg)

    def __getitem__(self, index: int | np.ndarray) -> BitmapMasks:
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self) -> Iterable:
        return iter(self.masks)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self) -> int:
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale: float | int | tuple[int, int], interpolation: str = "nearest") -> BitmapMasks:
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack([mmcv.imrescale(mask, scale, interpolation=interpolation) for mask in self.masks])
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape: tuple[int, int], interpolation: str = "nearest") -> BitmapMasks:
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack(
                [mmcv.imresize(mask, out_shape[::-1], interpolation=interpolation) for mask in self.masks],
            )
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction: str = "horizontal") -> BitmapMasks:
        """See :func:`BaseInstanceMasks.flip`."""
        if flip_direction not in ("horizontal", "vertical", "diagonal"):
            msg = f"Invalid flip_direction {flip_direction}"
            raise ValueError(msg)

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([mmcv.imflip(mask, direction=flip_direction) for mask in self.masks])
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape: tuple[int, int], pad_val: int = 0) -> BitmapMasks:
        """See :func:`BaseInstanceMasks.pad`."""
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack([mmcv.impad(mask, shape=out_shape, pad_val=pad_val) for mask in self.masks])
        return BitmapMasks(padded_masks, *out_shape)

    def crop(self, bbox: np.ndarray) -> BitmapMasks:
        """See :func:`BaseInstanceMasks.crop`."""
        if not isinstance(bbox, np.ndarray):
            msg = f"bbox must be a ndarray, but got {type(bbox)}"
            raise TypeError(msg)
        if bbox.ndim != 1:
            msg = f"bbox must be 1D array, but got {bbox.ndim}D"
            raise ValueError(msg)

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

    def crop_and_resize(
        self,
        bboxes: np.ndarray,
        out_shape: tuple[int, int],
        inds: np.ndarray,
        device: str = "cpu",
        interpolation: str = "bilinear",
        binarize: bool = True,
    ) -> BitmapMasks:
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
            resized_masks = (targets >= 0.5).cpu().numpy() if binarize else targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h: int, expanded_w: int, top: int, left: int) -> BitmapMasks:
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w), dtype=np.uint8)
        else:
            expanded_mask = np.zeros((len(self), expanded_h, expanded_w), dtype=np.uint8)
            expanded_mask[:, top : top + self.height, left : left + self.width] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(
        self,
        out_shape: tuple[int, int],
        offset: int,
        direction: str = "horizontal",
        border_value: int = 0,
        interpolation: str = "bilinear",
    ) -> BitmapMasks:
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

    def shear(
        self,
        out_shape: tuple[int, int],
        magnitude: int | float,
        direction: str = "horizontal",
        border_value: int = 0,
        interpolation: str = "bilinear",
    ) -> BitmapMasks:
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

    def rotate(
        self,
        out_shape: tuple[int, int],
        angle: int | float,
        center: tuple[float, float] | None = None,
        scale: float = 1.0,
        border_value: int = 0,
        interpolation: str = "bilinear",
    ) -> BitmapMasks:
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
    def areas(self) -> np.ndarray:
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self) -> np.ndarray:
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return torch.tensor(self.masks, dtype=dtype, device=device)

    @classmethod
    def cat(cls: type[BitmapMasks], masks: Sequence[BitmapMasks]) -> BitmapMasks:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[BitmapMasks]): A sequence of mask instances.

        Returns:
            BitmapMasks: Concatenated mask instance.
        """
        if not isinstance(masks, Sequence):
            msg = f"masks must be a sequence, but got {type(masks)}"
            raise TypeError(msg)
        if len(masks) == 0:
            msg = "masks should not be an empty list."
            raise ValueError(msg)
        if not all(isinstance(m, cls) for m in masks):
            msg = "All masks should be BitmapMasks instances."
            raise TypeError(msg)

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
            compose the object, the third level to the poly PolygonMasks
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

    def __init__(self, masks: list, height: int, width: int) -> None:
        if not isinstance(masks, list):
            msg = f"masks must be a list, but got {type(masks)}"
            raise TypeError(msg)
        if len(masks) > 0:
            if not isinstance(masks[0], list):
                msg = f"masks[0] must be a list, but got {type(masks[0])}"
                raise TypeError(msg)

            if not isinstance(masks[0][0], np.ndarray):
                msg = f"masks[0][0] must be a ndarray, but got {type(masks[0][0])}"
                raise TypeError(msg)

        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index: int | np.ndarray) -> PolygonMasks:
        """Index the polygon masks.

        Args:
            index (ndarray | List): The indices.

        Returns:
            :obj:`PolygonMasks`: The indexed polygon masks.
        """
        if isinstance(index, np.ndarray):
            index = np.where(index)[0].tolist() if index.dtype == bool else index.tolist()

        masks = [self.masks[i] for i in index] if isinstance(index, list) else self.masks[index]

        if len(masks) and isinstance(masks[0], np.ndarray):
            masks = [masks]  # ensure a list of three levels
        return PolygonMasks(masks, self.height, self.width)

    def __iter__(self) -> Iterable:
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

    def rescale(self, scale: float | int | tuple[int, int], interpolation: str | None = None) -> PolygonMasks:
        """See :func:`BaseInstanceMasks.rescale`."""
        new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
        return PolygonMasks([], new_h, new_w) if len(self.masks) == 0 else self.resize((new_h, new_w))

    def resize(self, out_shape: tuple[int, int], interpolation: str | None = None) -> PolygonMasks:
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            _resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks: list[np.ndarray] = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()  # noqa: PLW2901
                    p[0::2] = p[0::2] * w_scale
                    p[1::2] = p[1::2] * h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            _resized_masks = PolygonMasks(resized_masks, *out_shape)
        return _resized_masks

    def flip(self, flip_direction: str = "horizontal") -> PolygonMasks:
        """See :func:`BaseInstanceMasks.flip`."""
        if flip_direction not in ("horizontal", "vertical", "diagonal"):
            msg = f"Invalid flip_direction {flip_direction}"
            raise ValueError(msg)
        if len(self.masks) == 0:
            _flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            flipped_masks: list[list[np.ndarray]] = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()  # noqa: PLW2901
                    if flip_direction == "horizontal":
                        p[0::2] = self.width - p[0::2]
                    elif flip_direction == "vertical":
                        p[1::2] = self.height - p[1::2]
                    else:
                        p[0::2] = self.width - p[0::2]
                        p[1::2] = self.height - p[1::2]
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            _flipped_masks = PolygonMasks(flipped_masks, self.height, self.width)
        return _flipped_masks

    def crop(self, bbox: np.ndarray) -> PolygonMasks:
        """See :func:`BaseInstanceMasks.crop`."""
        if not isinstance(bbox, np.ndarray):
            msg = f"bbox must be a ndarray, but got {type(bbox)}"
            raise TypeError(msg)
        if bbox.ndim != 1:
            msg = f"bbox must be 1-dimensional, but got {bbox.ndim}"
            raise ValueError(msg)

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
            cropped_masks_list = []
            # suppress shapely warnings util it incorporates GEOS>=3.11.2
            # reference: https://github.com/shapely/shapely/issues/1345
            initial_settings = np.seterr()
            np.seterr(invalid="ignore")
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()  # noqa: PLW2901
                    p = geometry.Polygon(p.reshape(-1, 2)).buffer(0.0)  # noqa: PLW2901
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
                cropped_masks_list.append(cropped_poly_per_obj)
            np.seterr(**initial_settings)
            cropped_masks = PolygonMasks(cropped_masks_list, h, w)
        return cropped_masks

    def pad(self, out_shape: tuple[int, int], pad_val: int = 0) -> PolygonMasks:
        """Padding has no effect on polygons`."""
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs) -> PolygonMasks:
        """Expanding has no effect on polygons."""
        raise NotImplementedError

    def crop_and_resize(
        self,
        bboxes: np.ndarray,
        out_shape: tuple[int, int],
        inds: np.ndarray,
        device: str = "cpu",
        interpolation: str = "bilinear",
        binarize: bool = True,
    ) -> PolygonMasks:
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        out_h, out_w = out_shape
        if len(self.masks) == 0:
            return PolygonMasks([], out_h, out_w)

        if not binarize:
            msg = "Polygons are always binary, setting binarize=False is unsupported"
            raise ValueError(msg)

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
                p = p.copy()  # noqa: PLW2901
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

    def translate(
        self,
        out_shape: tuple[int, int],
        offset: int,
        direction: str = "horizontal",
        border_value: int | None = None,
        interpolation: str | None = None,
    ) -> PolygonMasks:
        """Translate the PolygonMasks.

        Example:
            >>> self = PolygonMasks.random(dtype=np.int64)
            >>> out_shape = (self.height, self.width)
            >>> new = self.translate(out_shape, 4., direction='horizontal')
            >>> assert np.all(new.masks[0][0][1::2] == self.masks[0][0][1::2])
            >>> assert np.all(new.masks[0][0][0::2] == self.masks[0][0][0::2] + 4)  # noqa: E501
        """
        if border_value is not None and border_value != 0:
            msg = "border_value is not supported for PolygonMasks"
            raise NotImplementedError(msg)
        if len(self.masks) == 0:
            translated_masks = PolygonMasks([], *out_shape)
        else:
            translated_masks_list = []
            for poly_per_obj in self.masks:
                translated_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()  # noqa: PLW2901
                    if direction == "horizontal":
                        p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                    elif direction == "vertical":
                        p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                    translated_poly_per_obj.append(p)
                translated_masks_list.append(translated_poly_per_obj)
            translated_masks = PolygonMasks(translated_masks_list, *out_shape)
        return translated_masks

    def shear(
        self,
        out_shape: tuple[int, int],
        magnitude: int | float,
        direction: str = "horizontal",
        border_value: int = 0,
        interpolation: str = "bilinear",
    ) -> PolygonMasks:
        """See :func:`BaseInstanceMasks.shear`."""
        if len(self.masks) == 0:
            sheared_masks = PolygonMasks([], *out_shape)
        else:
            sheared_masks_list = []
            if direction == "horizontal":
                shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(np.float32)
            elif direction == "vertical":
                shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
            for poly_per_obj in self.masks:
                sheared_poly = []
                for p in poly_per_obj:
                    p = np.stack([p[0::2], p[1::2]], axis=0)  # noqa: PLW2901
                    new_coords = np.matmul(shear_matrix, p)  # [2, n]
                    new_coords[0, :] = np.clip(new_coords[0, :], 0, out_shape[1])
                    new_coords[1, :] = np.clip(new_coords[1, :], 0, out_shape[0])
                    sheared_poly.append(new_coords.transpose((1, 0)).reshape(-1))
                sheared_masks_list.append(sheared_poly)
            sheared_masks = PolygonMasks(sheared_masks_list, *out_shape)
        return sheared_masks

    def rotate(
        self,
        out_shape: tuple[int, int],
        angle: float,
        center: tuple[float, float] | None = None,
        scale: float = 1.0,
        border_value: int = 0,
        interpolation: str = "bilinear",
    ) -> PolygonMasks:
        """See :func:`BaseInstanceMasks.rotate`."""
        if len(self.masks) == 0:
            return PolygonMasks([], *out_shape)
        rotated_masks = []
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        for poly_per_obj in self.masks:
            rotated_poly = []
            for p in poly_per_obj:
                p = p.copy()  # noqa: PLW2901
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
        return PolygonMasks(rotated_masks, *out_shape)

    def to_bitmap(self) -> BitmapMasks:
        """Convert polygon masks to bitmap masks."""
        bitmap_masks = self.to_ndarray()
        return BitmapMasks(bitmap_masks, self.height, self.width)

    @property
    def areas(self) -> np.ndarray:
        """Compute areas of masks.

        This func is modified from `detectron2
        <https://github.com/facebookresearch/detectron2/blob/ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9/detectron2/structures/masks.py#L387>`_.
        The function only works with Polygons using the shoelace formula.

        Return:
            ndarray: areas of each instance
        """
        area = []
        for polygons_per_obj in self.masks:
            area_per_obj = 0.0
            for p in polygons_per_obj:
                area_per_obj += self._polygon_area(p[0::2], p[1::2])
            area.append(area_per_obj)
        return np.asarray(area)

    def _polygon_area(self, x: np.ndarray, y: np.ndarray) -> float:
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

    def to_ndarray(self) -> np.ndarray:
        """Convert masks to the format of ndarray."""
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = [polygon_to_bitmap(poly_per_obj, self.height, self.width) for poly_per_obj in self.masks]
        return np.stack(bitmap_masks)

    def to_tensor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """See :func:`BaseInstanceMasks.to_tensor`."""
        if len(self.masks) == 0:
            return torch.empty((0, self.height, self.width), dtype=dtype, device=device)
        ndarray_masks = self.to_ndarray()
        return torch.tensor(ndarray_masks, dtype=dtype, device=device)

    @classmethod
    def cat(cls: type[PolygonMasks], masks: Sequence[PolygonMasks]) -> PolygonMasks:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[PolygonMasks]): A sequence of mask instances.

        Returns:
            PolygonMasks: Concatenated mask instance.
        """
        if not isinstance(masks, Sequence):
            msg = f"masks must be a sequence, but got {type(masks)}"
            raise TypeError(msg)
        if len(masks) == 0:
            msg = "masks should not be an empty list."
            raise ValueError(msg)
        if not all(isinstance(m, cls) for m in masks):
            msg = "All masks should be PolygonMasks instances."
            raise TypeError(msg)

        mask_list = list(itertools.chain(*[m.masks for m in masks]))
        return cls(mask_list, masks[0].height, masks[0].width)


def polygon_to_bitmap(polygons: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)
