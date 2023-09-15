"""MMdet3.x transforms."""
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
from typing import Dict, List, Sequence, Tuple, TypeVar, Union

import mmcv
import numpy as np
from mmdet.core import PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import Mosaic
from numpy import random

T = TypeVar("T")


def rescale(boxes: np.ndarray, scale_factor: List[float]) -> np.ndarray:
    """Rescale boxes w.r.t. rescale_factor.

    Note:
        Both ``rescale`` and ``resize_`` will enlarge or shrink boxes
        w.r.t ``scale_facotr``. The difference is that ``resize_`` only
        changes the width and the height of boxes, but ``rescale`` also
        rescales the box centers simultaneously.

    Args:
        boxes (np.ndarray): The boxes to be rescaled.
        scale_factor (Tuple[float, float]): factors for scaling boxes.
            The length should be 2.

    Returns:
        np.ndarray: The rescaled boxes.
    """
    scale_factor = np.tile(scale_factor, 2)
    boxes = boxes * scale_factor
    return boxes


def translate(boxes: np.ndarray, distances: List[float]) -> np.ndarray:
    """Translate boxes.

    Args:
        boxes (np.ndarray): The boxes to be rescaled.
        distances (Tuple[float, float]): translate distances. The first
            is horizontal distance and the second is vertical distance.

    Returns:
        np.ndarray: The translated boxes.
    """
    assert len(distances) == 2
    distances = np.tile(distances, 2)
    boxes = boxes + distances
    return boxes


def flip(boxes: np.ndarray, img_shape: List[int], direction: str = "horizontal") -> np.ndarray:
    """Flip boxes horizontally or vertically.

    Args:
        boxes (np.ndarray): The boxes to be rescaled.
        img_shape (Tuple[int, int]): A tuple of image height and width.
        direction (str): Flip direction, options are "horizontal",
            "vertical" and "diagonal". Defaults to "horizontal"

    Returns:
        boxes (np.ndarray): The flipped boxes.
    """
    assert direction in ["horizontal", "vertical", "diagonal"]
    flipped = boxes
    boxes = flipped.copy()
    if direction == "horizontal":
        flipped[..., 0] = img_shape[1] - boxes[..., 2]
        flipped[..., 2] = img_shape[1] - boxes[..., 0]
    elif direction == "vertical":
        flipped[..., 1] = img_shape[0] - boxes[..., 3]
        flipped[..., 3] = img_shape[0] - boxes[..., 1]
    else:
        flipped[..., 0] = img_shape[1] - boxes[..., 2]
        flipped[..., 1] = img_shape[0] - boxes[..., 3]
        flipped[..., 2] = img_shape[1] - boxes[..., 0]
        flipped[..., 3] = img_shape[0] - boxes[..., 1]
    return flipped


def clip(boxes: np.ndarray, img_shape: List[int]) -> np.ndarray:
    """Clip boxes according to the image shape.

    Args:
        boxes (np.ndarray): The boxes to be rescaled.
        img_shape (Tuple[int, int]): A tuple of image height and width.

    Returns:
        boxes (np.ndarray): The clipped boxes.
    """
    boxes[..., 0::2] = boxes[..., 0::2].clip(0, img_shape[1])
    boxes[..., 1::2] = boxes[..., 1::2].clip(0, img_shape[0])
    return boxes


def is_inside(boxes: np.ndarray, img_shape: List[int], all_inside: bool = False, allowed_border: int = 0) -> np.ndarray:
    """Find boxes inside the image.

    Args:
        boxes (np.ndarray): boxes.
        img_shape (Tuple[int, int]): A tuple of image height and width.
        all_inside (bool): Whether the boxes are all inside the image or
            part inside the image. Defaults to False.
        allowed_border (int): Boxes that extend beyond the image shape
            boundary by more than ``allowed_border`` are considered
            "outside" Defaults to 0.


    Returns:
        BoolTensor: A BoolTensor indicating whether the box is inside
        the image. Assuming the original boxes have shape (m, n, 4),
        the output has shape (m, n).
    """
    img_h, img_w = img_shape
    if all_inside:
        return (
            (boxes[:, 0] >= -allowed_border)
            & (boxes[:, 1] >= -allowed_border)
            & (boxes[:, 2] < img_w + allowed_border)
            & (boxes[:, 3] < img_h + allowed_border)
        )
    else:
        return (
            (boxes[..., 0] < img_w + allowed_border)
            & (boxes[..., 1] < img_h + allowed_border)
            & (boxes[..., 2] > -allowed_border)
            & (boxes[..., 3] > -allowed_border)
        )


def cat_mask(masks: Sequence[PolygonMasks]) -> PolygonMasks:
    """Concatenate a sequence of masks into one single mask instance.

    Args:
        masks (Sequence[PolygonMasks]): A sequence of mask instances.

    Returns:
        PolygonMasks: Concatenated mask instance.
    """
    mask_list = list(itertools.chain(*[m.masks for m in masks]))
    return PolygonMasks(mask_list, masks[0].height, masks[0].width)


def polygon__get_item(self, index: Union[np.ndarray, List[float]]):
    """Index the polygon masks.

    Note:
        Overwrite the original function to support indexing with a list of bool.

    Args:
        self (PolygonMasks): The polygon masks.
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


PolygonMasks.__getitem__ = polygon__get_item


@PIPELINES.register_module()
class CachedMosaic(Mosaic):
    """Cached mosaic augmentation.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The cached mosaic transform steps are as follows:

         1. Append the results from the last transform into the cache.
         2. Choose the mosaic center as the intersections of 4 images
         3. Get the left top image according to the index, and randomly
            sample another 3 images from the result cache.
         4. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (np.float32) (optional)
    - gt_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
    """

    def __init__(self, *args, max_cached_images: int = 40, random_pop: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.results_cache: List[Dict] = []
        self.random_pop = random_pop
        assert max_cached_images >= 4, "The length of cache must >= 4, " f"but got {max_cached_images}."
        self.max_cached_images = max_cached_images

    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    def __call__(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return results

        if random.uniform(0, 1) > self.prob:
            return results
        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        # mosaic_ignore_flags = []
        mosaic_masks = []
        with_mask = True if "gt_masks" in results else False

        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3), self.pad_val, dtype=results["img"].dtype
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)), self.pad_val, dtype=results["img"].dtype
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(mix_results[i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i, self.img_scale[0] / w_i)
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_bboxes_labels_i = results_patch["gt_labels"]
            # gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i = rescale(gt_bboxes_i, [scale_ratio_i, scale_ratio_i])
            gt_bboxes_i = translate(gt_bboxes_i, [padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            # mosaic_ignore_flags.append(gt_ignore_flags_i)
            if with_mask and results_patch.get("gt_masks", None) is not None:
                gt_masks_i = results_patch["gt_masks"]
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction="horizontal",
                )
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction="vertical",
                )
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        # mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes = clip(mosaic_bboxes, [2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = is_inside(mosaic_bboxes, [2 * self.img_scale[1], 2 * self.img_scale[0]])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        # mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape[:2]
        results["gt_bboxes"] = mosaic_bboxes.astype(np.float32)  # type: ignore[attr-defined]
        results["gt_labels"] = mosaic_bboxes_labels
        # results['gt_ignore_flags'] = mosaic_ignore_flags

        if with_mask:
            mosaic_masks = cat_mask(mosaic_masks)
            results["gt_masks"] = mosaic_masks[inside_inds]
        return results

    def __repr__(self):
        """Return repr str.

        Returns:
            repr_str: Return the string of repr.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"max_cached_images={self.max_cached_images}, "
        repr_str += f"random_pop={self.random_pop})"
        return repr_str


@PIPELINES.register_module()
class CachedMixUp:
    """Cached mixup data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The cached mixup transform steps are as follows:

        1. Append the results from the last transform into the cache.
        2. Another random image is picked from the cache and embedded in
           the top left patch(after padding and resizing)
        3. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (np.float32) (optional)
    - gt_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        ratio_range: Tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        prob: float = 1.0,
    ) -> None:
        assert isinstance(img_scale, tuple)
        assert max_cached_images >= 2, "The length of cache must >= 2, " f"but got {max_cached_images}."
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]. " f"got {prob}."
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.results_cache: List[Dict] = []

        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = prob

    def get_indexes(self, cache: list) -> int:
        """Call function to collect indexes.

        Args:
            cache (list): The result cache.

        Returns:
            int: index.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(cache) - 1)
            gt_bboxes_i = cache[index]["gt_bboxes"]
            if len(gt_bboxes_i) != 0:
                break
        return index

    def __call__(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return results

        if random.uniform(0, 1) > self.prob:
            return results

        index = self.get_indexes(self.results_cache)
        retrieve_results = copy.deepcopy(self.results_cache[index])

        # TODO: refactor mixup to reuse these code.
        if retrieve_results["gt_bboxes"].shape[0] == 0:
            # empty bbox
            return results

        retrieve_img = retrieve_results["img"]
        with_mask = True if "gt_masks" in results else False

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones((self.dynamic_scale[1], self.dynamic_scale[0], 3), dtype=retrieve_img.dtype) * self.pad_val
            )
        else:
            out_img = np.ones(self.dynamic_scale[::-1], dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[1] / retrieve_img.shape[0], self.dynamic_scale[0] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio), int(retrieve_img.shape[0] * scale_ratio))
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_bboxes = rescale(retrieve_gt_bboxes, [scale_ratio, scale_ratio])
        if with_mask:
            retrieve_gt_masks = retrieve_results["gt_masks"].rescale(scale_ratio)

        if self.bbox_clip_border:
            retrieve_gt_bboxes = clip(retrieve_gt_bboxes, [origin_h, origin_w])

        if is_flip:
            retrieve_gt_bboxes = flip(retrieve_gt_bboxes, [origin_h, origin_w], direction="horizontal")
            if with_mask:
                retrieve_gt_masks = retrieve_gt_masks.flip()

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes = translate(cp_retrieve_gt_bboxes, [-x_offset, -y_offset])
        if with_mask:
            retrieve_gt_masks = retrieve_gt_masks.translate(
                out_shape=(target_h, target_w), offset=-x_offset, direction="horizontal"
            )
            retrieve_gt_masks = retrieve_gt_masks.translate(
                out_shape=(target_h, target_w), offset=-y_offset, direction="vertical"
            )

        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes = clip(cp_retrieve_gt_bboxes, [target_h, target_w])

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results["gt_labels"]
        # retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = np.concatenate((results["gt_bboxes"], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_bboxes_labels = np.concatenate((results["gt_labels"], retrieve_gt_bboxes_labels), axis=0)
        # mixup_gt_ignore_flags = np.concatenate(
        #     (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        if with_mask:
            mixup_gt_masks = cat_mask([results["gt_masks"], retrieve_gt_masks])

        # remove outside bbox
        inside_inds = is_inside(mixup_gt_bboxes, [target_h, target_w])
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
        # mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]
        if with_mask:
            mixup_gt_masks = mixup_gt_masks[inside_inds]

        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape[:2]
        results["gt_bboxes"] = mixup_gt_bboxes.astype(np.float32)
        results["gt_labels"] = mixup_gt_bboxes_labels
        # results['gt_ignore_flags'] = mixup_gt_ignore_flags
        if with_mask:
            results["gt_masks"] = mixup_gt_masks
        return results

    def __repr__(self):
        """Return repr str.

        Returns:
            repr_str: Return the string of repr.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(dynamic_scale={self.dynamic_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_iters={self.max_iters}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border}, "
        repr_str += f"max_cached_images={self.max_cached_images}, "
        repr_str += f"random_pop={self.random_pop}, "
        repr_str += f"prob={self.prob})"
        return repr_str
