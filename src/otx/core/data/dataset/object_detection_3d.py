# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX3DObjectDetectionDataset."""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Union, TYPE_CHECKING

import numpy as np
import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.data.transform_libs.torchvision import Compose
from otx.core.types.image import ImageColorChannel
from otx.core.types.label import LabelInfo
from torchvision import tv_tensors
from otx.core.data.dataset.kitti_3d.kitti_utils import Calibration, angle2class, affine_transform, get_affine_transform
from datumaro import Bbox, Image

from .base import OTXDataset

if TYPE_CHECKING:
    from datumaro import DatasetSubset


Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


class OTX3DObjectDetectionDataset(OTXDataset[Det3DDataEntity]):
    """OTXDataset class for detection task."""
    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        max_objects = 50,
        depth_threshold = 65,
        resolution = (1280, 384), # (W, H)
    ) -> None:
        super().__init__(
            dm_subset,
            transforms,
            mem_cache_handler,
            mem_cache_img_max_size,
            max_refetch,
            image_color_channel,
            stack_images,
            to_tv_image,
        )
        self.max_objects = max_objects
        self.depth_threshold = depth_threshold
        self.resolution = np.array(resolution)

    def _get_item_impl(self, index: int) -> Det3DDataEntity | None:
        entity = self.dm_subset[index]
        inputs = entity.media_as(Image)
        inputs, img_shape = self._get_img_data_and_shape(inputs)
        calib = Calibration(entity.attributes["calib_path"])
        original_kitti_format, targets, calib_matrix, resized_img_shape = self._decode_item(inputs, img_shape, annotations, calib)
        entity = Det3DDataEntity(
            image=torch.tensor(inputs),
            img_info=ImageInfo(
                img_idx=index,
                img_shape=resized_img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=[],
            ),
            boxes=tv_tensors.BoundingBoxes(
                targets["boxes"],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=resized_img_shape,
                dtype=torch.float32,
            ),
            labels=torch.as_tensor(targets["labels"], dtype=torch.long),
            calib_matrix=torch.as_tensor(calib_matrix, dtype=torch.float32),
            boxes_3d=torch.as_tensor(targets["boxes_3d"], dtype=torch.float32),
            size_2d=torch.as_tensor(targets["size_2d"], dtype=torch.float32),
            size_3d=torch.as_tensor(targets["size_3d"], dtype=torch.float32),
            depth=torch.as_tensor(targets["depth"], dtype=torch.float32),
            heading_angle=torch.as_tensor(
                np.concatenate([targets["heading_bin"], targets["heading_res"]], axis=1),
                dtype=torch.float32,
            ),
            kitti_label_object=original_kitti_format,
        )

        return entity

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return partial(Det3DBatchDataEntity.collate_fn, stack_images=self.stack_images)


    def _decode_item(self, img, img_size, annotations, calib):
        #  ============================   get inputs   ===========================
        # data augmentation for image
        bbox2d = np.array([ann.points for ann in annotations])
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag = False
        if self.transforms:
            if np.random.random() < 0.5:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < 0.5:
                scale = 0.05
                shift = 0.05
                crop_scale = np.clip(np.random.randn() * scale + 1, 1 - scale, 1 + scale)
                crop_size = img_size * crop_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * shift, -2 * shift, 2 * shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * shift, -2 * shift, 2 * shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(
            tuple(self.resolution.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR,
        )

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W -> (384 * 1280)

        #  ============================   get labels   ==============================
        # data augmentation for labels
        if random_flip_flag:
            for annotation, box in zip(annotations, bbox2d):
                [x1, _, x2, _] = box
                box[0], box[2] = img_size[0] - x2, img_size[0] - x1
                annotation["alpha"] = np.pi - annotation["alpha"]
                annotation["rotation_y"] = np.pi - annotation["rotation_y"]
                if annotation["alpha"] > np.pi:
                    annotation["alpha"] -= 2 * np.pi  # check range
                if annotation["alpha"] < -np.pi:
                    annotation["alpha"] += 2 * np.pi
                if annotation["rotation_y"] > np.pi:
                    annotation["rotation_y"] -= 2 * np.pi
                if annotation["rotation_y"] < -np.pi:
                    annotation["rotation_y"] += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objects, 3, 4), dtype=np.float32)
        mask_2d = np.zeros((self.max_objects), dtype=bool)
        labels = np.zeros((self.max_objects), dtype=np.int8)
        depth = np.zeros((self.max_objects, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objects, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objects, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objects, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objects, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objects, 6), dtype=np.float32)

        object_num = len(annotations) if len(annotations) < self.max_objects else self.max_objects

        for i in range(object_num):
            # ignore the samples beyond the threshold [hard encoding]
            if annotations["location"][-1] > self.depth_threshold and annotations["location"][-1] < 2:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = bbox2d[i].copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array(
                [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                dtype=np.float32,
            )  # W * H
            corner_2d = bbox_2d.copy()

            center_3d = annotations["location"] + [0, -annotations[i]["dimensions"][0] / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if proj_inside_img == False:
                continue

            # class
            cls_id = annotations[i]["label"]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1.0 * w, 1.0 * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0:2] = corner_2d[0:2] / self.resolution
            corner_2d_norm[2:4] = corner_2d[2:4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                continue

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            depth[i] = annotations["location"][-1] * crop_scale

            # encoding heading angle
            heading_angle = calib.ry2alpha(annotations[i]["rotation_y"], (bbox2d[i][0] + bbox2d[i][2]) / 2)
            if heading_angle > np.pi:
                heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi:
                heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([annotations[i]["dimensions"]], dtype=np.float32)
            size_3d[i] = src_size_3d[i]

            # filter out the samples with truncated or occluded
            if annotations[i]["truncated"] <= 0.5 and annotations[i]["occluded"] <= 2:
                mask_2d[i] = 1

            calibs[i] = calib.P2

        # collect return data
        inputs = img
        targets_for_train = {
            "labels": labels[mask_2d],
            "boxes": boxes[mask_2d],
            "boxes_3d": boxes_3d[mask_2d],
            "depth": depth[mask_2d],
            "size_2d": size_2d[mask_2d],
            "size_3d": size_3d[mask_2d],
            "heading_bin": heading_bin[mask_2d],
            "heading_res": heading_res[mask_2d],
        }

        return inputs, calib, targets_for_train
