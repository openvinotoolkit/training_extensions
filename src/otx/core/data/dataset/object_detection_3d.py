# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX3DObjectDetectionDataset."""

# mypy: ignore-errors

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Union

import numpy as np
import torch
from datumaro import Image
from otx.core.data.dataset.utils.kitti_utils import Calibration, affine_transform, angle2class, get_affine_transform
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.data.transform_libs.torchvision import Compose
from otx.core.types.image import ImageColorChannel
from PIL import Image as PILImage
from torchvision import tv_tensors

from .base import OTXDataset

if TYPE_CHECKING:
    from datumaro import Bbox, DatasetSubset


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
        max_objects: int = 50,
        depth_threshold: int = 65,
        resolution: tuple[int, int] = (1280, 384),  # (W, H)
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
        self.resolution = np.array(resolution)  # TODO(Kirill): make it configurable
        self.subset_type = list(self.dm_subset.get_subset_info())[-1].split(":")[0]

    def _get_item_impl(self, index: int) -> Det3DDataEntity | None:
        entity = self.dm_subset[index]
        image = entity.media_as(Image)
        image = self._get_img_data_and_shape(image)[0]
        calib = Calibration(entity.attributes["calib_path"])
        original_kitti_format = None  # don't use for training
        if self.subset_type != "train":
            annotations_copy = deepcopy(entity.annotations)
            original_kitti_format = [obj.attributes for obj in annotations_copy]
            # decode original kitti format for metric calculation
            for i, anno_dict in enumerate(original_kitti_format):
                anno_dict["name"] = self.label_info.label_names[annotations_copy[i].label]
                anno_dict["bbox"] = annotations_copy[i].points
                dimension = anno_dict["dimensions"]
                anno_dict["dimensions"] = [dimension[2], dimension[0], dimension[1]]
            original_kitti_format = self._reformate_for_kitti_metric(original_kitti_format)
        # decode labels for training
        inputs, targets, ori_img_shape = self._decode_item(
            PILImage.fromarray(image),
            entity.annotations,
            calib,
        )
        # normilize image
        inputs = self._apply_transforms(torch.as_tensor(inputs, dtype=torch.float32))
        return Det3DDataEntity(
            image=inputs,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=inputs.shape[1:],
                ori_shape=ori_img_shape,  # TODO(Kirill): curently we use WxH here, make it HxW
                image_color_channel=self.image_color_channel,
                ignored_labels=[],
            ),
            boxes=tv_tensors.BoundingBoxes(
                targets["boxes"],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=inputs.shape[1:],
                dtype=torch.float32,
            ),
            labels=torch.as_tensor(targets["labels"], dtype=torch.long),
            calib_matrix=torch.as_tensor(calib.P2, dtype=torch.float32),
            boxes_3d=torch.as_tensor(targets["boxes_3d"], dtype=torch.float32),
            size_2d=torch.as_tensor(targets["size_2d"], dtype=torch.float32),
            size_3d=torch.as_tensor(targets["size_3d"], dtype=torch.float32),
            depth=torch.as_tensor(targets["depth"], dtype=torch.float32),
            heading_angle=torch.as_tensor(
                np.concatenate([targets["heading_bin"], targets["heading_res"]], axis=1),
                dtype=torch.float32,
            ),
            original_kitti_format=original_kitti_format,
        )

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return partial(Det3DBatchDataEntity.collate_fn, stack_images=self.stack_images)

    def _decode_item(self, img: PILImage, annotations: list[Bbox], calib: Calibration) -> tuple:  # noqa: C901
        """Decode item for training."""
        # data augmentation for image
        img_size = np.array(img.size)
        bbox2d = np.array([ann.points for ann in annotations])
        center = img_size / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag = False
        # TODO(Kirill): add data augmentation for 3d, remove them from here.
        if self.subset_type == "train":
            if np.random.random() < 0.5:
                random_flip_flag = True
                img = img.transpose(PILImage.FLIP_LEFT_RIGHT)

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
            method=PILImage.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=PILImage.BILINEAR,
        )
        img = np.array(img).astype(np.float32)
        img = img.transpose(2, 0, 1)  # C * H * W -> (384 * 1280)
        #  ============================   get labels   ==============================
        # data augmentation for labels
        annotations_list: list[dict[str, Any]] = [ann.attributes for ann in annotations]
        for i, obj in enumerate(annotations_list):
            obj["label"] = annotations[i].label
            obj["location"] = np.array(obj["location"])

        if random_flip_flag:
            for i in range(bbox2d.shape[0]):
                [x1, _, x2, _] = bbox2d[i]
                bbox2d[i][0], bbox2d[i][2] = img_size[0] - x2, img_size[0] - x1
                annotations_list[i]["alpha"] = np.pi - annotations_list[i]["alpha"]
                annotations_list[i]["rotation_y"] = np.pi - annotations_list[i]["rotation_y"]
                if annotations_list[i]["alpha"] > np.pi:
                    annotations_list[i]["alpha"] -= 2 * np.pi  # check range
                if annotations_list[i]["alpha"] < -np.pi:
                    annotations_list[i]["alpha"] += 2 * np.pi
                if annotations_list[i]["rotation_y"] > np.pi:
                    annotations_list[i]["rotation_y"] -= 2 * np.pi
                if annotations_list[i]["rotation_y"] < -np.pi:
                    annotations_list[i]["rotation_y"] += 2 * np.pi

        # labels encoding
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
            cur_obj = annotations_list[i]
            # ignore the samples beyond the threshold [hard encoding]
            if cur_obj["location"][-1] > self.depth_threshold and cur_obj["location"][-1] < 2:
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

            center_3d = np.array(
                cur_obj["location"]
                + [
                    0,
                    -cur_obj["dimensions"][0] / 2,
                    0,
                ],
            )  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if proj_inside_img:
                continue

            # class
            labels[i] = cur_obj["label"]

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1.0 * w, 1.0 * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0:2] = corner_2d[0:2] / self.resolution
            corner_2d_norm[2:4] = corner_2d[2:4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            k, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if k < 0 or r < 0 or t < 0 or b < 0:
                continue

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], k, r, t, b

            # encoding depth
            depth[i] = cur_obj["location"][-1] * crop_scale

            # encoding heading angle
            heading_angle = calib.ry2alpha(cur_obj["rotation_y"], (bbox2d[i][0] + bbox2d[i][2]) / 2)
            if heading_angle > np.pi:
                heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi:
                heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([cur_obj["dimensions"]], dtype=np.float32)
            size_3d[i] = src_size_3d[i]

            # filter out the samples with truncated or occluded
            if cur_obj["truncated"] <= 0.5 and cur_obj["occluded"] <= 2:
                mask_2d[i] = 1

        # collect return data
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

        return img, targets_for_train, img_size

    def _reformate_for_kitti_metric(self, annotations: dict[str, Any]) -> dict[str, np.array]:
        """Reformat the annotation for KITTI metric."""
        return {
            "name": np.array([obj["name"] for obj in annotations]),
            "alpha": np.array([obj["alpha"] for obj in annotations]),
            "bbox": np.array([obj["bbox"] for obj in annotations]).reshape(-1, 4),
            "dimensions": np.array([obj["dimensions"] for obj in annotations]).reshape(-1, 3),
            "location": np.array([obj["location"] for obj in annotations]).reshape(-1, 3),
            "rotation_y": np.array([obj["rotation_y"] for obj in annotations]),
            "occluded": np.array([obj["occluded"] for obj in annotations]),
            "truncated": np.array([obj["truncated"] for obj in annotations]),
        }
