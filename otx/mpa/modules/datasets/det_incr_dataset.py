# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmdet.datasets import DATASETS, CocoDataset

from otx.mpa.modules.utils.task_adapt import map_cat_and_cls_as_order


@DATASETS.register_module()
class DetIncrCocoDataset(CocoDataset):
    """COCO dataset w/ pseudo label augmentation"""

    def __init__(self, img_ids_dict, **kwargs):

        # Build org dataset
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop("org_type", None)
        # Load pre-stage results
        self.img_ids_dict = img_ids_dict
        self.img_incr_ids = self.img_ids_dict["img_ids"]
        self.img_ids_old = self.img_ids_dict.get("img_ids_old", None)
        self.img_ids_new = self.img_ids_dict.get("img_ids_new", None)
        self.src_classes = self.img_ids_dict["old_classes"]
        self.new_classes = self.img_ids_dict["new_classes"]
        dataset_cfg["classes"] = self.src_classes + self.new_classes
        super().__init__(**dataset_cfg)
        self.cat2label, self.cat_ids = map_cat_and_cls_as_order(self.CLASSES, self.coco.cats)
        self.old_cat_ids = self.cat_ids[0 : len(self.src_classes)]
        print(f"SamplingIncrDataset!!: {self.CLASSES}")
        self.img_indices = dict(old=[], new=[])

        self.statistics()

    def statistics(self):
        num_old_labels = 0
        num_new_labels = 0
        num_src_classes = len(self.src_classes)
        for idx in range(len(self.img_incr_ids)):
            ann_info = self.get_ann_info(idx)
            flag = False
            for label in ann_info["labels"]:
                if label < num_src_classes:
                    num_old_labels += 1
                else:
                    num_new_labels += 1
                    flag = True
            if flag:
                self.img_indices["new"].append(idx)
            else:
                self.img_indices["old"].append(idx)

        print("incr learning stat")
        print(f"- # images: {len(self.img_incr_ids)}")
        print(f"- # old bboxes: {num_old_labels}")
        print(f"- # new bboxes: {num_new_labels}")

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]["id"]
        if img_id in self.img_ids_old:
            cat_ids = self.old_cat_ids
        else:
            cat_ids = self.cat_ids

        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        ann_info = self._parse_ann_info(self.data_infos[idx], ann_info, cat_ids)
        return ann_info

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            # To load selected images
            if self.filter_empty_gt and img_id not in self.img_incr_ids:
                continue
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info, cat_ids):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if self.min_size is not None:
                if w < self.min_size or h < self.min_size:
                    continue
            # to load annotations of selected classes for each image.
            if ann["category_id"] not in cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info["filename"].replace("jpg", "png")

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann, seg_map=seg_map
        )

        return ann
