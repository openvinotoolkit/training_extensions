# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import json
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.polygon import Polygon, Point
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    DirectoryPathCheck,
    OptionalDirectoryPathCheck,
    JsonFilePathCheck,
    check_input_parameters_type,
)
from ote_sdk.utils.shape_factory import ShapeFactory
from pycocotools.coco import COCO

from mmdet.core import BitmapMasks, PolygonMasks

@check_input_parameters_type({"path": JsonFilePathCheck})
def get_classes_from_annotation(path):
    with open(path) as read_file:
        content = json.load(read_file)
        categories = [
            v["name"] for v in sorted(content["categories"], key=lambda x: x["id"])
        ]
    return categories


class LoadAnnotations:
    @check_input_parameters_type()
    def __init__(self, with_bbox: bool = True, with_label: bool = True, with_mask: bool = False):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask

    def _load_bboxes(self, results):
        ann_info = results["ann_info"]
        results["gt_bboxes"] = ann_info["bboxes"].copy()

        gt_bboxes_ignore = ann_info.get("bboxes_ignore", None)
        if gt_bboxes_ignore is not None:
            results["gt_bboxes_ignore"] = gt_bboxes_ignore.copy()
            results["bbox_fields"].append("gt_bboxes_ignore")
        results["bbox_fields"].append("gt_bboxes")
        return results

    def _load_labels(self, results):
        results["gt_labels"] = results["ann_info"]["labels"].copy()
        return results

    def _load_masks(self, results):
        gt_masks = results['ann_info']['masks']
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label})"
        return repr_str


class CocoDataset:
    @check_input_parameters_type({"ann_file": JsonFilePathCheck,
                                  "data_root": OptionalDirectoryPathCheck})
    def __init__(
        self,
        ann_file: str,
        classes: Optional[Sequence[str]] = None,
        data_root: Optional[str] = None,
        img_prefix: str = "",
        test_mode: bool = False,
        filter_empty_gt: bool = True,
        min_size: Optional[int] = None,
        with_mask: bool = False,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.classes = self.get_classes(classes)
        self.min_size = min_size
        self.with_mask = with_mask

        if self.data_root is not None:
            # if not osp.isabs(self.ann_file):
            #     self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        self.data_infos = self.load_annotations(self.ann_file)

        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

    def __len__(self):
        return len(self.data_infos)

    @check_input_parameters_type()
    def pre_pipeline(self, results: Dict[str, Any]):
        results["img_prefix"] = self.img_prefix
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    @check_input_parameters_type()
    def __getitem__(self, idx: int):
        return self.prepare_img(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @check_input_parameters_type()
    def prepare_img(self, idx: int):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return LoadAnnotations(with_mask=self.with_mask)(results)

    @check_input_parameters_type()
    def get_classes(self, classes: Optional[Sequence[str]] = None):
        if classes is None:
            return get_classes_from_annotation(self.ann_file)

        if isinstance(classes, (tuple, list)):
            return classes

        raise ValueError(f"Unsupported type {type(classes)} of classes.")

    @check_input_parameters_type({"ann_file": JsonFilePathCheck})
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.classes)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info["filename"] = info["file_name"]
            data_infos.append(info)
        return data_infos

    @check_input_parameters_type()
    def get_ann_info(self, idx: int):
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    @check_input_parameters_type()
    def get_cat_ids(self, idx: int):
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann["category_id"] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_["image_id"] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for ann in ann_info:
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
            if ann["category_id"] not in self.cat_ids:
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
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
        )

        return ann


@check_input_parameters_type()
def find_label_by_name(labels: Sequence[LabelEntity], name: str, domain: Domain):
    matching_labels = [label for label in labels if label.name == name]
    if len(matching_labels) == 1:
        return matching_labels[0]
    elif len(matching_labels) == 0:
        label = LabelEntity(name=name, domain=domain, id=ID(len(labels)))
        labels.append(label)
        return label
    else:
        raise ValueError("Found multiple matching labels")


@check_input_parameters_type({"ann_file_path": JsonFilePathCheck,
                              "data_root_dir": DirectoryPathCheck})
def load_dataset_items_coco_format(
    ann_file_path: str,
    data_root_dir: str,
    domain: Domain,
    subset: Subset = Subset.NONE,
    labels_list: Optional[List[LabelEntity]] = None,
    with_mask: bool = False,
):
    test_mode = subset in {Subset.VALIDATION, Subset.TESTING}

    coco_dataset = CocoDataset(
        ann_file=ann_file_path,
        data_root=data_root_dir,
        classes=None,
        test_mode=test_mode,
        with_mask=with_mask,
    )
    coco_dataset.test_mode = False
    for label_name in coco_dataset.classes:
        find_label_by_name(labels_list, label_name, domain)

    dataset_items = []
    for item in coco_dataset:

        def create_gt_box(x1, y1, x2, y2, label_name):
            return Annotation(
                Rectangle(x1=x1, y1=y1, x2=x2, y2=y2),
                labels=[ScoredLabel(label=find_label_by_name(labels_list, label_name, domain))],
            )

        def create_gt_polygon(polygon_group, label_name):
            if len(polygon_group) != 1:
                raise RuntimeError("Complex instance segmentation masks consisting of several polygons are not supported.")

            return Annotation(
                Polygon(points=polygon_group[0]),
                labels=[ScoredLabel(label=find_label_by_name(labels_list, label_name, domain))],
            )

        img_height = item["img_info"].get("height")
        img_width = item["img_info"].get("width")
        divisor = np.array(
            [img_width, img_height, img_width, img_height],
            dtype=item["gt_bboxes"].dtype,
        )
        bboxes = item["gt_bboxes"] / divisor
        labels = item["gt_labels"]

        assert len(bboxes) == len(labels)
        if with_mask:
            polygons = item["gt_masks"]
            assert len(bboxes) == len(polygons)
            normalized_polygons = []
            for polygon_group in polygons:
                normalized_polygons.append([])
                for polygon in polygon_group:
                    normalized_polygon = [p / divisor[i % 2] for i, p in enumerate(polygon)]
                    points = [Point(normalized_polygon[i], normalized_polygon[i + 1]) for i in range(0, len(polygon), 2)]
                    normalized_polygons[-1].append(points)

        if item["img_prefix"] is not None:
            filename = osp.join(item["img_prefix"], item["img_info"]["filename"])
        else:
            filename = item["img_info"]["filename"]

        if with_mask:
            shapes = [
                create_gt_polygon(polygon_group, coco_dataset.classes[label_id])
                for polygon_group, label_id in zip(normalized_polygons, labels)
            ]
        else:
            shapes = [
                create_gt_box(x1, y1, x2, y2, coco_dataset.classes[label_id])
                for (x1, y1, x2, y2), label_id in zip(bboxes, labels)
            ]

        dataset_item = DatasetItemEntity(
            media=Image(file_path=filename),
            annotation_scene=AnnotationSceneEntity(
                annotations=shapes, kind=AnnotationSceneKind.ANNOTATION
            ),
            subset=subset,
        )
        dataset_items.append(dataset_item)

    return dataset_items


@check_input_parameters_type({"dataset": DatasetParamTypeCheck})
def get_sizes_from_dataset_entity(dataset: DatasetEntity, target_wh: List[int]):
    """
    Function to get sizes of instances in DatasetEntity and to resize it to the target size.

    :param dataset: DatasetEntity in which to get statistics
    :param target_wh: target width and height of the dataset
    :return list: tuples with width and height of each instance
    """
    wh_stats = []
    for item in dataset:
        for ann in item.get_annotations(include_empty=False):
            has_detection_labels = any(label.domain == Domain.DETECTION for label in ann.get_labels(include_empty=False))
            if has_detection_labels:
                box = ShapeFactory.shape_as_rectangle(ann.shape)
                w = box.width * target_wh[0]
                h = box.height * target_wh[1]
                wh_stats.append((w, h))
    return wh_stats


@check_input_parameters_type()
def get_anchor_boxes(wh_stats: List[tuple], group_as: List[int]):
    from sklearn.cluster import KMeans
    kmeans = KMeans(init='k-means++', n_clusters=sum(group_as), random_state=0).fit(wh_stats)
    centers = kmeans.cluster_centers_

    areas = np.sqrt(np.prod(centers, axis=1))
    idx = np.argsort(areas)

    widths = centers[idx, 0]
    heights = centers[idx, 1]

    group_as = np.cumsum(group_as[:-1])
    widths, heights = np.split(widths, group_as), np.split(heights, group_as)
    return widths, heights


@check_input_parameters_type()
def format_list_to_str(value_lists: list):
    """ Decrease floating point digits in logs """
    str_value = ''
    for value_list in value_lists:
        str_value += '[' + ', '.join(f'{value:.2f}' for value in value_list) + '], '
    return f'[{str_value[:-2]}]'
