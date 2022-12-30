"""Convert dataset: Public dataset (Jester[RawFrames], AVA) --> Datumaro dataset (CVAT).

It contains lots of hard-coded to make .xml file consumed on Datumaro.

Current Datumaro format for video (CVAT)

root
|- video_0
    |- images
        |- frames_001.png
        |- frames_002.png
    |- annotations.xml
|- video_1
    |- images
    |- annotations.xml
|- video_2

"""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, Dict, List, Optional, Sequence
import os.path as osp
import json
import numpy as np
import argparse
from pycocotools.coco import COCO
from otx.api.utils.argument_checks import (
    JsonFilePathCheck,
    OptionalDirectoryPathCheck,
    check_input_parameters_type,
)

# pylint: disable=too-many-instance-attributes, too-many-arguments
@check_input_parameters_type({"path": JsonFilePathCheck})
def get_classes_from_annotation(path):
    """Return classes from annotation."""
    with open(path, encoding="UTF-8") as read_file:
        content = json.load(read_file)
        categories = [v["name"] for v in sorted(content["categories"], key=lambda x: x["id"])]
    return categories

class LoadAnnotations:
    """Load Annotations class."""

    @check_input_parameters_type()
    def __init__(self, with_bbox: bool = True, with_label: bool = True, with_mask: bool = False):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask

    @staticmethod
    def _load_bboxes(results):
        ann_info = results["ann_info"]
        results["gt_bboxes"] = ann_info["bboxes"].copy()

        gt_bboxes_ignore = ann_info.get("bboxes_ignore", None)
        if gt_bboxes_ignore is not None:
            results["gt_bboxes_ignore"] = gt_bboxes_ignore.copy()
            results["bbox_fields"].append("gt_bboxes_ignore")
        results["bbox_fields"].append("gt_bboxes")
        return results

    @staticmethod
    def _load_labels(results):
        results["gt_labels"] = results["ann_info"]["labels"].copy()
        return results

    @staticmethod
    def _load_masks(results):
        gt_masks = results["ann_info"]["masks"]
        results["gt_masks"] = gt_masks
        results["mask_fields"].append("gt_masks")
        return results

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadAnnotations."""
        if self.with_bbox:
            results = LoadAnnotations._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = LoadAnnotations._load_labels(results)
        if self.with_mask:
            results = LoadAnnotations._load_masks(results)

        return results

    def __repr__(self):
        """String function of LoadAnnotations."""
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label})"
        return repr_str

class CocoDataset:
    """CocoDataset."""

    @check_input_parameters_type({"ann_file": JsonFilePathCheck, "data_root": OptionalDirectoryPathCheck})
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
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        self.data_infos = self.load_annotations(self.ann_file)

        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

    def __len__(self):
        """Length of CocoDataset."""
        return len(self.data_infos)

    @check_input_parameters_type()
    def pre_pipeline(self, results: Dict[str, Any]):
        """Initialize pipeline."""
        results["img_prefix"] = self.img_prefix
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []

    def _rand_another(self, idx):
        """Get Random indices."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    @check_input_parameters_type()
    def __getitem__(self, idx: int):
        """Return dataset item from index."""
        return self.prepare_img(idx)

    def __iter__(self):
        """Iterator of CocoDataset."""
        for i in range(len(self)):
            yield self[i]

    @check_input_parameters_type()
    def prepare_img(self, idx: int):
        """Load Annotations function with images."""
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return LoadAnnotations(with_mask=self.with_mask)(results)

    @check_input_parameters_type()
    def get_classes(self, classes: Optional[Sequence[str]] = None):
        """Return classes function."""
        if classes is None:
            return get_classes_from_annotation(self.ann_file)

        if isinstance(classes, (tuple, list)):
            return classes

        raise ValueError(f"Unsupported type {type(classes)} of classes.")

    @check_input_parameters_type({"ann_file": JsonFilePathCheck})
    def load_annotations(self, ann_file):
        """Load annotations function from coco."""
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
        """Getting Annotation info."""
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    @check_input_parameters_type()
    def get_cat_ids(self, idx: int):
        """Getting cat_ids."""
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

    def _parse_ann_info(self, img_info, ann_info):  # pylint: disable=too-many-locals, too-many-branches
        """Parse annotation info."""
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for ann in ann_info:
            if ann.get("ignore", False):
                continue
            x1, y1, width, height = ann["bbox"]
            inter_w = max(0, min(x1 + width, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + height, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or width < 1 or height < 1:
                continue
            if self.min_size is not None:
                if width < self.min_size or height < self.min_size:
                    continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + width, y1 + height]
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


multilabel_ann_format = {
    "info": {},
    "categories": {
        "label":{
            "label_groups" : [],
            "labels": [],
            "attributes": [],
        }
    },
    "items":[]
}

def coco_to_datumaro_multilabel(
    ann_file_path: str, 
    data_root_dir: str, 
    output: str
):
    coco_dataset = CocoDataset(
        ann_file=ann_file_path,
        data_root=data_root_dir,
        classes=None,
        test_mode=False,
        with_mask=False,
    )
    
    overall_classes = coco_dataset.get_classes()
    for cl in overall_classes:
        multilabel_ann_format["categories"]["label"]["label_groups"].append(
            {
                "name": cl,
                "group_type": "exclusive",
                "labels": [cl]
            }
        )
    print(overall_classes)
    for item in coco_dataset:
        filename = item["img_info"]["filename"]
        file_id = filename.split('.')[0]
        labels = item["gt_labels"] 

        print(filename, labels)
        raise
    pass

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ann_file_path", required=True, type=str)
    parser.add_argument("--data_root_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--data_format", type=str, default='coco')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if args.data_format == "coco":
        coco_to_datumaro_multilabel(args.ann_file_path, args.data_root_dir, args.output)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()