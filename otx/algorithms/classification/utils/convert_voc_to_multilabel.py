"""Convert dataset: Public dataset (COCO) --> Multi-label dataset (Datumaro format).

this script contains a lot of hard-coding to create .json file that Datumaro can consume.
"""

import argparse
import json
import os.path as osp

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
from typing import Any, Dict, List

multilabel_ann_format = {
    "info": {},
    "categories": {
        "label": {
            "label_groups": [],
            "labels": [],
            "attributes": [],
        }
    },
    "items": [],
}  # type: Dict[str, Any]


def load_annotation(annot_path, data_dir):
    out_data = []
    with open(annot_path) as f:
        annotation = json.load(f)
        classes = sorted(annotation["classes"])
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        images_info = annotation["images"]
        img_wo_objects = 0
        for img_info in images_info:
            rel_image_path, img_labels = img_info
            full_image_path = osp.join(data_dir, rel_image_path)
            labels_idx = [class_to_idx[lbl] for lbl in img_labels if lbl in class_to_idx]
            assert full_image_path
            if not labels_idx:
                img_wo_objects += 1
            out_data.append((full_image_path, tuple(labels_idx)))
    if img_wo_objects:
        print(f"WARNING: there are {img_wo_objects} images without labels and will be treated as negatives")
    return out_data, class_to_idx, idx_to_class


def coco_to_datumaro_multilabel(ann_file_path: str, data_root_dir: str, output: str, test_mode: bool = False):
    """Convert coco dataset to datumaro multi-label format.

    Args:
        ann_file_path (str): The path of annotation file (COCO)
        data_root_dir (str): The path of images folder (COCO)
        output (str): Destination path of converted data (CVAT multi-label format)
        test_mode (bool): Omit filtering irrelevant images during COCO dataset initialization for testing purposes.
    """

    # Prepare COCO dataset to load annotations
    out_data, class_to_idx, idx_to_class = load_annotation(ann_file_path, data_root_dir)

    # Fill the categories part
    # For the multi-label classification,
    # Datumaro will make label_groups and labels
    overall_classes = list(class_to_idx.keys())  # type: List
    for class_name in overall_classes:
        multilabel_ann_format["categories"]["label"]["label_groups"].append(
            {"name": str(class_name), "group_type": "exclusive", "labels": [str(class_name)]}
        )

        multilabel_ann_format["categories"]["label"]["labels"].append(
            {"name": class_name, "parent": "", "attributes": []}
        )

    # Fill the items part
    for filename, labels in out_data:
        _, short_fname = osp.split(filename)
        file_id = short_fname.split(".")[0]

        annotations = []
        for i, label in enumerate(labels):
            annotations.append({"id": int(i), "type": "label", "group": 0, "label_id": int(label)})

        multilabel_ann_format["items"].append(
            {"id": str(file_id), "annotations": annotations, "image": {"path": str(short_fname)}}
        )
    print(f"Saving logfile to: {output}")
    with open(output, "w", encoding="utf-8") as out_file:
        json.dump(multilabel_ann_format, out_file)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ann_file_path", required=True, type=str)
    parser.add_argument("--data_root_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--data_format", type=str, default="coco")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if args.data_format == "coco":
        coco_to_datumaro_multilabel(args.ann_file_path, args.data_root_dir, args.output)
    else:
        raise ValueError(f"Unknown data format {args.data_format}.This script only support `coco`.")


if __name__ == "__main__":
    main()
