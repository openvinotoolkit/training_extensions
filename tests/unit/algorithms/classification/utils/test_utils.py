# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
from pathlib import Path

from otx.algorithms.classification.utils.convert_coco_to_multilabel import (
    coco_to_datumaro_multilabel,
    multilabel_ann_format,
)
from otx.algorithms.classification.utils.labels_utils import get_multihead_class_info
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    generate_cls_dataset,
    generate_label_schema,
)


@e2e_pytest_unit
def test_coco_conversion(tmp_dir_path):
    path_to_coco_example_ann_file = (
        Path("tests/assets/car_tree_bug/annotations/instances_train.json").absolute().as_posix()
    )
    path_to_coco_example_image_dir = Path("tests/assets/car_tree_bug/images/train").absolute().as_posix()
    output_path = Path(tmp_dir_path) / "annotations.json"
    coco_to_datumaro_multilabel(
        path_to_coco_example_ann_file, path_to_coco_example_image_dir, output_path, test_mode=True
    )
    assert Path.exists(output_path)
    with open(output_path, "r") as f:
        coco_ann = json.load(f)
        for key in multilabel_ann_format:
            assert key in coco_ann.keys()
        assert len(coco_ann["items"]) > 0


@e2e_pytest_unit
def test_get_multihead_class_info():
    hierarchical_dataset = generate_cls_dataset(hierarchical=True)
    label_schema = generate_label_schema(hierarchical_dataset.get_labels(), multilabel=False, hierarchical=True)
    class_info = get_multihead_class_info(label_schema)
    assert (
        len(class_info["label_to_idx"])
        == len(class_info["class_to_group_idx"])
        == len(hierarchical_dataset.get_labels())
    )
