# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import tempfile

import pytest

from otx.algorithms.detection.utils import generate_label_schema
from otx.algorithms.detection.utils.data import (
    find_label_by_name,
    format_list_to_str,
    get_anchor_boxes,
    get_sizes_from_dataset_entity,
    load_dataset_items_coco_format,
)
from otx.api.entities.label import Domain
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from otx.api.entities.subset import Subset
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    create_dummy_coco_json,
    generate_det_dataset,
)


@e2e_pytest_unit
@pytest.mark.parametrize("name", ["rectangle", "something"])
def test_find_label_by_name(name):
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))
    out = find_label_by_name(label_schema.get_labels(include_empty=False), name, Domain.DETECTION)
    assert out.name == name


@e2e_pytest_unit
def test_find_label_by_name_error():
    classes = ("rectangle", "rectangle", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))
    with pytest.raises(ValueError):
        find_label_by_name(label_schema.get_labels(include_empty=False), "rectangle", Domain.DETECTION)


@e2e_pytest_unit
@pytest.mark.parametrize(
    "task_type, domain",
    [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
)
def test_load_dataset_items_coco_format(task_type, domain):
    _, labels = generate_det_dataset(task_type=task_type)
    tmp_dir = tempfile.TemporaryDirectory()
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_json_file = os.path.join(tmp_dir, "fake_data.json")
        create_dummy_coco_json(fake_json_file)
        data_root_dir = "./some_data_root_dir"
        with_mask = True if domain == Domain.INSTANCE_SEGMENTATION else False
        out = load_dataset_items_coco_format(
            fake_json_file,
            data_root_dir,
            subset=Subset.TRAINING,
            domain=domain,
            with_mask=with_mask,
            labels_list=labels,
        )
    assert out is not None


@e2e_pytest_unit
def test_get_sizes_from_dataset_entity():
    dataset, _ = generate_det_dataset(task_type=TaskType.DETECTION)
    out = get_sizes_from_dataset_entity(dataset, [480, 640])
    assert out is not None


@e2e_pytest_unit
def test_get_anchor_boxes():
    out = get_anchor_boxes([(100, 120), (100, 120)], [1, 1])
    expected_out = ([[100.0], [100.0]], [[120.0], [120.0]])
    assert out == expected_out


@e2e_pytest_unit
def test_format_list_to_str():
    out = format_list_to_str([[0.1839128319, 0.47398123]])
    expected_out = "[[0.18, 0.47]]"
    assert out == expected_out
