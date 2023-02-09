# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from otx.algorithms.detection.utils import generate_label_schema
from otx.algorithms.detection.utils.data import (
    find_label_by_name,
    format_list_to_str,
    get_anchor_boxes,
    get_sizes_from_dataset_entity,
)
from otx.api.entities.label import Domain
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import generate_det_dataset


# TODO: Need to add adaptive_tile_params unit-test
class TestDetectionDataUtils:
    @e2e_pytest_unit
    @pytest.mark.parametrize("name", ["rectangle", "something"])
    def test_find_label_by_name(self, name):
        classes = ("rectangle", "ellipse", "triangle")
        label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))
        find_label_by_name(label_schema.get_labels(include_empty=False), name, Domain.DETECTION)

    @e2e_pytest_unit
    def test_find_label_by_name_error(self):
        classes = ("rectangle", "rectangle", "triangle")
        label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))
        with pytest.raises(ValueError):
            find_label_by_name(label_schema.get_labels(include_empty=False), "rectangle", Domain.DETECTION)

    @e2e_pytest_unit
    def test_get_sizes_from_dataset_entity(self):
        dataset, labels = generate_det_dataset(task_type=TaskType.DETECTION)
        get_sizes_from_dataset_entity(dataset, [480, 640])

    @e2e_pytest_unit
    def test_get_anchor_boxes(self):
        get_anchor_boxes([(100, 120), (100, 120)], [1, 1])

    @e2e_pytest_unit
    def test_format_list_to_str(self):
        format_list_to_str([[0.1239128319, 0.12398123]])
