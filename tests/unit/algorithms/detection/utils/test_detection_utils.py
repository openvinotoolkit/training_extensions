# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.detection.utils.utils import (
    ColorPalette,
    generate_label_schema,
    get_det_model_api_configuration,
)
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


@e2e_pytest_unit
def test_get_det_model_api_configuration():
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))
    det_thr = 0.5
    model_api_cfg = get_det_model_api_configuration(label_schema, TaskType.DETECTION, det_thr)

    assert len(model_api_cfg) > 0
    assert model_api_cfg[("model_info", "confidence_threshold")] == str(det_thr)
