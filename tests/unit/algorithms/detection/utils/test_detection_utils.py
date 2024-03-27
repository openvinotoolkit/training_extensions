# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import addict

from otx.algorithms.detection.utils.utils import (
    generate_label_schema,
    get_det_model_api_configuration,
)
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_det_model_api_configuration():
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))
    det_thr = 0.5
    tiling_parameters = addict.Addict(
        {
            "enable_tiling": True,
            "tile_size": 10,
            "tile_overlap": 0.1,
            "tile_ir_scale_factor": 1.0,
            "tile_max_number": 100,
        }
    )
    model_api_cfg = get_det_model_api_configuration(
        label_schema, TaskType.DETECTION, det_thr, tiling_parameters, use_ellipse_shapes=False
    )

    assert len(model_api_cfg) > 0
    assert model_api_cfg[("model_info", "confidence_threshold")] == str(det_thr)
    assert model_api_cfg[("model_info", "tiles_overlap")] == str(
        tiling_parameters.tile_overlap / tiling_parameters.tile_ir_scale_factor
    )
    assert model_api_cfg[("model_info", "max_pred_number")] == str(tiling_parameters.tile_max_number)
    assert ("model_info", "labels") in model_api_cfg
    assert ("model_info", "label_ids") in model_api_cfg
    assert model_api_cfg[("model_info", "use_ellipse_shapes")] == "False"
    assert len(label_schema.get_labels(include_empty=False)) == len(model_api_cfg[("model_info", "labels")].split())
    assert len(label_schema.get_labels(include_empty=False)) == len(model_api_cfg[("model_info", "label_ids")].split())
