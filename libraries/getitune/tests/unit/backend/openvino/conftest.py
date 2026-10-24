# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import openvino as ov
import pytest

from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo


@pytest.fixture(autouse=True)
def get_dummy_ov_cls_model() -> ov.Model:
    param_node = ov.op.Parameter(ov.Type.i32, ov.Shape([1, 3, 1, 1]))
    ov_model = ov.Model(param_node, [param_node])
    ov_model.outputs[0].tensor.set_names({"output"})
    model_params = TaskLevelExportParameters(
        model_type="Classification",
        model_name="dummy_model",
        task_type="classification",
        multilabel=True,
        label_info=LabelInfo(["car", "truck"], ["0", "1"], [["car"], ["truck"]]),
        optimization_config={},
    )
    for k, data in model_params.to_metadata().items():
        ov_model.set_rt_info(data, list(k))

    return ov_model
