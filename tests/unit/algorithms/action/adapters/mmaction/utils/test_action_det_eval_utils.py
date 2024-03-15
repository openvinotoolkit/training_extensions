"""Unit Test for otx.algorithms.action.adapters.mmaction.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from typing import Any, Dict

import numpy as np

from otx.algorithms.action.adapters.mmaction.data import OTXActionDetDataset
from otx.algorithms.action.adapters.mmaction.utils import det_eval
from otx.algorithms.common.utils.utils import is_xpu_available
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
import pytest
from tests.test_suite.e2e_test_system import e2e_pytest_unit

FULL_BOX = np.array([[0, 0, 1, 1]])

if is_xpu_available():
    pytest.skip("Action task is not supported on XPU", allow_module_level=True)


class MockDataInfoProxy(OTXActionDetDataset._DataInfoProxy):
    """Mock clsass for data proxy in OTXActionDetDataset."""

    def __init__(self):
        self.video_infos = [
            {"img_key": "video_0,0", "gt_bboxes": FULL_BOX, "gt_labels": np.array([[0, 1, 0, 0]])},
            {"img_key": "video_0,1", "gt_bboxes": FULL_BOX, "gt_labels": np.array([[0, 0, 1, 0]])},
            {"img_key": "video_0,2", "gt_bboxes": FULL_BOX, "gt_labels": np.array([[0, 0, 0, 1]])},
        ]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.video_infos[index]


@e2e_pytest_unit
def test_det_eval() -> None:
    """Test for det_eval function.

    <Step>
        1. Generate sample predictions: Tuple[defaultdict]
        2. Generate sample labels: List[LabelEntity]
        3. Generate mock video_infos
    """

    pred_bboxes = defaultdict()
    pred_bboxes["video_0,0"] = [FULL_BOX[0], FULL_BOX[0], FULL_BOX[0]]
    pred_bboxes["video_0,1"] = [FULL_BOX[0], FULL_BOX[0], FULL_BOX[0]]
    pred_bboxes["video_0,2"] = [FULL_BOX[0], FULL_BOX[0], FULL_BOX[0]]
    pred_labels = defaultdict()
    pred_labels["video_0,0"] = [1, 2, 3]
    pred_labels["video_0,1"] = [1, 2, 3]
    pred_labels["video_0,2"] = [1, 2, 3]
    pred_confs = defaultdict()
    pred_confs["video_0,0"] = [1, 0, 0]
    pred_confs["video_0,1"] = [0, 1, 0]
    pred_confs["video_0,2"] = [0, 0, 1]
    predictions = (pred_bboxes, pred_labels, pred_confs)

    labels = [
        LabelEntity(name="0", domain=Domain.ACTION_DETECTION, id=ID(1)),
        LabelEntity(name="1", domain=Domain.ACTION_DETECTION, id=ID(2)),
        LabelEntity(name="2", domain=Domain.ACTION_DETECTION, id=ID(3)),
    ]

    video_infos = MockDataInfoProxy()
    custom_classes = [0, 1, 2, 3]

    out = det_eval(predictions, "mAP", labels, video_infos, None, True, custom_classes)
    assert out["mAP@0.5IOU"] == 1.0
