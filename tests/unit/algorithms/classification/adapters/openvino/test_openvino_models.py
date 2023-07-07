# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.classification.adapters.openvino.model_wrappers import (
    OTXClassification,
)
from otx.algorithms.classification.utils.cls_utils import get_multihead_class_info
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    generate_cls_dataset,
    generate_label_schema,
)


class MockClassification(OTXClassification):
    def __init__(self):
        self.out_layer_names = ["logits"]
        hierarchical_dataset = generate_cls_dataset(hierarchical=True)
        label_schema = generate_label_schema(hierarchical_dataset.get_labels(), multilabel=False, hierarchical=True)
        self.multihead_class_info = get_multihead_class_info(label_schema)


class TestOTXClassification:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.cls_ov_model = MockClassification()
        self.fake_logits = np.array([-0.12, 0.92])
        self.fake_logits_hierarchical = np.array([3.01, -2.75, -0.63, 0.74, 0.017, -0.12, -0.42, -0.88])

    @pytest.mark.parametrize("multilabel, hierarchical", [(False, False), (True, False), (False, True)])
    @e2e_pytest_unit
    def test_postprocess(self, multilabel, hierarchical):
        self.cls_ov_model.multilabel = multilabel
        self.cls_ov_model.hierarchical = hierarchical
        fake_model_out = {"logits": self.fake_logits}
        if self.cls_ov_model.hierarchical:
            fake_model_out = {"logits": self.fake_logits_hierarchical}

        fake_metadata = {"original_shape": (254, 320, 3), "resized_shape": (224, 224, 3)}
        prediction = self.cls_ov_model.postprocess(fake_model_out, fake_metadata)
        if self.cls_ov_model.hierarchical:
            assert len(prediction) == 3
        else:
            assert prediction[0][0] == 1
            assert len(prediction) == 1

    @pytest.mark.parametrize("multilabel, hierarchical", [(False, False), (True, False), (False, True)])
    @e2e_pytest_unit
    def test_postprocess_aux_outputs(self, multilabel, hierarchical):
        self.cls_ov_model.multilabel = multilabel
        self.cls_ov_model.hierarchical = hierarchical
        fake_map = np.random.rand(1, 100, 100)
        fake_feature_vector = np.random.rand(1, 100)
        fake_logits = np.array([-0.012, 0.092])
        if self.cls_ov_model.hierarchical:
            fake_logits = np.array(
                [3.018761, -2.7551947, -0.6320721, 0.7461215, 0.01764688, -0.1284887, -0.42783177, -0.8848756]
            )

        fake_output = {"logits": fake_logits, "feature_vector": fake_feature_vector, "saliency_map": fake_map}
        fake_metadata = {"original_shape": (254, 320, 3), "resized_shape": (224, 224, 3)}
        out = self.cls_ov_model.postprocess_aux_outputs(fake_output, fake_metadata)
        assert len(out) == 4
        for item in out:
            assert item is not None
