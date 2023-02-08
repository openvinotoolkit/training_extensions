import pytest
import numpy as np

from openvino.model_zoo.model_api.adapters import create_core
from otx.algorithms.classification.tasks.openvino import OpenvinoAdapter
from openvino.model_zoo.model_api.models.classification import Classification
from otx.algorithms.classification.adapters.openvino.model_wrappers import (
    OTXClassification,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from tests.unit.algorithms.classification.test_helper import (
    DEFAULT_CLS_TEMPLATE,
    init_environment
)

class MockClassification(OTXClassification, Classification):
    def __init__(self):
        self.out_layer_name = 'logits'
        self.multihead_class_info = {'num_multiclass_heads': 3,
                                  'num_multilabel_classes': 2,
                                  'head_idx_to_logits_range': {0: (0, 2), 1: (2, 4), 2: (4, 6)},
                                  'num_single_label_classes': 6,
                                  'class_to_group_idx': {'equilateral': (0, 0),
                                                         'right': (0, 1),
                                                         'non_square': (1, 0),
                                                         'square': (1, 1),
                                                         'rectangle': (2, 0),
                                                         'triangle': (2, 1),
                                                         'multi a': (3, 0),
                                                         'multi b': (3, 1)},
                                  'all_groups': [['equilateral', 'right'],
                                                 ['non_square', 'square'],
                                                 ['rectangle', 'triangle'],
                                                 ['multi a'], ['multi b']],
                                  'label_to_idx': {'right': 0,
                                                   'multi a': 1,
                                                   'multi b': 2,
                                                   'equilateral': 3,
                                                   'square': 4,
                                                   'triangle': 5,
                                                   'non_square': 6,
                                                   'rectangle': 7}}

class TestOTXClassification:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        self.cls_ov_model = MockClassification()
        self.fake_logits = np.array([-0.12, 0.92])
        self.fake_logits_hierarchical = np.array([3.01, -2.75, -0.63, 0.74, 0.017, -0.12, -0.42, -0.88])

    @pytest.mark.parametrize("multilabel, hierarchical", [(False, False), (True, False), (False, True)])
    @e2e_pytest_unit
    def test_postprocess(self, multilabel, hierarchical, mocker):
        self.cls_ov_model.multilabel = multilabel
        self.cls_ov_model.hierarchical = hierarchical
        fake_model_out = {'logits': self.fake_logits}
        if self.cls_ov_model.hierarchical:
            fake_model_out = {'logits': self.fake_logits_hierarchical}

        fake_metadata = {'original_shape': (254, 320, 3), 'resized_shape': (224, 224, 3)}
        prediction = self.cls_ov_model.postprocess(fake_model_out, fake_metadata)
        if self.cls_ov_model.hierarchical:
            print(prediction)
            assert len(prediction) == 3
        else:
            assert prediction[0][0] == 1
            assert len(prediction) == 1

    @pytest.mark.parametrize("multilabel, hierarchical", [(False, False), (True, False), (False, True)])
    @e2e_pytest_unit
    def test_aux_postprocess(self, multilabel, hierarchical, mocker):
        self.cls_ov_model.multilabel = multilabel
        self.cls_ov_model.hierarchical = hierarchical
        fake_map = np.random.rand(1,100,100)
        fake_feature_vector = np.random.rand(1,100)
        fake_logits = np.array([-0.012, 0.092])
        if self.cls_ov_model.hierarchical:
            fake_logits = np.array([3.018761, -2.7551947, -0.6320721, 0.7461215, 0.01764688, -0.1284887, -0.42783177, -0.8848756])

        fake_output = {"logits": fake_logits, 'feature_vector': fake_feature_vector, 'saliency_map': fake_map}
        fake_metadata = {'original_shape': (254, 320, 3), 'resized_shape': (224, 224, 3)}
        out = self.cls_ov_model.postprocess_aux_outputs(fake_output, fake_metadata)
        assert len(out) == 4
        for item in out:
            assert item is not None
