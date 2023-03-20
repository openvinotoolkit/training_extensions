import copy
import os

import numpy as np
import pytest
from addict import Dict as ADDict
from openvino.model_zoo.model_api.models import Model

from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.tasks.openvino import (
    OpenVINODetectionTask,
    OpenVINOMaskInferencer,
    OpenVINOTileClassifierWrapper,
)
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.configuration.helper import create
from otx.api.entities.label import LabelEntity
from otx.api.entities.model_template import (
    TaskType,
    parse_model_template,
    task_type_to_label_domain,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_ISEG_TEMPLATE_DIR,
    generate_det_dataset,
    init_environment,
)


class TestTilingTileClassifier:
    """Test the tile classifier"""

    @pytest.fixture(autouse=True)
    def setUp(self, otx_model) -> None:
        classes = ("rectangle", "ellipse", "triangle")
        self.ov_inferencer = dict()
        task_type = TaskType.INSTANCE_SEGMENTATION
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        self.label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
        self.task_env = init_environment(hyper_parameters, model_template, task_type=task_type)
        self.params = DetectionConfig(header=hyper_parameters.header)
        self.params.tiling_parameters.enable_tiling = True
        self.params.tiling_parameters.enable_tile_classifier = True
        self.task_env.model = otx_model

    @e2e_pytest_unit
    def test_openvino(self, mocker):
        mocker.patch("otx.algorithms.detection.tasks.openvino.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model", return_value=mocker.MagicMock(spec=Model))
        ov_mask_inferencer = OpenVINOMaskInferencer(self.params, self.label_schema, "")
        ov_mask_inferencer.model.preprocess.return_value = ({"foo": "bar"}, {"baz": "qux"})
        ov_mask_inferencer.model.postprocess.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.uint32),
            np.zeros((0, 4), dtype=np.float32),
            [],
        )
        ov_inferencer = OpenVINOTileClassifierWrapper(ov_mask_inferencer, "", "")
        ov_inferencer.model.__model__ = "OTX_MaskRCNN"
        mock_predict = mocker.patch.object(ov_inferencer.tile_classifier, "infer_sync", return_value={'tile_prob': 0.5})
        mocker.patch.object(OpenVINODetectionTask, "load_inferencer", return_value=ov_inferencer)
        mocker.patch.object(OpenVINODetectionTask, "load_config", return_value=ADDict(vars(self.params)))
        ov_task = OpenVINODetectionTask(self.task_env)
        dataset, labels = generate_det_dataset(task_type=TaskType.INSTANCE_SEGMENTATION)
        updated_dataset = ov_task.infer(dataset)

        mock_predict.assert_called()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name=labels[0].name, domain="DETECTION")])

        output_model = copy.deepcopy(self.task_env.model)
        ov_task.model.set_data("openvino.bin", b"foo")
        ov_task.model.set_data("openvino.xml", b"bar")
        ov_task.model.set_data("tile_classifier.bin", b"foo")
        ov_task.model.set_data("tile_classifier.xml", b"bar")
        ov_task.deploy(output_model)
        assert output_model.exportable_code is not None

    @e2e_pytest_unit
    def test_export(self):
        pass