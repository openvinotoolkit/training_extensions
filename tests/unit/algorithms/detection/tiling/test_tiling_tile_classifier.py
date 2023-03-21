# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import copy
import os

import numpy as np
import pytest
import torch
from addict import Dict as ADDict
from openvino.model_zoo.model_api.models import Model

from otx.algorithms.detection.adapters.mmdet.utils import build_detector
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.tasks import DetectionTrainTask
from otx.algorithms.detection.tasks.openvino import (
    OpenVINODetectionTask,
    OpenVINOMaskInferencer,
    OpenVINOTileClassifierWrapper,
)
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.configuration.helper import create
from otx.api.entities.label import LabelEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import (
    TaskType,
    parse_model_template,
    task_type_to_label_domain,
)
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.mpa.utils.config_utils import MPAConfig
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
        dataset, labels = generate_det_dataset(task_type=TaskType.INSTANCE_SEGMENTATION)
        self.dataset = dataset
        self.labels = labels

    @e2e_pytest_unit
    def test_openvino(self, mocker):
        """Test OpenVINO tile classifier

        Args:
            mocker (_type_): pytest mocker from fixture
        """
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
        mock_predict = mocker.patch.object(ov_inferencer.tile_classifier, "infer_sync", return_value={"tile_prob": 0.5})
        mocker.patch.object(OpenVINODetectionTask, "load_inferencer", return_value=ov_inferencer)
        mocker.patch.object(OpenVINODetectionTask, "load_config", return_value=ADDict(vars(self.params)))
        ov_task = OpenVINODetectionTask(self.task_env)
        updated_dataset = ov_task.infer(self.dataset)

        mock_predict.assert_called()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name=self.labels[0].name, domain="DETECTION")])

        output_model = copy.deepcopy(self.task_env.model)
        ov_task.model.set_data("openvino.bin", b"foo")
        ov_task.model.set_data("openvino.xml", b"bar")
        ov_task.model.set_data("tile_classifier.bin", b"foo")
        ov_task.model.set_data("tile_classifier.xml", b"bar")
        ov_task.deploy(output_model)
        assert output_model.exportable_code is not None

    @e2e_pytest_unit
    def test_load_tile_classifier_parameters(self, tmp_dir_path):
        maskrcnn_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
        detector = build_detector(maskrcnn_cfg)
        model_ckpt = os.path.join(tmp_dir_path, "maskrcnn_without_tile_classifier.pth")
        torch.save({"state_dict": detector.state_dict()}, model_ckpt)

        # Enable tiling and save weights
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.tiling_parameters.enable_tiling = True
        hyper_parameters.tiling_parameters.enable_tile_classifier = True
        task_env = init_environment(hyper_parameters, model_template)
        output_model = ModelEntity(self.dataset, task_env.get_model_configuration())
        task = DetectionTrainTask(task_env, output_path=str(tmp_dir_path))
        task._model_ckpt = model_ckpt
        task.save_model(output_model)
        for filename, model_adapter in output_model.model_adapters.items():
            with open(os.path.join(tmp_dir_path, filename), "wb") as write_file:
                write_file.write(model_adapter.data)

        # Read tiling parameters from weights
        with open(os.path.join(tmp_dir_path, "weights.pth"), "rb") as f:
            bin_data = f.read()
            model = ModelEntity(
                self.dataset,
                configuration=task_env.get_model_configuration(),
                model_adapters={"weights.pth": ModelAdapter(bin_data)},
            )
        task_env.model = model
        with pytest.raises(RuntimeError) as e:
            task = DetectionTrainTask(task_env, output_path=str(tmp_dir_path))
            assert (
                str(e.value)
                == "Tile classifier is enabled but not found in the trained model. Please retrain your model."
            )

        maskrcnn_classifier_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
        maskrcnn_classifier_cfg.model.type = "CustomMaskRCNNTileOptimised"
        tile_classifier_detector = build_detector(maskrcnn_classifier_cfg)
        tile_classifier_ckpt = os.path.join(tmp_dir_path, "maskrcnn_with_tile_classifier.pth")
        torch.save({"state_dict": tile_classifier_detector.state_dict()}, tile_classifier_ckpt)

        task_env = init_environment(hyper_parameters, model_template)
        output_model = ModelEntity(self.dataset, task_env.get_model_configuration())
        task = DetectionTrainTask(task_env, output_path=str(tmp_dir_path))
        task._model_ckpt = tile_classifier_ckpt
        task.save_model(output_model)
        for filename, model_adapter in output_model.model_adapters.items():
            with open(os.path.join(tmp_dir_path, filename), "wb") as write_file:
                write_file.write(model_adapter.data)

        # Read tiling parameters from weights
        with open(os.path.join(tmp_dir_path, "weights.pth"), "rb") as f:
            bin_data = f.read()
            model = ModelEntity(
                self.dataset,
                configuration=task_env.get_model_configuration(),
                model_adapters={"weights.pth": ModelAdapter(bin_data)},
            )
        task_env.model = model
        task = DetectionTrainTask(task_env, output_path=str(tmp_dir_path))
