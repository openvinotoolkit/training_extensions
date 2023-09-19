# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from functools import partial

import numpy as np
import pytest
import torch
from openvino.model_api.models import ImageModel, Model

from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.algorithms.detection.adapters.mmdet.utils import build_detector, patch_tiling
from openvino.model_api.models import MaskRCNNModel
from otx.algorithms.detection.adapters.openvino.task import (
    OpenVINODetectionTask,
    OpenVINOMaskInferencer,
    OpenVINOTileClassifierWrapper,
)
from otx.algorithms.detection.configs.base import DetectionConfig
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
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_ISEG_TEMPLATE_DIR,
    generate_det_dataset,
    init_environment,
)
from openvino.model_api.models.utils import InstanceSegmentationResult


class TestTilingTileClassifier:
    """Test the tile classifier"""

    @pytest.fixture(autouse=True)
    def setUp(self, otx_model) -> None:
        """Set up the test

        Args:
            otx_model (mocker): Mocked model
        """
        classes = ("rectangle", "ellipse", "triangle")
        self.ov_inferencer = dict()
        task_type = TaskType.INSTANCE_SEGMENTATION
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        self.hyper_parameters = create(model_template.hyper_parameters.data)
        self.hyper_parameters.tiling_parameters.enable_tiling = True
        self.hyper_parameters.tiling_parameters.enable_tile_classifier = True
        self.label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
        self.task_env = init_environment(self.hyper_parameters, model_template, task_type=task_type)
        self.task_env.model = otx_model
        dataset, labels = generate_det_dataset(task_type=TaskType.INSTANCE_SEGMENTATION)
        self.dataset = dataset
        self.labels = labels

    @e2e_pytest_unit
    def test_openvino_sync(self, mocker):
        """Test OpenVINO tile classifier

        Args:
            mocker (_type_): pytest mocker from fixture
        """
        mocker.patch("otx.algorithms.detection.adapters.openvino.task.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        adapter_mock = mocker.Mock(
            set_callback=mocker.Mock(return_value=None), get_rt_info=mocker.Mock(return_value=np.array([1]))
        )
        mocker.patch.object(ImageModel, "__init__", return_value=None)
        mocker.patch.object(Model, "__init__", return_value=None)
        mocked_model.return_value = mocker.MagicMock(
            spec=MaskRCNNModel, inference_adapter=adapter_mock, postprocess_semantic_masks=False
        )
        params = DetectionConfig(header=self.hyper_parameters.header)
        ov_mask_inferencer = OpenVINOMaskInferencer(params, self.label_schema, "")
        original_shape = (self.dataset[0].media.width, self.dataset[0].media.height, 3)
        ov_mask_inferencer.model.resize_mask = False
        ov_mask_inferencer.model.preprocess.return_value = (
            {"foo": "bar"},
            {"baz": "qux", "original_shape": original_shape},
        )
        ov_mask_inferencer.model.postprocess.return_value = (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.uint32),
            np.zeros((0, 4), dtype=np.float32),
            [],
        )
        ov_inferencer = OpenVINOTileClassifierWrapper(
            ov_mask_inferencer, tile_classifier_model_file="", tile_classifier_weight_file="", mode="sync"
        )
        ov_inferencer.model.__model__ = "MaskRCNN"
        mock_predict = mocker.patch.object(
            ov_inferencer.tiler.tile_classifier_model, "infer_sync", return_value={"tile_prob": 0.5}
        )
        mocker.patch.object(ov_inferencer.tiler, "_postprocess_tile", return_value={})
        mocker.patch.object(
            ov_inferencer.tiler,
            "_merge_results",
            return_value=InstanceSegmentationResult(
                [], [np.zeros((0, 4), dtype=np.float32)], np.zeros((0, 4), dtype=np.float32)
            ),
        )
        ov_inferencer.tiler.model.infer_sync.return_value = {
            "feature_vector": np.zeros((1, 5), dtype=np.float32),
            "saliency_map": np.zeros((1, 1, 2, 2), dtype=np.float32),
        }
        mocker.patch.object(OpenVINODetectionTask, "load_inferencer", return_value=ov_inferencer)
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
        """Test loading tile classifier parameters

        Args:
            tmp_dir_path (str): Path to temporary directory
        """
        maskrcnn_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
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
        task = MMDetectionTask(task_env, output_path=str(tmp_dir_path))
        task._model_ckpt = model_ckpt
        task._init_task()
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
            task = MMDetectionTask(task_env, output_path=str(tmp_dir_path))
            assert (
                str(e.value)
                == "Tile classifier is enabled but not found in the trained model. Please retrain your model."
            )

        maskrcnn_classifier_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
        maskrcnn_classifier_cfg.model.type = "CustomMaskRCNNTileOptimized"
        tile_classifier_detector = build_detector(maskrcnn_classifier_cfg)
        tile_classifier_ckpt = os.path.join(tmp_dir_path, "maskrcnn_with_tile_classifier.pth")
        torch.save({"state_dict": tile_classifier_detector.state_dict()}, tile_classifier_ckpt)

        task_env = init_environment(hyper_parameters, model_template)
        output_model = ModelEntity(self.dataset, task_env.get_model_configuration())
        task = MMDetectionTask(task_env, output_path=str(tmp_dir_path))
        task._model_ckpt = tile_classifier_ckpt
        task._init_task()
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
        task = MMDetectionTask(task_env, output_path=str(tmp_dir_path))

    @e2e_pytest_unit
    def test_patch_tiling_func(self):
        """Test that patch_tiling function works correctly"""
        cfg = OTXConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
        data_pipeline_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "tile_pipeline.py"))
        cfg.merge_from_dict(data_pipeline_cfg)
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.tiling_parameters.enable_tiling = True
        hyper_parameters.tiling_parameters.enable_tile_classifier = True
        patch_tiling(cfg, hyper_parameters, self.dataset)
