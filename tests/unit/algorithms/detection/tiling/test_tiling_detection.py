# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import List

import numpy as np
import pytest
import torch
from mmcv import Config, ConfigDict
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import DETECTORS
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from torch import nn

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.common.adapters.mmdeploy.apis import MMdeployExporter
from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.algorithms.detection.adapters.mmdet.utils import build_detector, patch_tiling
from otx.api.configuration.helper import create
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity, DatasetPurpose
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.subset import Subset
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_helpers import generate_random_annotated_image
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_ISEG_TEMPLATE_DIR,
    init_environment,
)


@DETECTORS.register_module(force=True)
class MockDetModel(nn.Module):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.box_dummy = torch.nn.AdaptiveAvgPool2d((1, 5))
        self.label_dummy = torch.nn.AdaptiveAvgPool2d((1))
        self.mask_dummy = torch.nn.AdaptiveAvgPool2d((28, 28))

    def forward(self, *args, **kwargs):
        img = args[0]
        x = self.conv(img)
        boxes = self.box_dummy(x).mean(1)
        labels = self.label_dummy(x).mean(1)
        masks = self.mask_dummy(x).mean(1)
        return boxes, labels, masks


def create_otx_dataset(height: int, width: int, labels: List[str], domain: Domain = Domain.DETECTION):
    """Create a random OTX dataset.

    Args:
        height (int): The height of the image
        width (int): The width of the image

    Returns:
        DatasetEntity: OTX dataset entity
        List[LabelEntity]: The list of labels
    """
    labels = []
    for label in ["rectangle", "ellipse", "triangle"]:
        labels.append(LabelEntity(name=label, domain=domain))
    image, anno_list = generate_random_annotated_image(width, height, labels)
    image = Image(data=image)
    annotation_scene = AnnotationSceneEntity(annotations=anno_list, kind=AnnotationSceneKind.ANNOTATION)
    dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=Subset.TRAINING)
    return DatasetEntity([dataset_item]), labels


class TestTilingDetection:
    """Test the tiling detection algorithm."""

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """Setup the test case."""
        self.height = 1024
        self.width = 1024
        self.label_names = ["rectangle", "ellipse", "triangle"]
        self.tile_cfg = dict(
            tile_size=np.random.randint(low=100, high=500),
            overlap_ratio=np.random.uniform(low=0.0, high=0.5),
            max_per_img=np.random.randint(low=1, high=10000),
            max_annotation=1000,
        )
        self.dataloader_cfg = dict(samples_per_gpu=1, workers_per_gpu=1)
        self.otx_dataset, self.labels = create_otx_dataset(self.height, self.width, self.label_names)

        img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

        self.train_data_cfg = ConfigDict(
            dict(
                type="ImageTilingDataset",
                filter_empty_gt=False,
                pipeline=[
                    dict(type="Resize", img_scale=(self.height, self.width), keep_ratio=False),
                    dict(type="RandomFlip", flip_ratio=0.5),
                    dict(type="Pad", size_divisor=32),
                    dict(type="DefaultFormatBundle"),
                    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
                ],
                dataset=dict(
                    type="OTXDetDataset",
                    pipeline=[
                        dict(type="LoadImageFromOTXDataset"),
                        dict(
                            type="LoadAnnotationFromOTXDataset",
                            with_bbox=True,
                            with_mask=False,
                            domain=Domain.DETECTION,
                            min_size=-1,
                        ),
                    ],
                    otx_dataset=self.otx_dataset,
                    labels=self.labels,
                    domain=Domain.DETECTION,
                ),
                **self.tile_cfg
            )
        )

        self.test_data_cfg = ConfigDict(
            dict(
                type="ImageTilingDataset",
                filter_empty_gt=False,
                pipeline=[
                    dict(
                        type="MultiScaleFlipAug",
                        img_scale=(self.height, self.width),
                        flip=False,
                        transforms=[
                            dict(type="Resize", keep_ratio=False),
                            dict(type="Normalize", **img_norm_cfg),
                            dict(type="ImageToTensor", keys=["img"]),
                            dict(type="Collect", keys=["img"]),
                        ],
                    )
                ],
                dataset=dict(
                    type="OTXDetDataset",
                    pipeline=[dict(type="LoadImageFromOTXDataset")],
                    otx_dataset=self.otx_dataset.with_empty_annotations(),
                    labels=list(self.labels),
                    domain=Domain.DETECTION,
                ),
                test_mode=True,
                **self.tile_cfg
            )
        )

    @e2e_pytest_unit
    def test_tiling_train_dataloader(self):
        """Test that the training dataloader is built correctly for tiling."""

        dataset = build_dataset(self.train_data_cfg)
        train_dataloader = build_dataloader(dataset, **self.dataloader_cfg)
        for data in train_dataloader:
            assert isinstance(data["img"].data[0], torch.Tensor)
            assert isinstance(data["gt_bboxes"].data[0][0], torch.Tensor)
            assert isinstance(data["gt_labels"].data[0][0], torch.Tensor)

    @e2e_pytest_unit
    def test_tiling_test_dataloader(self):
        """Test that the testing dataloader is built correctly for tiling."""

        dataset = build_dataset(self.test_data_cfg)
        stride = int((1 - self.tile_cfg["overlap_ratio"]) * self.tile_cfg["tile_size"])
        num_tile_rows = (self.height + stride - 1) // stride
        num_tile_cols = (self.width + stride - 1) // stride
        assert len(dataset) == (num_tile_rows * num_tile_cols), "Incorrect number of tiles"

        test_dataloader = build_dataloader(dataset, **self.dataloader_cfg)
        for data in test_dataloader:
            assert isinstance(data["img"][0], torch.Tensor)
            assert "gt_bboxes" not in data
            assert "gt_labels" not in data

    @e2e_pytest_unit
    def test_tiling_sampling(self):
        self.train_data_cfg.sampling_ratio = 0.5
        dataset = build_dataset(self.train_data_cfg)
        assert len(dataset) == max(int(len(dataset.tile_dataset.tiles_all) * 0.5), 1)

        self.train_data_cfg.sampling_ratio = 0.01
        dataset = build_dataset(self.train_data_cfg)
        assert len(dataset) == max(int(len(dataset.tile_dataset.tiles_all) * 0.01), 1)

        self.test_data_cfg.sampling_ratio = 0.1
        self.test_data_cfg.dataset.otx_dataset[0].subset = Subset.TESTING
        dataset = build_dataset(self.test_data_cfg)
        assert len(dataset) == len(dataset.tile_dataset.tiles_all)

    @e2e_pytest_unit
    def test_inference_merge(self):
        """Test that the inference merge works correctly."""
        dataset = build_dataset(self.test_data_cfg)

        # create simulated inference results
        results: List[List[np.ndarray]] = []
        for i in range(len(dataset)):
            results.append([])
            for _ in range(len(self.labels)):
                results[i].append(np.zeros((0, 5), dtype=np.float32))

        # generate tile predictions
        for i in range(len(dataset)):
            img_width, img_height = self.tile_cfg["tile_size"], self.tile_cfg["tile_size"]
            if i == 0:
                # first index belongs is the full image
                img_width, img_height = self.width, self.height

            _, anno_list = generate_random_annotated_image(img_width, img_height, self.labels)
            for anno in anno_list:
                shape = ShapeFactory.shape_as_rectangle(anno.shape)
                bbox = np.array([shape.x1, shape.y1, shape.x2, shape.y2], np.float32)
                bbox *= np.tile([img_width, img_height], 2)
                score_bbox = np.array([*bbox, np.random.rand()], np.float32)
                label_idx = self.label_names.index(anno.get_labels()[0].name)
                results[i][label_idx] = np.append(results[i][label_idx], [score_bbox], axis=0)

        merged_bbox_results = dataset.merge(results)
        assert len(merged_bbox_results) == dataset.num_samples

    @e2e_pytest_unit
    def test_merge_feature_vectors(self):
        """Test that the merge feature vectors works correctly."""
        dataset = build_dataset(self.test_data_cfg)

        # create simulated vectors results
        feature_vectors: List[np.ndarray] = []
        vectors_per_image = 5
        vector_length = 10
        feature_vectors = [np.zeros((vectors_per_image, vector_length), dtype=np.float32) for _ in range(len(dataset))]

        # Test merge_vectors if vectors are to be returned
        merged_vectors = dataset.merge_vectors(feature_vectors, dump_vectors=True)
        assert len(merged_vectors) == dataset.num_samples

        # Test merge_vectors if merged vectors should be a list of None
        merged_vectors = dataset.merge_vectors(feature_vectors, dump_vectors=False)
        assert len(merged_vectors) == dataset.num_samples

    @e2e_pytest_unit
    def test_merge_saliency_maps(self):
        """Test that the inference merge works correctly."""
        dataset = build_dataset(self.test_data_cfg)

        # create simulated maps results
        saliency_maps: List[np.ndarray] = []
        num_classes = len(dataset.CLASSES)
        feature_map_size = (num_classes, 2, 2)
        features_per_image = 5
        saliency_maps = [np.zeros(feature_map_size, dtype=np.float32) for _ in range(len(dataset) * features_per_image)]

        # Test merge_maps if maps are to be processed
        merged_maps = dataset.merge_maps(saliency_maps, dump_maps=True)
        assert len(merged_maps) == dataset.num_samples

        # Test merge_maps if maps should be a list of None
        merged_maps = dataset.merge_maps(saliency_maps, dump_maps=False)
        assert len(merged_maps) == dataset.num_samples

    @e2e_pytest_unit
    def test_load_tiling_parameters(self, tmp_dir_path):
        maskrcnn_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
        detector = build_detector(maskrcnn_cfg)

        # Enable tiling and save weights
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.tiling_parameters.enable_tiling = True
        task_env = init_environment(hyper_parameters, model_template)
        output_model = ModelEntity(self.otx_dataset, task_env.get_model_configuration())
        task = MMDetectionTask(task_env, output_path=str(tmp_dir_path))
        model_ckpt = os.path.join(tmp_dir_path, "maskrcnn.pth")
        task._init_task()
        torch.save(detector.state_dict(), model_ckpt)
        task._model_ckpt = model_ckpt
        task.save_model(output_model)
        for filename, model_adapter in output_model.model_adapters.items():
            with open(os.path.join(tmp_dir_path, filename), "wb") as write_file:
                write_file.write(model_adapter.data)

        # Read tiling parameters from weights
        with open(os.path.join(tmp_dir_path, "weights.pth"), "rb") as f:
            bin_data = f.read()
            model = ModelEntity(
                self.otx_dataset,
                configuration=task_env.get_model_configuration(),
                model_adapters={"weights.pth": ModelAdapter(bin_data)},
            )
        task_env.model = model
        task = MMDetectionTask(task_env, output_path=str(tmp_dir_path))

    @e2e_pytest_unit
    def test_patch_tiling_func(self):
        """Test that patch_tiling function works correctly."""
        cfg = MPAConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.tiling_parameters.enable_tiling = True

        self.otx_dataset.purpose = DatasetPurpose.TRAINING
        patch_tiling(cfg, hyper_parameters, self.otx_dataset)

        self.otx_dataset.purpose = DatasetPurpose.INFERENCE
        patch_tiling(cfg, hyper_parameters, self.otx_dataset)

    @e2e_pytest_unit
    @pytest.mark.parametrize("scale_factor", [1, 1.5, 2, 3, 4])
    def test_tile_ir_scale_deploy(self, tmp_dir_path, scale_factor):
        """Test that the IR scale factor is correctly applied during inference."""
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.tiling_parameters.enable_tiling = True
        hyper_parameters.tiling_parameters.tile_ir_scale_factor = scale_factor
        task_env = init_environment(hyper_parameters, model_template)
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        task = MMDetectionTask(task_env)
        pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=False),
                    dict(type="RandomFlip"),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="Pad", size_divisor=32),
                    dict(type="DefaultFormatBundle"),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ]
        config = Config(
            dict(model=dict(type="MockDetModel", backbone=dict(init_cfg=None)), data=dict(test=dict(pipeline=pipeline)))
        )

        deploy_cfg = task._init_deploy_cfg(config)
        onnx_path = MMdeployExporter.torch2onnx(
            tmp_dir_path,
            np.zeros((50, 50, 3), dtype=np.float32),
            config,
            deploy_cfg,
        )
        assert isinstance(onnx_path, str)
        assert os.path.exists(onnx_path)

        openvino_paths = MMdeployExporter.onnx2openvino(
            tmp_dir_path,
            onnx_path,
            deploy_cfg,
        )
        for openvino_path in openvino_paths:
            assert os.path.exists(openvino_path)

        task._init_task()
        original_width, original_height = task._recipe_cfg.data.test.pipeline[0].img_scale  # w, h

        model_adapter = OpenvinoAdapter(create_core(), openvino_paths[0], openvino_paths[1])

        ir_input_shape = model_adapter.get_input_layers()["image"].shape
        _, _, ir_height, ir_width = ir_input_shape
        assert ir_height == original_height * scale_factor
        assert ir_width == original_width * scale_factor

    @e2e_pytest_unit
    @pytest.mark.skip(reason="Issue#2245: Sporadic failure of tiling max ann.")
    def test_max_annotation(self, max_annotation=200):
        otx_dataset, labels = create_otx_dataset(
            self.height, self.width, self.label_names, Domain.INSTANCE_SEGMENTATION
        )
        factor = int(max_annotation / len(otx_dataset[0].annotation_scene.annotations)) + 2
        otx_dataset[0].annotation_scene.annotations = otx_dataset[0].annotation_scene.annotations * factor
        dataloader_cfg = dict(samples_per_gpu=1, workers_per_gpu=1)

        tile_cfg = dict(
            tile_size=np.random.randint(low=100, high=500),
            overlap_ratio=np.random.uniform(low=0.0, high=0.5),
            max_per_img=np.random.randint(low=1, high=10000),
            max_annotation=max_annotation,
        )
        train_data_cfg = ConfigDict(
            dict(
                type="ImageTilingDataset",
                pipeline=[
                    dict(type="Resize", img_scale=(self.height, self.width), keep_ratio=False),
                    dict(type="RandomFlip", flip_ratio=0.5),
                    dict(type="Pad", size_divisor=32),
                    dict(type="DefaultFormatBundle"),
                    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
                ],
                dataset=dict(
                    type="OTXDetDataset",
                    pipeline=[
                        dict(type="LoadImageFromOTXDataset"),
                        dict(
                            type="LoadAnnotationFromOTXDataset",
                            with_bbox=True,
                            with_mask=True,
                            domain=Domain.INSTANCE_SEGMENTATION,
                            min_size=-1,
                        ),
                    ],
                    otx_dataset=otx_dataset,
                    labels=labels,
                    domain=Domain.INSTANCE_SEGMENTATION,
                ),
                **tile_cfg
            )
        )

        # original annotation over the limitation
        assert len(otx_dataset[0].annotation_scene.annotations) > max_annotation
        dataset = build_dataset(train_data_cfg)
        train_dataloader = build_dataloader(dataset, **dataloader_cfg)
        # check gt annotation is under the limitation
        for data in train_dataloader:
            assert len(data["gt_bboxes"].data[0][0]) <= max_annotation
            assert len(data["gt_labels"].data[0][0]) <= max_annotation
            assert len(data["gt_masks"].data[0][0]) <= max_annotation
