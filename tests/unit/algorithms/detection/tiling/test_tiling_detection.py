# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import List

import numpy as np
import pytest
import torch
from mmcv import ConfigDict
from mmdet.datasets import build_dataloader, build_dataset

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
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
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_helpers import generate_random_annotated_image
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_ISEG_TEMPLATE_DIR,
    init_environment,
)


def create_otx_dataset(height: int, width: int, labels: List[str]):
    """Create a random OTX dataset

    Args:
        height (int): The height of the image
        width (int): The width of the image

    Returns:
        DatasetEntity: OTX dataset entity
        List[LabelEntity]: The list of labels
    """
    labels = []
    for label in ["rectangle", "ellipse", "triangle"]:
        labels.append(LabelEntity(name=label, domain=Domain.DETECTION))
    image, anno_list = generate_random_annotated_image(width, height, labels)
    image = Image(data=image)
    annotation_scene = AnnotationSceneEntity(annotations=anno_list, kind=AnnotationSceneKind.ANNOTATION)
    dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene)
    return DatasetEntity([dataset_item]), labels


class TestTilingDetection:
    """Test the tiling functionality"""

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """Setup the test case"""
        self.height = 1024
        self.width = 1024
        self.label_names = ["rectangle", "ellipse", "triangle"]
        self.tile_cfg = dict(
            tile_size=np.random.randint(low=100, high=500),
            overlap_ratio=np.random.uniform(low=0.0, high=0.5),
            max_per_img=np.random.randint(low=1, high=10000),
        )
        self.dataloader_cfg = dict(samples_per_gpu=1, workers_per_gpu=1)
        self.otx_dataset, self.labels = create_otx_dataset(self.height, self.width, self.label_names)

        img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

        self.train_data_cfg = ConfigDict(
            dict(
                type="ImageTilingDataset",
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
        """Test that the training dataloader is built correctly for tiling"""

        dataset = build_dataset(self.train_data_cfg)
        train_dataloader = build_dataloader(dataset, **self.dataloader_cfg)
        for data in train_dataloader:
            assert isinstance(data["img"].data[0], torch.Tensor)
            assert isinstance(data["gt_bboxes"].data[0][0], torch.Tensor)
            assert isinstance(data["gt_labels"].data[0][0], torch.Tensor)

    @e2e_pytest_unit
    def test_tiling_test_dataloader(self):
        """Test that the testing dataloader is built correctly for tiling"""

        dataset = build_dataset(self.test_data_cfg)
        stride = int((1 - self.tile_cfg["overlap_ratio"]) * self.tile_cfg["tile_size"])
        num_tile_rows = ((self.height - self.tile_cfg["tile_size"]) // stride) + 1
        num_tile_cols = ((self.width - self.tile_cfg["tile_size"]) // stride) + 1
        # +1 for the original image
        assert len(dataset) == (num_tile_rows * num_tile_cols) + 1, "Incorrect number of tiles"

        test_dataloader = build_dataloader(dataset, **self.dataloader_cfg)
        for data in test_dataloader:
            assert isinstance(data["img"][0], torch.Tensor)
            assert "gt_bboxes" not in data
            assert "gt_labels" not in data

    @e2e_pytest_unit
    def test_inference_merge(self):
        """Test that the inference merge works correctly"""
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
        """Test that patch_tiling function works correctly"""
        cfg = MPAConfig.fromfile(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "model.py"))
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.tiling_parameters.enable_tiling = True

        self.otx_dataset.purpose = DatasetPurpose.TRAINING
        patch_tiling(cfg, hyper_parameters, self.otx_dataset)

        self.otx_dataset.purpose = DatasetPurpose.INFERENCE
        patch_tiling(cfg, hyper_parameters, self.otx_dataset)

    @e2e_pytest_unit
    def test_openvino(self):
        # TODO[EUGENE]: implement unittest for tiling prediction with openvino
        pass
