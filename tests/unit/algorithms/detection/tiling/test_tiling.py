import copy
import unittest

import numpy as np
import pytest
import torch
from mmcv import ConfigDict
from mmdet.datasets import build_dataloader, build_dataset
from skimage.draw import random_shapes

from otx.algorithms.detection.adapters.mmdet.data import (  # noqa: F401
    ImageTilingDataset,
)
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle


class TilingTest(unittest.TestCase):
    """Test the tiling functionality"""

    def create_otx_dataset(self, height, width, min_shapes=1, max_shapes=10):
        """Create a test dataset"""
        image, labels = random_shapes((height, width), min_shapes=min_shapes, max_shapes=max_shapes, allow_overlap=True)
        image = Image(data=image)
        label_dict = {}
        annotations = []
        for label, bbox in labels:
            if label not in label_dict:
                label_dict[label] = LabelEntity(name=label, domain=Domain.DETECTION)
            y1, y2 = bbox[0]
            x1, x2 = bbox[1]
            annotation = Annotation(shape=Rectangle(x1, y1, x2, y2), labels=[ScoredLabel(label_dict[label])])
            annotations.append(annotation)
        annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)
        dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene)
        return DatasetEntity([dataset_item]), label_dict

    def setUp(self) -> None:
        """Set up the test case"""
        self.height = 1024
        self.width = 1024
        self.tile_cfg = dict(
            tile_size=np.random.randint(low=100, high=500),
            overlap_ratio=np.random.rand(),
            max_per_img=np.random.randint(low=1, high=10000),
        )
        self.dataloader_cfg = dict(samples_per_gpu=1, workers_per_gpu=1)
        self.otx_dataset, self.label_dict = self.create_otx_dataset(self.height, self.width)

    def test_tiling_train_dataloader(self):
        """Test that the tiling dataloader is built correctly"""
        train_data_cfg = ConfigDict(
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
                    type="MPADetDataset",
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
                    labels=list(self.label_dict.values()),
                    domain=Domain.DETECTION,
                ),
                **self.tile_cfg
            )
        )

        dataset = build_dataset(train_data_cfg)
        train_dataloader = build_dataloader(dataset, **self.dataloader_cfg)
        for data in train_dataloader:
            self.assertIsInstance(data["img"].data[0], torch.Tensor)
            self.assertIsInstance(data["gt_bboxes"].data[0][0], torch.Tensor)
            self.assertIsInstance(data["gt_labels"].data[0][0], torch.Tensor)

    def test_tiling_test_dataloader(self):
        """Test that the tiling dataset is built correctly"""

        img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
        test_data_cfg = ConfigDict(
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
                    type="MPADetDataset",
                    pipeline=[dict(type="LoadImageFromOTXDataset")],
                    otx_dataset=self.otx_dataset,
                    labels=list(self.label_dict.values()),
                    domain=Domain.DETECTION,
                ),
                test_mode=True,
                **self.tile_cfg
            )
        )

        dataset = build_dataset(test_data_cfg)

        stride = (1 - self.tile_cfg["overlap_ratio"]) * self.tile_cfg["tile_size"]
        num_tile_rows = ((self.height - self.tile_cfg["tile_size"]) // stride) + 1
        num_tile_cols = ((self.width - self.tile_cfg["tile_size"]) // stride) + 1
        # +1 for the original image
        self.assertEqual(len(dataset), (num_tile_rows * num_tile_cols) + 1, "Incorrect number of tiles")

        test_dataloader = build_dataloader(dataset, **self.dataloader_cfg)
        for data in test_dataloader:
            self.assertIsInstance(data["img"][0], torch.Tensor)

    @pytest.xfail(reason="Inference merge not implemented yet")
    def test_inference_merge(self):
        """Test that the inference merge works correctly"""
        dataset_cfg = copy.deepcopy(self.dataset_cfg)
        dataset_cfg["test_mode"] = True
        dataset = build_dataset(dataset_cfg)
        # TODO: Implement inference merge check
        dataset.merge()
