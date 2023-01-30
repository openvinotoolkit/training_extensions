import copy
import unittest

import numpy as np
from mmdet.datasets import build_dataloader, build_dataset
from skimage.draw import random_shapes

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

    def setUp(self) -> None:
        """Set up the test case"""
        self.height = 1024
        self.width = 1024
        self.tile_cfg = dict(
            tile_size=np.random.randint(low=100, high=500),
            overlap_ratio=np.random.rand(),
            max_per_img=np.random.randint(low=1, high=10000),
        )

        image, labels = random_shapes((self.height, self.width), min_shapes=1, max_shapes=10, allow_overlap=True)
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
        otx_dataset = DatasetEntity([dataset_item])

        self.dataloader_cfg = dict(
            samples_per_gpu=1,
            workers_per_gpu=1,
        )

        self.dataset_cfg = dict(
            type="ImageTilingDataset",
            pipeline=[
                dict(type="Resize", img_scale=(1024, 1024), keep_ratio=False),
                dict(type="RandomFlip", flip_ratio=0.5),
                dict(type="Normalize", mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), to_rgb=False),
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
                otx_dataset=otx_dataset,
                labels=list(label_dict.values()),
                domain=Domain.DETECTION,
            ),
            **self.tile_cfg
        )

    def test_build_tiling_dataset(self):
        """ Test that the tiling dataset is built correctly """
        dataset_cfg = copy.deepcopy(self.dataset_cfg)
        dataset = build_dataset(dataset_cfg)

        stride = (1 - self.tile_cfg["overlap_ratio"]) * self.tile_cfg["tile_size"]

        assert len(dataset) == ((self.height - self.tile_cfg["tile_size"]) // stride) * (
                (self.width - self.tile_cfg["tile_size"]) // stride
        )

    def test_build_tiling_dataloader(self):
        """ Test that the tiling dataloader is built correctly """
        dataset = build_dataset(self.dataset_cfg)
        dataloader = build_dataloader(dataset, **self.dataloader_cfg)
        for _ in dataloader:
            pass

    def test_inference_merge(self):
        """Test that the inference merge works correctly"""
        dataset_cfg = copy.deepcopy(self.dataset_cfg)
        dataset_cfg["test_mode"] = True
        dataset = build_dataset(dataset_cfg)
        # TODO: Implement inference merge check
        # print(dataset.merge([[]]))
