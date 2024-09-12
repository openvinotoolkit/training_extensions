# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
import shapely.geometry as sg
import torch
from datumaro import Dataset as DmDataset
from datumaro import Polygon
from omegaconf import DictConfig, OmegaConf
from otx.algo.detection.atss import MobileNetV2ATSS
from otx.algo.instance_segmentation.maskrcnn import MaskRCNNEfficientNet
from otx.core.config.data import (
    SamplerConfig,
    TileConfig,
    VisualPromptingConfig,
)
from otx.core.data.dataset.tile import OTXTileTransform
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.tile import TileBatchDetDataEntity
from otx.core.data.module import OTXDataModule
from otx.core.model.detection import OTXDetectionModel
from otx.core.types.task import OTXTaskType
from torchvision import tv_tensors

from tests.test_helpers import generate_random_bboxes


class TestOTXTiling:
    @pytest.fixture()
    def mock_otx_det_model(self) -> OTXDetectionModel:
        return create_autospec(OTXDetectionModel)

    @pytest.fixture()
    def fxt_det_transform_config(self) -> DictConfig:
        config = OmegaConf.load("src/otx/recipe/_base_/data/detection_tile.yaml")
        config.train_subset.input_size = config.input_size
        config.val_subset.input_size = config.input_size
        config.test_subset.input_size = config.input_size
        config.train_subset.sampler = SamplerConfig(**config.train_subset.sampler)
        return config

    @pytest.fixture()
    def fxt_det_data_config(self, fxt_det_transform_config) -> dict:
        data_root = Path(__file__).parent.parent.parent.parent / "assets" / "car_tree_bug"

        return {
            "data_format": "coco_instances",
            "data_root": data_root,
            "train_subset": fxt_det_transform_config.train_subset,
            "val_subset": fxt_det_transform_config.val_subset,
            "test_subset": fxt_det_transform_config.test_subset,
            "tile_config": TileConfig(),
            "vpm_config": VisualPromptingConfig(),
        }

    def det_dummy_forward(self, x: DetBatchDataEntity) -> DetBatchPredEntity:
        """Dummy detection forward function for testing.

        This function creates random bounding boxes for each image in the batch.
        Args:
            x (DetBatchDataEntity): Input batch data entity.

        Returns:
            DetBatchPredEntity: Output batch prediction entity.
        """
        bboxes = []
        labels = []
        scores = []
        saliency_maps = []
        feature_vectors = []
        for img_info in x.imgs_info:
            img_h, img_w = img_info.ori_shape
            img_bboxes = generate_random_bboxes(
                image_width=img_w,
                image_height=img_h,
                num_boxes=100,
            )
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    img_bboxes,
                    canvas_size=img_info.ori_shape,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    dtype=torch.float64,
                ),
            )
            labels.append(
                torch.LongTensor(len(img_bboxes)).random_(3),
            )
            scores.append(
                torch.rand(len(img_bboxes), dtype=torch.float64),
            )
            if self.explain_mode:
                saliency_maps.append(np.zeros((3, 7, 7)))
                feature_vectors.append(np.zeros((1, 32)))

        pred_entity = DetBatchPredEntity(
            batch_size=x.batch_size,
            images=x.images,
            imgs_info=x.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )
        if self.explain_mode:
            pred_entity.saliency_map = saliency_maps
            pred_entity.feature_vector = feature_vectors

        return pred_entity

    def inst_seg_dummy_forward(self, x: InstanceSegBatchDataEntity) -> InstanceSegBatchPredEntity:
        """Dummy instance segmantation forward function for testing.

        This function creates random bounding boxes/masks for each image in the batch.
        Args:
            x (InstanceSegBatchDataEntity): Input batch data entity.

        Returns:
            InstanceSegBatchPredEntity: Output batch prediction entity.
        """
        bboxes = []
        labels = []
        scores = []
        masks = []
        feature_vectors = []

        for img_info in x.imgs_info:
            img_h, img_w = img_info.ori_shape
            img_bboxes = generate_random_bboxes(
                image_width=img_w,
                image_height=img_h,
                num_boxes=100,
            )
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    img_bboxes,
                    canvas_size=img_info.ori_shape,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    dtype=torch.float64,
                ),
            )
            labels.append(
                torch.LongTensor(len(img_bboxes)).random_(3),
            )
            scores.append(
                torch.rand(len(img_bboxes), dtype=torch.float64),
            )
            masks.append(
                tv_tensors.Mask(
                    torch.randint(0, 2, (len(img_bboxes), img_h, img_w)),
                    dtype=torch.bool,
                ),
            )
            if self.explain_mode:
                feature_vectors.append(np.zeros((1, 32)))

        pred_entity = InstanceSegBatchPredEntity(
            batch_size=x.batch_size,
            images=x.images,
            imgs_info=x.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
            polygons=x.polygons,
        )
        if self.explain_mode:
            pred_entity.saliency_map = []
            pred_entity.feature_vector = feature_vectors

        return pred_entity

    def test_tile_transform(self):
        dataset = DmDataset.import_from("tests/assets/car_tree_bug", format="coco_instances")
        first_item = next(iter(dataset), None)
        height, width = first_item.media.data.shape[:2]

        rng = np.random.default_rng()
        tile_size = rng.integers(low=100, high=500, size=(2,))
        overlap = rng.random(2)
        overlap = overlap.clip(0, 0.9)
        threshold_drop_ann = rng.random()
        tiled_dataset = DmDataset.import_from("tests/assets/car_tree_bug", format="coco_instances")
        tiled_dataset.transform(
            OTXTileTransform,
            tile_size=tile_size,
            overlap=overlap,
            threshold_drop_ann=threshold_drop_ann,
            with_full_img=True,
        )

        h_stride = max(int((1 - overlap[0]) * tile_size[0]), 1)
        w_stride = max(int((1 - overlap[1]) * tile_size[1]), 1)
        num_tile_rows = (height + h_stride - 1) // h_stride
        num_tile_cols = (width + w_stride - 1) // w_stride
        assert len(tiled_dataset) == (num_tile_rows * num_tile_cols * len(dataset)) + len(
            dataset,
        ), "Incorrect number of tiles"

        tiled_dataset = DmDataset.import_from("tests/assets/car_tree_bug", format="coco_instances")
        tiled_dataset.transform(
            OTXTileTransform,
            tile_size=tile_size,
            overlap=overlap,
            threshold_drop_ann=threshold_drop_ann,
            with_full_img=False,
        )
        assert len(tiled_dataset) == (num_tile_rows * num_tile_cols * len(dataset)), "Incorrect number of tiles"

    def test_tile_polygon_func(self):
        points = np.array([(1, 2), (3, 5), (4, 2), (4, 6), (1, 6)])
        polygon = Polygon(points=points.flatten().tolist())
        roi = sg.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        inter_polygon = OTXTileTransform._tile_polygon(polygon, roi, threshold_drop_ann=0.0)
        assert isinstance(inter_polygon, Polygon), "Intersection should be a Polygon"
        assert inter_polygon.get_area() > 0, "Intersection area should be greater than 0"

        assert (
            OTXTileTransform._tile_polygon(polygon, roi, threshold_drop_ann=1.0) is None
        ), "Intersection should be None"

        invalid_polygon = Polygon(points=[0, 0, 5, 0, 5, 5, 5, 0])
        assert OTXTileTransform._tile_polygon(invalid_polygon, roi) is None, "Invalid polygon should be None"

    def test_adaptive_tiling(self, fxt_det_data_config):
        # Enable tile adapter
        fxt_det_data_config["tile_config"] = TileConfig(enable_tiler=True, enable_adaptive_tiling=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **fxt_det_data_config,
        )
        tile_datamodule.prepare_data()

        assert tile_datamodule.tile_config.tile_size == (6750, 6750), "Tile size should be [6750, 6750]"
        assert pytest.approx(tile_datamodule.tile_config.overlap, rel=1e-3) == 0.03608, "Overlap should be 0.03608"
        assert tile_datamodule.tile_config.max_num_instances == 3, "Max num instances should be 3"

    def test_tile_sampler(self, fxt_det_data_config):
        rng = np.random.default_rng()
        sampling_ratio = rng.random()
        fxt_det_data_config["tile_config"] = TileConfig(
            enable_tiler=True,
            enable_adaptive_tiling=False,
            sampling_ratio=sampling_ratio,
        )
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **fxt_det_data_config,
        )
        tile_datamodule.prepare_data()
        sampled_count = max(
            1,
            int(len(tile_datamodule._get_dataset("train")) * sampling_ratio),
        )

        count = 0
        for batch in tile_datamodule.train_dataloader():
            count += batch.batch_size
            assert isinstance(batch, DetBatchDataEntity)

        assert sampled_count == count, "Sampled count should be equal to the count of the dataloader batch size"

    def test_train_dataloader(self, fxt_det_data_config) -> None:
        # Enable tile adapter
        fxt_det_data_config["tile_config"] = TileConfig(enable_tiler=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **fxt_det_data_config,
        )
        tile_datamodule.prepare_data()
        for batch in tile_datamodule.train_dataloader():
            assert isinstance(batch, DetBatchDataEntity)

    def test_val_dataloader(self, fxt_det_data_config) -> None:
        # Enable tile adapter
        fxt_det_data_config["tile_config"] = TileConfig(enable_tiler=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **fxt_det_data_config,
        )
        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            assert isinstance(batch, TileBatchDetDataEntity)

    def test_det_tile_merge(self, fxt_det_data_config):
        model = MobileNetV2ATSS(
            label_info=3,
        )  # updated from OTXDetectionModel to avoid NotImplementedError in _build_model
        # Enable tile adapter
        fxt_det_data_config["tile_config"] = TileConfig(enable_tiler=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **fxt_det_data_config,
        )

        self.explain_mode = False
        model.forward = self.det_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            model.forward_tiles(batch)

    def test_explain_det_tile_merge(self, fxt_det_data_config):
        model = MobileNetV2ATSS(
            label_info=3,
        )  # updated from OTXDetectionModel to avoid NotImplementedError in _build_model
        # Enable tile adapter
        fxt_det_data_config["tile_config"] = TileConfig(enable_tiler=True, enable_adaptive_tiling=False)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **fxt_det_data_config,
        )

        self.explain_mode = model.explain_mode = True
        model.forward_explain = self.det_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            prediction = model.forward_tiles(batch)
            assert prediction.saliency_map[0].ndim == 3
        self.explain_mode = False

    def test_instseg_tile_merge(self, fxt_det_data_config):
        model = MaskRCNNEfficientNet(label_info=3)
        # Enable tile adapter
        fxt_det_data_config["tile_config"] = TileConfig(enable_tiler=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.INSTANCE_SEGMENTATION,
            **fxt_det_data_config,
        )

        self.explain_mode = False
        model.forward = self.inst_seg_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            model.forward_tiles(batch)

    def test_explain_instseg_tile_merge(self, fxt_det_data_config):
        model = MaskRCNNEfficientNet(label_info=3)
        # Enable tile adapter
        fxt_det_data_config["tile_config"] = TileConfig(enable_tiler=True, enable_adaptive_tiling=False)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.INSTANCE_SEGMENTATION,
            **fxt_det_data_config,
        )

        self.explain_mode = model.explain_mode = True
        model.forward_explain = self.inst_seg_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            prediction = model.forward_tiles(batch)
            assert prediction.saliency_map[0].ndim == 3
        self.explain_mode = False
