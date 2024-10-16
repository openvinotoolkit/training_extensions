# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import shapely.geometry as sg
import torch
from datumaro import Dataset as DmDataset
from datumaro import Polygon
from model_api.models import Model
from model_api.models.utils import ImageResultWithSoftPrediction
from model_api.tilers import SemanticSegmentationTiler
from omegaconf import OmegaConf
from otx.algo.detection.atss import ATSS
from otx.algo.instance_segmentation.maskrcnn import MaskRCNN
from otx.algo.segmentation.litehrnet import LiteHRNet
from otx.core.config.data import (
    SamplerConfig,
    TileConfig,
    VisualPromptingConfig,
)
from otx.core.data.dataset.tile import OTXTileTransform
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity
from otx.core.data.entity.tile import TileBatchDetDataEntity, TileBatchInstSegDataEntity, TileBatchSegDataEntity
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
    def fxt_data_roots(self) -> dict[OTXTaskType, Path]:
        parent_root = Path(__file__).parent.parent.parent.parent / "assets"
        return {
            OTXTaskType.DETECTION: parent_root / "car_tree_bug",
            OTXTaskType.INSTANCE_SEGMENTATION: parent_root / "car_tree_bug",
            OTXTaskType.SEMANTIC_SEGMENTATION: parent_root / "common_semantic_segmentation_dataset" / "supervised",
        }

    @pytest.fixture()
    def fxt_data_config(self, fxt_data_roots) -> dict[dict]:
        torchvision_base = OmegaConf.load("src/otx/recipe/_base_/data/torchvision_base.yaml")
        transforms = torchvision_base.train_subset.transforms
        transforms.append(
            {
                "class_path": "torchvision.transforms.v2.ToDtype",
                "init_args": {
                    "dtype": "${as_torch_dtype:torch.float32}",
                    "scale": True,
                },
            },
        )

        batch_size = 8
        num_workers = 0

        train_subset = SubsetConfig(
            subset_name="train",
            batch_size=batch_size,
            num_workers=num_workers,
            transform_lib_type=TransformLibType.TORCHVISION,
            transforms=transforms,
        )

        val_subset = SubsetConfig(
            subset_name="val",
            batch_size=batch_size,
            num_workers=num_workers,
            transform_lib_type=TransformLibType.TORCHVISION,
            transforms=transforms,
        )

        test_subset = SubsetConfig(
            subset_name="test",
            batch_size=batch_size,
            num_workers=num_workers,
            transform_lib_type=TransformLibType.TORCHVISION,
            transforms=transforms,
        )

        return {
            OTXTaskType.DETECTION: {
                "data_format": "coco_instances",
                "data_root": fxt_data_roots[OTXTaskType.DETECTION],
                "train_subset": train_subset,
                "val_subset": val_subset,
                "test_subset": test_subset,
                "tile_config": TileConfig(),
                "vpm_config": VisualPromptingConfig(),
            },
            OTXTaskType.INSTANCE_SEGMENTATION: {
                "data_format": "coco_instances",
                "data_root": fxt_data_roots[OTXTaskType.INSTANCE_SEGMENTATION],
                "train_subset": train_subset,
                "val_subset": val_subset,
                "test_subset": test_subset,
                "tile_config": TileConfig(),
                "vpm_config": VisualPromptingConfig(),
            },
            OTXTaskType.SEMANTIC_SEGMENTATION: {
                "data_format": "common_semantic_segmentation_with_subset_dirs",
                "data_root": fxt_data_roots[OTXTaskType.SEMANTIC_SEGMENTATION],
                "train_subset": train_subset,
                "val_subset": val_subset,
                "test_subset": test_subset,
                "tile_config": TileConfig(),
                "vpm_config": VisualPromptingConfig(),
            },
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

    @pytest.mark.parametrize(
        "task",
        [OTXTaskType.DETECTION, OTXTaskType.INSTANCE_SEGMENTATION, OTXTaskType.SEMANTIC_SEGMENTATION],
    )
    def test_tile_transform(self, task, fxt_data_roots):
        if task in (OTXTaskType.INSTANCE_SEGMENTATION, OTXTaskType.DETECTION):
            dataset_format = "coco_instances"
        elif task == OTXTaskType.SEMANTIC_SEGMENTATION:
            dataset_format = "common_semantic_segmentation_with_subset_dirs"
        else:
            pytest.skip("Task not supported")

        data_root = str(fxt_data_roots[task])
        dataset = DmDataset.import_from(data_root, format=dataset_format)

        rng = np.random.default_rng()
        tile_size = rng.integers(low=50, high=128, size=(2,))
        overlap = rng.random(2)
        overlap = overlap.clip(0, 0.9)
        threshold_drop_ann = rng.random()
        tiled_dataset = DmDataset.import_from(data_root, format=dataset_format)
        tiled_dataset.transform(
            OTXTileTransform,
            tile_size=tile_size,
            overlap=overlap,
            threshold_drop_ann=threshold_drop_ann,
            with_full_img=True,
        )

        h_stride = max(int((1 - overlap[0]) * tile_size[0]), 1)
        w_stride = max(int((1 - overlap[1]) * tile_size[1]), 1)

        num_tiles = 0
        for dataset_item in dataset:
            height, width = dataset_item.media.data.shape[:2]
            for _ in range(0, height, h_stride):
                for _ in range(0, width, w_stride):
                    num_tiles += 1

        assert len(tiled_dataset) == num_tiles + len(dataset), "Incorrect number of tiles"

        tiled_dataset = DmDataset.import_from(data_root, format=dataset_format)
        tiled_dataset.transform(
            OTXTileTransform,
            tile_size=tile_size,
            overlap=overlap,
            threshold_drop_ann=threshold_drop_ann,
            with_full_img=False,
        )
        assert len(tiled_dataset) == num_tiles, "Incorrect number of tiles"

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

    def test_adaptive_tiling(self, fxt_data_config):
        for task, data_config in fxt_data_config.items():
            # Enable tile adapter
            data_config["tile_config"] = TileConfig(enable_tiler=True, enable_adaptive_tiling=True)

            if task is OTXTaskType.DETECTION:
                tile_datamodule = OTXDataModule(
                    task=OTXTaskType.DETECTION,
                    **data_config,
                )
                tile_datamodule.prepare_data()
                assert tile_datamodule.tile_config.tile_size == (8380, 8380), "Tile size should be [8380, 8380]"
                assert (
                    pytest.approx(tile_datamodule.tile_config.overlap, rel=1e-3) == 0.04255
                ), "Overlap should be 0.04255"
                assert tile_datamodule.tile_config.max_num_instances == 3, "Max num instances should be 3"
            elif task is OTXTaskType.INSTANCE_SEGMENTATION:
                tile_datamodule = OTXDataModule(
                    task=OTXTaskType.INSTANCE_SEGMENTATION,
                    **data_config,
                )
                tile_datamodule.prepare_data()
                assert tile_datamodule.tile_config.tile_size == (6750, 6750), "Tile size should be [6750, 6750]"
                assert (
                    pytest.approx(tile_datamodule.tile_config.overlap, rel=1e-3) == 0.03608
                ), "Overlap should be 0.03608"
                assert tile_datamodule.tile_config.max_num_instances == 3, "Max num instances should be 3"
            elif task is OTXTaskType.SEMANTIC_SEGMENTATION:
                tile_datamodule = OTXDataModule(
                    task=OTXTaskType.SEMANTIC_SEGMENTATION,
                    **data_config,
                )
                tile_datamodule.prepare_data()
                assert tile_datamodule.tile_config.tile_size == (2878, 2878), "Tile size should be [6750, 6750]"
                assert (
                    pytest.approx(tile_datamodule.tile_config.overlap, rel=1e-3) == 0.04412
                ), "Overlap should be 0.04412"
                assert tile_datamodule.tile_config.max_num_instances == 2, "Max num instances should be 3"
            else:
                pytest.skip("Task not supported")

    def test_tile_sampler(self, fxt_data_config):
        for task, data_config in fxt_data_config.items():
            rng = np.random.default_rng()
            sampling_ratio = rng.random()
            data_config["tile_config"] = TileConfig(
                enable_tiler=True,
                enable_adaptive_tiling=False,
                sampling_ratio=sampling_ratio,
            )
            tile_datamodule = OTXDataModule(
                task=task,
                **data_config,
            )
            tile_datamodule.prepare_data()
            sampled_count = max(
                1,
                int(len(tile_datamodule._get_dataset("train")) * sampling_ratio),
            )

            count = 0
            for batch in tile_datamodule.train_dataloader():
                count += batch.batch_size
                if task is OTXTaskType.DETECTION:
                    assert isinstance(batch, DetBatchDataEntity)
                elif task is OTXTaskType.INSTANCE_SEGMENTATION:
                    assert isinstance(batch, InstanceSegBatchDataEntity)
                elif task is OTXTaskType.SEMANTIC_SEGMENTATION:
                    assert isinstance(batch, SegBatchDataEntity)
                else:
                    pytest.skip("Task not supported")

            assert sampled_count == count, "Sampled count should be equal to the count of the dataloader batch size"

    def test_train_dataloader(self, fxt_data_config) -> None:
        for task, data_config in fxt_data_config.items():
            # Enable tile adapter
            data_config["tile_config"] = TileConfig(enable_tiler=True)
            tile_datamodule = OTXDataModule(
                task=task,
                **data_config,
            )
            tile_datamodule.prepare_data()
            for batch in tile_datamodule.train_dataloader():
                if task is OTXTaskType.DETECTION:
                    assert isinstance(batch, DetBatchDataEntity)
                elif task is OTXTaskType.INSTANCE_SEGMENTATION:
                    assert isinstance(batch, InstanceSegBatchDataEntity)
                elif task is OTXTaskType.SEMANTIC_SEGMENTATION:
                    assert isinstance(batch, SegBatchDataEntity)
                else:
                    pytest.skip("Task not supported")

    def test_val_dataloader(self, fxt_data_config) -> None:
        for task, data_config in fxt_data_config.items():
            # Enable tile adapter
            data_config["tile_config"] = TileConfig(enable_tiler=True)
            tile_datamodule = OTXDataModule(
                task=task,
                **data_config,
            )
            tile_datamodule.prepare_data()
            for batch in tile_datamodule.val_dataloader():
                if task is OTXTaskType.DETECTION:
                    assert isinstance(batch, TileBatchDetDataEntity)
                elif task is OTXTaskType.INSTANCE_SEGMENTATION:
                    assert isinstance(batch, TileBatchInstSegDataEntity)
                elif task is OTXTaskType.SEMANTIC_SEGMENTATION:
                    assert isinstance(batch, TileBatchSegDataEntity)
                else:
                    pytest.skip("Task not supported")

    def test_det_tile_merge(self, fxt_data_config):
        data_config = fxt_data_config[OTXTaskType.DETECTION]
        model = ATSS(
            model_name="atss_mobilenetv2",
            label_info=3,
        )  # updated from OTXDetectionModel to avoid NotImplementedError in _build_model
        # Enable tile adapter
        data_config["tile_config"] = TileConfig(enable_tiler=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **data_config,
        )

        self.explain_mode = False
        model.forward = self.det_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            model.forward_tiles(batch)

    def test_explain_det_tile_merge(self, fxt_data_config):
        data_config = fxt_data_config[OTXTaskType.DETECTION]
        model = ATSS(
            model_name="atss_mobilenetv2",
            label_info=3,
        )  # updated from OTXDetectionModel to avoid NotImplementedError in _build_model
        # Enable tile adapter
        data_config["tile_config"] = TileConfig(enable_tiler=True, enable_adaptive_tiling=False)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            **data_config,
        )

        self.explain_mode = model.explain_mode = True
        model.forward_explain = self.det_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            prediction = model.forward_tiles(batch)
            assert prediction.saliency_map[0].ndim == 3
        self.explain_mode = False

    def test_instseg_tile_merge(self, fxt_data_config):
        data_config = fxt_data_config[OTXTaskType.INSTANCE_SEGMENTATION]
        model = MaskRCNN(label_info=3, model_name="maskrcnn_efficientnet_b2b")
        # Enable tile adapter
        data_config["tile_config"] = TileConfig(enable_tiler=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.INSTANCE_SEGMENTATION,
            **data_config,
        )

        self.explain_mode = False
        model.forward = self.inst_seg_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            model.forward_tiles(batch)

    def test_explain_instseg_tile_merge(self, fxt_data_config):
        data_config = fxt_data_config[OTXTaskType.INSTANCE_SEGMENTATION]
        model = MaskRCNN(label_info=3, model_name="maskrcnn_efficientnet_b2b")
        # Enable tile adapter
        data_config["tile_config"] = TileConfig(enable_tiler=True, enable_adaptive_tiling=False)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.INSTANCE_SEGMENTATION,
            **data_config,
        )

        self.explain_mode = model.explain_mode = True
        model.forward_explain = self.inst_seg_dummy_forward

        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            prediction = model.forward_tiles(batch)
            assert prediction.saliency_map[0].ndim == 3
        self.explain_mode = False

    def test_seg_tile_merge(self, fxt_data_config):
        data_config = fxt_data_config[OTXTaskType.SEMANTIC_SEGMENTATION]
        model = LiteHRNet(label_info=3, model_name="lite_hrnet_18")
        # Enable tile adapter
        data_config["tile_config"] = TileConfig(enable_tiler=True)
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.SEMANTIC_SEGMENTATION,
            **data_config,
        )

        self.explain_mode = False
        model.eval()
        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            model.forward_tiles(batch)

    def test_seg_tiler(self, mocker):
        rng = np.random.default_rng()
        rnd_tile_size = rng.integers(low=100, high=500)
        rnd_tile_overlap = rng.random() * 0.75
        image_size = rng.integers(low=1000, high=5000)
        np_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        mock_model = MagicMock(spec=Model)
        mocker.patch("model_api.tilers.tiler.Tiler.__init__", return_value=None)
        mocker.patch.multiple(SemanticSegmentationTiler, __abstractmethods__=set())

        num_labels = rng.integers(low=1, high=10)

        tiler = SemanticSegmentationTiler(model=mock_model)
        tiler.model = mock_model
        tiler.model.labels = [f"label{i}" for i in range(num_labels)]
        tiler.tile_with_full_img = True
        tiler.tile_size = rnd_tile_size
        tiler.tiles_overlap = rnd_tile_overlap

        tile_results = []
        for coord in tiler._tile(image=np_image):
            x1, y1, x2, y2 = coord
            h, w = y2 - y1, x2 - x1
            tile_predictions = ImageResultWithSoftPrediction(
                resultImage=np.zeros((h, w), dtype=np.uint8),
                soft_prediction=np.random.rand(h, w, num_labels),
                feature_vector=np.array([]),
                saliency_map=np.array([]),
            )
            tile_result = tiler._postprocess_tile(tile_predictions, coord)
            tile_results.append(tile_result)
        tiler._merge_results(tile_results, np_image.shape)
