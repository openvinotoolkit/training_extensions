# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
from typing import Any, Dict, Type

import datumaro as dm
import pytest
import torch
from mmcv import ConfigDict
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models.builder import build_detector
from mmdet.models.detectors import BaseDetector

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain


class TestLossDynamicsTrackingMixin:
    @pytest.fixture()
    def dataloader(self, fxt_det_dataset_entity: DatasetEntity):
        dataloader_cfg = dict(samples_per_gpu=len(fxt_det_dataset_entity), workers_per_gpu=1)
        dataset_cfg = ConfigDict(
            dict(
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
                    dict(type="RandomFlip", flip_ratio=0.5),
                    dict(type="DefaultFormatBundle"),
                    dict(
                        type="Collect",
                        keys=["img", "gt_bboxes", "gt_labels"],
                        meta_keys=(
                            "filename",
                            "ori_shape",
                            "img_shape",
                            "pad_shape",
                            "scale_factor",
                            "flip",
                            "img_norm_cfg",
                            "gt_ann_ids",
                        ),
                    ),
                ],
                otx_dataset=fxt_det_dataset_entity,
                labels=fxt_det_dataset_entity.get_labels(),
                domain=Domain.DETECTION,
            )
        )

        dataset = build_dataset(dataset_cfg)
        dataloader = build_dataloader(dataset, **dataloader_cfg)

        return dataloader

    @pytest.fixture()
    def detector(self, request: Type[pytest.FixtureRequest], fxt_det_dataset_entity: DatasetEntity) -> BaseDetector:
        fxt_cfg_detector = request.getfixturevalue(request.param)
        fxt_cfg_detector["track_loss_dynamics"] = True

        detector = build_detector(fxt_cfg_detector)
        detector.loss_dyns_tracker.init_with_otx_dataset(fxt_det_dataset_entity)
        return detector

    TESTCASE = [
        "fxt_cfg_custom_atss",
        "fxt_cfg_custom_ssd",
        "fxt_cfg_custom_vfnet",
        "fxt_cfg_custom_yolox",
    ]

    @torch.no_grad()
    @pytest.mark.parametrize("detector", TESTCASE, indirect=True)
    def test_train_step(self, detector, dataloader: Dict[str, Any], tmp_dir_path: str):
        for data in dataloader:
            outputs = detector.train_step({k: v.data[0] for k, v in data.items()}, None)

        output_keys = {key for key in outputs.keys()}
        for loss_type in detector.TRACKING_LOSS_TYPE:
            assert loss_type in output_keys

        n_steps = 3
        for iter in range(n_steps):
            detector.loss_dyns_tracker.accumulate(outputs, iter)

        export_dir = osp.join(tmp_dir_path, "noisy_label_detection")
        detector.loss_dyns_tracker.export(export_dir)

        dataset = dm.Dataset.import_from(export_dir, format="datumaro")

        cnt = 0
        for item in dataset:
            for ann in item.annotations:
                has_attrs = False
                for v in ann.attributes.values():
                    assert set(list(ann.attributes.keys())) == {
                        "iters",
                        *[f"loss_dynamics_{loss_type.name}" for loss_type in detector.TRACKING_LOSS_TYPE],
                    }
                    assert len(v) == n_steps
                    has_attrs = True
                if has_attrs:
                    cnt += 1

        for loss_type, values in outputs.items():
            if loss_type in detector.TRACKING_LOSS_TYPE:
                assert cnt == len(
                    values
                ), "The number of accumulated statistics is equal to the number of Datumaro items which have attirbutes."
