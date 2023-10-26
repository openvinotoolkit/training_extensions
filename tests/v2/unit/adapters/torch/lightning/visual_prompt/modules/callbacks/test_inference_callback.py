"""Tests inference callback for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import numpy as np
import pytest
import torch
from bson import ObjectId
from otx.v2.adapters.torch.lightning.visual_prompt.modules.callbacks import (
    InferenceCallback,
)
from otx.v2.api.entities.annotation import Annotation
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.image import Image
from otx.v2.api.entities.label import Domain, LabelEntity
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.shapes.polygon import Point, Polygon

from tests.v2.unit.adapters.torch.lightning.visual_prompt.test_helpers import (
    generate_visual_prompting_dataset,
)


class TestInferenceCallback:
    @pytest.fixture(autouse=True)
    def setup(self, mocker, monkeypatch) -> None:
        monkeypatch.setattr(
            "otx.v2.api.entities.utils.segmentation_utils.create_annotation_from_segmentation_map",
            lambda *_, **__: Annotation(
                shape=Image(data=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), size=(3, 3)),
                labels=[ScoredLabel(label=LabelEntity("foreground", domain=Domain.VISUAL_PROMPTING), probability=0.9)],
                id=ID(ObjectId()),
            ),
        )
        monkeypatch.setattr(
            "otx.v2.api.entities.utils.segmentation_utils.create_hard_prediction_from_soft_prediction",
            lambda *_, **__: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        )

        self.mocker_trainer = mocker.patch("pytorch_lightning.Trainer")
        self.mocker_lightning_module = mocker.patch("pytorch_lightning.LightningModule")

    @pytest.mark.parametrize(
        ("use_mask", "expected"),
        [
            (True, np.ones((3, 3), dtype=np.uint8)),
            (
                False,
                [
                    Point(0.0, 0.0),
                    Point(0.0, 0.5),
                    Point(0.0, 1.0),
                    Point(0.5, 1.0),
                    Point(1.0, 1.0),
                    Point(1.0, 0.5),
                    Point(1.0, 0.0),
                    Point(0.5, 0.0),
                ],
            ),
        ],
    )
    def test_on_predict_epoch_end(self, use_mask: bool, expected: tuple | list) -> None:
        """Test on_predict_epoch_end."""
        otx_dataset = generate_visual_prompting_dataset(use_mask=use_mask)
        inference_callback = InferenceCallback(otx_dataset)

        outputs = [
            [
                {
                    "masks": [torch.Tensor([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]])],
                    "iou_predictions": [torch.Tensor([[0.9]])],
                    "labels": [
                        [ScoredLabel(label=LabelEntity("foreground", domain=Domain.VISUAL_PROMPTING), probability=0.0)],
                    ],
                }
            ]
        ]

        inference_callback.on_predict_epoch_end(self.mocker_trainer, self.mocker_lightning_module, outputs)
        predicted_otx_dataset = inference_callback.otx_dataset

        assert len(predicted_otx_dataset) == 4
        dataset_item = predicted_otx_dataset[0]
        assert len(dataset_item.annotation_scene.annotations) == 1

        annotation = dataset_item.annotation_scene.annotations[0]
        assert isinstance(annotation, Annotation)
        if use_mask:
            assert isinstance(annotation.shape, Image)
            assert (annotation.shape.numpy == expected).all()
        else:
            assert isinstance(annotation.shape, Polygon)
            assert annotation.shape.points == expected
            assert annotation.get_labels()[0].name == "foreground"
            assert annotation.get_labels()[0].probability == 0.5
