"""Tests Segment Anything."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from collections import OrderedDict
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything import (
    SegmentAnything,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockImageEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return torch.Tensor([[1]])


class MockPromptEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return torch.Tensor([[1]]), torch.Tensor([[1]])

    def get_dense_pe(self):
        return torch.Tensor([[1]])


class MockMaskDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return torch.Tensor([[1]]), torch.Tensor([[1]])


class TestSegmentAnything:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMImageEncoder",
            MockImageEncoder,
        )
        monkeypatch.setattr(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMPromptEncoder",
            MockPromptEncoder,
        )
        monkeypatch.setattr(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMMaskDecoder",
            MockMaskDecoder,
        )

        self.base_config = DictConfig(
            dict(
                model=dict(
                    backbone="vit_b",
                    image_size=1024,
                    freeze_image_encoder=True,
                    freeze_prompt_encoder=True,
                    freeze_mask_decoder=False,
                    loss_type="sam",
                    checkpoint=None,
                    mask_threshold=0.0,
                    return_logits=False,
                )
            )
        )

    @e2e_pytest_unit
    @pytest.mark.parametrize("backbone", ["vit_b", "resnet"])
    def test_set_models(self, mocker, backbone: str) -> None:
        """Test set_models."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.freeze_networks"
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.set_metrics"
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )

        config = self.base_config.copy()
        config.model.update(dict(backbone=backbone))
        if backbone == "resnet":
            with pytest.raises(NotImplementedError):
                sam = SegmentAnything(config)
        else:
            # backbone == vit_b
            sam = SegmentAnything(config)

            assert isinstance(sam.image_encoder, MockImageEncoder)
            assert isinstance(sam.prompt_encoder, MockPromptEncoder)
            assert isinstance(sam.mask_decoder, MockMaskDecoder)

    @e2e_pytest_unit
    @pytest.mark.parametrize("freeze_image_encoder", [True, False])
    @pytest.mark.parametrize("freeze_prompt_encoder", [True, False])
    @pytest.mark.parametrize("freeze_mask_decoder", [True, False])
    def test_freeze_networks(
        self, mocker, freeze_image_encoder: bool, freeze_prompt_encoder: bool, freeze_mask_decoder: bool
    ):
        """Test freeze_networks."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.set_metrics"
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )

        config = self.base_config.copy()
        config.model.update(
            dict(
                freeze_image_encoder=freeze_image_encoder,
                freeze_prompt_encoder=freeze_prompt_encoder,
                freeze_mask_decoder=freeze_mask_decoder,
            )
        )
        sam = SegmentAnything(config)

        for param in sam.image_encoder.parameters():
            if freeze_image_encoder:
                assert param.requires_grad == False
            else:
                assert param.requires_grad == True

        for param in sam.prompt_encoder.parameters():
            if freeze_prompt_encoder:
                assert param.requires_grad == False
            else:
                assert param.requires_grad == True

        for param in sam.mask_decoder.parameters():
            if freeze_mask_decoder:
                assert param.requires_grad == False
            else:
                assert param.requires_grad == True

    @e2e_pytest_unit
    @pytest.mark.parametrize("loss_type", ["sam", "medsam"])
    def test_set_metrics(self, mocker, loss_type: str):
        """Test set_metrics."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.set_models"
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.freeze_networks"
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )

        config = self.base_config.copy()
        config.model.update(dict(loss_type=loss_type))
        sam = SegmentAnything(config)

        if loss_type == "sam":
            assert "train_loss_focal" in sam.train_metrics
            assert "train_loss_iou" in sam.train_metrics

        elif loss_type == "medsam":
            assert "train_loss_ce" in sam.train_metrics

        else:
            assert 0

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "is_backbone_arg,state_dict",
        [
            (
                False,
                OrderedDict(
                    [
                        ("image_encoder.weight", Tensor([[0.0]])),
                        ("image_encoder.bias", Tensor([0.0])),
                        ("prompt_encoder.layer.weight", Tensor([[0.0]])),
                        ("prompt_encoder.layer.bias", Tensor([0.0])),
                        ("mask_decoder.layer.weight", Tensor([[0.0]])),
                        ("mask_decoder.layer.bias", Tensor([0.0])),
                    ]
                ),
            ),
            (
                True,
                OrderedDict(
                    [
                        ("image_encoder.backbone.weight", Tensor([[1.0]])),
                        ("image_encoder.backbone.bias", Tensor([1.0])),
                        ("prompt_encoder.layer.weight", Tensor([[1.0]])),
                        ("prompt_encoder.layer.bias", Tensor([1.0])),
                        ("mask_decoder.layer.weight", Tensor([[1.0]])),
                        ("mask_decoder.layer.bias", Tensor([1.0])),
                    ]
                ),
            ),
        ],
    )
    def test_load_checkpoint_with_state_dict(self, mocker, is_backbone_arg: bool, state_dict: OrderedDict):
        """Test load_checkpoint with state_dict."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.freeze_networks"
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.set_metrics"
        )

        sam = SegmentAnything(self.base_config, state_dict=state_dict)
        sam_state_dict = sam.state_dict()

        for k, v in state_dict.items():
            if not is_backbone_arg:
                k = k.replace("image_encoder", "image_encoder.backbone")
            assert k in sam_state_dict
            assert v == sam_state_dict[k]

    @e2e_pytest_unit
    @pytest.mark.parametrize("checkpoint", [None, "checkpoint", "http://checkpoint"])
    def test_load_checkpoint(self, mocker, monkeypatch, checkpoint: str):
        """Test load_checkpoint."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.freeze_networks"
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.set_metrics"
        )
        if checkpoint is not None:
            monkeypatch.setattr("torch.hub.load_state_dict_from_url", lambda *args, **kwargs: OrderedDict())
            monkeypatch.setattr("torch.load", lambda *args, **kwargs: None)

            mocker_load_state_dict = mocker.patch(
                "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_state_dict"
            )
            if checkpoint.startswith("http"):
                mocker_load_from_checkpoint = mocker.patch(
                    "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_from_checkpoint",
                    side_effect=ValueError(),
                )
            else:
                mocker_load_from_checkpoint = mocker.patch(
                    "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_from_checkpoint"
                )

        config = self.base_config.copy()
        config.model.update(dict(checkpoint=checkpoint))

        sam = SegmentAnything(config, state_dict=None)

        if checkpoint is None:
            assert True
        elif checkpoint.startswith("http"):
            mocker_load_state_dict.assert_called_once()
        else:
            mocker_load_from_checkpoint.assert_called_once()

    @e2e_pytest_unit
    def test_forward(self) -> None:
        """Test forward."""
        sam = SegmentAnything(config=self.base_config)
        images = torch.zeros((1))
        bboxes = torch.zeros((1))

        results = sam.forward(images=images, bboxes=bboxes, points=None)
        pred_masks, ious = results

        assert len(bboxes) == len(pred_masks) == len(ious)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "loss_type,expected", [("sam", torch.tensor(2.4290099144)), ("medsam", torch.tensor(0.9650863409))]
    )
    def test_training_step(self, mocker, loss_type: str, expected: Tensor) -> None:
        """Test training_step."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.forward",
            return_value=([torch.Tensor([[0, 1, 1, 0] for _ in range(4)])], [torch.tensor(1.0)]),
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.postprocess_masks",
            return_value=torch.Tensor([[0, 1, 1, 0] for _ in range(4)]),
        )

        config = self.base_config.copy()
        config.model.update(dict(loss_type=loss_type))
        sam = SegmentAnything(config=config)
        batch = dict(
            images=torch.ones((1, 3, 4, 4)),
            gt_masks=[torch.Tensor([[0, 1, 1, 0] for _ in range(4)]).to(torch.int32)],
            bboxes=torch.Tensor([[0, 0, 1, 1]]),
            points=[],
        )

        results = sam.training_step(batch, None)

        assert torch.equal(results, expected)

    @e2e_pytest_unit
    def test_training_epoch_end(self) -> None:
        """Test training_epoch_end."""
        sam = SegmentAnything(config=self.base_config)
        for k, v in sam.train_metrics.items():
            v.update(torch.zeros((2, 2)), torch.zeros((2, 2), dtype=torch.int32))

        sam.training_epoch_end(None)

        assert sam.train_metrics["train_Dice"].compute() == 0.0
        assert sam.train_metrics["train_F1"].compute() == 0.0
        assert sam.train_metrics["train_IoU"].compute().isnan()
        assert sam.train_metrics["train_loss"].compute().isnan()
        assert sam.train_metrics["train_loss_dice"].compute().isnan()
        assert sam.train_metrics["train_loss_focal"].compute().isnan()
        assert sam.train_metrics["train_loss_iou"].compute().isnan()

    @e2e_pytest_unit
    def test_validation_step(self, mocker) -> None:
        """Test validation_step."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.forward",
            return_value=([torch.Tensor([[0, 1, 1, 0] for _ in range(4)])], [torch.tensor(1.0)]),
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.postprocess_masks",
            return_value=torch.Tensor([[0, 1, 1, 0] for _ in range(4)]),
        )
        sam = SegmentAnything(config=self.base_config)
        batch = dict(
            images=torch.ones((1, 3, 4, 4)),
            gt_masks=[torch.Tensor([[0, 1, 1, 0] for _ in range(4)]).to(torch.int32)],
            bboxes=torch.Tensor([[0, 0, 1, 1]]),
            points=[],
            path=None,
            labels=None,
            padding=[0],
            original_size=[0],
        )

        results = sam.validation_step(batch, None)

        assert torch.equal(results["val_Dice"].compute(), torch.tensor(0.6666666865))
        assert torch.equal(results["val_F1"].compute(), torch.tensor(1.0))
        assert torch.equal(results["val_IoU"].compute(), torch.tensor(1.0))

    @e2e_pytest_unit
    def test_validation_epoch_end(self) -> None:
        """Test validation_epoch_end."""
        sam = SegmentAnything(config=self.base_config)
        for k, v in sam.val_metrics.items():
            v.update(torch.zeros((2, 2)), torch.zeros((2, 2), dtype=torch.int32))

        sam.validation_epoch_end(None)

        assert sam.val_metrics["val_Dice"].compute() == 0.0
        assert sam.val_metrics["val_F1"].compute() == 0.0
        assert sam.val_metrics["val_IoU"].compute().isnan()

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "return_logits,expected",
        [
            (True, torch.Tensor([[0.5 for _ in range(4)] for _ in range(4)])),
            (False, torch.Tensor([[False for _ in range(4)] for _ in range(4)])),
        ],
    )
    def test_predict_step(self, mocker, return_logits: bool, expected: Tensor) -> None:
        """Test predict_step."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.forward",
            return_value=([torch.zeros((4, 4))], [torch.tensor(1.0)]),
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.postprocess_masks",
            return_value=torch.zeros((4, 4)),
        )

        config = self.base_config.copy()
        config.model.update(dict(return_logits=return_logits))
        sam = SegmentAnything(config=config)
        batch = dict(
            images=torch.zeros((1, 3, 4, 4)),
            bboxes=torch.Tensor([[0, 0, 1, 1]]),
            points=[],
            path=None,
            labels=None,
            padding=[0],
            original_size=[0],
        )

        results = sam.predict_step(batch, None)

        assert torch.equal(results["masks"][0], expected)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "input_size,original_size,is_predict,expected",
        [
            ((6, 6), (8, 8), True, (8, 8)),
            ((6, 6), (8, 8), False, (6, 6)),
        ],
    )
    def test_postprocess_masks(
        self, input_size: Tuple[int], original_size: Tuple[int], is_predict: bool, expected: Tuple[int]
    ) -> None:
        """Test postprocess_masks."""
        sam = SegmentAnything(config=self.base_config)
        masks = torch.zeros((1, 1, 4, 4))

        results = sam.postprocess_masks(masks, input_size, original_size=original_size, is_predict=is_predict)

        assert results.shape[1:] == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "inputs,targets,expected",
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.25])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.3888888359])),
        ],
    )
    def test_calculate_dice_loss(self, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_dice_loss."""
        sam = SegmentAnything(config=self.base_config)

        results = sam.calculate_dice_loss(inputs, targets, num_masks=1)

        assert torch.isclose(results, expected)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "inputs,targets,expected",
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0098766042])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0226361733])),
        ],
    )
    def test_calculate_sigmoid_ce_focal_loss(self, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_sigmoid_ce_focal_loss."""
        sam = SegmentAnything(config=self.base_config)

        results = sam.calculate_sigmoid_ce_focal_loss(inputs, targets, num_masks=1)

        assert torch.isclose(results, expected)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "inputs,targets,expected",
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([1.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([1.0])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
        ],
    )
    def test_calculate_iou(self, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_iou."""
        sam = SegmentAnything(config=self.base_config)

        results = sam.calculate_iou(inputs, targets)

        assert results == expected
