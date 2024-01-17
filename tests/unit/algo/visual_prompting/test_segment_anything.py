# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from otx.algo.visual_prompting.segment_anything import OTXSegmentAnything, SegmentAnything
from otx.core.data.entity.visual_prompting import VisualPromptingBatchPredEntity
from torch import Tensor
from torchvision import tv_tensors


class TestSegmentAnything:
    @pytest.mark.parametrize(("backbone", "expected_backbone"), [("tiny_vit", "TinyViT")])
    @pytest.mark.parametrize("freeze_image_encoder", [True, False])
    @pytest.mark.parametrize("freeze_prompt_encoder", [True, False])
    @pytest.mark.parametrize("freeze_mask_decoder", [True, False])
    def test_init(
        self,
        mocker,
        backbone: str,
        expected_backbone: str,
        freeze_image_encoder: bool,
        freeze_prompt_encoder: bool,
        freeze_mask_decoder: bool,
    ) -> None:
        """Test __init__."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_state_dict")
        mocker_load_state_dict_from_url = mocker.patch(
            "otx.algo.visual_prompting.segment_anything.torch.hub.load_state_dict_from_url",
        )
        segment_anything = SegmentAnything(
            backbone=backbone,
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
        )

        # check import modules
        assert hasattr(segment_anything, "image_encoder")
        assert segment_anything.image_encoder.__class__.__name__ == expected_backbone
        assert hasattr(segment_anything, "prompt_encoder")
        assert hasattr(segment_anything, "mask_decoder")

        # check load_checkpoint
        mocker_load_state_dict_from_url.assert_called_once()

        # check freeze_networks
        for param in segment_anything.image_encoder.parameters():
            assert param.requires_grad == (freeze_image_encoder is False)

        for param in segment_anything.prompt_encoder.parameters():
            assert param.requires_grad == (freeze_prompt_encoder is False)

        for param in segment_anything.mask_decoder.parameters():
            assert param.requires_grad == (freeze_mask_decoder is False)

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize(
        "ori_shapes",
        [
            [torch.tensor([512, 256])],
            [torch.tensor([256, 512])],
            [torch.tensor([1536, 1280])],
            [torch.tensor([1280, 1536])],
        ],
    )
    def test_forward(self, mocker, training: bool, ori_shapes: list[Tensor]) -> None:
        """Test forward."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")
        segment_anything.training = training

        images = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
        bboxes = [torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)]
        gt_masks = [torch.zeros((1, *os)) for os in ori_shapes] if training else None

        results = segment_anything(
            images=images,
            ori_shapes=ori_shapes,
            bboxes=bboxes,
            points=None,  # TODO(sungchul): enable point prompts # noqa: TD003
            gt_masks=gt_masks,
        )

        if training:
            # check loss
            assert isinstance(results, dict)
            for metric in ["loss", "loss_focal", "loss_dice", "loss_iou"]:
                assert isinstance(results[metric], Tensor)
            assert results["loss"].ndim == 0
        else:
            assert isinstance(results, tuple)
            assert all([isinstance(r, list) for r in results])  # noqa: C419
            assert all([isinstance(r[0], Tensor) for r in results])  # noqa: C419

            # check post_processed_pred_masks
            assert results[0][0].shape == torch.Size(ori_shapes[0])

            # check ious
            assert results[1][0].ndim == 0

    @pytest.mark.parametrize(
        ("inputs", "targets", "expected"),
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.25])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.3888888359])),
        ],
    )
    def test_calculate_dice_loss(self, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_dice_loss."""
        segment_anything = SegmentAnything(backbone="tiny_vit")

        results = segment_anything.calculate_dice_loss(inputs, targets, num_masks=1)

        assert torch.isclose(results, expected)

    @pytest.mark.parametrize(
        ("inputs", "targets", "expected"),
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0098766042])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0226361733])),
        ],
    )
    def test_calculate_sigmoid_ce_focal_loss(self, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_sigmoid_ce_focal_loss."""
        segment_anything = SegmentAnything(backbone="tiny_vit")

        results = segment_anything.calculate_sigmoid_ce_focal_loss(inputs, targets, num_masks=1)

        assert torch.isclose(results, expected)

    @pytest.mark.parametrize(
        ("inputs", "targets", "expected"),
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([1.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([1.0])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
        ],
    )
    def test_calculate_iou(self, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_iou."""
        segment_anything = SegmentAnything(backbone="tiny_vit")

        results = segment_anything.calculate_iou(inputs, targets)

        assert results == expected

    @pytest.mark.parametrize(
        ("input_size", "orig_size", "expected"),
        [
            (6, torch.tensor((8, 8)), torch.Size((8, 8))),
            (6, torch.tensor((10, 8)), torch.Size((10, 8))),
            (6, torch.tensor((8, 10)), torch.Size((8, 10))),
        ],
    )
    def test_postprocess_masks(
        self,
        input_size: int,
        orig_size: Tensor,
        expected: torch.Size,
    ) -> None:
        """Test postprocess_masks."""
        segment_anything = SegmentAnything(backbone="tiny_vit")
        masks = torch.zeros((1, 1, 4, 4))

        results = segment_anything.postprocess_masks(masks, input_size, orig_size).squeeze()

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("input_image_size", "expected"),
        [
            (torch.tensor((2, 4)), torch.tensor((3, 6))),
            (torch.tensor((4, 2)), torch.tensor((6, 3))),
        ],
    )
    def test_get_prepadded_size(self, input_image_size: Tensor, expected: Tensor) -> None:
        """Test get_prepadded_size."""
        segment_anything = SegmentAnything(backbone="tiny_vit")

        longest_side = 6

        results = segment_anything.get_prepadded_size(input_image_size, longest_side)

        assert torch.all(results == expected)


class TestOTXSegmentAnything:
    @pytest.fixture()
    def config(self) -> DictConfig:
        return OmegaConf.load("src/otx/recipe/visual_prompting/sam_tiny_vit.yaml").model.otx_model.config

    @pytest.fixture()
    def model(self, config) -> OTXSegmentAnything:
        return OTXSegmentAnything(num_classes=1, config=config)

    def test_create_model(self, model) -> None:
        """Test _create_model."""
        segment_anything = model._create_model()
        assert segment_anything is not None
        assert isinstance(segment_anything, torch.nn.Module)
        assert segment_anything.__class__.__name__ == "SegmentAnything"

    def test_customize_inputs(self, model, fxt_vpm_data_entity) -> None:
        """Test _customize_inputs."""
        output_data = model._customize_inputs(fxt_vpm_data_entity[2])
        assert output_data is not None
        assert output_data["images"].shape[-2:] == torch.Size(output_data["ori_shapes"][0])
        assert output_data["gt_masks"][0].shape[-2:] == torch.Size(output_data["ori_shapes"][0])

    def test_customize_outputs(self, model, fxt_vpm_data_entity) -> None:
        """Test _customize_outputs."""
        outputs = {"loss": torch.tensor(1.0)}
        result = model._customize_outputs(outputs, fxt_vpm_data_entity[2])
        assert isinstance(result, dict)
        assert "loss" in result

        model.training = False
        outputs = (torch.tensor([1]), torch.tensor([1]))
        result = model._customize_outputs(outputs, fxt_vpm_data_entity[2])
        assert isinstance(result, VisualPromptingBatchPredEntity)
        assert result.masks[0].data == outputs[0]
        assert result.scores[0] == outputs[1]

    def test_inspect_prompts(self, model) -> None:
        """Test _inspect_prompts."""
        # TODO(sungchul): Add point prompts # noqa: TD003
        prompts: list[tv_tensors.BoundingBoxes] = [
            tv_tensors.BoundingBoxes(
                [[0, 0, 1, 1]],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(2, 2),
                dtype=torch.float32,
            ),
            tv_tensors.BoundingBoxes(torch.zeros((0, 4)), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(2, 2)),
        ]

        result = model._inspect_prompts(prompts)

        assert torch.all(result[0] == prompts[0])
        assert result[1] is None
