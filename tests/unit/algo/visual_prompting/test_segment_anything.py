# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from otx.algo.visual_prompting.segment_anything import OTXSegmentAnything, SegmentAnything
from otx.core.data.entity.base import Points
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

    @pytest.mark.parametrize(
        "ori_shape",
        [
            torch.tensor([512, 256]),
            torch.tensor([256, 512]),
            torch.tensor([1536, 1280]),
            torch.tensor([1280, 1536]),
        ],
    )
    def test_forward_inference(self, mocker, ori_shape: Tensor) -> None:
        """Test forward_inference."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")
        segment_anything.training = False

        image_embeddings = torch.zeros(1, 256, 64, 64, dtype=torch.float32)
        point_coords = torch.tensor([[[0, 0], [10, 10]]], dtype=torch.float32)
        point_labels = torch.tensor([[2, 3]], dtype=torch.float32)
        mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
        has_mask_inputs = torch.tensor([[0.0]])

        results = segment_anything.forward_inference(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_inputs[0],
            ori_shape=ori_shape,
        )

        assert results[0].shape[2:] == torch.Size(ori_shape)

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
    def test_forward_train(self, mocker, training: bool, ori_shapes: list[Tensor]) -> None:
        """Test forward_train."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")
        segment_anything.training = training

        images = tv_tensors.Image(torch.zeros((1, 3, 1024, 1024), dtype=torch.float32))
        bboxes = [
            tv_tensors.BoundingBoxes(
                torch.tensor([[0, 0, 10, 10]]),
                format="xyxy",
                canvas_size=(1024, 1024),
                dtype=torch.float32,
            ),
        ]
        points = [Points(torch.tensor([[5, 5]]), canvas_size=(1024, 1024), dtype=torch.float32)]
        labels = [torch.as_tensor([1, 1])]
        gt_masks = [torch.zeros((2, *os)) for os in ori_shapes] if training else None

        results = segment_anything.forward_train(
            images=images,
            ori_shapes=ori_shapes,
            bboxes=bboxes,
            points=points,
            labels=labels,
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
            assert results[0][0][0].shape == torch.Size(ori_shapes[0])

            # check ious
            assert results[1][0].ndim == 2

            # check labels
            assert torch.all(results[2][0] == labels[0])

    @pytest.mark.parametrize(
        ("point_coords", "point_labels", "expected"),
        [
            (Tensor([[[1, 1]]]), Tensor([[1]]), (1, 1, 256)),
            (Tensor([[[1, 1], [2, 2]]]), Tensor([[2, 3]]), (1, 2, 256)),
        ],
    )
    def test_embed_points(self, mocker, point_coords: Tensor, point_labels: Tensor, expected: tuple[int]) -> None:
        """Test _embed_points."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")

        results = segment_anything._embed_points(point_coords, point_labels)

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("masks_input", "has_mask_input", "expected"),
        [
            (torch.randn(1, 1, 4, 4, dtype=torch.float), torch.tensor([1], dtype=torch.float), (1, 256, 1, 1)),
        ],
    )
    def test_embed_masks(self, mocker, masks_input: Tensor, has_mask_input: Tensor, expected: tuple[int]) -> None:
        """Test _embed_masks."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")

        results = segment_anything._embed_masks(masks_input, has_mask_input)

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("inputs", "targets", "expected"),
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.25])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.3888888359])),
        ],
    )
    def test_calculate_dice_loss(self, mocker, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_dice_loss."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
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
    def test_calculate_sigmoid_ce_focal_loss(self, mocker, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_sigmoid_ce_focal_loss."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
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
    def test_calculate_iou(self, mocker, inputs: Tensor, targets: Tensor, expected: Tensor) -> None:
        """Test calculate_iou."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
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
    def test_postprocess_masks(self, mocker, input_size: int, orig_size: Tensor, expected: torch.Size) -> None:
        """Test postprocess_masks."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
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
    def test_get_prepadded_size(self, mocker, input_image_size: Tensor, expected: Tensor) -> None:
        """Test get_prepadded_size."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")

        longest_side = 6

        results = segment_anything.get_prepadded_size(input_image_size, longest_side)

        assert torch.all(results == expected)

    @pytest.mark.parametrize(
        ("masks", "expected"),
        [
            (Tensor([[[-2, -2], [2, 2]]]), 1),
            (Tensor([[[-2, -2], [1, 1]]]), 0),
        ],
    )
    def test_calculate_stability_score(self, mocker, masks: Tensor, expected: int) -> None:
        """Test calculate_stability_score."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")

        results = segment_anything.calculate_stability_score(masks, mask_threshold=0.0, threshold_offset=1.0)

        assert results == expected

    def test_select_masks(self, mocker) -> None:
        """Test select_masks."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        segment_anything = SegmentAnything(backbone="tiny_vit")

        masks = Tensor([[[[1]], [[2]], [[3]], [[4]]]])
        iou_preds = Tensor([[0.1, 0.2, 0.3, 0.4]])
        num_points = 1

        selected_mask, selected_iou_pred = segment_anything.select_masks(masks, iou_preds, num_points)

        assert masks[:, -1, :, :] == selected_mask
        assert iou_preds[:, -1] == selected_iou_pred


class TestOTXSegmentAnything:
    @pytest.fixture()
    def model(self) -> OTXSegmentAnything:
        return OTXSegmentAnything(backbone="tiny_vit", num_classes=0)

    def test_create_model(self, model) -> None:
        """Test _create_model."""
        segment_anything = model._create_model()
        assert segment_anything is not None
        assert isinstance(segment_anything, torch.nn.Module)
        assert segment_anything.__class__.__name__ == "SegmentAnything"

    def test_customize_inputs(self, model, fxt_vpm_data_entity) -> None:
        """Test _customize_inputs."""
        output_data = model._customize_inputs(fxt_vpm_data_entity[1])
        assert output_data is not None
        assert output_data["mode"] == "finetuning"
        assert isinstance(output_data["ori_shapes"][0], Tensor)
        assert output_data["images"].shape[-2:] == torch.Size(output_data["ori_shapes"][0])
        assert isinstance(output_data["images"], tv_tensors.Image)
        assert output_data["gt_masks"][0].shape[-2:] == torch.Size(output_data["ori_shapes"][0])
        assert isinstance(output_data["bboxes"][0], tv_tensors.BoundingBoxes)
        assert isinstance(output_data["points"][0], tuple)
        assert isinstance(output_data["points"][0][0], Points)
        assert isinstance(output_data["points"][0][1], Tensor)
        assert isinstance(output_data["labels"][0], Tensor)

    def test_customize_outputs(self, model, fxt_vpm_data_entity) -> None:
        """Test _customize_outputs."""
        # training
        outputs = {"loss": torch.tensor(1.0)}
        result = model._customize_outputs(outputs, fxt_vpm_data_entity[1])
        assert isinstance(result, dict)
        assert "loss" in result

        # inference
        model.training = False
        outputs = (torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
        result = model._customize_outputs(outputs, fxt_vpm_data_entity[1])
        assert isinstance(result, VisualPromptingBatchPredEntity)
        assert result.masks[0].data == outputs[0]
        assert result.scores[0] == outputs[1]
        assert result.labels[0] == outputs[2]

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
