# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pytest
import torch
from otx.algo.visual_prompting.zero_shot_segment_anything import (
    OTXZeroShotSegmentAnything,
    PromptGetter,
    ZeroShotSegmentAnything,
)
from otx.core.data.entity.base import Points
from otx.core.data.entity.visual_prompting import ZeroShotVisualPromptingBatchPredEntity
from torch import Tensor
from torchvision import tv_tensors


class TestPromptGetter:
    @pytest.fixture()
    def prompt_getter(self) -> PromptGetter:
        return PromptGetter(image_size=3, downsizing=1)

    def test_set_default_thresholds(self, prompt_getter) -> None:
        """Test set_default_thresholds."""
        assert prompt_getter.default_threshold_reference == 0.3
        assert prompt_getter.default_threshold_target == 0.65

        prompt_getter.set_default_thresholds(default_threshold_reference=0.5, default_threshold_target=0.7)

        assert prompt_getter.default_threshold_reference == 0.5
        assert prompt_getter.default_threshold_target == 0.7

    @pytest.mark.parametrize(
        "result_point_selection",
        [torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), torch.tensor([[-1, -1, -1]])],
    )
    def test_forward(self, mocker, prompt_getter, result_point_selection: Tensor) -> None:
        """Test forward."""
        mocker.patch("otx.algo.visual_prompting.zero_shot_segment_anything.ZeroShotSegmentAnything")
        mocker.patch.object(prompt_getter, "_point_selection", return_value=(result_point_selection, torch.zeros(1, 2)))

        image_embeddings = torch.ones(1, 4, 4, 4)
        reference_feat = torch.rand(1, 4)
        ori_shape = torch.tensor((prompt_getter.image_size, prompt_getter.image_size), dtype=torch.int64)

        points_scores, bg_coords = prompt_getter(
            image_embeddings=image_embeddings,
            reference_feat=reference_feat,
            ori_shape=ori_shape,
        )

        assert torch.all(points_scores == result_point_selection)
        assert torch.all(bg_coords == torch.zeros(1, 2))

    @pytest.mark.parametrize(
        "result_point_selection",
        [torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), torch.tensor([[-1, -1, -1]])],
    )
    def test_get_prompt_candidates(self, mocker, prompt_getter, result_point_selection: Tensor) -> None:
        """Test get_prompt_candidates."""
        mocker.patch.object(prompt_getter, "_point_selection", return_value=(result_point_selection, torch.zeros(1, 2)))
        image_embeddings = torch.ones(1, 4, 4, 4)
        reference_feats = torch.rand(1, 1, 4)
        used_indices = torch.as_tensor([0])
        ori_shape = torch.tensor([prompt_getter.image_size, prompt_getter.image_size], dtype=torch.int64)

        total_points_scores, total_bg_coords = prompt_getter.get_prompt_candidates(
            image_embeddings=image_embeddings,
            reference_feats=reference_feats,
            used_indices=used_indices,
            ori_shape=ori_shape,
        )

        assert total_points_scores[0].shape[0] == len(result_point_selection)
        assert total_bg_coords[0].shape[0] == 1

    @pytest.mark.parametrize(
        ("mask_sim", "expected"),
        [
            (
                torch.arange(0.1, 1.0, 0.1).reshape(3, 3),
                torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]),
            ),
            (torch.zeros(3, 3), torch.tensor([[-1, -1, -1]])),
        ],
    )
    def test_point_selection(self, prompt_getter, mask_sim: torch.Tensor, expected: torch.Tensor) -> None:
        """Test _point_selection."""
        points_scores, bg_coords = prompt_getter._point_selection(
            mask_sim=mask_sim,
            ori_shape=torch.tensor([prompt_getter.image_size, prompt_getter.image_size]),
            threshold=torch.tensor([[0.5]]),
            num_bg_points=torch.tensor([[1]], dtype=torch.int64),
        )

        assert torch.equal(points_scores, expected)


class TestZeroShotSegmentAnything:
    @pytest.fixture()
    def build_zero_shot_segment_anything(self) -> Callable:
        def _build_zero_shot_segment_anything(
            backbone: str = "tiny_vit",
            freeze_image_encoder: bool = True,
            freeze_prompt_encoder: bool = True,
            freeze_mask_decoder: bool = True,
            default_threshold_reference: float = 0.3,
            default_threshold_target: float = 0.65,
        ) -> ZeroShotSegmentAnything:
            return ZeroShotSegmentAnything(
                backbone=backbone,
                freeze_image_encoder=freeze_image_encoder,
                freeze_prompt_encoder=freeze_prompt_encoder,
                freeze_mask_decoder=freeze_mask_decoder,
                default_threshold_reference=default_threshold_reference,
                default_threshold_target=default_threshold_target,
            )

        return _build_zero_shot_segment_anything

    @pytest.mark.parametrize(("backbone", "expected_backbone"), [("tiny_vit", "TinyViT")])
    @pytest.mark.parametrize("freeze_image_encoder", [True, False])
    @pytest.mark.parametrize("freeze_prompt_encoder", [True, False])
    @pytest.mark.parametrize("freeze_mask_decoder", [True, False])
    def test_init(
        self,
        mocker,
        build_zero_shot_segment_anything,
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
        zero_shot_segment_anything = build_zero_shot_segment_anything(
            backbone=backbone,
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
            default_threshold_reference=0.3,
            default_threshold_target=0.65,
        )

        # check import modules
        assert hasattr(zero_shot_segment_anything, "image_encoder")
        assert zero_shot_segment_anything.image_encoder.__class__.__name__ == expected_backbone
        assert hasattr(zero_shot_segment_anything, "prompt_encoder")
        assert hasattr(zero_shot_segment_anything, "mask_decoder")

        # check load_checkpoint
        mocker_load_state_dict_from_url.assert_called_once()

        # check freeze_networks
        for param in zero_shot_segment_anything.image_encoder.parameters():
            assert not param.requires_grad

        for param in zero_shot_segment_anything.prompt_encoder.parameters():
            assert not param.requires_grad

        for param in zero_shot_segment_anything.mask_decoder.parameters():
            assert not param.requires_grad

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {
                "backbone": "tiny_vit",
            },
            {
                "mask_threshold": 0.0,
                "use_stability_score": True,
                "return_single_mask": True,
                "return_extra_metrics": True,
                "stability_score_offset": 2.0,
            },
        ],
    )
    def test_set_default_config(self, mocker, kwargs: dict[str, Any]) -> None:
        """Test set_default_config."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        zero_shot_segment_anything = ZeroShotSegmentAnything(**kwargs)

        assert zero_shot_segment_anything.image_encoder.__class__.__name__ == "TinyViT"
        for param in zero_shot_segment_anything.image_encoder.parameters():
            assert not param.requires_grad

        for param in zero_shot_segment_anything.prompt_encoder.parameters():
            assert not param.requires_grad

        for param in zero_shot_segment_anything.mask_decoder.parameters():
            assert not param.requires_grad

        for key, value in kwargs.items():
            if key in ["backbone"]:
                continue
            assert getattr(zero_shot_segment_anything, key) == value

    @pytest.mark.parametrize("new_largest_label", [0, 3])
    def test_expand_reference_info(self, build_zero_shot_segment_anything, new_largest_label: int) -> None:
        """Test expand_reference_info."""
        zero_shot_segment_anything = build_zero_shot_segment_anything()
        reference_feats = torch.zeros(0, 1, 256)

        results = zero_shot_segment_anything.expand_reference_info(
            reference_feats=reference_feats,
            new_largest_label=new_largest_label,
        )

        assert len(results) == new_largest_label + 1

    def test_learn(self, mocker, build_zero_shot_segment_anything) -> None:
        """Test learn."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        zero_shot_segment_anything = build_zero_shot_segment_anything()
        images = [tv_tensors.Image(torch.zeros((1, 3, 1024, 1024), dtype=torch.float32))]
        processed_prompts = [
            {
                torch.tensor(0): [
                    tv_tensors.BoundingBoxes(
                        torch.tensor([[0, 0, 10, 10]]),
                        format="xyxy",
                        canvas_size=(1024, 1024),
                        dtype=torch.float32,
                    ),
                    Points(torch.tensor([[5, 5]]), canvas_size=(1024, 1024), dtype=torch.float32),
                ],
            },
        ]
        reference_feats = torch.zeros(0, 1, 256)
        used_indices = torch.tensor([], dtype=torch.int64)
        ori_shapes = [torch.tensor((1024, 1024))]

        reference_info, ref_masks = zero_shot_segment_anything.learn(
            images=images,
            processed_prompts=processed_prompts,
            reference_feats=reference_feats,
            used_indices=used_indices,
            ori_shapes=ori_shapes,
        )

        assert reference_info["reference_feats"].shape == torch.Size((1, 1, 256))
        assert ref_masks[0].shape == torch.Size((1, *ori_shapes[0]))
        assert 0 in reference_info["used_indices"]

        new_processed_prompts = [
            {
                torch.tensor(1): [
                    tv_tensors.BoundingBoxes(
                        torch.tensor([[0, 0, 10, 10]]),
                        format="xyxy",
                        canvas_size=(1024, 1024),
                        dtype=torch.float32,
                    ),
                    Points(torch.tensor([[5, 5]]), canvas_size=(1024, 1024), dtype=torch.float32),
                ],
            },
        ]

        reference_info, ref_masks = zero_shot_segment_anything.learn(
            images=images,
            processed_prompts=new_processed_prompts,
            reference_feats=reference_info["reference_feats"],
            used_indices=reference_info["used_indices"],
            ori_shapes=ori_shapes,
        )

        assert reference_info["reference_feats"].shape == torch.Size((2, 1, 256))
        assert ref_masks[0].shape == torch.Size((2, *ori_shapes[0]))
        assert 0 in reference_info["used_indices"]
        assert 1 in reference_info["used_indices"]

    def test_infer(self, mocker, build_zero_shot_segment_anything) -> None:
        """Test infer."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        zero_shot_segment_anything = build_zero_shot_segment_anything()
        mocker.patch.object(
            zero_shot_segment_anything.prompt_getter,
            "get_prompt_candidates",
            return_value=({0: torch.tensor([[0, 0, 0.5], [1000, 1000, 0.7]])}, {0: torch.tensor([[500, 500]])}),
        )

        def _patch_predict_masks(**kwargs) -> Tensor:
            point_coords = kwargs.get("point_coords")
            mask = torch.zeros(*kwargs["ori_shape"], dtype=torch.bool)
            mask[int(point_coords[0, 0, 1]), int(point_coords[0, 0, 0])] = True
            return mask

        zero_shot_segment_anything._predict_masks = _patch_predict_masks

        images = [tv_tensors.Image(torch.zeros((1, 3, 1024, 1024), dtype=torch.float32))]
        reference_feats = torch.rand(1, 1, 1, 256)
        used_indices = {0: [0]}
        ori_shapes = [torch.tensor((1024, 1024))]

        results = zero_shot_segment_anything.infer(
            images=images,
            reference_feats=reference_feats,
            used_indices=used_indices,
            ori_shapes=ori_shapes,
        )

        for predicted_masks, used_points in results:
            for label, predicted_mask in predicted_masks.items():
                for pm, up in zip(predicted_mask, used_points[label]):
                    assert pm[int(up[1]), int(up[0])] == up[2]

    def test_inspect_overlapping_areas(self, mocker, build_zero_shot_segment_anything) -> None:
        """Test _inspect_overlapping_areas."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        zero_shot_segment_anything = build_zero_shot_segment_anything()
        predicted_masks = {
            0: [
                torch.tensor(
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
                torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
                torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                    ],
                ),
            ],
            1: [
                torch.tensor(
                    [
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
                torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 1],
                    ],
                ),
                torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                    ],
                ),
                torch.tensor(
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
            ],
        }
        used_points = {
            0: [
                torch.tensor([0, 0, 0.5]),  # to be removed
                torch.tensor([2, 2, 0.5]),
                torch.tensor([1, 4, 0.5]),
            ],
            1: [
                torch.tensor([3, 0, 0.5]),
                torch.tensor([4, 4, 0.5]),
                torch.tensor([1, 4, 0.3]),  # to be removed
                torch.tensor([0, 0, 0.7]),
            ],
        }

        zero_shot_segment_anything._inspect_overlapping_areas(predicted_masks, used_points, threshold_iou=0.5)

        assert len(predicted_masks[0]) == 2
        assert len(predicted_masks[1]) == 3
        assert all(torch.tensor([2, 2, 0.5]) == used_points[0][0])
        assert all(torch.tensor([0, 0, 0.7]) == used_points[1][2])

    def test_predict_masks(self, mocker, build_zero_shot_segment_anything) -> None:
        """Test _predict_masks."""
        mocker.patch(
            "otx.algo.visual_prompting.segment_anything.SegmentAnything.forward",
            return_value=(torch.ones(1, 4, 8, 8), torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4)),
        )

        zero_shot_segment_anything = build_zero_shot_segment_anything()
        zero_shot_segment_anything.image_size = 6

        mask = zero_shot_segment_anything._predict_masks(
            mode="infer",
            image_embeddings=torch.rand(1),
            point_coords=torch.rand(1, 2, 2),
            point_labels=torch.randint(low=0, high=2, size=(1, 2)),
            ori_shape=torch.tensor([8, 8], dtype=torch.int64),
        )
        assert mask.shape == (8, 8)

    @pytest.mark.parametrize(
        ("masks", "logits", "expected"),
        [
            (torch.ones(1, 4, 8, 8), torch.ones(1, 4, 4, 4), torch.ones(8, 8)),
            (torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 4, 4), torch.zeros(8, 8)),
        ],
    )
    def test_decide_cascade_results(
        self,
        mocker,
        build_zero_shot_segment_anything,
        masks: Tensor,
        logits: Tensor,
        expected: Tensor,
    ) -> None:
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        zero_shot_segment_anything = build_zero_shot_segment_anything()
        scores = torch.tensor([[0.0, 0.1, 0.2, 0.3]])

        _, result = zero_shot_segment_anything._decide_cascade_results(masks, logits, scores)

        assert torch.equal(result, expected)


class TestOTXZeroShotSegmentAnything:
    @pytest.fixture()
    def model(self) -> OTXZeroShotSegmentAnything:
        return OTXZeroShotSegmentAnything(backbone="tiny_vit")

    def test_create_model(self, model) -> None:
        """Test _create_model."""
        zero_shot_segment_anything = model._create_model()
        assert zero_shot_segment_anything is not None
        assert isinstance(zero_shot_segment_anything, torch.nn.Module)
        assert zero_shot_segment_anything.__class__.__name__ == "ZeroShotSegmentAnything"

    @pytest.mark.parametrize("training", [True, False])
    def test_forward(self, mocker, model, training: bool) -> None:
        """Test forward."""
        mocker_learn = mocker.patch.object(model, "learn")
        mocker_infer = mocker.patch.object(model, "infer")
        model.training = training

        model.forward(None)

        if training:
            mocker_learn.assert_called_once()
        else:
            mocker_infer.assert_called_once()

    @pytest.mark.parametrize("reset_feat", [True, False])
    def test_learn(self, mocker, model, reset_feat: bool) -> None:
        """Test learn."""
        mocker_initialize_reference_info = mocker.patch.object(model, "initialize_reference_info")
        mocker_learn = mocker.patch.object(model.model, "learn")
        mocker_customize_inputs = mocker.patch.object(model, "_customize_inputs")
        mocker_customize_outputs = mocker.patch.object(model, "_customize_outputs")

        model.learn(None, reset_feat=reset_feat)

        if reset_feat:
            mocker_initialize_reference_info.assert_called_once()
        else:
            mocker_initialize_reference_info.assert_not_called()
        mocker_learn.assert_called_once()
        mocker_customize_inputs.assert_called_once()
        mocker_customize_outputs.assert_called_once()

    def test_infer(self, mocker, model) -> None:
        """Test infer."""
        mocker_infer = mocker.patch.object(model.model, "infer")
        mocker_customize_inputs = mocker.patch.object(model, "_customize_inputs")
        mocker_customize_outputs = mocker.patch.object(model, "_customize_outputs")

        model.infer(None)

        mocker_infer.assert_called_once()
        mocker_customize_inputs.assert_called_once()
        mocker_customize_outputs.assert_called_once()

    @pytest.mark.parametrize("is_training", [True, False])
    def test_customize_inputs_learn(
        self,
        model: OTXZeroShotSegmentAnything,
        fxt_zero_shot_vpm_data_entity,
        is_training: bool,
    ) -> None:
        """Test _customize_inputs with training=True."""
        model.training = is_training
        model.initialize_reference_info()
        output_data = model._customize_inputs(fxt_zero_shot_vpm_data_entity[1])

        assert output_data is not None
        assert isinstance(output_data["images"][0], tv_tensors.Image)
        assert output_data["images"][0].shape[-2:] == torch.Size(output_data["ori_shapes"][0])
        assert isinstance(output_data["ori_shapes"][0], Tensor)
        assert isinstance(output_data["reference_feats"], Tensor)
        assert torch.all(output_data["reference_feats"] == model.reference_feats)
        assert isinstance(output_data["used_indices"], Tensor)
        assert torch.all(output_data["used_indices"] == model.used_indices)

        if is_training:
            assert "processed_prompts" in output_data

    def test_customize_inputs_infer(self, model: OTXZeroShotSegmentAnything, fxt_zero_shot_vpm_data_entity) -> None:
        """Test _customize_inputs with training=False."""
        model.training = False
        model.reference_feats = torch.rand(1, 1, 256)
        model.used_indices = torch.tensor([0.0])
        output_data = model._customize_inputs(fxt_zero_shot_vpm_data_entity[1])

        assert output_data is not None
        assert isinstance(output_data["images"][0], tv_tensors.Image)
        assert output_data["images"][0].shape[-2:] == torch.Size(output_data["ori_shapes"][0])
        assert isinstance(output_data["ori_shapes"][0], Tensor)
        assert "reference_feats" in output_data
        assert torch.all(output_data["reference_feats"] == model.reference_feats)
        assert torch.all(output_data["used_indices"] == model.used_indices)

    def test_customize_outputs(self, model, fxt_zero_shot_vpm_data_entity) -> None:
        """Test _customize_outputs."""

        # training
        outputs = [
            {"reference_feats": torch.zeros(0, 1, 256)},
            {"used_indices": torch.tensor([1, 1, 1])},
            "reference_masks",
        ]
        model.training = True
        result = model._customize_outputs(outputs, fxt_zero_shot_vpm_data_entity[1])
        assert result == outputs

        # inference
        label = 0
        outputs = [[{label: [torch.tensor(0)]}, {label: [torch.tensor([1, 1, 1])]}]]
        model.training = False
        result = model._customize_outputs(outputs, fxt_zero_shot_vpm_data_entity[1])
        assert isinstance(result, ZeroShotVisualPromptingBatchPredEntity)
        assert result.masks[0].data == outputs[0][0][label][0]
        assert result.scores[0] == outputs[0][1][label][0][2]
        assert result.labels[0] == label
        assert torch.all(result.prompts[0].data == outputs[0][1][label][0][:2].unsqueeze(0))

    def test_gather_prompts_with_labels(self, model, fxt_zero_shot_vpm_data_entity) -> None:
        """Test _gather_prompts_with_labels."""
        entity = deepcopy(fxt_zero_shot_vpm_data_entity[1])

        results = model._gather_prompts_with_labels(entity)

        assert torch.all(results[0][1][0] == entity.prompts[0][0])
        assert torch.all(results[0][1][1] == entity.masks[0])
        assert torch.all(results[0][2][0] == entity.prompts[0][1])
        assert results[0][2][1] == entity.polygons[0][0]

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (tv_tensors.Image(torch.zeros(3, 2, 4)), (3, 4, 8)),
            (tv_tensors.Image(torch.zeros(3, 12, 16)), (3, 6, 8)),
        ],
    )
    def test_apply_image(self, model, image: tv_tensors.Image, expected: tuple[int, ...]) -> None:
        """Test apply_image."""
        results = model.apply_image(image, target_length=8)

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("coords", "ori_shape", "expected"),
        [
            (torch.tensor([[1, 1], [2, 2]]), (4, 4), torch.tensor([[2, 2], [4, 4]])),
            (torch.tensor([[4, 4], [8, 8]]), (16, 16), torch.tensor([[2, 2], [4, 4]])),
        ],
    )
    def test_apply_points(self, model, coords: Tensor, ori_shape: tuple[int, int], expected: Tensor) -> None:
        """Test apply_points."""
        result = model.apply_points(Points(coords, canvas_size=ori_shape), ori_shape, target_length=8)

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        ("boxes", "ori_shape", "expected"),
        [
            (torch.tensor([[1, 1, 2, 2], [2, 2, 3, 3]]), (4, 4), torch.tensor([[2, 2, 4, 4], [4, 4, 6, 6]])),
            (torch.tensor([[4, 4, 8, 8], [8, 8, 12, 12]]), (16, 16), torch.tensor([[2, 2, 4, 4], [4, 4, 6, 6]])),
        ],
    )
    def test_apply_boxes(self, model, boxes: Tensor, ori_shape: tuple[int, int], expected: Tensor) -> None:
        """Test apply_boxes."""
        result = model.apply_boxes(
            tv_tensors.BoundingBoxes(boxes, format="xyxy", canvas_size=ori_shape),
            ori_shape,
            target_length=8,
        )

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        ("prompts", "ori_shape", "expected"),
        [
            (
                [
                    Points([[4, 4], [8, 8]], canvas_size=(16, 16)),
                    tv_tensors.BoundingBoxes([[4, 4, 8, 8], [8, 8, 12, 12]], format="xyxy", canvas_size=(16, 16)),
                ],
                (16, 16),
                [
                    Points([[2, 2], [4, 4]], canvas_size=(8, 8)),
                    tv_tensors.BoundingBoxes([[2, 2, 4, 4], [4, 4, 6, 6]], format="xyxy", canvas_size=(8, 8)),
                ],
            ),
        ],
    )
    def test_apply_prompts(
        self,
        model,
        prompts: list[Points | tv_tensors.BoundingBoxes],
        ori_shape: tuple[int, int],
        expected: list[Points | tv_tensors.BoundingBoxes],
    ) -> None:
        """Test apply_prompts."""
        results = model.apply_prompts(prompts, ori_shape, target_length=8)

        for r, e in zip(results, expected):
            assert torch.all(r == e)

    @pytest.mark.parametrize(
        ("oldh", "oldw", "expected"),
        [
            (3, 4, (6, 8)),
            (12, 16, (6, 8)),
        ],
    )
    def test_get_preprocess_shape(self, model, oldh: int, oldw: int, expected: tuple[int, int]):
        """Test get_preprocess_shape."""
        results = model.get_preprocess_shape(oldh, oldw, target_length=8)

        assert results == expected

    @pytest.mark.parametrize("image", (tv_tensors.Image(torch.zeros(1, 3, 2, 4, dtype=torch.uint8))))
    def test_preprocess(self, model, image: tv_tensors.Image) -> None:
        """Test preprocess."""
        model.pixel_mean = torch.ones_like(model.pixel_mean)
        model.pixel_std = torch.ones_like(model.pixel_std) * 2
        model.model.image_size = 8

        results = model.preprocess(image)

        assert results.shape == (3, 8, 8)
        assert torch.all(torch.unique(results) == torch.tensor((-0.5, 0.0)))

    def test_initialize_reference_info(self, model) -> None:
        """Test initialize_reference_info."""
        model.initialize_reference_info()

        assert model.reference_feats.shape == (0, 1, 256)
        assert model.used_indices.shape == (0,)

    def test_save_reference_info(self, mocker, tmpdir, model) -> None:
        """Test save_reference_info."""
        model.root_reference_info = tmpdir
        model.reference_feats = torch.tensor(1)
        model.used_indices = torch.tensor(1)
        mocker_mkdir = mocker.patch("pathlib.Path.mkdir")
        mocker.patch("pathlib.Path.open")
        mocker_torch_save = mocker.patch("torch.save")
        mocker_pickle_dump = mocker.patch("pickle.dump")

        model.save_reference_info(".")

        mocker_mkdir.assert_called_once()
        mocker_torch_save.assert_called_once()
        mocker_pickle_dump.assert_called_once()

    def test_load_reference_info(self, mocker, model) -> None:
        """Test load_reference_info."""
        # get previously saved reference info
        mocker.patch(
            "torch.load",
            return_value={"reference_feats": torch.zeros((1, 1, 256)), "used_indices": torch.tensor([0.0])},
        )
        mocker.patch("pathlib.Path.is_file", return_value=True)

        model.load_reference_info(".")
        assert model.reference_feats.shape == (1, 1, 256)
        assert model.used_indices.shape == (1,)

        # no saved reference info
        mocker.patch("pathlib.Path.is_file", return_value=False)

        model.initialize_reference_info()
        model.load_reference_info(".")

        assert model.reference_feats.shape == (0, 1, 256)
        assert model.used_indices.shape == (0,)
