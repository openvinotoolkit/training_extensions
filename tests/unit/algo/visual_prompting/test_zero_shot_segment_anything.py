# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock

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
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.prompt_getter = PromptGetter(image_size=3, downsizing=1)

    def test_set_default_thresholds(self) -> None:
        """Test set_default_thresholds."""
        assert self.prompt_getter.default_threshold_reference == 0.3
        assert self.prompt_getter.default_threshold_target == 0.65

        self.prompt_getter.set_default_thresholds(default_threshold_reference=0.5, default_threshold_target=0.7)

        assert self.prompt_getter.default_threshold_reference == 0.5
        assert self.prompt_getter.default_threshold_target == 0.7

    def test_forward(self, mocker) -> None:
        """Test forward."""
        mocker.patch.object(
            self.prompt_getter,
            "_get_prompt_candidates",
            return_value=(torch.tensor([[0, 0, 0.5], [1, 1, 0.7]]), torch.tensor([[2, 2]])),
        )
        image_embedding = torch.ones(1, 4, 4, 4)
        reference_feats = torch.rand(1, 1, 4)
        used_indices = [0]
        ori_shape = torch.tensor((self.prompt_getter.image_size, self.prompt_getter.image_size), dtype=torch.int64)

        total_points_scores, total_bg_coords = self.prompt_getter(
            image_embedding=image_embedding,
            reference_feats=reference_feats,
            used_indices=used_indices,
            ori_shape=ori_shape,
        )

        assert total_points_scores.shape == torch.Size((1, 2, 3))
        assert total_bg_coords.shape == torch.Size((1, 1, 2))

    def test_get_prompt_candidates(self, mocker) -> None:
        """Test _get_prompt_candidates."""
        mocker.patch.object(self.prompt_getter, "_point_selection", return_value=("points_scores", "bg_coords"))
        image_embedding = torch.ones(1, 4, 4, 4)
        reference_feat = torch.rand(1, 4)
        ori_shape = torch.tensor(
            [self.prompt_getter.image_size, self.prompt_getter.image_size],
            dtype=torch.int64,
        )

        points_scores, bg_coords = self.prompt_getter._get_prompt_candidates(
            image_embedding=image_embedding,
            reference_feat=reference_feat,
            ori_shape=ori_shape,
            threshold=torch.tensor([[0.0]], dtype=torch.float32),
            num_bg_points=torch.tensor([[1]], dtype=torch.int64),
        )

        assert points_scores == "points_scores"
        assert bg_coords == "bg_coords"

    def test_point_selection(self) -> None:
        """Test _point_selection."""
        mask_sim = torch.arange(0.1, 1.0, 0.1).reshape(self.prompt_getter.image_size, self.prompt_getter.image_size)

        points_scores, bg_coords = self.prompt_getter._point_selection(
            mask_sim=mask_sim,
            ori_shape=torch.tensor([self.prompt_getter.image_size, self.prompt_getter.image_size]),
            threshold=torch.tensor([[0.5]]),
            num_bg_points=torch.tensor([[1]], dtype=torch.int64),
        )

        assert torch.equal(points_scores, torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]))
        assert torch.equal(bg_coords, torch.tensor([[0, 0]]))


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

        assert zero_shot_segment_anything.reference_info["reference_feats"] is None
        assert zero_shot_segment_anything.reference_info["used_indices"] is None

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

    def test_initialize_reference_info_expand_reference_info(self, build_zero_shot_segment_anything) -> None:
        """Test initialize_reference_info and expand_reference_info."""
        zero_shot_segment_anything = build_zero_shot_segment_anything()

        zero_shot_segment_anything.initialize_reference_info(largest_label=0)

        assert isinstance(zero_shot_segment_anything.reference_info["reference_feats"], Tensor)
        assert zero_shot_segment_anything.reference_info["reference_feats"].shape == torch.Size((1, 1, 256))
        assert isinstance(zero_shot_segment_anything.reference_info["used_indices"], set)

        zero_shot_segment_anything.expand_reference_info(new_largest_label=3)

        assert zero_shot_segment_anything.reference_info["reference_feats"].shape == torch.Size((4, 1, 256))

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
        ori_shapes = [torch.tensor((1024, 1024))]

        _, ref_masks = zero_shot_segment_anything.learn(
            images=images,
            processed_prompts=processed_prompts,
            ori_shapes=ori_shapes,
            return_outputs=True,
        )

        assert zero_shot_segment_anything.reference_info["reference_feats"].shape == torch.Size((1, 1, 256))
        assert ref_masks[0].shape == torch.Size((1, *ori_shapes[0]))
        assert 0 in zero_shot_segment_anything.reference_info["used_indices"]

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

        _, ref_masks = zero_shot_segment_anything.learn(
            images=images,
            processed_prompts=new_processed_prompts,
            ori_shapes=ori_shapes,
            return_outputs=True,
        )

        assert zero_shot_segment_anything.reference_info["reference_feats"].shape == torch.Size((2, 1, 256))
        assert ref_masks[0].shape == torch.Size((2, *ori_shapes[0]))
        assert 0 in zero_shot_segment_anything.reference_info["used_indices"]
        assert 1 in zero_shot_segment_anything.reference_info["used_indices"]

    def test_infer(self, mocker, build_zero_shot_segment_anything) -> None:
        """Test infer."""
        mocker.patch("otx.algo.visual_prompting.segment_anything.SegmentAnything.load_checkpoint")
        zero_shot_segment_anything = build_zero_shot_segment_anything()
        zero_shot_segment_anything.prompt_getter = MagicMock(
            spec=PromptGetter,
            return_value=(torch.tensor([[[0, 0, 0.5], [1000, 1000, 0.7]]]), torch.tensor([[[500, 500]]])),
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

    def test_inspect_overlapping_areas(self) -> None:
        """Test __inspect_overlapping_areas."""


class TestOTXZeroShotSegmentAnything:
    @pytest.fixture()
    def model(self) -> OTXZeroShotSegmentAnything:
        return OTXZeroShotSegmentAnything(backbone="tiny_vit", num_classes=0)

    def test_create_model(self, model) -> None:
        """Test _create_model."""
        zero_shot_segment_anything = model._create_model()
        assert zero_shot_segment_anything is not None
        assert isinstance(zero_shot_segment_anything, torch.nn.Module)
        assert zero_shot_segment_anything.__class__.__name__ == "ZeroShotSegmentAnything"

    def test_customize_inputs_learn(self, model: OTXZeroShotSegmentAnything, fxt_zero_shot_vpm_data_entity) -> None:
        """Test _customize_inputs with training=True."""
        model.training = True
        output_data = model._customize_inputs(fxt_zero_shot_vpm_data_entity[1])

        assert output_data is not None
        assert isinstance(output_data["images"][0], tv_tensors.Image)
        assert output_data["images"][0].shape[-2:] == torch.Size(output_data["ori_shapes"][0])
        assert isinstance(output_data["ori_shapes"][0], Tensor)
        assert "processed_prompts" in output_data

    def test_customize_inputs_infer(self, model: OTXZeroShotSegmentAnything, fxt_zero_shot_vpm_data_entity) -> None:
        """Test _customize_inputs with training=False."""
        model.training = False
        model.model.reference_info["reference_feats"] = torch.rand(1, 1, 1, 256)
        model.model.reference_info["used_indices"] = {0: [0]}
        output_data = model._customize_inputs(fxt_zero_shot_vpm_data_entity[1])

        assert output_data is not None
        assert isinstance(output_data["images"][0], tv_tensors.Image)
        assert output_data["images"][0].shape[-2:] == torch.Size(output_data["ori_shapes"][0])
        assert isinstance(output_data["ori_shapes"][0], Tensor)
        assert "reference_feats" in output_data
        assert torch.all(output_data["reference_feats"] == model.model.reference_info["reference_feats"])
        assert output_data["used_indices"] == model.model.reference_info["used_indices"]

    def test_customize_outputs(self, model, fxt_zero_shot_vpm_data_entity) -> None:
        """Test _customize_outputs."""
        label = torch.tensor(0)
        outputs = [[{label: [torch.tensor(0)]}, {label: [torch.tensor([1, 1, 1])]}]]

        # training
        model.training = True
        result = model._customize_outputs(outputs, fxt_zero_shot_vpm_data_entity[1])
        assert result == outputs

        # inference
        model.training = False
        result = model._customize_outputs(outputs, fxt_zero_shot_vpm_data_entity[1])
        assert isinstance(result, ZeroShotVisualPromptingBatchPredEntity)
        assert result.masks[0].data == outputs[0][0][label][0]
        assert result.scores[0] == outputs[0][1][label][0][2]
        assert result.labels[0] == label
        assert torch.all(result.prompts[0].data == outputs[0][1][label][0][:2].unsqueeze(0))

    def test_gather_prompts_with_labels(self, model) -> None:
        """Test _gather_prompts_with_labels."""
        prompts = [[torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(2), torch.tensor(4)]]
        labels = [torch.tensor([0, 1, 2, 2, 4])]

        results = model._gather_prompts_with_labels(prompts, labels)

        assert results[0][0][0] == prompts[0][0]
        assert results[0][1][0] == prompts[0][1]
        assert results[0][2] == prompts[0][2:4]
        assert results[0][4][0] == prompts[0][4]
