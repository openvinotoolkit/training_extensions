"""Tests Segment Anything for zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from typing import Dict, Any, Optional
from collections import OrderedDict
from tests.test_suite.e2e_test_system import e2e_pytest_unit
import torch
from torch import nn
from omegaconf import DictConfig

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything import (
    SegmentAnything,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything import (
    PromptGetter,
    ZeroShotSegmentAnything,
)
from tests.unit.algorithms.visual_prompting.test_helpers import (
    MockScoredLabel,
    MockImageEncoder,
    MockPromptGetter,
    MockMaskDecoder,
)


class TestPromptGetter:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.prompt_getter = PromptGetter(image_size=3, downsizing=1)

    @e2e_pytest_unit
    def test_set_default_thresholds(self) -> None:
        """Test set_default_thresholds."""
        assert self.prompt_getter.default_threshold_reference == 0.3
        assert self.prompt_getter.default_threshold_target == 0.65

        self.prompt_getter.set_default_thresholds(default_threshold_reference=0.5, default_threshold_target=0.7)

        assert self.prompt_getter.default_threshold_reference == 0.5
        assert self.prompt_getter.default_threshold_target == 0.7

    @e2e_pytest_unit
    @pytest.mark.parametrize("result_point_selection", [torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), torch.tensor([[-1, -1, -1]])])
    def test_forward(self, mocker, result_point_selection: torch.Tensor) -> None:
        """Test forward."""
        mocker.patch.object(
            self.prompt_getter,
            "get_prompt_candidates",
            return_value=(result_point_selection, torch.zeros(1, 2)))
        image_embedding = torch.ones(1, 4, 4, 4)
        reference_feats = torch.rand(1, 1, 4)
        used_indices = torch.as_tensor([[0]])
        original_size = torch.tensor((self.prompt_getter.image_size, self.prompt_getter.image_size), dtype=torch.int64)

        total_points_scores, total_bg_coords = self.prompt_getter(
            image_embedding=image_embedding, reference_feats=reference_feats, used_indices=used_indices, original_size=original_size
        )
        
        assert total_points_scores.shape[0] == 1
        assert total_bg_coords.shape[0] == 1

    @e2e_pytest_unit
    @pytest.mark.parametrize("result_point_selection", [torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), torch.tensor([[-1, -1, -1]])])
    def test_get_prompt_candidates(self, mocker, result_point_selection: torch.Tensor) -> None:
        """Test get_prompt_candidates."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.ZeroShotSegmentAnything"
        )
        mocker.patch.object(self.prompt_getter, "_point_selection", return_value=(result_point_selection, torch.zeros(1, 2)))
        image_embedding = torch.ones(1, 4, 4, 4)
        reference_feat = torch.rand(1, 4)
        original_size = torch.tensor(
            [[self.prompt_getter.image_size, self.prompt_getter.image_size]], dtype=torch.int64
        )

        points_scores, bg_coords = self.prompt_getter.get_prompt_candidates(
            image_embedding=image_embedding, reference_feat=reference_feat, original_size=original_size
        )

        assert torch.all(points_scores == result_point_selection)
        assert torch.all(bg_coords == torch.zeros(1, 2))

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "mask_sim,expected",
        [
            (torch.arange(0.1, 1.0, 0.1).reshape(3, 3), torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]])),
            (torch.zeros(3, 3), torch.tensor([[-1, -1, -1]]))
        ])
    def test_point_selection(self, mask_sim: torch.Tensor, expected: torch.Tensor) -> None:
        """Test _point_selection."""
        points_scores, bg_coords = self.prompt_getter._point_selection(
            mask_sim=mask_sim,
            original_size=torch.tensor([self.prompt_getter.image_size, self.prompt_getter.image_size]),
            threshold=torch.tensor([[0.5]]),
        )

        assert torch.equal(points_scores, expected)


class TestZeroShotSegmentAnything:
    @pytest.fixture
    def set_zero_shot_segment_anything(self, monkeypatch):
        def zero_shot_segment_anything(state_dict: Optional[OrderedDict] = None):
            monkeypatch.setattr(
                "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMImageEncoder",
                MockImageEncoder,
            )
            monkeypatch.setattr(
                "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMMaskDecoder",
                MockMaskDecoder,
            )
            return ZeroShotSegmentAnything(state_dict=state_dict)

        return zero_shot_segment_anything

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "state_dict",
        [
            None,
            {
                "reference_info.reference_feats": torch.zeros(1),
                "reference_info.used_indices": torch.zeros(1, dtype=torch.int64),
            },
        ],
    )
    def test_init(self, set_zero_shot_segment_anything, state_dict: Optional[Dict[str, Any]]) -> None:
        """Test __init__."""
        if state_dict is not None:
            zero_shot_segment_anything_for_init_weights = set_zero_shot_segment_anything().state_dict()
            zero_shot_segment_anything_for_init_weights.update(state_dict)
            state_dict = zero_shot_segment_anything_for_init_weights
        
        zero_shot_segment_anything = set_zero_shot_segment_anything(state_dict=state_dict)

        assert zero_shot_segment_anything.config.model.freeze_image_encoder
        assert zero_shot_segment_anything.config.model.freeze_prompt_encoder
        assert zero_shot_segment_anything.config.model.freeze_mask_decoder

        if state_dict:
            assert zero_shot_segment_anything.reference_info.reference_feats == torch.zeros(1)
            assert zero_shot_segment_anything.reference_info.used_indices == torch.zeros(1, dtype=torch.int64)

        assert zero_shot_segment_anything.reference_info.reference_feats.dtype == torch.float32
        assert zero_shot_segment_anything.reference_info.used_indices.dtype == torch.int64

    @e2e_pytest_unit
    def test_set_default_config(self, set_zero_shot_segment_anything) -> None:
        """Test set_default_config."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()

        default_config = zero_shot_segment_anything.set_default_config()

        assert isinstance(default_config, DictConfig)
        assert "model" in default_config
        assert "backbone" in default_config.model
        assert "checkpoint" in default_config.model
        assert "default_threshold_reference" in default_config.model
        assert "default_threshold_target" in default_config.model
        assert "freeze_image_encoder" in default_config.model
        assert "freeze_mask_decoder" in default_config.model
        assert "freeze_prompt_encoder" in default_config.model
        assert "image_size" in default_config.model
        assert "mask_threshold" in default_config.model

    @e2e_pytest_unit
    def test_learn(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test learn."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        mocker.patch.object(
            zero_shot_segment_anything,
            "_predict_masks",
            return_value=torch.tensor([[[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]]),
        )
        mocker.patch.object(zero_shot_segment_anything, "_generate_masked_features", return_value=torch.ones(1, 256))

        processed_prompts = {MockScoredLabel(label=0, name="label"): [{"box": torch.tensor([[0, 0, 1, 1]])}]}
        zero_shot_segment_anything.learn(
            images=torch.ones((1, 3, 4, 4)),
            processed_prompts=processed_prompts,
            original_size=torch.tensor((4, 4)),
        )

        assert zero_shot_segment_anything.reference_info.reference_feats.shape == (1, 1, 256)
        assert zero_shot_segment_anything.reference_info.used_indices == torch.as_tensor([0])

    @e2e_pytest_unit
    @pytest.mark.parametrize("expected", [[torch.ones((4, 4)) / 2, torch.tensor([0.0, 0.0, 0.5])]])
    def test_infer(self, monkeypatch, mocker, set_zero_shot_segment_anything, expected: torch.Tensor) -> None:
        """Test infer."""
        monkeypatch.setattr(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.PromptGetter",
            MockPromptGetter,
        )

        zero_shot_segment_anything = set_zero_shot_segment_anything()
        reference_feats = nn.Parameter(torch.rand(1, 1, 256), requires_grad=False)
        used_indices = nn.Parameter(torch.as_tensor([[0]], dtype=torch.int64), requires_grad=False)
        mocker.patch.object(
            SegmentAnything, "forward", return_value=(torch.ones(1, 4, 4, 4), torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4))
        )

        total_results = zero_shot_segment_anything.infer(
            images=torch.ones((1, 3, 4, 4)), 
            reference_feats=reference_feats,
            used_indices=used_indices,
            original_size=torch.tensor([[4, 4]], dtype=torch.int64)
        )

        for i, results in enumerate(total_results[0]):
            for _, result in results.items():
                assert torch.equal(result[0], expected[i])
                
    @e2e_pytest_unit
    def test_inspect_overlapping_areas(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test _inspect_overlapping_areas."""
        mocker.patch("otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint")
        zero_shot_segment_anything = set_zero_shot_segment_anything()
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

    @e2e_pytest_unit
    def test_predict_masks(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test _predict_masks."""
        mocker.patch.object(
            SegmentAnything, "forward", return_value=(torch.ones(1, 4, 8, 8), torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4))
        )

        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 6

        mask = zero_shot_segment_anything._predict_masks(
            image_embeddings=torch.rand(1),
            point_coords=torch.rand(1, 2, 2),
            point_labels=torch.randint(low=0, high=2, size=(1, 2)),
            original_size=torch.tensor([[8, 8]], dtype=torch.int64),
        )
        assert mask.shape == (8, 8)

    @e2e_pytest_unit
    def test_preprocess_prompts(self, set_zero_shot_segment_anything) -> None:
        """Test _preprocess_prompts.

        TODO (sungchul)
        - get inputs grouped as label and prompts
        - use points and annotations.
        """
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        bboxes = [torch.tensor([0, 0, 1, 1])]
        labels = [MockScoredLabel(label=1)]
        processed_prompts = zero_shot_segment_anything._preprocess_prompts(
            bboxes=bboxes,
            labels=labels,
        )

        # processed_prompts = {labels[0]: [{"box": torch.tensor([[0, 0, 1, 1]])}]}
        assert torch.equal(processed_prompts[labels[0]][0].get("box")[0], bboxes[0])

    @e2e_pytest_unit
    def test_generate_masked_features(self, set_zero_shot_segment_anything) -> None:
        """Test _generate_masked_features."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 16
        feats = torch.rand((8, 8, 1))
        masks = torch.zeros((16, 16), dtype=torch.float32)
        masks[4:12, 4:12] = 1.0

        masked_feat = zero_shot_segment_anything._generate_masked_features(feats=feats, masks=masks, threshold_mask=0.3)

        assert masked_feat.shape == (1, 1)

    @e2e_pytest_unit
    def test_pad_to_square(self, set_zero_shot_segment_anything) -> None:
        """Test _pad_to_square."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 16

        result = zero_shot_segment_anything._pad_to_square(x=torch.ones(1, 1, 8, 8))

        assert result[:8, :8].sum() == 8**2
        assert result[:8, 8:].sum() == 0
        assert result[8:, :8].sum() == 0
        assert result[8:, 8:].sum() == 0

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "masks,logits,expected",
        [
            (torch.ones(1, 4, 8, 8), torch.ones(1, 4, 4, 4), torch.ones(8, 8)),
            (torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 4, 4), torch.zeros(8, 8)),
        ],
    )
    def test_postprocess_masks(
        self, set_zero_shot_segment_anything, masks: torch.Tensor, logits: torch.Tensor, expected: torch.Tensor
    ) -> None:
        """Test _postprocess_masks."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 4
        scores = torch.tensor([[0.0, 0.1, 0.2, 0.3]])

        _, result = zero_shot_segment_anything._postprocess_masks(masks, logits, scores)

        assert torch.equal(result, expected)

    @e2e_pytest_unit
    @pytest.mark.parametrize("use_only_background", [True, False])
    def test_merge_prompts(self, set_zero_shot_segment_anything, use_only_background: bool) -> None:
        """Test _merge_prompts."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()

        input_prompts = {"point_coords": torch.tensor([1]), "point_labels": torch.tensor([1])}
        processed_prompts = {
            MockScoredLabel(label=0): [{"point_coords": torch.tensor([0]), "point_labels": torch.tensor([0])}],
            MockScoredLabel(label=2): [{"point_coords": torch.tensor([2]), "point_labels": torch.tensor([1])}],
        }

        merged_input_prompts = zero_shot_segment_anything._merge_prompts(
            label=MockScoredLabel(label=1),
            input_prompts=input_prompts,
            processed_prompts=processed_prompts,
            use_only_background=use_only_background,
        )

        if use_only_background:
            assert torch.equal(merged_input_prompts.get("point_coords"), torch.tensor([1, 0]))
            assert torch.equal(merged_input_prompts.get("point_labels"), torch.tensor([1, 0]))
        else:
            assert torch.equal(merged_input_prompts.get("point_coords"), torch.tensor([1, 0, 2]))
            assert torch.equal(merged_input_prompts.get("point_labels"), torch.tensor([1, 0, 0]))
