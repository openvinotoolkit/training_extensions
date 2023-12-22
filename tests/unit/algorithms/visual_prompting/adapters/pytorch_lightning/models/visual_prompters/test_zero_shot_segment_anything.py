"""Tests Segment Anything for zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from typing import Dict, Any, Optional
from collections import OrderedDict
from tests.test_suite.e2e_test_system import e2e_pytest_unit
import torch
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
    def test_initialize(self) -> None:
        """Test initialize."""
        assert not self.prompt_getter.reference_feats
        assert not self.prompt_getter.reference_prompts

    @e2e_pytest_unit
    def test_set_default_thresholds(self) -> None:
        """Test set_default_thresholds."""
        assert self.prompt_getter.default_threshold_reference == 0.3
        assert self.prompt_getter.default_threshold_target == 0.65

        self.prompt_getter.set_default_thresholds(default_threshold_reference=0.5, default_threshold_target=0.7)

        assert self.prompt_getter.default_threshold_reference == 0.5
        assert self.prompt_getter.default_threshold_target == 0.7

    @e2e_pytest_unit
    def test_set_reference(self) -> None:
        """Test set_reference."""
        self.prompt_getter.set_reference(
            label=MockScoredLabel(label=1),
            reference_feats=torch.ones((self.prompt_getter.image_size, self.prompt_getter.image_size)),
            reference_prompts=torch.ones((self.prompt_getter.image_size, self.prompt_getter.image_size)),
        )

        assert self.prompt_getter.reference_feats[0].sum() == 0
        assert self.prompt_getter.reference_prompts[0].sum() == 0
        assert self.prompt_getter.reference_feats[1].sum() == 9
        assert self.prompt_getter.reference_prompts[1].sum() == 9

        self.prompt_getter.set_reference(
            label=MockScoredLabel(label=3),
            reference_feats=torch.ones((self.prompt_getter.image_size, self.prompt_getter.image_size)),
            reference_prompts=torch.ones((self.prompt_getter.image_size, self.prompt_getter.image_size)),
        )

        assert self.prompt_getter.reference_feats[2].sum() == 0
        assert self.prompt_getter.reference_prompts[2].sum() == 0
        assert self.prompt_getter.reference_feats[3].sum() == 9
        assert self.prompt_getter.reference_prompts[3].sum() == 9

    @e2e_pytest_unit
    def test_forward(self, mocker) -> None:
        """Test forward."""
        mocker.patch.object(
            self.prompt_getter,
            "get_prompt_candidates",
            return_value=(torch.tensor([[[0, 0, 0.5], [1, 1, 0.7]]]), torch.tensor([[[2, 2]]])),
        )
        image_embeddings = torch.ones(1, 4, 4, 4)
        self.prompt_getter.reference_feats = torch.rand(1, 1, 4)
        original_size = torch.tensor((self.prompt_getter.image_size, self.prompt_getter.image_size), dtype=torch.int64)

        total_points_scores, total_bg_coords = self.prompt_getter(
            image_embeddings=image_embeddings, original_size=original_size
        )

        assert total_points_scores.shape[0] == 1
        assert total_bg_coords.shape[0] == 1

    @e2e_pytest_unit
    def test_get_prompt_candidates(self, mocker) -> None:
        """Test get_prompt_candidates."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.ZeroShotSegmentAnything"
        )
        mocker.patch.object(self.prompt_getter, "_point_selection", return_value=("points_scores", "bg_coords"))
        image_embeddings = torch.ones(1, 4, 4, 4)
        self.prompt_getter.reference_feats = torch.rand(1, 1, 4)
        label = torch.tensor([[0]], dtype=torch.int64)
        original_size = torch.tensor(
            [[self.prompt_getter.image_size, self.prompt_getter.image_size]], dtype=torch.int64
        )

        points_scores, bg_coords = self.prompt_getter.get_prompt_candidates(
            image_embeddings=image_embeddings, label=label, original_size=original_size
        )

        assert points_scores == "points_scores"
        assert bg_coords == "bg_coords"

    @e2e_pytest_unit
    def test_point_selection(self) -> None:
        """Test _point_selection."""
        mask_sim = torch.arange(0.1, 1.0, 0.1).reshape(self.prompt_getter.image_size, self.prompt_getter.image_size)

        points_scores, bg_coords = self.prompt_getter._point_selection(
            mask_sim=mask_sim,
            original_size=torch.tensor([self.prompt_getter.image_size, self.prompt_getter.image_size]),
            threshold=torch.tensor([[0.5]]),
        )

        assert torch.equal(points_scores, torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]))
        assert torch.equal(bg_coords, torch.tensor([[0, 0]]))


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
                "prompt_getter.reference_feats": "prompt_getter.reference_feats",
                "prompt_getter.reference_prompts": "prompt_getter.reference_prompts",
            },
        ],
    )
    def test_init(self, set_zero_shot_segment_anything, state_dict: Dict[str, Any]) -> None:
        """Test __init__."""
        zero_shot_segment_anything = set_zero_shot_segment_anything(state_dict=state_dict)

        assert zero_shot_segment_anything.config.model.freeze_image_encoder
        assert zero_shot_segment_anything.config.model.freeze_prompt_encoder
        assert zero_shot_segment_anything.config.model.freeze_mask_decoder

        if state_dict:
            zero_shot_segment_anything.prompt_getter.reference_feats = "prompt_getter.reference_feats"
            zero_shot_segment_anything.prompt_getter.reference_prompts = "prompt_getter.reference_prompts"

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
            return_value=torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]),
        )

        processed_prompts = {MockScoredLabel(label=1, name="label"): [{"box": torch.tensor([[0, 0, 1, 1]])}]}
        zero_shot_segment_anything.learn(
            images=torch.ones((1, 3, 8, 8)),
            processed_prompts=processed_prompts,
            padding=(0, 0, 0, 0),
            original_size=(8, 8),
        )

        assert zero_shot_segment_anything.prompt_getter.reference_feats.shape == (2, 1, 2)
        assert zero_shot_segment_anything.prompt_getter.reference_prompts.shape == (2, 8, 8)

    @e2e_pytest_unit
    @pytest.mark.parametrize("expected", [[torch.ones((8, 8)), torch.tensor([0.0, 0.0, 0.5])]])
    def test_infer(self, monkeypatch, mocker, set_zero_shot_segment_anything, expected: torch.Tensor) -> None:
        """Test infer."""
        monkeypatch.setattr(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.PromptGetter",
            MockPromptGetter,
        )

        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.prompt_getter.reference_feats = torch.rand(1, 1, 4)
        zero_shot_segment_anything.prompt_getter.reference_prompts = torch.zeros((8, 8))
        mocker.patch.object(
            SegmentAnything, "forward", return_value=(torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4))
        )

        total_results = zero_shot_segment_anything.infer(
            images=torch.ones((1, 3, 8, 8)), original_size=torch.tensor([[8, 8]], dtype=torch.int64)
        )

        for i, results in enumerate(total_results[0]):
            for _, result in results.items():
                assert torch.equal(result[0], expected[i])

    @e2e_pytest_unit
    @pytest.mark.parametrize("is_postprocess", [True, False])
    def test_predict_masks(self, mocker, set_zero_shot_segment_anything, is_postprocess: bool) -> None:
        """Test _predict_masks."""
        mocker.patch.object(
            SegmentAnything, "forward", return_value=(torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4))
        )

        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 6

        mask = zero_shot_segment_anything._predict_masks(
            image_embeddings=torch.rand(1),
            point_coords=torch.rand(1, 2, 2),
            point_labels=torch.randint(low=0, high=2, size=(1, 2)),
            original_size=torch.tensor((8, 8), dtype=torch.int64),
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
    def test_preprocess_masks(self, set_zero_shot_segment_anything) -> None:
        """Test _preprocess_masks."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 16

        result = zero_shot_segment_anything._preprocess_masks(x=torch.ones(1, 1, 8, 8))

        assert result[:8, :8].sum() == 8**2
        assert result[:8, 8:].sum() == 0
        assert result[8:, :8].sum() == 0
        assert result[8:, 8:].sum() == 0

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "logits,expected",
        [
            (torch.ones(1, 4, 4, 4), torch.ones(4, 4, dtype=torch.bool)),
            (torch.zeros(1, 4, 4, 4), torch.zeros(4, 4, dtype=torch.bool)),
        ],
    )
    def test_postprocess_masks(
        self, set_zero_shot_segment_anything, logits: torch.Tensor, expected: torch.Tensor
    ) -> None:
        """Test _postprocess_masks."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 4
        scores = torch.tensor([[0.0, 0.1, 0.2, 0.3]])
        original_size = torch.tensor([4, 4], dtype=torch.int64)

        _, result = zero_shot_segment_anything._postprocess_masks(logits, scores, original_size)

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
