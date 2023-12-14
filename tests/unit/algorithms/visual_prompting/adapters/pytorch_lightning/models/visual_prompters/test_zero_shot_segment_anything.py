"""Tests Segment Anything for zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from typing import Dict, Any, Optional
from collections import OrderedDict
from tests.test_suite.e2e_test_system import e2e_pytest_unit
import torch
from omegaconf import DictConfig

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything import (
    PromptGetter,
    ZeroShotSegmentAnything,
)
from tests.unit.algorithms.visual_prompting.test_helpers import MockScoredLabel, MockImageEncoder, MockPromptGetter


class TestPromptGetter:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.prompt_getter = PromptGetter(image_size=3)

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
            reference_prompts=torch.zeros((self.prompt_getter.image_size, self.prompt_getter.image_size)),
        )

        assert self.prompt_getter.reference_feats[1].sum() == 9
        assert self.prompt_getter.reference_prompts[1].sum() == 0

    @e2e_pytest_unit
    def test_forward(self, mocker) -> None:
        """Test forward."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.ZeroShotSegmentAnything"
        )
        mocker.patch.object(self.prompt_getter, "_point_selection", return_value=("points_scores", "bg_coords"))

        image_embeddings = torch.rand(1, 2, self.prompt_getter.image_size, self.prompt_getter.image_size)
        self.prompt_getter.reference_feats = {1: torch.rand(1, 2)}

        prompts = self.prompt_getter(
            image_embeddings=image_embeddings,
            padding=(0, 0, 0, 0),
            original_size=(self.prompt_getter.image_size, self.prompt_getter.image_size),
        )

        assert 1 in prompts
        assert prompts[1] == ("points_scores", "bg_coords")

    @e2e_pytest_unit
    def test_preprocess_target_feat(self) -> None:
        """Test _preprocess_target_feat."""
        old_target_feat = torch.arange(1, self.prompt_getter.image_size**2 + 1, dtype=torch.float).reshape(
            1, 1, self.prompt_getter.image_size, self.prompt_getter.image_size
        )
        new_target_feat = self.prompt_getter._preprocess_target_feat(
            target_feat=old_target_feat,
            c_feat=1,
            h_feat=self.prompt_getter.image_size,
            w_feat=self.prompt_getter.image_size,
        )

        assert new_target_feat.sum() == 9
        assert new_target_feat.shape == (1, self.prompt_getter.image_size**2)

    @e2e_pytest_unit
    def test_point_selection(self) -> None:
        """Test _point_selection."""
        mask_sim = torch.arange(0.1, 1.0, 0.1).reshape(self.prompt_getter.image_size, self.prompt_getter.image_size)

        points_scores, bg_coords = self.prompt_getter._point_selection(
            mask_sim=mask_sim,
            original_size=(self.prompt_getter.image_size, self.prompt_getter.image_size),
            threshold=0.5,
            downsizing=1,
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
            "_predict_mask",
            return_value=(
                torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]),
                torch.tensor([1, 0, 0]),
                torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]),
            ),
        )

        processed_prompts = {MockScoredLabel(label=1, name="label"): [{"box": torch.tensor([[0, 0, 1, 1]])}]}
        zero_shot_segment_anything.learn(
            images=torch.ones((1, 3, 8, 8)),
            processed_prompts=processed_prompts,
            padding=(0, 0, 0, 0),
            original_size=(8, 8),
        )

        assert zero_shot_segment_anything.prompt_getter.reference_feats.get(1).shape == (1, 2)
        assert zero_shot_segment_anything.prompt_getter.reference_prompts.get(1).shape == (8, 8)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "expected", [[torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), torch.tensor([0.0, 0.0, 0.5])]]
    )
    def test_infer(self, monkeypatch, mocker, set_zero_shot_segment_anything, expected: torch.Tensor) -> None:
        """Test infer."""
        monkeypatch.setattr(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.PromptGetter",
            MockPromptGetter,
        )

        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.prompt_getter.reference_feats = {1: torch.rand((1, 2))}
        zero_shot_segment_anything.prompt_getter.reference_prompts = {1: torch.zeros((8, 8))}
        mocker.patch.object(
            zero_shot_segment_anything,
            "_predict_mask",
            return_value=(
                torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]),
                torch.tensor([1, 0, 0]),
                torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]),
            ),
        )

        total_results = zero_shot_segment_anything.infer(
            images=torch.ones((1, 3, 8, 8)), padding=(0, 0, 0, 0), original_size=(8, 8)
        )

        for i, results in enumerate(total_results[0]):
            for _, result in results.items():
                assert torch.equal(result[0], expected[i])

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
    def test_preprocess_mask(self, set_zero_shot_segment_anything) -> None:
        """Test _preprocess_mask."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 16

        result = zero_shot_segment_anything._preprocess_mask(x=torch.ones(1, 1, 8, 8))

        assert result[:8, :8].sum() == 8**2
        assert result[:8, 8:].sum() == 0
        assert result[8:, :8].sum() == 0
        assert result[8:, 8:].sum() == 0

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

    @e2e_pytest_unit
    def test_predict_target_mask(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test _predict_target_mask."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        mocker.patch.object(
            zero_shot_segment_anything,
            "_predict_mask",
            return_value=(
                torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]),
                torch.tensor([1, 0, 0]),
                torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]),
            ),
        )

        mask = zero_shot_segment_anything._predict_target_mask(
            image_embeddings=torch.rand(1), input_prompts={}, padding=(0, 0, 0, 0), original_size=(1, 1)
        )

        assert mask.shape == (3, 3)

    @e2e_pytest_unit
    def test_predict_mask(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test _predict_mask."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        mocker.patch.object(zero_shot_segment_anything, "postprocess_masks", return_value=torch.Tensor([[1]]))

        masks, scores, low_res_masks = zero_shot_segment_anything._predict_mask(
            image_embeddings=torch.rand(1), input_prompts={}, padding=(0, 0, 0, 0), original_size=(1, 1)
        )

        assert masks.dtype == torch.bool
        assert scores.shape[1] == 3
        assert low_res_masks.shape[1] == 3
