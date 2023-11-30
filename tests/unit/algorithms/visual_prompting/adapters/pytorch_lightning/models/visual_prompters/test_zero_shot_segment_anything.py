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
    ZeroShotSegmentAnything
)
from tests.unit.algorithms.visual_prompting.test_helpers import MockScoredLabel, MockImageEncoder, MockPromptEncoder, MockMaskDecoder


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
            reference_prompts=torch.zeros((self.prompt_getter.image_size, self.prompt_getter.image_size))
        )
        
        assert self.prompt_getter.reference_feats[1].sum() == 9
        assert self.prompt_getter.reference_prompts[1].sum() == 0
        
    @e2e_pytest_unit
    def test_forward(self, mocker) -> None:
        """Test forward."""
        mocker.patch("otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.ZeroShotSegmentAnything")
        mocker.patch.object(self.prompt_getter, "_point_selection", return_value=("points_scores", "bg_coords"))
        
        image_embeddings = torch.rand(1, 2, self.prompt_getter.image_size, self.prompt_getter.image_size)
        self.prompt_getter.reference_feats = {1: torch.rand(1, 2)}
        
        prompts = self.prompt_getter(
            image_embeddings=image_embeddings,
            padding=(0, 0, 0, 0),
            original_size=(self.prompt_getter.image_size, self.prompt_getter.image_size))
        
        assert 1 in prompts
        assert prompts[1] == ("points_scores", "bg_coords")

    @e2e_pytest_unit
    def test_preprocess_target_feat(self) -> None:
        """Test _preprocess_target_feat."""
        old_target_feat = torch.arange(1, self.prompt_getter.image_size**2 + 1, dtype=torch.float).reshape(1, 1, self.prompt_getter.image_size, self.prompt_getter.image_size)
        new_target_feat = self.prompt_getter._preprocess_target_feat(
            target_feat=old_target_feat,
            c_feat=1,
            h_feat=self.prompt_getter.image_size,
            w_feat=self.prompt_getter.image_size)

        assert new_target_feat.sum() == 9
        assert new_target_feat.shape == (1, self.prompt_getter.image_size**2)
        
    @e2e_pytest_unit
    def test_point_selection(self) -> None:
        """Test _point_selection."""
        mask_sim = torch.arange(0.1, 1., 0.1).reshape(self.prompt_getter.image_size, self.prompt_getter.image_size)
        
        points_scores, bg_coords = self.prompt_getter._point_selection(
            mask_sim=mask_sim,
            original_size=(self.prompt_getter.image_size, self.prompt_getter.image_size),
            threshold=0.5,
            downsizing=1)
        
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
                "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMPromptEncoder",
                MockPromptEncoder,
            )
            monkeypatch.setattr(
                "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMMaskDecoder",
                MockMaskDecoder,
            )
            return ZeroShotSegmentAnything(state_dict=state_dict)
        return zero_shot_segment_anything
    
    @e2e_pytest_unit
    @pytest.mark.parametrize("state_dict", [
        None,
        {
            "prompt_getter.reference_feats": "prompt_getter.reference_feats",
            "prompt_getter.reference_prompts": "prompt_getter.reference_prompts",
        }
    ])
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
    def test_learn(self) -> None:
        """Test learn."""
    
    @e2e_pytest_unit
    def test_infer(self) -> None:
        """Test infer."""
        
    @e2e_pytest_unit
    def test_forward(self) -> None:
        """Test forward."""
        
    @e2e_pytest_unit
    def test_training_step(self) -> None:
        """Test training_step."""
        
    @e2e_pytest_unit
    def test_predict_step(self) -> None:
        """Test predict_step."""
        
    @e2e_pytest_unit
    def test_preprocess_prompts(self) -> None:
        """Test _preprocess_prompts."""
        
    @e2e_pytest_unit
    def test_generate_masked_features(self) -> None:
        """Test _generate_masked_features."""
        
    @e2e_pytest_unit
    def test_preprocess_mask(self) -> None:
        """Test _preprocess_mask."""
        
    @e2e_pytest_unit
    def test_update_value(self) -> None:
        """Test _update_value."""
        
    @e2e_pytest_unit
    def test_merge_prompts(self) -> None:
        """Test _merge_prompts."""
        
    @e2e_pytest_unit
    def test_predict_target_mask(self) -> None:
        """Test _predict_target_mask."""
        
    @e2e_pytest_unit
    def test_predict_mask(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test _predict_mask."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        mocker.patch.object(zero_shot_segment_anything, "postprocess_masks", return_value=torch.Tensor([[1]]))
        
        masks, scores, low_res_masks = zero_shot_segment_anything._predict_mask(
            image_embeddings=torch.rand(1),
            input_prompts={},
            padding=(0, 0, 0, 0),
            original_size=(1, 1))
        
        assert masks.dtype == torch.bool
        assert scores == torch.tensor([[1.]])
        assert low_res_masks == torch.tensor([[1.]])
