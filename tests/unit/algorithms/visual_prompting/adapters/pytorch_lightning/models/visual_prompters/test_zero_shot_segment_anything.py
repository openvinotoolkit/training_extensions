"""Tests Segment Anything for zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from typing import Dict, Any, Optional
from collections import OrderedDict
from tests.test_suite.e2e_test_system import e2e_pytest_unit
import torch
import numpy as np
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
from pytorch_lightning import Trainer


class TestPromptGetter:
    @pytest.fixture
    def prompt_getter(self) -> PromptGetter:
        return PromptGetter(image_size=4, downsizing=1)

    @e2e_pytest_unit
    def test_set_default_thresholds(self, prompt_getter) -> None:
        """Test set_default_thresholds."""
        assert prompt_getter.default_threshold_reference == 0.3
        assert prompt_getter.default_threshold_target == 0.65

        prompt_getter.set_default_thresholds(default_threshold_reference=0.5, default_threshold_target=0.7)

        assert prompt_getter.default_threshold_reference == 0.5
        assert prompt_getter.default_threshold_target == 0.7

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "result_point_selection",
        [torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), torch.tensor([[-1, -1, -1]])],
    )
    def test_forward(self, mocker, prompt_getter, result_point_selection: torch.Tensor) -> None:
        """Test forward."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.ZeroShotSegmentAnything"
        )
        mocker.patch.object(prompt_getter, "_point_selection", return_value=(result_point_selection, torch.zeros(1, 2)))
        image_embeddings = torch.ones(1, 4, 4, 4)
        reference_feat = torch.rand(1, 4)
        original_size = torch.tensor([[prompt_getter.image_size, prompt_getter.image_size]], dtype=torch.int64)

        points_scores, bg_coords = prompt_getter(
            image_embeddings=image_embeddings, reference_feat=reference_feat, original_size=original_size
        )

        assert torch.all(points_scores == result_point_selection)
        assert torch.all(bg_coords == torch.zeros(1, 2))

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "result_point_selection",
        [torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), torch.tensor([[-1, -1, -1]])],
    )
    def test_get_prompt_candidates(self, mocker, prompt_getter, result_point_selection: torch.Tensor) -> None:
        """Test get_prompt_candidates."""
        mocker.patch.object(prompt_getter, "_point_selection", return_value=(result_point_selection, torch.zeros(1, 2)))
        image_embeddings = torch.ones(1, 4, 4, 4)
        reference_feats = torch.rand(1, 1, 4)
        used_indices = torch.as_tensor([[0]])
        original_size = torch.tensor((prompt_getter.image_size, prompt_getter.image_size), dtype=torch.int64)

        total_points_scores, total_bg_coords = prompt_getter.get_prompt_candidates(
            image_embeddings=image_embeddings,
            reference_feats=reference_feats,
            used_indices=used_indices,
            original_size=original_size,
        )

        assert total_points_scores[0].shape[0] == len(result_point_selection)
        assert total_bg_coords[0].shape[0] == 1

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "mask_sim,expected",
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
            original_size=torch.tensor([prompt_getter.image_size, prompt_getter.image_size]),
            threshold=torch.tensor([[0.5]]),
        )

        assert torch.equal(points_scores, expected)


class TestZeroShotSegmentAnything:
    @pytest.fixture
    def set_zero_shot_segment_anything(self, monkeypatch):
        def zero_shot_segment_anything(
            manual_config_update: Optional[Dict] = None, state_dict: Optional[OrderedDict] = None
        ):
            monkeypatch.setattr(
                "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMImageEncoder",
                MockImageEncoder,
            )
            monkeypatch.setattr(
                "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SAMMaskDecoder",
                MockMaskDecoder,
            )
            return ZeroShotSegmentAnything(manual_config_update=manual_config_update, state_dict=state_dict)

        return zero_shot_segment_anything

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "state_dict",
        [
            None,
            {},
        ],
    )
    def test_init(self, set_zero_shot_segment_anything, state_dict: Optional[Dict[str, Any]]) -> None:
        """Test __init__."""
        if state_dict is not None:
            state_dict = set_zero_shot_segment_anything().state_dict()
            state_dict.pop("reference_info.reference_feats")
            state_dict.pop("reference_info.used_indices")

        zero_shot_segment_anything = set_zero_shot_segment_anything(state_dict=state_dict)

        assert zero_shot_segment_anything.config.model.freeze_image_encoder
        assert zero_shot_segment_anything.config.model.freeze_prompt_encoder
        assert zero_shot_segment_anything.config.model.freeze_mask_decoder

        if state_dict:
            assert zero_shot_segment_anything.reference_info.reference_feats is not None
            assert zero_shot_segment_anything.reference_info.used_indices is not None

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
    def test_expand_reference_info(self, set_zero_shot_segment_anything):
        """Test expand_reference_info."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.reference_info["reference_feats"] = torch.ones((3, 2, 2))
        new_largest_label = 5

        zero_shot_segment_anything.expand_reference_info(new_largest_label)

        assert zero_shot_segment_anything.reference_info["reference_feats"].shape == (6, 2, 2)
        assert torch.all(zero_shot_segment_anything.reference_info["reference_feats"][:3] == 1.0)
        assert torch.all(zero_shot_segment_anything.reference_info["reference_feats"][3:] == 0.0)

    @e2e_pytest_unit
    def test_learn(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test learn."""
        zero_shot_segment_anything = set_zero_shot_segment_anything(manual_config_update={"model.image_size": 4})
        mocker.patch.object(
            zero_shot_segment_anything,
            "_predict_masks",
            return_value=torch.tensor([[[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]]),
        )
        mocker.patch.object(zero_shot_segment_anything, "_generate_masked_features", return_value=torch.ones(1, 256))

        batch = [
            {
                "images": np.ones((4, 4, 3), dtype=np.uint8),
                "gt_masks": np.ones((4, 4), dtype=np.uint8),
                "bboxes": np.array([[0, 0, 1, 1]], dtype=np.float32),
                "points": np.zeros((0, 2), dtype=np.float32),
                "labels": {"bboxes": [MockScoredLabel(label=0, name="label")]},
                "original_size": np.array([4, 4], dtype=np.int64),
            }
        ]
        zero_shot_segment_anything.learn(batch=batch, reset_feat=True)

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

        zero_shot_segment_anything = set_zero_shot_segment_anything(manual_config_update={"model.image_size": 4})
        reference_feats = nn.Parameter(torch.rand(1, 1, 256), requires_grad=False)
        used_indices = nn.Parameter(torch.as_tensor([[0]], dtype=torch.int64), requires_grad=False)
        mocker.patch.object(
            SegmentAnything,
            "forward",
            return_value=(torch.ones(1, 4, 4, 4), torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4)),
        )

        batch = [
            {
                "images": np.ones((4, 4, 3), dtype=np.uint8),
                "gt_masks": np.ones((4, 4), dtype=np.uint8),
                "original_size": np.array([4, 4], dtype=np.int64),
            }
        ]
        total_results = zero_shot_segment_anything.infer(
            batch=batch,
            reference_feats=reference_feats,
            used_indices=used_indices,
        )

        for i, results in enumerate(total_results[0]):
            for _, result in results.items():
                assert torch.equal(result[0], expected[i])

    @e2e_pytest_unit
    def test_inspect_overlapping_areas(self, mocker, set_zero_shot_segment_anything) -> None:
        """Test _inspect_overlapping_areas."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )
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
            SegmentAnything,
            "forward",
            return_value=(torch.ones(1, 4, 8, 8), torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4)),
        )

        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.image_size = 6

        mask = zero_shot_segment_anything._predict_masks(
            image_embeddings=torch.rand(1),
            point_coords=torch.rand(1, 2, 2),
            point_labels=torch.randint(low=0, high=2, size=(1, 2)),
            original_size=torch.tensor([8, 8], dtype=torch.int64),
        )
        assert mask.shape == (8, 8)

    @e2e_pytest_unit
    def test_preprocess_prompts(self, set_zero_shot_segment_anything) -> None:
        """Test _preprocess_prompts."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        transformed_batch = {
            "bboxes": torch.tensor([[0, 0, 1, 1]]),
            "points": torch.tensor([[2, 2]]),
            "labels": {"bboxes": [MockScoredLabel(label=1)], "points": [MockScoredLabel(label=1)]},
        }
        processed_prompts = zero_shot_segment_anything._preprocess_prompts(transformed_batch)

        for prompts in processed_prompts.values():
            for prompt in prompts:
                if "bboxes" in prompt:
                    prompt["bboxes"]["point_coords"].shape == (1, 2, 2)
                elif "points" in prompt:
                    prompt["points"]["point_coords"].shape == (1, 1, 2)

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
    def test_find_latest_reference_info(self, mocker, set_zero_shot_segment_anything):
        """Test _find_latest_reference_info."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.os.path.isdir",
            return_value=True,
        )

        # there are some saved reference info
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.os.listdir",
            return_value=["1", "2"],
        )
        results = zero_shot_segment_anything._find_latest_reference_info()
        assert results == "2"

        # there are no saved reference info
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.os.listdir",
            return_value=[],
        )
        results = zero_shot_segment_anything._find_latest_reference_info()
        assert results is None

    @e2e_pytest_unit
    def test_on_predict_start(self, mocker, set_zero_shot_segment_anything):
        """Test on_predict_start."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.os.path.isdir",
            return_value=True,
        )

        # get previously saved reference info
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.os.listdir",
            return_value=["1", "2"],
        )
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.torch.load",
            return_value=torch.nn.ParameterDict(
                {"reference_feats": torch.zeros((1, 1, 256)), "used_indices": torch.tensor([0.0])}
            ),
        )
        mocker.patch("builtins.open", return_value="Mocked data")

        zero_shot_segment_anything.on_predict_start()
        assert isinstance(zero_shot_segment_anything.reference_info, torch.nn.ParameterDict)
        assert zero_shot_segment_anything.reference_info["reference_feats"].shape == (1, 1, 256)
        assert zero_shot_segment_anything.reference_info["used_indices"].shape == (1,)

        # no saved reference info
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.os.listdir",
            return_value=[],
        )

        zero_shot_segment_anything.set_empty_reference_info()
        zero_shot_segment_anything.on_predict_start()

        assert zero_shot_segment_anything.reference_info["reference_feats"].shape == (0,)
        assert zero_shot_segment_anything.reference_info["used_indices"].shape == (0,)

    @e2e_pytest_unit
    def test_training_epoch_end(self, mocker, set_zero_shot_segment_anything):
        """Test training_epoch_end."""
        zero_shot_segment_anything = set_zero_shot_segment_anything()
        zero_shot_segment_anything.config.model.save_outputs = True

        mocker_makedirs = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.os.makedirs"
        )
        mocker_save = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.torch.save"
        )
        mocker_open = mocker.patch("builtins.open", return_value="Mocked data")
        mocker_pickle_dump = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.pickle.dump"
        )
        mocker_json_dump = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.zero_shot_segment_anything.json.dump"
        )

        from unittest.mock import Mock

        zero_shot_segment_anything._trainer = Mock(autospec=Trainer)
        zero_shot_segment_anything.training_epoch_end(None)

        mocker_makedirs.assert_called_once()
        mocker_save.assert_called_once()
        mocker_open.assert_called()
        assert mocker_open.call_count == 2
        mocker_pickle_dump.assert_called_once()
        mocker_json_dump.assert_called_once()
