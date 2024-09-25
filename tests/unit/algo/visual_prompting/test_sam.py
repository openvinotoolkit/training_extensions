# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from unittest import mock

import pytest
import torch
from otx.algo.visual_prompting.backbones.tiny_vit import TinyViT
from otx.algo.visual_prompting.decoders.sam_mask_decoder import SAMMaskDecoder
from otx.algo.visual_prompting.encoders.sam_prompt_encoder import SAMPromptEncoder
from otx.algo.visual_prompting.losses.sam_loss import SAMCriterion
from otx.algo.visual_prompting.sam import SAM, CommonSettingMixin, ZeroShotSAM
from otx.core.data.entity.base import Points
from torch import Tensor, nn
from torchvision import tv_tensors


class TestCommonSettingMixin:
    def test_load_checkpoint_success(self, mocker) -> None:
        # Mock torch.hub.load_state_dict_from_url
        mock_load_state_dict_from_url = mocker.patch("torch.hub.load_state_dict_from_url")

        # Mock state dictionary returned by load_state_dict_from_url
        mock_state_dict = {
            "image_encoder.norm_head.weight": torch.tensor([1.0]),
            "image_encoder.norm_head.bias": torch.tensor([1.0]),
            "image_encoder.head.weight": torch.tensor([1.0]),
            "image_encoder.head.bias": torch.tensor([1.0]),
            "some_other_key": torch.tensor([1.0]),
        }
        mock_load_state_dict_from_url.return_value = mock_state_dict

        # Create an instance of CommonSettingMixin and set the mock model
        mixin = CommonSettingMixin()
        mixin.load_state_dict = mock.Mock()

        # Call the load_checkpoint method
        mixin.load_checkpoint("https://example.com/checkpoint.pth")

        # Assertions
        mock_load_state_dict_from_url.assert_called_once_with("https://example.com/checkpoint.pth")
        mixin.load_state_dict.assert_called_once_with(mock_state_dict)

    def test_load_checkpoint_failure(self, mocker) -> None:
        mock_load_state_dict_from_url = mocker.patch(
            "torch.hub.load_state_dict_from_url",
            side_effect=ValueError("Invalid URL"),
        )
        mock_log_info = mocker.patch("logging.info")

        mixin = CommonSettingMixin()
        mixin.load_checkpoint("invalid_url")

        mock_load_state_dict_from_url.assert_called_once_with("invalid_url")
        mock_log_info.assert_called_once_with(
            "Invalid URL: invalid_url is not desirable format for torch.hub.load_state_dict_from_url. "
            "To manually load invalid_url, try to set it to trainer.checkpoint.",
        )

    @pytest.mark.parametrize("freeze_image_encoder", [True, False])
    @pytest.mark.parametrize("freeze_prompt_encoder", [True, False])
    @pytest.mark.parametrize("freeze_mask_decoder", [True, False])
    def test_freeze_networks(
        self,
        freeze_image_encoder: bool,
        freeze_prompt_encoder: bool,
        freeze_mask_decoder: bool,
    ) -> None:
        class MockModel:
            def __init__(self):
                self.image_encoder = nn.Linear(10, 10)
                self.prompt_encoder = nn.Linear(10, 10)
                self.mask_decoder = nn.Linear(10, 10)

        mock_model = MockModel()
        mixin = CommonSettingMixin()
        mixin.model = mock_model

        mixin.freeze_networks(
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
        )

        for param in mock_model.image_encoder.parameters():
            assert param.requires_grad != freeze_image_encoder

        for param in mock_model.prompt_encoder.parameters():
            assert param.requires_grad != freeze_prompt_encoder

        for param in mock_model.mask_decoder.parameters():
            assert param.requires_grad != freeze_mask_decoder

    def test_forward_for_tracing(self, mocker) -> None:
        mixin = CommonSettingMixin()
        mixin.model = mock.Mock()
        mock_forward_for_tracing = mocker.patch.object(mixin.model, "forward_for_tracing")

        image_embeddings = torch.zeros((1, 256, 64, 64))
        point_coords = torch.zeros((1, 10, 2))
        point_labels = torch.zeros((1, 10))
        mask_input = torch.zeros((1, 1, 256, 256))
        has_mask_input = torch.zeros((1, 1))
        ori_shape = torch.zeros((1, 2))

        mixin.forward_for_tracing(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            ori_shape=ori_shape,
        )

        mock_forward_for_tracing.assert_called_once_with(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            ori_shape=ori_shape,
        )


class TestSAM:
    @pytest.fixture()
    def sam(self) -> SAM:
        return SAM(backbone_type="tiny_vit")

    def test_initialization(self, mocker) -> None:
        mock_freeze_networks = mocker.patch.object(CommonSettingMixin, "freeze_networks")
        mock_load_checkpoint = mocker.patch.object(CommonSettingMixin, "load_checkpoint")

        sam = SAM(backbone_type="tiny_vit")

        assert sam.backbone_type == "tiny_vit"
        assert sam.image_size == 1024
        assert sam.image_embedding_size == 64
        assert sam.use_stability_score is False
        assert sam.return_single_mask is True
        assert sam.return_extra_metrics is False
        assert sam.stability_score_offset == 1.0

        mock_load_checkpoint.assert_called_once_with(load_from=sam.load_from["tiny_vit"])
        mock_freeze_networks.assert_called_once_with(True, True, False)

    def test_build_model(self, sam: SAM) -> None:
        segment_anything = sam._build_model()
        assert segment_anything is not None
        assert isinstance(segment_anything, torch.nn.Module)
        assert segment_anything.__class__.__name__ == "SegmentAnything"

        assert isinstance(segment_anything.image_encoder, TinyViT)
        assert isinstance(segment_anything.prompt_encoder, SAMPromptEncoder)
        assert isinstance(segment_anything.mask_decoder, SAMMaskDecoder)
        assert isinstance(segment_anything.criterion, SAMCriterion)


class TestZeroShotSAM:
    @pytest.fixture()
    def zero_shot_sam(self) -> ZeroShotSAM:
        return ZeroShotSAM(backbone_type="tiny_vit")

    def test_initialization(self, zero_shot_sam: ZeroShotSAM) -> None:
        assert zero_shot_sam.backbone_type == "tiny_vit"
        assert zero_shot_sam.image_size == 1024
        assert zero_shot_sam.image_embedding_size == 64
        assert zero_shot_sam.default_threshold_reference == 0.3
        assert zero_shot_sam.default_threshold_target == 0.65
        assert zero_shot_sam.use_stability_score is False
        assert zero_shot_sam.return_single_mask is False
        assert zero_shot_sam.return_extra_metrics is False
        assert zero_shot_sam.stability_score_offset == 1.0

        assert isinstance(zero_shot_sam.model.image_encoder, nn.Module)
        assert isinstance(zero_shot_sam.model.prompt_encoder, nn.Module)
        assert isinstance(zero_shot_sam.model.mask_decoder, nn.Module)
        assert zero_shot_sam.model.image_size == 1024
        assert zero_shot_sam.model.prompt_getter.default_threshold_reference == 0.3
        assert zero_shot_sam.model.prompt_getter.default_threshold_target == 0.65
        assert zero_shot_sam.model.use_stability_score is False
        assert zero_shot_sam.model.return_single_mask is False
        assert zero_shot_sam.model.return_extra_metrics is False
        assert zero_shot_sam.model.stability_score_offset == 1.0

    def test_build_model(self, zero_shot_sam: ZeroShotSAM) -> None:
        model = zero_shot_sam._build_model()
        assert isinstance(model, nn.Module)
        assert isinstance(model.image_encoder, nn.Module)
        assert isinstance(model.prompt_encoder, nn.Module)
        assert isinstance(model.mask_decoder, nn.Module)
        assert model.image_size == 1024
        assert model.prompt_getter.default_threshold_reference == 0.3
        assert model.prompt_getter.default_threshold_target == 0.65
        assert model.use_stability_score is False
        assert model.return_single_mask is False
        assert model.return_extra_metrics is False
        assert model.stability_score_offset == 1.0

    @pytest.mark.parametrize("training", [True, False])
    def test_forward(self, mocker, zero_shot_sam: ZeroShotSAM, training: bool) -> None:
        """Test forward."""
        mocker_learn = mocker.patch.object(zero_shot_sam, "learn")
        mocker_infer = mocker.patch.object(zero_shot_sam, "infer")
        zero_shot_sam.training = training

        zero_shot_sam.forward(None)

        if training:
            mocker_learn.assert_called_once()
        else:
            mocker_infer.assert_called_once()

    @pytest.mark.parametrize("reset_feat", [True, False])
    def test_learn(self, mocker, zero_shot_sam: ZeroShotSAM, reset_feat: bool) -> None:
        """Test learn."""
        mocker_initialize_reference_info = mocker.patch.object(zero_shot_sam, "initialize_reference_info")
        mocker_learn = mocker.patch.object(zero_shot_sam.model, "learn")
        mocker_customize_inputs = mocker.patch.object(zero_shot_sam, "_customize_inputs")
        mocker_customize_outputs = mocker.patch.object(zero_shot_sam, "_customize_outputs")

        zero_shot_sam.learn(None, reset_feat=reset_feat)

        if reset_feat:
            mocker_initialize_reference_info.assert_called_once()
        else:
            mocker_initialize_reference_info.assert_not_called()
        mocker_learn.assert_called_once()
        mocker_customize_inputs.assert_called_once()
        mocker_customize_outputs.assert_called_once()

    def test_infer(self, mocker, zero_shot_sam: ZeroShotSAM) -> None:
        """Test infer."""
        mocker_infer = mocker.patch.object(zero_shot_sam.model, "infer")
        mocker_customize_inputs = mocker.patch.object(zero_shot_sam, "_customize_inputs")
        mocker_customize_outputs = mocker.patch.object(zero_shot_sam, "_customize_outputs")

        zero_shot_sam.infer(None)

        mocker_infer.assert_called_once()
        mocker_customize_inputs.assert_called_once()
        mocker_customize_outputs.assert_called_once()

    def test_gather_prompts_with_labels(self, zero_shot_sam: ZeroShotSAM, fxt_zero_shot_vpm_data_entity) -> None:
        """Test _gather_prompts_with_labels."""
        entity = deepcopy(fxt_zero_shot_vpm_data_entity[1])

        results = zero_shot_sam._gather_prompts_with_labels(entity)

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
    def test_apply_image(self, zero_shot_sam: ZeroShotSAM, image: tv_tensors.Image, expected: tuple[int, ...]) -> None:
        """Test apply_image."""
        results = zero_shot_sam.apply_image(image, target_length=8)

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("coords", "ori_shape", "expected"),
        [
            (torch.tensor([[1, 1], [2, 2]]), (4, 4), torch.tensor([[2, 2], [4, 4]])),
            (torch.tensor([[4, 4], [8, 8]]), (16, 16), torch.tensor([[2, 2], [4, 4]])),
        ],
    )
    def test_apply_points(
        self,
        zero_shot_sam: ZeroShotSAM,
        coords: Tensor,
        ori_shape: tuple[int, int],
        expected: Tensor,
    ) -> None:
        """Test apply_points."""
        result = zero_shot_sam.apply_points(Points(coords, canvas_size=ori_shape), ori_shape, target_length=8)

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        ("boxes", "ori_shape", "expected"),
        [
            (torch.tensor([[1, 1, 2, 2], [2, 2, 3, 3]]), (4, 4), torch.tensor([[2, 2, 4, 4], [4, 4, 6, 6]])),
            (torch.tensor([[4, 4, 8, 8], [8, 8, 12, 12]]), (16, 16), torch.tensor([[2, 2, 4, 4], [4, 4, 6, 6]])),
        ],
    )
    def test_apply_boxes(
        self,
        zero_shot_sam: ZeroShotSAM,
        boxes: Tensor,
        ori_shape: tuple[int, int],
        expected: Tensor,
    ) -> None:
        """Test apply_boxes."""
        result = zero_shot_sam.apply_boxes(
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
        zero_shot_sam: ZeroShotSAM,
        prompts: list[Points | tv_tensors.BoundingBoxes],
        ori_shape: tuple[int, int],
        expected: list[Points | tv_tensors.BoundingBoxes],
    ) -> None:
        """Test apply_prompts."""
        results = zero_shot_sam.apply_prompts(prompts, ori_shape, target_length=8)

        for r, e in zip(results, expected):
            assert torch.all(r == e)

    @pytest.mark.parametrize(
        ("oldh", "oldw", "expected"),
        [
            (3, 4, (6, 8)),
            (12, 16, (6, 8)),
        ],
    )
    def test_get_preprocess_shape(self, zero_shot_sam: ZeroShotSAM, oldh: int, oldw: int, expected: tuple[int, int]):
        """Test get_preprocess_shape."""
        results = zero_shot_sam.get_preprocess_shape(oldh, oldw, target_length=8)

        assert results == expected

    @pytest.mark.parametrize("image", (tv_tensors.Image(torch.zeros(1, 3, 2, 4, dtype=torch.uint8))))
    def test_preprocess(self, zero_shot_sam: ZeroShotSAM, image: tv_tensors.Image) -> None:
        """Test preprocess."""
        zero_shot_sam.pixel_mean = torch.ones_like(zero_shot_sam.pixel_mean)
        zero_shot_sam.pixel_std = torch.ones_like(zero_shot_sam.pixel_std) * 2
        zero_shot_sam.model.image_size = 8

        results = zero_shot_sam.preprocess(image)

        assert results.shape == (3, 8, 8)
        assert torch.all(torch.unique(results) == torch.tensor((-0.5, 0.0)))

    def test_initialize_reference_info(self, zero_shot_sam: ZeroShotSAM) -> None:
        """Test initialize_reference_info."""
        zero_shot_sam.initialize_reference_info()

        assert zero_shot_sam.reference_feats.shape == (0, 1, 256)
        assert zero_shot_sam.used_indices.shape == (0,)

    def test_save_reference_info(self, mocker, tmpdir, zero_shot_sam: ZeroShotSAM) -> None:
        """Test save_reference_info."""
        zero_shot_sam.root_reference_info = tmpdir
        zero_shot_sam.reference_feats = torch.tensor(1)
        zero_shot_sam.used_indices = torch.tensor(1)
        mocker_mkdir = mocker.patch("pathlib.Path.mkdir")
        mocker.patch("pathlib.Path.open")
        mocker_torch_save = mocker.patch("torch.save")
        mocker_pickle_dump = mocker.patch("pickle.dump")

        zero_shot_sam.save_reference_info(".")

        mocker_mkdir.assert_called_once()
        mocker_torch_save.assert_called_once()
        mocker_pickle_dump.assert_called_once()

    def test_load_reference_info(self, mocker, zero_shot_sam: ZeroShotSAM) -> None:
        """Test load_reference_info."""
        # get previously saved reference info
        mocker.patch(
            "torch.load",
            return_value={"reference_feats": torch.zeros((1, 1, 256)), "used_indices": torch.tensor([0.0])},
        )
        mocker.patch("pathlib.Path.is_file", return_value=True)

        zero_shot_sam.load_reference_info(".")
        assert zero_shot_sam.reference_feats.shape == (1, 1, 256)
        assert zero_shot_sam.used_indices.shape == (1,)

        # no saved reference info
        mocker.patch("pathlib.Path.is_file", return_value=False)

        zero_shot_sam.initialize_reference_info()
        zero_shot_sam.load_reference_info(".")

        assert zero_shot_sam.reference_feats.shape == (0, 1, 256)
        assert zero_shot_sam.used_indices.shape == (0,)
