# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from otx.algo.visual_prompting.decoders.sam_mask_decoder import SAMMaskDecoder
from otx.algo.visual_prompting.encoders.sam_image_encoder import SAMImageEncoder
from otx.algo.visual_prompting.encoders.sam_prompt_encoder import SAMPromptEncoder
from otx.algo.visual_prompting.losses.sam_loss import SAMCriterion
from otx.algo.visual_prompting.visual_prompters.segment_anything import (
    PromptGetter,
    SegmentAnything,
    ZeroShotSegmentAnything,
)
from otx.core.data.entity.base import Points
from torch import Tensor, nn
from torchvision import tv_tensors


class TestSegmentAnything:
    @pytest.fixture()
    def segment_anything(self, mocker) -> SegmentAnything:
        image_encoder = mocker.Mock(
            spec=SAMImageEncoder,
            return_value=torch.randn(1, 256, 64, 64),
        )

        prompt_encoder = mocker.Mock(
            spec=SAMPromptEncoder,
            return_value=(torch.randn(1, 256, 64, 64), torch.randn(1, 256, 64, 64)),
        )

        mask_decoder = mocker.Mock(
            spec=SAMMaskDecoder,
            return_value=(torch.randn(1, 1, 256, 256), torch.randn(1, 1)),
        )

        criterion = mocker.Mock(
            spec=SAMCriterion,
            return_value={
                "loss": torch.tensor(1.0),
                "loss_focal": torch.tensor(1.0),
                "loss_dice": torch.tensor(1.0),
                "loss_iou": torch.tensor(1.0),
            },
        )

        return SegmentAnything(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            criterion=criterion,
        )

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
    def test_forward(
        self,
        segment_anything: SegmentAnything,
        training: bool,
        ori_shapes: list[Tensor],
    ) -> None:
        """Test the forward method of SegmentAnything."""
        segment_anything.training = training

        images = tv_tensors.Image(torch.zeros((2, 3, 1024, 1024), dtype=torch.float32))
        bboxes = [
            tv_tensors.BoundingBoxes(
                torch.tensor([[0, 0, 10, 10]]),
                format="xyxy",
                canvas_size=(1024, 1024),
                dtype=torch.float32,
            ),
        ] * 2
        points = [Points(torch.tensor([[5, 5]]), canvas_size=(1024, 1024), dtype=torch.float32)] * 2
        gt_masks = [torch.zeros((2, *os)) for os in ori_shapes] * 2 if training else None

        results = segment_anything(
            images=images,
            ori_shapes=ori_shapes,
            bboxes=bboxes,
            points=points,
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

    @pytest.mark.parametrize(
        "ori_shape",
        [
            torch.tensor([512, 256]),
            torch.tensor([256, 512]),
            torch.tensor([1536, 1280]),
            torch.tensor([1280, 1536]),
        ],
    )
    def test_forward_for_tracing(self, mocker, segment_anything: SegmentAnything, ori_shape: Tensor) -> None:
        """Test forward_for_tracing."""
        mocker.patch.object(segment_anything, "_embed_points", return_value=torch.randn(1, 2, 256))
        mocker.patch.object(segment_anything, "_embed_masks", return_value=torch.randn(1, 256, 64, 64))
        mocker.patch.object(
            segment_anything.mask_decoder,
            "predict_masks",
            return_value=(torch.randn(1, 4, 256, 256), torch.rand(1, 4)),
        )
        segment_anything.training = False

        image_embeddings = torch.zeros(1, 256, 64, 64, dtype=torch.float32)
        point_coords = torch.tensor([[[0, 0], [10, 10]]], dtype=torch.float32)
        point_labels = torch.tensor([[2, 3]], dtype=torch.float32)
        mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
        has_mask_inputs = torch.tensor([[0.0]])

        results = segment_anything.forward_for_tracing(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_inputs[0],
            ori_shape=ori_shape,
        )

        assert results[0].shape[2:] == torch.Size(ori_shape)

    @pytest.mark.parametrize(
        ("point_coords", "point_labels", "expected"),
        [
            (Tensor([[[1, 1]]]), Tensor([[1]]), (1, 1, 256)),
            (Tensor([[[1, 1], [2, 2]]]), Tensor([[2, 3]]), (1, 2, 256)),
        ],
    )
    def test_embed_points(
        self,
        mocker,
        segment_anything: SegmentAnything,
        point_coords: Tensor,
        point_labels: Tensor,
        expected: tuple[int],
    ) -> None:
        """Test _embed_points."""
        segment_anything.prompt_encoder.pe_layer = mocker.Mock()
        segment_anything.prompt_encoder.pe_layer._pe_encoding.return_value = torch.randn(*expected)
        segment_anything.prompt_encoder.not_a_point_embed = mocker.Mock
        segment_anything.prompt_encoder.not_a_point_embed.weight = torch.randn(1, 256)
        segment_anything.prompt_encoder.num_point_embeddings = 4
        segment_anything.prompt_encoder.point_embeddings = nn.ModuleList([nn.Embedding(1, 256) for i in range(4)])

        results = segment_anything._embed_points(point_coords, point_labels)

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("masks_input", "has_mask_input", "expected"),
        [
            (torch.randn(1, 1, 4, 4, dtype=torch.float), torch.tensor([1], dtype=torch.float), (1, 256, 64, 64)),
        ],
    )
    def test_embed_masks(
        self,
        mocker,
        segment_anything: SegmentAnything,
        masks_input: Tensor,
        has_mask_input: Tensor,
        expected: tuple[int],
    ) -> None:
        """Test _embed_masks."""
        segment_anything.prompt_encoder.mask_downscaling = mocker.Mock(return_value=torch.randn(1, 256, 64, 64))
        segment_anything.prompt_encoder.no_mask_embed = mocker.Mock()
        segment_anything.prompt_encoder.no_mask_embed.weight = torch.randn(1, 256)

        results = segment_anything._embed_masks(masks_input, has_mask_input)

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("masks", "expected"),
        [
            (Tensor([[[-2, -2], [2, 2]]]), 1),
            (Tensor([[[-2, -2], [1, 1]]]), 0),
        ],
    )
    def test_calculate_stability_score(self, segment_anything: SegmentAnything, masks: Tensor, expected: int) -> None:
        """Test calculate_stability_score."""
        results = segment_anything.calculate_stability_score(masks, mask_threshold=0.0, threshold_offset=1.0)

        assert results == expected

    def test_select_masks(self, segment_anything: SegmentAnything) -> None:
        """Test select_masks."""
        segment_anything.mask_decoder.num_mask_tokens = 4

        masks = Tensor([[[[1]], [[2]], [[3]], [[4]]]])
        iou_preds = Tensor([[0.1, 0.2, 0.3, 0.4]])
        num_points = 1

        selected_mask, selected_iou_pred = segment_anything.select_masks(masks, iou_preds, num_points)

        assert masks[:, -1, :, :] == selected_mask
        assert iou_preds[:, -1] == selected_iou_pred


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
        "result_point_selection",
        [torch.tensor([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), torch.tensor([[-1, -1, -1]])],
    )
    def test_forward(self, mocker, prompt_getter, result_point_selection: Tensor) -> None:
        """Test forward."""
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
    def zero_shot_segment_anything(self, mocker) -> ZeroShotSegmentAnything:
        return ZeroShotSegmentAnything(
            image_encoder=mocker.Mock(return_value=torch.randn(1, 256, 64, 64)),
            prompt_encoder=mocker.Mock(return_value=(torch.randn(1, 256, 64, 64), torch.randn(1, 256, 64, 64))),
            mask_decoder=mocker.Mock(return_value=(torch.randn(1, 1, 256, 256), torch.randn(1, 1))),
            criterion=mocker.Mock(
                return_value={
                    "loss": torch.tensor(1.0),
                    "loss_focal": torch.tensor(1.0),
                    "loss_dice": torch.tensor(1.0),
                    "loss_iou": torch.tensor(1.0),
                },
            ),
        )

    @pytest.mark.parametrize("new_largest_label", [0, 3])
    def test_expand_reference_info(
        self,
        zero_shot_segment_anything: ZeroShotSegmentAnything,
        new_largest_label: int,
    ) -> None:
        """Test expand_reference_info."""
        reference_feats = torch.zeros(0, 1, 256)

        results = zero_shot_segment_anything.expand_reference_info(
            reference_feats=reference_feats,
            new_largest_label=new_largest_label,
        )

        assert len(results) == new_largest_label + 1

    def test_learn(self, mocker, zero_shot_segment_anything: ZeroShotSegmentAnything) -> None:
        """Test learn."""
        mocker.patch.object(
            zero_shot_segment_anything,
            "forward_for_tracing",
            return_value=(torch.rand(1, 4, 1024, 1024), torch.rand(1, 4), torch.rand(1, 4, 256, 256)),
        )

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

    def test_infer(self, mocker, zero_shot_segment_anything: ZeroShotSegmentAnything) -> None:
        """Test infer."""
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

    def test_inspect_overlapping_areas(self, mocker, zero_shot_segment_anything: ZeroShotSegmentAnything) -> None:
        """Test _inspect_overlapping_areas."""
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

    def test_predict_masks(self, mocker, zero_shot_segment_anything: ZeroShotSegmentAnything) -> None:
        """Test _predict_masks."""
        mocker.patch.object(
            zero_shot_segment_anything,
            "forward_for_tracing",
            return_value=(torch.ones(1, 4, 8, 8), torch.tensor([[0.1, 0.2, 0.5, 0.7]]), torch.ones(1, 4, 4, 4)),
        )

        zero_shot_segment_anything.image_size = 6

        mask = zero_shot_segment_anything._predict_masks(
            image_embeddings=torch.rand(1),
            point_coords=torch.rand(1, 2, 2),
            point_labels=torch.randint(low=0, high=2, size=(1, 2)),
            ori_shape=torch.tensor([8, 8], dtype=torch.int64),
        )
        assert mask.shape == (8, 8)

    def test_preprocess_coords(self, zero_shot_segment_anything: ZeroShotSegmentAnything) -> None:
        coords = torch.tensor([[[1, 1], [2, 2]]])
        ori_shape = torch.tensor([512, 512])
        target_length = 1024

        preprocessed_coords = zero_shot_segment_anything._preprocess_coords(coords, ori_shape, target_length)

        assert preprocessed_coords.shape == (1, 2, 2)
        assert preprocessed_coords.max() <= target_length

    def test_generate_masked_features(self, zero_shot_segment_anything: ZeroShotSegmentAnything) -> None:
        feats = torch.randn(256, 256)
        masks = torch.zeros(256, 256)
        threshold_mask = 0.5

        masked_feats = zero_shot_segment_anything._generate_masked_features(feats, masks, threshold_mask)

        assert masked_feats is None

    def test_pad_to_square(self, zero_shot_segment_anything: ZeroShotSegmentAnything) -> None:
        x = torch.randn(1, 1, 1023, 1024)  # Example input tensor

        padded_x = zero_shot_segment_anything.pad_to_square(x)

        assert padded_x.shape == (1, 1, 1024, 1024)  # Check if the output tensor has the expected shape
        assert padded_x[:, :, :1023].allclose(x)  # Check if the original tensor is preserved
        assert padded_x[:, :, 1023:].sum() == 0  # Check if the padding region is filled with zeros

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
        zero_shot_segment_anything: ZeroShotSegmentAnything,
        masks: Tensor,
        logits: Tensor,
        expected: Tensor,
    ) -> None:
        scores = torch.tensor([[0.0, 0.1, 0.2, 0.3]])

        _, result = zero_shot_segment_anything._decide_cascade_results(masks, logits, scores)

        assert torch.equal(result, expected)
