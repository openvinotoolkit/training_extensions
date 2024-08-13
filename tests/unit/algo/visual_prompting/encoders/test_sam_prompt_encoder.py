# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from otx.algo.visual_prompting.encoders.sam_prompt_encoder import PositionEmbeddingRandom, SAMPromptEncoder
from torch import nn


class TestSAMPromptEncoder:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.image_embedding_size = 4
        self.input_image_size = (4, 4)
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=4,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=self.input_image_size,
            mask_in_chans=4,
        )

    def test_init(self) -> None:
        """Test init."""
        assert isinstance(self.prompt_encoder.pe_layer, PositionEmbeddingRandom)
        assert isinstance(self.prompt_encoder.point_embeddings, nn.ModuleList)
        assert isinstance(self.prompt_encoder.not_a_point_embed, nn.Embedding)
        assert isinstance(self.prompt_encoder.mask_input_size, tuple)
        assert isinstance(self.prompt_encoder.mask_downscaling, nn.Sequential)
        assert isinstance(self.prompt_encoder.no_mask_embed, nn.Embedding)

    def test_get_dense_pe(self) -> None:
        """Test get_dense_pe."""
        results = self.prompt_encoder.get_dense_pe()

        assert results.shape == torch.Size(
            (1, self.image_embedding_size, self.image_embedding_size, self.image_embedding_size),
        )

    @pytest.mark.parametrize(
        ("points", "labels", "pad", "expected"),
        [
            (torch.ones((1, 2, 2), dtype=torch.float32), torch.Tensor([[0, 1]]), True, torch.Size((1, 3, 4))),
            (torch.ones((1, 2, 2), dtype=torch.float32), torch.Tensor([[0, 1]]), False, torch.Size((1, 2, 4))),
        ],
    )
    def test_embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool, expected: torch.Size) -> None:
        """Test _embed_points."""
        results = self.prompt_encoder._embed_points(points, labels, pad)

        assert results.shape == expected

    @pytest.mark.parametrize(("boxes", "expected"), [(torch.Tensor([[0, 0, 1, 1]]), torch.Size((1, 2, 4)))])
    def test_embed_boxes(self, boxes: torch.Tensor, expected: torch.Size) -> None:
        """Test _embed_boxes."""
        results = self.prompt_encoder._embed_boxes(boxes)

        assert results.shape == expected

    @pytest.mark.parametrize(("mask", "expected"), [(torch.zeros((1, 4, 4)), torch.Size((4, 1, 1)))])
    def test_embed_masks(self, mask: torch.Tensor, expected: torch.Size) -> None:
        """Test _embed_masks."""
        results = self.prompt_encoder._embed_masks(mask)

        assert results.shape == expected

    @pytest.mark.parametrize(
        ("points", "boxes", "masks", "expected"),
        [
            ((torch.tensor([1]), torch.tensor([1])), None, None, 1),
            (None, torch.Tensor([[0, 0, 1, 1]]), None, 1),
            (None, None, torch.zeros((1, 2, 2)), 1),
            (None, None, None, 1),
        ],
    )
    def test_get_batch_size(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
        expected: int,
    ) -> None:
        """Test _get_batch_size."""
        results = self.prompt_encoder._get_batch_size(points, boxes, masks)

        assert results == expected

    @pytest.mark.parametrize(
        ("points", "boxes", "masks", "expected"),
        [
            (
                (torch.ones((1, 2, 2), dtype=torch.float32), torch.Tensor([[0, 1]])),
                None,
                None,
                (torch.Size((1, 3, 4)), torch.Size((1, 4, 4, 4))),
            ),
            (None, torch.Tensor([[0, 0, 1, 1]]), None, (torch.Size((1, 2, 4)), torch.Size((1, 4, 4, 4)))),
            (None, None, torch.zeros((1, 4, 4)), (torch.Size((1, 0, 4)), torch.Size((4, 1, 1)))),
            (None, None, None, (torch.Size((1, 0, 4)), torch.Size((1, 4, 4, 4)))),
        ],
    )
    def test_forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
        expected: tuple[torch.Size, ...],
    ) -> None:
        """Test forward."""
        results = self.prompt_encoder.forward(points, boxes, masks)
        sparse_embeddings, dense_embeddings = results

        assert sparse_embeddings.shape == expected[0]
        assert dense_embeddings.shape == expected[1]


class TestPositionEmbeddingRandom:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.position_embedding_random = PositionEmbeddingRandom(num_pos_feats=4)

    def test_init(self) -> None:
        """Test init."""
        assert hasattr(self.position_embedding_random, "positional_encoding_gaussian_matrix")
        assert self.position_embedding_random.positional_encoding_gaussian_matrix.shape == torch.Size((2, 4))

    def test_pe_encoding(self) -> None:
        """Test _pe_encoding."""
        results = self.position_embedding_random._pe_encoding(torch.ones((2, 2, 2), dtype=torch.float32))

        assert results.shape == torch.Size((2, 2, 8))

    def test_forward(self) -> None:
        """Test forward."""
        results = self.position_embedding_random.forward(size=(2, 2))

        assert results.shape == torch.Size((8, 2, 2))

    def test_forward_with_coords(self) -> None:
        """Test forward_with_coords."""
        results = self.position_embedding_random.forward_with_coords(
            coords_input=torch.ones((2, 2, 2), dtype=torch.float32),
            image_size=(2, 2),
        )

        assert results.shape == torch.Size((2, 2, 8))
