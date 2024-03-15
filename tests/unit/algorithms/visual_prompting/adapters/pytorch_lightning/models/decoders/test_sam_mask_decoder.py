"""Tests sam mask decoder used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Tuple

import pytest
import torch
import torch.nn as nn

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.decoders.sam_mask_decoder import (
    MLP,
    Attention,
    SAMMaskDecoder,
    TwoWayAttentionBlock,
    TwoWayTransformer,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.utils import (
    MLPBlock,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSAMMaskDecoder:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.num_multimask_outputs = 3
        self.sam_mask_decoder = SAMMaskDecoder(
            transformer_dim=8,
            transformer_cfg=dict(depth=2, embedding_dim=8, mlp_dim=8, num_heads=2),
            num_multimask_outputs=self.num_multimask_outputs,
            iou_head_depth=1,
            iou_head_hidden_dim=8,
        )

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert isinstance(self.sam_mask_decoder.transformer, TwoWayTransformer)
        assert isinstance(self.sam_mask_decoder.iou_token, nn.Embedding)
        assert isinstance(self.sam_mask_decoder.mask_tokens, nn.Embedding)
        assert isinstance(self.sam_mask_decoder.output_upscaling, nn.Sequential)
        assert isinstance(self.sam_mask_decoder.output_hypernetworks_mlps, nn.ModuleList)
        assert len(self.sam_mask_decoder.output_hypernetworks_mlps) == self.num_multimask_outputs + 1
        assert isinstance(self.sam_mask_decoder.iou_prediction_head, MLP)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "image_embeddings,image_pe,sparse_prompt_embeddings,dense_prompt_embeddings,multimask_output,expected",
        [
            (
                torch.empty((1, 8, 2, 2)),
                torch.empty((1, 8, 2, 2)),
                torch.empty((1, 4, 8)),
                torch.empty((1, 8, 2, 2)),
                False,
                ((1, 1, 8, 8), (1, 1)),
            ),
            (
                torch.empty((1, 8, 2, 2)),
                torch.empty((1, 8, 2, 2)),
                torch.empty((1, 4, 8)),
                torch.empty((1, 8, 2, 2)),
                True,
                ((1, 3, 8, 8), (1, 3)),
            ),
        ],
    )
    def test_forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        expected: Tuple[Tuple[int]],
    ):
        """Test forward."""
        results = self.sam_mask_decoder.forward(
            image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output
        )
        masks, iou_pred = results

        assert masks.shape == expected[0]
        assert iou_pred.shape == expected[1]

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "image_embeddings,image_pe,sparse_prompt_embeddings,dense_prompt_embeddings,expected",
        [
            (
                torch.empty((1, 8, 2, 2)),
                torch.empty((1, 8, 2, 2)),
                torch.empty((1, 4, 8)),
                torch.empty((1, 8, 2, 2)),
                ((1, 4, 8, 8), (1, 4)),
            )
        ],
    )
    def test_predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        expected: Tuple[Tuple[int]],
    ):
        """Test predict_masks."""
        results = self.sam_mask_decoder.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings
        )
        masks, iou_pred = results

        assert masks.shape == expected[0]
        assert iou_pred.shape == expected[1]


class TestMLP:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mlp = MLP(input_dim=4, hidden_dim=4, output_dim=4, num_layers=2)

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert len(self.mlp.layers) == 2

    @e2e_pytest_unit
    @pytest.mark.parametrize("inputs,expected", [(torch.empty((1, 1, 4)), (1, 1, 4))])
    def test_forward(self, inputs: torch.Tensor, expected: Tuple[int]):
        """Test forward."""
        results = self.mlp.forward(inputs)

        assert results.shape == expected


class TestTwoWayTransformer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.depth = 2
        self.embedding_dim = 8
        self.num_heads = 2
        self.two_way_transformer = TwoWayTransformer(
            depth=self.depth, embedding_dim=self.embedding_dim, num_heads=self.num_heads, mlp_dim=self.embedding_dim
        )

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert len(self.two_way_transformer.layers) == self.depth
        assert isinstance(self.two_way_transformer.final_attn_token_to_image, Attention)
        assert isinstance(self.two_way_transformer.norm_final_attn, nn.LayerNorm)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "image_embedding,image_pe,point_embedding,expected",
        [(torch.empty((1, 8, 2, 2)), torch.empty((1, 8, 2, 2)), torch.empty((1, 1, 8)), ((1, 1, 8), (1, 4, 8)))],
    )
    def test_forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
        expected: Tuple[Tuple[int]],
    ):
        """Test forward."""
        results = self.two_way_transformer.forward(image_embedding, image_pe, point_embedding)
        queries, keys = results

        assert queries.shape == expected[0]
        assert keys.shape == expected[1]


class TestTwoWayAttentionBlock:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.embedding_dim = 8
        self.num_heads = 2
        self.two_way_attention_block = TwoWayAttentionBlock(
            embedding_dim=self.embedding_dim, num_heads=self.num_heads, mlp_dim=self.embedding_dim
        )

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert isinstance(self.two_way_attention_block.self_attn, Attention)
        assert isinstance(self.two_way_attention_block.norm1, nn.LayerNorm)
        assert isinstance(self.two_way_attention_block.cross_attn_token_to_image, Attention)
        assert isinstance(self.two_way_attention_block.norm2, nn.LayerNorm)
        assert isinstance(self.two_way_attention_block.mlp, MLPBlock)
        assert isinstance(self.two_way_attention_block.norm3, nn.LayerNorm)
        assert isinstance(self.two_way_attention_block.norm4, nn.LayerNorm)
        assert isinstance(self.two_way_attention_block.cross_attn_image_to_token, Attention)

    @e2e_pytest_unit
    @pytest.mark.parametrize("queries,keys,query_pe,key_pe", [[torch.empty((1, 8, 8)) for _ in range(4)]])
    def test_forward(self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor):
        """Test forward."""
        results = self.two_way_attention_block.forward(queries, keys, query_pe, key_pe)
        queries, keys = results

        assert queries.shape == (1, 8, 8)
        assert keys.shape == (1, 8, 8)


class TestAttention:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.embedding_dim = 8
        self.num_heads = 2
        self.attention = Attention(embedding_dim=self.embedding_dim, num_heads=self.num_heads)

    @e2e_pytest_unit
    def test_init(self):
        """Test init."""
        assert isinstance(self.attention.q_proj, nn.Linear)
        assert self.attention.q_proj.in_features == self.embedding_dim
        assert isinstance(self.attention.k_proj, nn.Linear)
        assert self.attention.k_proj.in_features == self.embedding_dim
        assert isinstance(self.attention.v_proj, nn.Linear)
        assert self.attention.v_proj.in_features == self.embedding_dim
        assert isinstance(self.attention.out_proj, nn.Linear)
        assert self.attention.out_proj.out_features == self.embedding_dim

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "x,expected",
        [
            (torch.empty((1, 4, 4)), (1, 2, 4, 2)),
            (torch.empty((1, 2, 2)), (1, 2, 2, 1)),
        ],
    )
    def test_separate_heads(self, x: torch.Tensor, expected: Tuple[int]):
        """Test _separate_heads."""
        results = self.attention._separate_heads(x, self.num_heads)

        assert results.shape == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "x,expected",
        [
            (torch.empty((1, 2, 4, 2)), (1, 4, 4)),
            (torch.empty((1, 2, 2, 1)), (1, 2, 2)),
        ],
    )
    def test_recombine_heads(self, x: torch.Tensor, expected: Tuple[int]):
        """Test _recombine_heads."""
        results = self.attention._recombine_heads(x)

        assert results.shape == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "q,k,v,expected", [(torch.empty((1, 1, 8)), torch.empty((1, 1, 8)), torch.empty((1, 1, 8)), (1, 1, 8))]
    )
    def test_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, expected: Tuple[int]):
        """Test forward."""
        results = self.attention.forward(q, k, v)

        assert results.shape == expected
