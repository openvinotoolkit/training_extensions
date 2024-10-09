# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for MonoDetr backbone."""
import pytest
import torch
from otx.algo.object_detection_3d.backbones.monodetr_resnet import BackboneBase, Joiner, PositionEmbeddingSine
from otx.algo.object_detection_3d.utils.utils import NestedTensor


class TestBackbone:
    @pytest.fixture()
    def backbone(self, mocker):
        mocker.patch("otx.algo.object_detection_3d.backbones.monodetr_resnet.IntermediateLayerGetter")
        model = BackboneBase(backbone=mocker.MagicMock(torch.nn.Module), train_backbone=True, return_interm_layers=True)
        model.body = mocker.MagicMock(return_value={"layer_0": torch.rand(1, 3, 256, 224)})
        return model

    def test_backbone_forward(self, backbone):
        images = torch.randn(1, 3, 224, 224)
        output = backbone(images)
        assert isinstance(output, dict)
        assert len(output) == 1
        assert all(isinstance(value, NestedTensor) for value in output.values())

    def test_position_embedding_sine(self):
        # Create a PositionEmbeddingSine instance
        position_embedding = PositionEmbeddingSine(num_pos_feats=128, temperature=10000, normalize=False, scale=None)

        # Create a dummy input tensor
        tensor_list = torch.randn(1, 512, 48, 160)
        nested_tensor = NestedTensor(tensor_list, mask=torch.ones(1, 48, 160).bool())

        # Forward pass
        output = position_embedding(nested_tensor)

        # Check output shape
        assert output.shape == (1, 256, 48, 160)
        # Check output type
        assert output.dtype == torch.float32
        # Check sine and cosine properties
        assert torch.allclose(
            output[:, :, :, :80].sin().pow(2) + output[:, :, :, 80:].cos().pow(2),
            torch.ones(1, 256, 48, 80),
        )


class TestJoiner:
    @pytest.fixture()
    def joiner(self, mocker):
        mocker.patch("otx.algo.object_detection_3d.backbones.monodetr_resnet.Backbone")
        mocker.patch("otx.algo.object_detection_3d.backbones.monodetr_resnet.PositionEmbeddingSine")
        backbone = mocker.MagicMock(torch.nn.Module)
        backbone.strides = [4, 8, 16]
        backbone.num_channels = [32, 64, 128]
        position_embedding = mocker.MagicMock(torch.nn.Module)
        return Joiner(backbone=backbone, position_embedding=position_embedding)

    def test_joiner_forward(self, joiner):
        images = torch.randn(1, 3, 224, 224)
        nested_tensors = [NestedTensor(torch.randn(1, 256, 56, 56), torch.ones(1, 56, 56).bool())]
        position_embeddings = [torch.randn(1, 256, 56, 56)]
        joiner[0].return_value = {0: nested_tensors[0]}
        joiner[1].return_value = position_embeddings[0]

        output_tensors, output_position_embeddings = joiner(images)

        assert isinstance(output_tensors, list)
        assert isinstance(output_position_embeddings, list)
        assert len(output_tensors) == 1
        assert len(output_position_embeddings) == 1
        assert isinstance(output_tensors[0], NestedTensor)
        assert isinstance(output_position_embeddings[0], torch.Tensor)
        assert output_tensors[0].tensors.shape == (1, 256, 56, 56)
        assert output_tensors[0].mask.shape == (1, 56, 56)
        assert output_position_embeddings[0].shape == (1, 256, 56, 56)
        assert output_tensors[0].tensors.dtype == images.dtype
        assert output_position_embeddings[0].dtype == images.dtype
