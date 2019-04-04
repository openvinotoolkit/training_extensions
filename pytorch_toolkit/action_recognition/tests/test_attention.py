import torch
import numpy as np

from action_recognition.models.modules.self_attention import ScaledDotProductAttention, MultiHeadAttention


class TestScaledDotProductAttention:
    def test_shapes(self):
        layer = ScaledDotProductAttention(16, attn_dropout=0)
        q = torch.zeros(4, 8, 16)
        k = torch.zeros(4, 4, 16)
        v = torch.zeros(4, 4, 16)

        with torch.no_grad():
            outputs, attns = layer(q, k, v)

        assert outputs.size() == q.size()
        assert attns.size(0) == v.size(0)
        assert attns.size(1) == q.size(1)
        assert attns.size(2) == k.size(1)

    def test_atten_range(self, rand):
        layer = ScaledDotProductAttention(16, attn_dropout=0)
        q = torch.from_numpy(rand.rand(2, 4, 16))
        k = torch.from_numpy(rand.rand(2, 4, 16))
        v = torch.from_numpy(rand.rand(2, 4, 16))

        with torch.no_grad():
            outputs, attns = layer(q, k, v)
        attns = attns.numpy()

        assert np.alltrue(attns >= 0)
        assert np.alltrue(attns <= 1)
        assert np.allclose(attns.sum(2), np.ones((2, 4)))


class TestMultiHeadAttention:
    def test_shapes(self):
        layer = MultiHeadAttention(
            n_head=2,
            input_size=16,
            output_size=16,
            d_k=8,
            d_v=8,
            dropout=0,
            use_proj=False,
            layer_norm=False
        )
        q = torch.zeros(2, 8, 16)
        k = torch.zeros(2, 4, 16)
        v = torch.zeros(2, 4, 16)

        with torch.no_grad():
            outputs, attns = layer(q, k, v)

        assert outputs.size() == q.size()

    def test_shapes_with_use_proj_set(self):
        layer = MultiHeadAttention(
            n_head=2,
            input_size=16,
            output_size=16,
            d_k=4,
            d_v=4,
            dropout=0,
            use_proj=True,
            layer_norm=False
        )
        q = torch.zeros(2, 8, 16)
        k = torch.zeros(2, 4, 16)
        v = torch.zeros(2, 4, 16)

        with torch.no_grad():
            outputs, attns = layer(q, k, v)

        assert outputs.size() == q.size()
