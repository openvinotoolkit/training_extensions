import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .transformer_encoder import TransformerFeedForward
from .embedding import TransformerEmbedding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1):
        super().__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)

        self.self_attn_layer_norm = nn.LayerNorm(size)
        self.enc_attn_layer_norm = nn.LayerNorm(size)
        self.ff_layer_norm = nn.LayerNorm(size)

        self.self_attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio)
        self.encoder_attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)

    def forward(self, src, src_mask, tgt, tgt_mask):
        # Self-attention layer
        _tgt, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.self_attn_layer_norm(tgt + self.dropout(_tgt))
        # Encoder attention layer
        _tgt, attention = self.encoder_attention(tgt, src, src, src_mask)
        tgt = self.enc_attn_layer_norm(tgt + self.dropout(_tgt))
        # Feed-forward layer
        _tgt = self.ff_layer(tgt)
        tgt = self.ff_layer_norm(tgt + self.dropout(_tgt))
        return tgt, attention


class TransformerDecoder(nn.Module):
    def __init__(self, embed_layer, size, n_layers, ff_size=None, n_att_head=8, dropout_ratio=0.1):
        super().__init__()
        self.embed_layer = embed_layer
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(size, ff_size, n_att_head, dropout_ratio)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src, src_mask, tgt, tgt_mask):
        if self.embed_layer is not None:
            tgt = self.embed_layer(tgt)
        for layer in self.layers:
            tgt, attention = layer(src, src_mask, tgt, tgt_mask)
        return tgt, attention
