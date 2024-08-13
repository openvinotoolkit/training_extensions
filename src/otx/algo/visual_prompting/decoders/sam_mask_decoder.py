# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM mask decoder model for the OTX visual prompting."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from otx.algo.visual_prompting.utils import LayerNorm2d, MLPBlock


class SAMMaskDecoder(nn.Module):
    """Predicts masks given an image and prompt embeddings, using a transformer architecture.

    Reference: https://github.com/facebookresearch/segment-anything

    Args:
        transformer_dim (int): Channel dimension of the transformer.
            Defaults to 256.
        transformer_cfg (dict): Configuration of the transformer.
            Defaults to `{"depth": 2, "embedding_dim": 256, "mlp_dim": 2048, "num_heads": 8}`.
        num_multimask_outputs (int): The number of masks to predict when disambiguating masks.
            Defaults to 3.
        activation (nn.Module): Type of activation to use when upscaling masks.
            Defaults to ``nn.GELU``.
        iou_head_depth (int): Depth of the MLP used to predict mask quality.
            Defaults to 3.
        iou_head_hidden_dim (int): Hidden dimension of the MLP used to predict mask quality.
            Defaults to 256.
    """

    def __init__(
        self,
        *,
        transformer_dim: int = 256,
        transformer_cfg: dict | None = None,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if transformer_cfg is None:
            transformer_cfg = {"depth": 2, "embedding_dim": 256, "mlp_dim": 2048, "num_heads": 8}

        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(**transformer_cfg)

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)],
        )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
    ) -> tuple[Tensor, Tensor]:
        """Predict masks given image and prompt embeddings.

        Args:
          image_embeddings (Tensor): Embeddings from the image encoder.
          image_pe (Tensor): Positional encoding with the shape of image_embeddings.
          sparse_prompt_embeddings (Tensor): Embeddings of the points and boxes.
          dense_prompt_embeddings (Tensor): Embeddings of the mask inputs.
          multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
          masks (Tensor): Batched predicted masks.
          iou_pred (Tensor): Batched predictions of mask quality.
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predicts masks. See 'forward' for more details.

        Args:
            image_embeddings (Tensor): Embeddings from the image encoder.
            image_pe (Tensor): Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (Tensor): Embeddings of the points and boxes.
            dense_prompt_embeddings (Tensor): Embeddings of the mask inputs.

        Returns:
            masks (Tensor): Batched predicted masks.
            iou_pred (Tensor): Batched predictions of mask quality.
        """
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: list[Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    """Simple MLP with ReLU activations.

    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        output_dim (int): Output dimension.
        num_layers (int): Number of layers.
        sigmoid_output (bool): Whether to apply sigmoid to the output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayTransformer(nn.Module):
    """A transformer decoder that attends to an input image using queries whose positional embedding is supplied.

    Args:
        depth (int): Number of layers in the transformer decoder.
        embedding_dim (int): Channel dimension for the input embeddings and the positional embeddings.
        num_heads (int): The number of heads for multihead attention. Must divide embedding_dim evenly.
        mlp_dim (int): Channel dimension internal to the MLP block in the transformer layers, defaults to 2048.
        activation (nn.Module): Activation to use in the MLP block, defaults to `nn.ReLU`.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                ),
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the transformer to the image and point embeddings.

        Args:
            image_embedding (Tensor): Image to attend to. Should be shape B x embedding_dim x h x w for any h and w.
            image_pe (Tensor): Positional encoding to add to the image. Must have the same shape as image_embedding.
            point_embedding (Tensor): Embedding to add to the query points. Must have shape B x N_points x embedding_dim
                for any N_points.

        Returns:
            Tensor: Processed point_embedding with shape B x N_points x embedding_dim for any N_points.
            Tensor: Processed image_embedding with shape B x embedding_dim x h x w for any h and w.
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """A transformer block with four layers.

    (1) self-attention of sparse inputs,
    (2) cross attention of sparse inputs to dense inputs,
    (3) mlp block on sparse inputs, and
    (4) cross attention of dense inputs to sparse inputs.

    Args:
        embedding_dim (int): Channel dimension of the embeddings in the transformer block.
        num_heads (int): The number of heads in the attention layers of the transformer block.
        mlp_dim (int): Hidden dimension of the mlp block, defaults to 2048.
        activation (nn.Module): Activation of the mlp block, defaults to `nn.ReLU`.
        skip_first_layer_pe (bool): Skip the PE on the first layer of the transformer block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the transformer block to the queries and keys.

        Args:
            queries (Tensor): Queries to attend to. Should be shape B x N_queries x C for any N_queries.
            keys (Tensor): Keys to attend to. Should be shape B x N_keys x C for any N_keys.
            query_pe (Tensor): Positional encoding to add to the queries. Must have the same shape as queries.
            key_pe (Tensor): Positional encoding to add to the keys. Must have the same shape as keys.

        Returns:
            Tensor: Processed queries.
            Tensor: Processed keys.
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """An attention layer.

    It allows for downscaling the size of the embedding after projection to queries, keys, and values.

    Args:
        embedding_dim (int): Channel dimension of the embeddings.
        num_heads (int): The number of heads in the attention layers.
        downsample_rate (int): The rate to downsample the embedding by after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."  # noqa: S101

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """Separate the heads of the input tensor.

        Args:
            x (Tensor): Input tensor of shape B x N_tokens x C.
            num_heads (int): The number of heads to separate the input tensor into.

        Returns:
            Tensor: The input tensor separated into heads. Shape B x N_heads x N_tokens x C_per_head.
        """
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """Recombine the heads of the input tensor.

        Args:
            x (Tensor): Input tensor of shape B x N_heads x N_tokens x C_per_head.

        Returns:
            Tensor: The input tensor recombined into tokens. Shape B x N_tokens x C.
        """
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Apply the attention layer to the queries, keys, and values.

        Args:
            q (Tensor): Queries to attend to. Should be shape B x N_queries x C for any N_queries.
            k (Tensor): Keys to attend to. Should be shape B x N_keys x C for any N_keys.
            v (Tensor): Values to attend to. Should be shape B x N_values x C for any N_values.

        Returns:
            Tensor: The output of the attention layer. Shape B x N_queries x C.
        """
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)
