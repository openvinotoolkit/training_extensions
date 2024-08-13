# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM prompt encoder model for the OTX visual prompting."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from otx.algo.visual_prompting.utils.layer_norm_2d import LayerNorm2d


class SAMPromptEncoder(nn.Module):
    """Encodes prompts for input to SAM's mask decoder.

    Reference: https://github.com/facebookresearch/segment-anything

    Args:
        image_embedding_size (tuple(int, int)): The spatial size of the image embedding, as (H, W).
        input_image_size (int): The padded size of the image as input to the image encoder, as (H, W).
        embed_dim (int): The prompts' embedding dimension.
            Defaults to 256.
        mask_in_chans (int): The number of hidden channels used for encoding input masks.
            Defaults to 16.
        activation (nn.Module): The activation to use when encoding input masks.
    """

    def __init__(
        self,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        embed_dim: int = 256,
        mask_in_chans: int = 16,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """Returns the positional encoding.

        It used to encode point prompts, applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape 1x(embed_dim)x(embedding_h)x(embedding_w).
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts.

        Args:
            points (torch.Tensor): A BxNx2 array of point prompts to the model.
                Each point is in (X,Y) in pixels.
            labels (torch.Tensor): A BxN array of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            pad (bool): Whether to pad the points with a zero point.

        Returns:
            torch.Tensor: The embedded points, as (N, embed_dim).
        """
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts.

        Args:
            boxes (torch.Tensor): A Bx4 array given a box prompt to the model, in XYXY format.

        Returns:
            torch.Tensor: The embedded boxes, as (N, embed_dim).
        """
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs.

        Args:
            masks (torch.Tensor): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form Bx1xHxW, where
                for SAM, H=W=256. Masks returned by a previous iteration of the
                predict method do not need further transformation.

        Returns:
            torch.Tensor: The embedded masks, as (N, embed_dim).
        """
        return self.mask_downscaling(masks)

    def _get_batch_size(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> int:
        """Gets the batch size of the output given the batch size of the input prompts.

        Args:
            points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates and labels to embed.
            boxes (torch.Tensor or none): boxes to embed.
            masks (torch.Tensor or none): masks to embed.

        Returns:
            int: The batch size of the output.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:  # noqa: RET505
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        """Gets the device of the embeddings.

        Returns:
            torch.device: The device of the embeddings.
        """
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (tuple(torch.Tensor, torch.Tensor) or none): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            boxes (torch.Tensor or none): A Bx4 array given a box prompt to the model, in XYXY format.
            masks (torch.Tensor or none): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form Bx1xHxW, where
                for SAM, H=W=256. Masks returned by a previous iteration of the
                predict method do not need further transformation.

        Returns:
            sparse_embeddings (torch.Tensor): sparse embeddings for the points and boxes, with shape Nx1x(embed_dim),
                where N is determined by the number of input points and boxes.
            dense_embeddings (torch.Tensor): dense embeddings for the masks,
                in the shape Nx(embed_dim)x(embed_H)x(embed_W).
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies.

    Args:
        num_pos_feats (int): The number of positional frequencies.
        scale (float): The scale of the positional encoding.
    """

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1].

        Args:
            coords (torch.Tensor): Stacked x-y grids, as (H, W, 2).

        Returns:
            torch.Tensor: The positional encoding, as (H, W, num_pos_feats * 2).
        """
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size.

        Args:
            size (tuple(int, int)): The size of the grid to generate the encoding for.

        Returns:
            torch.Tensor: The positional encoding, as (num_pos_feats * 2, H, W).
        """
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1].

        Args:
            coords_input (torch.Tensor): The coordinates to encode, as (N, 1, 2).
            image_size (tuple(int, int)): The size of the image the coordinates are from.

        Returns:
            torch.Tensor: The positional encoding, as (N, 1, num_pos_feats * 2).
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
