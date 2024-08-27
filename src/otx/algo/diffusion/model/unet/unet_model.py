"""UNetModel implementation for the diffusion model."""

from __future__ import annotations

import math

import torch
from torch import nn

from .res_block import ResBlock
from .spatial_transformer import SpatialTransformer


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Timestamp embedding function.

    https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/util.py#L207.
    """
    half = dim // 2
    freqs = (-math.log(max_period) * torch.arange(half) / half).exp().to(timesteps.device)
    args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat((args.cos(), args.sin()), dim=-1).to(timesteps.dtype)


class Upsample(nn.Module):
    """Upsample module for the UNetModel."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNetModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        bs, c, py, px = x.shape
        z = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py * 2, px * 2)
        return self.conv(z)


class UNetModel(nn.Module):
    """UNetModel class for the diffusion model."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        model_ch: int,
        attention_resolutions: list[int],
        num_res_blocks: int,
        channel_mult: list[int],
        transformer_depth: list[int],
        ctx_dim: int | list[int],
        use_linear: bool = False,
        d_head: int | None = None,
        n_heads: int | None = None,
    ):
        super().__init__()
        self.model_ch = model_ch
        self.num_res_blocks = [num_res_blocks] * len(channel_mult)

        self.attention_resolutions = attention_resolutions
        self.d_head = d_head
        self.n_heads = n_heads

        def get_d_and_n_heads(dims: int) -> tuple[int, int]:
            if self.d_head is None:
                if self.n_heads is None:
                    msg = "d_head and n_heads cannot both be None"
                    raise ValueError(msg)
                return dims // self.n_heads, self.n_heads
            if self.n_heads is not None:
                msg = "d_head and n_heads cannot both be non-None"
                raise ValueError(msg)
            return self.d_head, dims // self.d_head

        time_embed_dim = model_ch * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks: nn.ModuleList = nn.ModuleList([nn.ModuleList([nn.Conv2d(in_ch, model_ch, 3, padding=1)])])
        input_block_channels = [model_ch]
        ch = model_ch
        ds = 1
        for idx, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[idx]):
                layers = nn.ModuleList(
                    [
                        ResBlock(ch, time_embed_dim, model_ch * mult),
                    ],
                )
                ch = mult * model_ch
                if ds in attention_resolutions:
                    d_head, n_heads = get_d_and_n_heads(ch)
                    layers.append(
                        SpatialTransformer(
                            ch,
                            n_heads,
                            d_head,
                            ctx_dim,
                            use_linear,
                            depth=transformer_depth[idx],
                        ),
                    )

                self.input_blocks.append(layers)
                input_block_channels.append(ch)

            if idx != len(channel_mult) - 1:
                downsample = nn.ModuleDict({"op": nn.Conv2d(ch, ch, 3, stride=2, padding=1)})
                self.input_blocks.append(nn.ModuleList([downsample]))
                input_block_channels.append(ch)
                ds *= 2

        d_head, n_heads = get_d_and_n_heads(ch)
        self.middle_block: nn.ModuleList = nn.ModuleList(
            [
                ResBlock(ch, time_embed_dim, ch),
                SpatialTransformer(ch, n_heads, d_head, ctx_dim, use_linear, depth=transformer_depth[-1]),
                ResBlock(ch, time_embed_dim, ch),
            ],
        )
        self.output_blocks = nn.ModuleList()
        for idx, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[idx] + 1):
                ich = input_block_channels.pop()
                layers = nn.ModuleList(
                    [
                        ResBlock(ch + ich, time_embed_dim, model_ch * mult),
                    ],
                )
                ch = model_ch * mult

                if ds in attention_resolutions:
                    d_head, n_heads = get_d_and_n_heads(ch)
                    layers.append(
                        SpatialTransformer(
                            ch,
                            n_heads,
                            d_head,
                            ctx_dim,
                            use_linear,
                            depth=transformer_depth[idx],
                        ),
                    )

                if idx > 0 and i == self.num_res_blocks[idx]:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(layers)

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_ch, out_ch, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        tms: torch.Tensor,
        ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the UNetModel.

        Args:
            x (torch.Tensor): Input tensor.
            tms (torch.Tensor): Timestep tensor.
            ctx (torch.Tensor): Context tensor.
            y (torch.Tensor, optional): Label tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        t_emb = timestep_embedding(tms, self.model_ch)
        emb = self.time_embed(t_emb)
        dtype = next(self.parameters()).dtype
        emb = emb.to(dtype)
        ctx = ctx.to(dtype)
        x = x.to(dtype)

        def run(x: torch.Tensor, bb: nn.Module) -> torch.Tensor:
            if isinstance(bb, ResBlock):
                x = bb(x, emb)
            elif isinstance(bb, SpatialTransformer):
                x = bb(x, ctx)
            elif isinstance(bb, nn.ModuleDict):
                for m in bb.values():
                    x = m(x)
            else:
                x = bb(x)
            return x

        saved_inputs = []
        for b in self.input_blocks:
            for bb in b:
                x = run(x, bb)
            saved_inputs.append(x)
        for b in self.middle_block:
            x = run(x, b)
        for b in self.output_blocks:
            x = torch.cat((x, saved_inputs.pop()), dim=1)
            for bb in b:
                x = run(x, bb)
        return self.out(x)
