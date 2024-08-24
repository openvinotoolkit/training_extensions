"""This module contains the OTXStableDiffusion class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F  # noqa: N812

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.diffusion import DiffusionBatchDataEntity, DiffusionBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.model.diffusion import OTXDiffusionModel

from .clip_tokenizer import ClipTokenizer
from .ddim_scheduler import DDIMScheduler
from .model.autoencoder_kl import AutoencoderKL
from .model.clip import ClipTextTransformer
from .model.unet import UNetModel


class OTXStableDiffusion(OTXDiffusionModel):
    """OTX Stable Diffusion model."""

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
        **kwargs,
    ):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.model_ch = model_ch
        self.attention_resolutions = attention_resolutions
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.n_heads = n_heads
        self.transformer_depth = transformer_depth
        self.ctx_dim = ctx_dim
        self.use_linear = use_linear
        self.d_head = d_head

        super().__init__(**kwargs)
        self.tokenizer = ClipTokenizer()
        self.vae = AutoencoderKL().eval().to(self.device)
        self.noise_scheduler = DDIMScheduler()
        self.text_model = ClipTextTransformer().eval().to(self.device)

    def _create_model(self) -> UNetModel:
        return UNetModel(
            in_ch=self.in_ch,
            out_ch=self.out_ch,
            model_ch=self.model_ch,
            attention_resolutions=self.attention_resolutions,
            num_res_blocks=self.num_res_blocks,
            channel_mult=self.channel_mult,
            transformer_depth=self.transformer_depth,
            ctx_dim=self.ctx_dim,
            use_linear=self.use_linear,
            d_head=self.d_head,
            n_heads=self.n_heads,
        )

    def _customize_inputs(
        self,
        entity: DiffusionBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(
                entity.images,
                entity.imgs_info,
                pad_size_divisor=pad_size_divisor,
                pad_value=pad_value,
            )
        input_ids = list(map(self.tokenizer.encode, entity.captions))

        input_ids = torch.tensor(input_ids).to(self.device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (len(entity.images),),
            device=self.device,
        ).float()
        self.latents = self.vae.encoder(entity.images)
        self.text_model.to(self.device)
        encoder_hidden_states = self.text_model(input_ids).float()

        return {
            "x": self.latents,
            "tms": timesteps,
            "ctx": encoder_hidden_states,
        }

    def _customize_outputs(
        self,
        outputs: torch.Tensor,
        inputs: DiffusionBatchDataEntity,
    ) -> DiffusionBatchPredEntity | OTXBatchLossEntity:
        for output in outputs:
            if torch.isnan(output).any():
                error_msg = "NaN detected in the output."
                raise ValueError(error_msg)
        if not self.training:
            raise NotImplementedError

        return OTXBatchLossEntity(mse=F.mse_loss(outputs, self.latents))

    def validation_step(
        self,
        batch: DiffusionBatchDataEntity,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.noise_scheduler.set_timesteps(num_inference_steps)
        noise = torch.randn_like(batch.images, device=self.device)
        latents = self.vae.encoder(noise)
        input_ids = list(map(self.tokenizer.encode, batch.captions))
        input_ids = torch.tensor(input_ids, device=self.device)
        self.text_model.to(self.device)
        encoder_hidden_states = self.text_model(input_ids)
        uncond_encoder_input_ids = [self.tokenizer.encode("")]
        uncond_encoder_input_ids = torch.tensor(uncond_encoder_input_ids, device=self.device)
        uncond_encoder_hidden_states = self.text_model(uncond_encoder_input_ids)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.model(
                torch.cat([latents] * 2),
                torch.tensor([t]).float().to(self.device),
                torch.cat((uncond_encoder_hidden_states, encoder_hidden_states)).to(self.device),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.noise_scheduler.step(noise_pred, t, latents)
        latents = self.vae.post_quant_conv(1 / 0.18215 * latents)
        images = self.vae.decoder(latents)
        self.metric.update(images, real=False)

        # Save images
        Path(f"val_images/{self.epoch_idx}").mkdir(parents=True, exist_ok=True)
        for image, caption in zip(images, batch.captions):
            PIL.Image.fromarray(
                (image.permute(1, 2, 0).clip(0, 1) * 255).to(torch.uint8).detach().cpu().numpy(),
            ).save(f"val_images/{self.epoch_idx}/{caption}.png")
