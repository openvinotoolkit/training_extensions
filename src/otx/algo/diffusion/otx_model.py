"""This module contains the OTXStableDiffusion class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.diffusion import DiffusionBatchDataEntity, DiffusionBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.model.diffusion import OTXDiffusionModel

from .clip_tokenizer import CLIPTokenizer
from .ddim_scheduler import DDIMScheduler
from .model.autoencoder_kl import AutoencoderKL
from .model.clip import TextTransformer as CLIPTextTransformer
from .model.unet import UNetModel
from .otx_model_pretrained import PretrainedOTXStableDiffusion
from .utils.download import download


class OTXStableDiffusion(OTXDiffusionModel):
    """OTX Stable Diffusion model."""

    model: nn.Module

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
        checkpoint_url: str | None = None,
        val_num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
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
        self.val_num_inference_steps = val_num_inference_steps
        self.guidance_scale = guidance_scale

        super().__init__(**kwargs)
        self.strict_loading = False

        self.vae = AutoencoderKL()
        self.text_model = CLIPTextTransformer()
        self.tokenizer = CLIPTokenizer()
        self.noise_scheduler = DDIMScheduler()
        if checkpoint_url:
            pretrained = PretrainedOTXStableDiffusion(
                unet=self.model,
                vae=self.vae,
                text_model=self.text_model,
            )
            file_path = download(checkpoint_url, ".")
            pretrained.load_state_dict(
                torch.load(file_path, weights_only=False)["state_dict"],
                strict=self.strict_loading,
            )

        self.vae.eval()
        self.text_model.eval()

    def _create_model(self) -> nn.Module:
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
        self.text_model.to(self.device)
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(
                entity.images,
                entity.imgs_info,
                pad_size_divisor=pad_size_divisor,
                pad_value=pad_value,
            )

        encoder_hidden_states = torch.cat(tuple(map(self._encode_prompt, entity.captions)))

        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (len(entity.images),),
            device=self.device,
        ).long()
        latents = self._encode_images(entity.images)
        self.noise = torch.randn_like(latents)
        noisy_model_input = self.noise_scheduler.add_noise(latents, self.noise, timesteps)
        return {
            "x": noisy_model_input,
            "tms": timesteps.float(),
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

        return OTXBatchLossEntity(mse=F.mse_loss(outputs, self.noise))

    def validation_step(
        self,
        batch: DiffusionBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.noise_scheduler.set_timesteps(self.val_num_inference_steps)
        embeds = torch.cat(tuple(map(self._encode_prompt, batch.captions)))
        neg_embeds = torch.cat(tuple(map(self._encode_prompt, [""] * len(batch.captions))))
        latents = self._encode_images(batch.images)
        latents = torch.randn_like(latents)
        latents = torch.cat([latents] * 2)
        for t in self.noise_scheduler.timesteps:
            timestep = torch.tensor([t]).float().to(self.device)
            noise_pred = self.model(
                latents,
                timestep,
                torch.cat([neg_embeds, embeds]),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.noise_scheduler.step(noise_pred, t, latents)
        images = self._decode_latents(latents)
        self.metric.update(images, real=False)

        # Save images
        Path(f"val_images/{self.epoch_idx}").mkdir(parents=True, exist_ok=True)
        for image, caption in zip(images, batch.captions):
            PIL.Image.fromarray(
                (image.permute(1, 2, 0).clip(0, 1) * 255).to(torch.uint8).detach().cpu().numpy(),
            ).save(f"val_images/{self.epoch_idx}/{caption}.png")

    def _encode_images(
        self,
        images: torch.Tensor,
        sample_posterior: bool = True,
    ) -> torch.Tensor:
        mean, logvar = self.vae.encoder(images).chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(mean, device=self.device)
        else:
            z = mean

        return z * 1.0 / self.guidance_scale

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        return (image / 2 + 0.5).clamp(0, 1)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)

        return self.text_model(input_ids).float()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        if self.metric.fid.real_features_num_samples > 0:
            # validation is started after training, otherwise we cannot calculate the metric
            super().on_validation_epoch_end()
