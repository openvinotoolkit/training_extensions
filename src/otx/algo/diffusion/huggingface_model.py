# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging-Face pretrained model for the OTX Object Detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers import StableDiffusionPipeline
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import nn
from torch.optim.optimizer import Optimizer

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.diffusion import (
    DiffusionBatchDataEntity,
    DiffusionBatchPredEntity,
)
from otx.core.data.entity.utils import stack_batch
from otx.core.model.diffusion import OTXDiffusionModel

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from transformers import CLIPTextModel


class UNetWrapper(nn.Module):
    """StableDiffusion module for performing stable diffusion process."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
    ):
        """Initializes the StableDiffusion module.

        Args:
            text_encoder (CLIPTextModel): The text encoder model.
            vae (AutoencoderKL): The VAE (Variational Autoencoder) model.
            unet (UNet2DConditionModel): The UNet model for conditioning.
            noise_scheduler (DDPMScheduler): The noise scheduler.
        """
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward diffusion process.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            input_ids (torch.Tensor): The input IDs.

        Returns:
            tuple: A tuple containing the model predictions and the target values.
        """
        # Predict the noise residual
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class HuggingFaceModelForDiffusion(OTXDiffusionModel):
    """A class representing a Hugging Face model for diffusion.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        label_info (LabelInfoTypes): The label information for the model.
        optimizer (OptimizerCallable, optional): The optimizer for training the model.
            Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional):
            The learning rate scheduler for training the model. Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The evaluation metric for the model.
            Defaults to DiffusionMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

    Example:
        1. API
            >>> model = HuggingFaceModelForDiffusion(
            ...     model_name_or_path="CompVis/stable-diffusion-v1-4",
            ... )
        2. CLI
            >>> otx train \
            ... --model otx.algo.detection.huggingface_model.HuggingFaceModelForDiffusion \
            ... --model.model_name_or_path CompVis/stable-diffusion-v1-4
    """

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        self.model_name = model_name_or_path
        self.load_from = None
        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(model_name_or_path)
        super().__init__(*args, **kwargs)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()

    def _create_model(self) -> nn.Module:
        return UNetWrapper(self.pipe.unet)

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
        input_ids = self.pipe.tokenizer(
            entity.captions,
            max_length=self.pipe.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        latents = self.pipe.vae.encode(entity.images).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.pipe.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.pipe.text_encoder(input_ids, return_dict=False)[0]

        if self.pipe.scheduler.config.prediction_type == "epsilon":
            self.target = noise
        elif self.pipe.scheduler.config.prediction_type == "v_prediction":
            self.target = self.pipe.scheduler.get_velocity(latents, noise, timesteps)
        else:
            error_msg = f"Invalid prediction type: {self.pipe.scheduler.config.prediction_type}"
            raise ValueError(error_msg)

        return {
            "sample": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def _customize_outputs(
        self,
        outputs: torch.Tensor,
        inputs: DiffusionBatchDataEntity,
    ) -> DiffusionBatchPredEntity | OTXBatchLossEntity:
        if torch.isnan(outputs).any():
            error_msg = "NaN detected in the output."
            raise ValueError(error_msg)
        if not self.training:
            msg = "This should never raise since `validation_step` is overridden."
            raise NotImplementedError(msg)
        return OTXBatchLossEntity(mse=F.mse_loss(outputs, self.target, reduction="mean"))

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.

        If on DDP it is called on every process

        """
        self.pipe.to(self.device)
        return super().on_fit_start()

    def validation_step(self, batch: DiffusionBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # We use "latent" output type to overcome built-in NSFW filtering that blacks out some random images
        images = self.pipe(batch.captions, output_type="latent").images
        images = self.pipe.vae.decode(
            images / self.pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        images = self.pipe.image_processor.postprocess(
            images,
            output_type="pt",
            do_denormalize=[True] * len(images),
        )
        self.metric.update(images, real=False)

        # Save images
        for image, caption in zip(images, batch.captions):
            self.logger.experiment.add_image(f"val/images/{caption}", image.detach().cpu())
