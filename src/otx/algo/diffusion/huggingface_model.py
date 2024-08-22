# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging-Face pretrained model for the OTX Object Detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.pipelines import StableDiffusionPipeline
from torch import nn

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
    from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
    from transformers import CLIPTextModel

    from otx.core.metrics import MetricInput


# WEIGHT_DTYPE = torch.float16


class StableDiffusion(nn.Module):
    """StableDiffusion module for performing stable diffusion process."""

    def __init__(
        self,
        text_encoder: CLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        noise_scheduler: KarrasDiffusionSchedulers,
    ):
        """Initializes the StableDiffusion module.

        Args:
            text_encoder (CLIPTextModel): The text encoder model.
            vae (AutoencoderKL): The VAE (Variational Autoencoder) model.
            unet (UNet2DConditionModel): The UNet model for conditioning.
            noise_scheduler (DDPMScheduler): The noise scheduler.
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler

        # Freeze weights
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.train()

    def train(self, mode: bool = True) -> StableDiffusion:
        """Sets the training mode for the UNet model.

        Args:
            mode (bool, optional): Whether to set the training mode to True or False.
                Defaults to True.

        Returns:
            self: The StableDiffusion instance.
        """
        self.unet.train(mode)
        return self

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> tuple:
        """Performs the forward diffusion process.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            input_ids (torch.Tensor): The input IDs.

        Returns:
            tuple: A tuple containing the model predictions and the target values.
        """
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            error_msg = f"Invalid prediction type: {self.noise_scheduler.config.prediction_type}"
            raise ValueError(error_msg)

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            return_dict=False,
        )[0]
        return model_pred, target


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
            Defaults to MeanAveragePrecisionFMeasureCallable.
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
        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
        )
        super().__init__(*args, **kwargs)
        self.pipe.set_progress_bar_config(disable=True)

        self.epoch_idx = 0

    def _create_model(self) -> nn.Module:
        return StableDiffusion(
            self.pipe.text_encoder,
            self.pipe.vae,
            self.pipe.unet,
            self.pipe.scheduler,
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
        input_ids = (
            self.pipe.tokenizer(
                entity.captions,
                max_length=self.pipe.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids,
        )
        input_ids = torch.hstack(input_ids).to(self.device)

        return {
            "pixel_values": entity.images,
            "input_ids": input_ids,
        }

    def _customize_outputs(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        inputs: DiffusionBatchDataEntity,
    ) -> DiffusionBatchPredEntity | OTXBatchLossEntity:
        for output in outputs:
            if torch.isnan(output).any():
                error_msg = "NaN detected in the output."
                raise ValueError(error_msg)
        preds, targets = outputs
        if self.training:
            return OTXBatchLossEntity(
                {
                    "mse": F.mse_loss(preds, targets, reduction="mean"),
                },
            )

        images = self.pipe.vae.decode(
            preds / self.pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]

        return DiffusionBatchPredEntity(
            batch_size=inputs.batch_size,
            images=self.pipe.image_processor.postprocess(images, output_type="pt"),
            imgs_info=inputs.imgs_info,
            scores=[],
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: DiffusionBatchPredEntity,
        inputs: DiffusionBatchDataEntity,
    ) -> MetricInput:
        return {"imgs": preds.images, "real": False}

    def validation_step(self, batch: DiffusionBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.pipe.to(self.device)

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
        self.metric.update(images=images, text=batch.captions)

        # Save images
        images = self.pipe.image_processor.pt_to_numpy(images)
        images = self.pipe.image_processor.numpy_to_pil(images)
        Path(f"val_images/{self.epoch_idx}").mkdir(parents=True, exist_ok=True)
        for image, caption in zip(images, batch.captions):
            image.save(f"val_images/{self.epoch_idx}/{caption}.png")

    def test_step(self, batch: DiffusionBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        super().on_validation_epoch_start()
        self.epoch_idx += 1
