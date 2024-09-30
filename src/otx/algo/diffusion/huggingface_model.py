# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging-Face pretrained model for the OTX Object Detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers import StableDiffusionPipeline
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from otx.core.data.entity.diffusion import DiffusionBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.diffusion import DiffusionOTXModelExporter
from otx.core.model.diffusion import OTXDiffusionModel
from torch import nn

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from otx.core.data.entity.diffusion import (
        DiffusionBatchDataEntity,
    )
    from otx.core.exporter.base import OTXModelExporter


class DiffusionModule(nn.Module):
    """StableDiffusion module for performing stable diffusion process."""

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward diffusion process."""
        if self.training:
            return self.loss(sample, timestep, encoder_hidden_states, target_noise)
        return self.generate_sample(sample, timestep, encoder_hidden_states)


class UNetWrapper(DiffusionModule):
    """StableDiffusion module for performing stable diffusion process."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
    ):
        """Initializes the StableDiffusion module.

        Args:
            unet (UNet2DConditionModel): The UNet model for conditioning.
        """
        super().__init__()
        self.unet = unet

    def _forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward diffusion process.

        Args:
            sample (torch.Tensor): Latent dimension tensor for denoising.
            timestep (torch.Tensor): Noise scheduler timestep.
            encoder_hidden_states (torch.Tensor): Encoded text conditioning.

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

    def loss(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the loss for input batch."""
        return F.mse_loss(self._forward(sample, timestep, encoder_hidden_states), target_noise)

    def generate_sample(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Generates a sample from the input batch."""
        return self._forward(sample, timestep, encoder_hidden_states)


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

    def forward_for_tracing(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model(sample, timestep, encoder_hidden_states)

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
            target = noise
        elif self.pipe.scheduler.config.prediction_type == "v_prediction":
            target = self.pipe.scheduler.get_velocity(latents, noise, timesteps)
        else:
            error_msg = f"Invalid prediction type: {self.pipe.scheduler.config.prediction_type}"
            raise ValueError(error_msg)

        return {
            "sample": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "target_noise": target,
        }

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.

        If on DDP it is called on every process

        """
        self.pipe.to(self.device)
        return super().on_fit_start()

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self.pipe.to(self.device)
        return super().on_test_start()

    def on_predict_start(self) -> None:
        """Called at the beginning of prediction."""
        self.pipe.to(self.device)
        return super().on_predict_start()

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
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                for image, caption in zip(images, batch.captions):
                    logger.experiment.add_image(f"val/images/{caption}", image.detach().cpu())

    def predict_step(
        self,
        batch: DiffusionBatchDataEntity,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> DiffusionBatchPredEntity:
        """Step function called during PyTorch Lightning Trainer's predict."""
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
        return DiffusionBatchPredEntity(
            images=images,
            imgs_info=batch.imgs_info,
            batch_size=batch.batch_size,
            scores=[],
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        return DiffusionOTXModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 4, 64, 64),
            resize_mode="fit_to_window_letterbox",
            swap_rgb=True,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["sample", "timestep", "encoder_hidden_states"],
                "output_names": ["latents"],
                "dynamic_axes": {
                    "sample": {0: "batch", 2: "height", 3: "width"},
                    "timestep": {0: "batch"},
                    "encoder_hidden_states": {0: "batch"},
                },
                "autograd_inlining": False,
                "args": {
                    "sample": torch.randn(1, 4, 64, 64),
                    "timestep": torch.tensor([0]),
                    "encoder_hidden_states": torch.randn(1, 77, 768),
                },
            },
            output_names=None,
        )
