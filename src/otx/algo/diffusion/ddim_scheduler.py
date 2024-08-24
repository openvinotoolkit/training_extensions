"""This module provides the DDIM scheduler."""

from __future__ import annotations

import math

import numpy as np
import torch


def cosine_beta_schedule(
    timesteps: int, beta_start: float = 0.0, beta_end: float = 0.999, s: float = 0.008
) -> torch.Tensor:
    """Compute the cosine beta schedule.

    Args:
        timesteps (int): The number of timesteps.
        beta_start (float, optional): The starting value of beta. Defaults to 0.0.
        beta_end (float, optional): The ending value of beta. Defaults to 0.999.
        s (float, optional): The scaling factor. Defaults to 0.008.

    Returns:
        torch.Tensor: The computed beta values.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


class DDIMScheduler:
    """Class for DDIM scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.betas = (
            cosine_beta_schedule(num_train_timesteps, beta_start, beta_end)
            if beta_schedule == "cosine"
            else torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.timesteps = np.arange(num_train_timesteps)[::-1]

    def set_timesteps(self, num_inference_steps: int, offset: int = 0) -> None:
        """Sets the timesteps for the scheduler.

        Args:
            num_inference_steps (int): The number of inference steps.
            offset (int, optional): The offset value. Defaults to 0.

        Returns:
            None
        """
        self.timesteps = (
            np.arange(
                0,
                self.num_train_timesteps,
                self.num_train_timesteps // num_inference_steps,
            )[::-1]
            + offset
        )

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        return (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Performs a single step of the diffusion process.

        Args:
            model_output (torch.Tensor): The output of the model.
            timestep (int): The current timestep.
            sample (torch.Tensor): The input sample.
            eta (float, optional): The scaling factor for the variance. Defaults to 1.0.
            generator (torch.Generator | None, optional): The random number generator. Defaults to None.

        Returns:
            torch.Tensor: The previous sample after the diffusion step.
        """
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // len(self.timesteps)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # 4. Clamp "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**0.5
        # the model_output is always re-derived from the clipped x_0 in Glide
        model_output = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = torch.randn(model_output.shape, generator=generator).to(sample.device)
            prev_sample += std_dev_t * noise

        return prev_sample

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Adds noise to the original samples based on the given timesteps.

        Args:
            original_samples (torch.Tensor): The original samples to add noise to.
            noise (torch.Tensor): The noise tensor to add to the original samples.
            timesteps (torch.Tensor): The timesteps to determine the amount of noise to add.

        Returns:
            torch.Tensor: The samples with added noise.

        """
        self.alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        return (
            sqrt_alpha_prod[:, None, None, None] * original_samples
            + sqrt_one_minus_alpha_prod[:, None, None, None] * noise
        )
