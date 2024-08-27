"""This module contains the class for the pretrained OTX stable diffusion model."""

from torch import nn


class PretrainedOTXStableDiffusion(nn.Module):
    """This class represents a pretrained OTX stable diffusion model.

    It allows to load a pretrained model checkpoint compatible with original implementation: https://github.com/CompVis/stable-diffusion
    """

    def __init__(self, unet: nn.Module, vae: nn.Module, text_model: nn.Module):
        super().__init__()
        self.model = nn.ModuleDict({"diffusion_model": unet})
        self.first_stage_model = vae
        self.cond_stage_model = nn.ModuleDict({"transformer": nn.ModuleDict({"text_model": text_model})})
