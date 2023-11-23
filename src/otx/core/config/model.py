"""Config data type objects for model."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """DTO for model configuration."""

    _target_: str
    optimizer: dict
    scheduler: dict
    otx_model: dict
    torch_compile: bool
