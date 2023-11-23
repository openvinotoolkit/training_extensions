"""Config data type objects for trainer."""

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig


@dataclass
class TrainerConfig(DictConfig):
    """DTO for trainer configuration."""

    default_root_dir: Path
    accelerator: str
    precision: int
    max_epochs: int
    min_epochs: int
    devices: int
    check_val_every_n_epoch: int
    deterministic: bool

    _target_: str = "lightning.pytorch.trainer.Trainer"
