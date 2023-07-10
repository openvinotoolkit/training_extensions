from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from anomalib.models import get_model as anomalib_get_model


def get_model(
    config: Optional[Union[Dict[str, Any], DictConfig, str]] = None,
    checkpoint: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> torch.nn.Module:
    """_summary_.

    Args:
        config (Optional[Union[Dict[str, Any], DictConfig]], optional): _description_. Defaults to None.
        {"model": {"name"}}
        checkpoint (Optional[str], optional): _description_. Defaults to None.

    Returns:
        AnomalyModule: _description_
    """
    if isinstance(config, str):
        pass
    if checkpoint is not None:
        config["init_weights"] = checkpoint
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    return anomalib_get_model(config=config)


if __name__ == "__main__":
    model_config = {
        "model": {
            "name": "padim",
            "backbone": "resnet18",
            "pre_trained": True,
            "layers": ["layer1", "layer2", "layer3"],
            "normalization_method": "min_max",
            "input_size": [256, 256],
        }
    }
    model = get_model(model_config)
