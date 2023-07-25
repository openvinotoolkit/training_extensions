from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from anomalib.models import get_model as anomalib_get_model


def get_model(
    model: Optional[Union[Dict[str, Any], DictConfig, str]] = None,
    checkpoint: Optional[str] = None,
    **kwargs,
) -> torch.nn.Module:
    """_summary_.

    Args:
        config (Optional[Union[Dict[str, Any], DictConfig]], optional): _description_. Defaults to None.
        {"model": {"name"}}
        checkpoint (Optional[str], optional): _description_. Defaults to None.

    Returns:
        AnomalyModule: _description_
    """
    if isinstance(model, str):
        pass
    if not hasattr(model, "model"):
        model = DictConfig(content={"model": model})
    if checkpoint is not None:
        model["init_weights"] = checkpoint
    if isinstance(model, dict):
        model = OmegaConf.create(model)
    return anomalib_get_model(config=model)


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
