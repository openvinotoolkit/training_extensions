# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper script to download all the model pre-trained weights."""

import logging
from pathlib import Path

from importlib_resources import files
from omegaconf import OmegaConf
from otx.core.utils.instantiators import partial_instantiate_class

logging.basicConfig(
    level=logging.INFO,
    filename="download_pretrained_weights.log",
    filemode="w",
)

logger = logging.getLogger()


def download_all() -> None:
    """Download pre-trained weights of all models."""
    recipe_dir = Path(files("otx") / "recipe")

    for config_path in recipe_dir.glob("**/*.yaml"):
        if "_base_" in str(config_path):
            msg = f"Skip {config_path} since it is a base config."
            logger.warning(msg)
            continue
        if config_path.name == "openvino_model.yaml":
            msg = f"Skip {config_path} since it is not a PyTorch config."
            logger.warning(msg)
            continue
        if "anomaly_" in str(config_path) or "otx_dino_v2" in str(config_path) or "h_label_cls" in str(config_path):
            msg = f"Skip {config_path} since those models show errors on instantiation."
            logger.warning(msg)
            continue

        config = OmegaConf.load(config_path)
        init_model = next(iter(partial_instantiate_class(config.model)))
        try:
            model = init_model()
            msg = f"Downloaded pre-trained model weight of {model!s}"
            logger.info(msg)
        except Exception:
            msg = f"Error on instiating {config_path}"
            logger.exception(msg)


if __name__ == "__main__":
    download_all()
