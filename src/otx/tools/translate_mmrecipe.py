# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Python script to translate MMx recipe to OTX YAML recipe file."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

from mmengine.config import Config
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from mmengine.config import Config as MMConfig

parser = ArgumentParser()
parser.add_argument("-n", "--recipe-name", type=str, required=True)
parser.add_argument("-o", "--output-dir", type=Path, required=True)
parser.add_argument("-i", "--input-path", type=Path, required=True)

override = parser.add_argument_group("override")
override.add_argument("--base", type=str, default="detection")
override.add_argument("--data", type=str, default="mmdet")
override.add_argument("--model", type=str, default="mmdet")


def mmconfig_dict_to_dict(obj: MMConfig | list[MMConfig]) -> list | dict:
    """Convert MMEngine config object to Python dictionary."""
    if isinstance(obj, list):
        return [mmconfig_dict_to_dict(x) for x in obj]
    if hasattr(obj, "to_dict"):
        return {k: mmconfig_dict_to_dict(v) for k, v in obj.to_dict().items()}

    return obj


if __name__ == "__main__":
    args = parser.parse_args()

    config = Config.fromfile(args.input_path)
    config = mmconfig_dict_to_dict(config)

    omega_conf = OmegaConf.create(
        {
            "defaults": [
                {"override /base": args.base},
                {"override /data": args.data},
                {"override /model": args.model},
            ],
            "data": {
                "subsets": {
                    "train": {
                        "batch_size": config["train_dataloader"]["batch_size"],
                        "transforms": config["train_dataloader"]["dataset"]["pipeline"],
                    },
                    "val": {
                        "batch_size": config["val_dataloader"]["batch_size"],
                        "transforms": config["val_dataloader"]["dataset"]["pipeline"],
                    },
                },
            },
            "model": {"otx_model": {"config": config["model"]}},
        },
    )

    print(omega_conf)
    output_path: Path = args.output_dir / f"{args.recipe_name}.yaml"
    with output_path.open("w") as fp:
        fp.write("# @package _global_\n")
        OmegaConf.save(omega_conf, fp)
