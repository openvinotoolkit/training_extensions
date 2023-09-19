"""OTX adapters.torch.mmengine.mmdeploy.utils deploy config util functions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

from mmengine.config import ConfigDict

from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig as Config


def patch_input_preprocessing(
    deploy_cfg: ConfigDict,
    mean: List[float] = [],
    std: List[float] = [],
    to_rgb: bool = False,
):
    """Update backend configuration with input preprocessing options.

    - If `"to_rgb"` in Normalize config is truthy, it adds `"--reverse_input_channels"` as a flag.

    The function then sets default values for the backend configuration in `deploy_cfg`.

    Args:
        cfg (mmcv.ConfigDict): Config object containing test pipeline and other configurations.
        deploy_cfg (mmcv.ConfigDict): DeployConfig object containing backend configuration.

    Returns:
        None: This function updates the input `deploy_cfg` object directly.
    """
    # Set options based on Normalize config
    options = {
        "flags": ["--reverse_input_channels"] if to_rgb else [],
        "args": {
            "--mean_values": list(mean),
            "--scale_values": list(std),
        },
    }

    # Set default backend configuration
    mo_options = deploy_cfg.backend_config.get("mo_options", ConfigDict())
    mo_options = ConfigDict() if mo_options is None else mo_options
    mo_options.args = mo_options.get("args", ConfigDict())
    mo_options.flags = mo_options.get("flags", [])

    # Override backend configuration with options from Normalize config
    mo_options.args.update(options["args"])
    mo_options.flags = list(set(mo_options.flags + options["flags"]))

    deploy_cfg.backend_config.mo_options = mo_options


def patch_input_shape(deploy_cfg: Config, input_shape: Optional[Tuple[int, int]]) -> None:
    assert input_shape is not None
    assert all(isinstance(i, int) and i > 0 for i in input_shape)
    # default is static shape to prevent an unexpected error
    # when converting to OpenVINO IR
    deploy_cfg.backend_config.model_inputs = [ConfigDict(opt_shapes=ConfigDict(input=[1, 3, *input_shape]))]
