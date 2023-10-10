"""OTX adapters.torch.mmengine.mmdeploy.utils deploy config util functions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from mmengine.config import ConfigDict

from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config


def patch_input_preprocessing(
    deploy_cfg: ConfigDict,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    to_rgb: bool = False,
) -> None:
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
            "--mean_values": list(mean) if mean is not None else [],
            "--scale_values": list(std) if std is not None else [],
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


def patch_input_shape(deploy_cfg: Config, input_shape: Optional[tuple]) -> None:
    """Patch the input shape of the model in the deployment configuration.

    Args:
        deploy_cfg (DictConfig): The deployment configuration.
        input_shape (Optional[Tuple[int, int]]): The input shape of the model.

    Returns:
        None
    """
    # default is static shape to prevent an unexpected error
    # when converting to OpenVINO IR
    _input_shape = [1, 3, *input_shape] if input_shape is not None else [1, 3, 256, 256]
    deploy_cfg.backend_config.model_inputs = [ConfigDict(opt_shapes=ConfigDict(input=_input_shape))]
