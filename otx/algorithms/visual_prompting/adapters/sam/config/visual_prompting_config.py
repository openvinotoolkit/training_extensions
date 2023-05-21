"""Configurable parameter conversion between OTX and Visual Prompting."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Union

from omegaconf import DictConfig, ListConfig

from anomalib.config.config import get_configurable_parameters
from otx.api.configuration.configurable_parameters import ConfigurableParameters


def get_visual_prompting_config(task_name: str, otx_config: ConfigurableParameters) -> Union[DictConfig, ListConfig]:
    """Get anomalib configuration.

    Create an anomalib config object that matches the values specified in the
    OTX config.

    Args:
        task_name: Task name to load configuration from the Anomalib
        otx_config: ConfigurableParameters: OTX config object parsed from
            configuration.yaml file

    Returns:
        Anomalib config object for the specified model type with overwritten
        default values.
    """
    # TODO: remove hard coding
    config_path = "/home/cosmos/ws/otx-sam/otx/algorithms/visual_prompting/configs/lightning_sam/config.yaml"
    visual_prompting_config = get_configurable_parameters(model_name=task_name.lower(), config_path=config_path)
    update_visual_prompting_config(visual_prompting_config, otx_config)
    return visual_prompting_config


def _visual_prompting_config_mapper(anomalib_config: Union[DictConfig, ListConfig], otx_config: ConfigurableParameters):
    """Return mapping from learning parameters to anomalib parameters.

    Args:
        anomalib_config: DictConfig: Anomalib config object
        otx_config: ConfigurableParameters: OTX config object parsed from configuration.yaml file
    """
    parameters = otx_config.parameters
    groups = otx_config.groups
    for name in parameters:
        if name == "train_batch_size":
            anomalib_config.dataset["train_batch_size"] = getattr(otx_config, "train_batch_size")
        elif name == "max_epochs":
            anomalib_config.trainer["max_epochs"] = getattr(otx_config, "max_epochs")
        else:
            assert name in anomalib_config.model.keys(), f"Parameter {name} not present in anomalib config."
            sc_value = getattr(otx_config, name)
            sc_value = sc_value.value if hasattr(sc_value, "value") else sc_value
            anomalib_config.model[name] = sc_value
    for group in groups:
        update_visual_prompting_config(anomalib_config.model[group], getattr(otx_config, group))


def update_visual_prompting_config(anomalib_config: Union[DictConfig, ListConfig], otx_config: ConfigurableParameters):
    """Update anomalib configuration.

    Overwrite the default parameter values in the anomalib config with the
    values specified in the OTX config. The function is recursively called for
    each parameter group present in the OTX config.

    Args:
        anomalib_config: DictConfig: Anomalib config object
        otx_config: ConfigurableParameters: OTX config object parsed from
            configuration.yaml file
    """
    for param in otx_config.parameters:
        assert param in anomalib_config.keys(), f"Parameter {param} not present in anomalib config."
        sc_value = getattr(otx_config, param)
        sc_value = sc_value.value if hasattr(sc_value, "value") else sc_value
        anomalib_config[param] = sc_value
    for group in otx_config.groups:
        # Since pot_parameters and nncf_optimization are specific to OTX
        if group == "learning_parameters":
            _visual_prompting_config_mapper(anomalib_config, getattr(otx_config, "learning_parameters"))
        elif group not in ["pot_parameters", "nncf_optimization"]:
            update_visual_prompting_config(anomalib_config[group], getattr(otx_config, group))
