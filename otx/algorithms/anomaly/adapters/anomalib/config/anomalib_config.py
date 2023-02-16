"""Configurable parameter conversion between OTX and Anomalib."""

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

from pathlib import Path
from typing import Union

import anomalib
from anomalib.config.config import get_configurable_parameters
from omegaconf import DictConfig, ListConfig

from otx.api.configuration.configurable_parameters import ConfigurableParameters


def get_anomalib_config(task_name: str, otx_config: ConfigurableParameters) -> Union[DictConfig, ListConfig]:
    """Get anomalib configuration.

    Create an anomalib config object that matches the values specified in the
    OTX config.

    Args:
        otx_config: ConfigurableParameters: OTX config object parsed from
            configuration.yaml file

    Returns:
        Anomalib config object for the specified model type with overwritten
        default values.
    """
    config_path = Path(anomalib.__file__).parent / "models" / task_name.lower() / "config.yaml"
    anomalib_config = get_configurable_parameters(model_name=task_name.lower(), config_path=config_path)
    # TODO: remove this hard coding of the config location
    if anomalib_config.model.name == "draem":
        anomalib_config.dataset.transform_config.train = (
            f"otx/algorithms/adapters/anomalib/configs/{task_name.lower()}/draem/transform_config.yaml"
        )
        anomalib_config.dataset.transform_config.val = (
            f"otx/algorithms/adapters/anomalib/configs/{task_name.lower()}/draem/transform_config.yaml"
        )
    else:
        anomalib_config.dataset.transform_config.train = None
        anomalib_config.dataset.transform_config.val = None
    update_anomalib_config(anomalib_config, otx_config)
    return anomalib_config


def _anomalib_config_mapper(anomalib_config: Union[DictConfig, ListConfig], otx_config: ConfigurableParameters):
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
        update_anomalib_config(anomalib_config.model[group], getattr(otx_config, group))


def update_anomalib_config(anomalib_config: Union[DictConfig, ListConfig], otx_config: ConfigurableParameters):
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
            _anomalib_config_mapper(anomalib_config, getattr(otx_config, "learning_parameters"))
        elif group not in ["pot_parameters", "nncf_optimization"]:
            update_anomalib_config(anomalib_config[group], getattr(otx_config, group))
