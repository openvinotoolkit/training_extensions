# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.algorithms.anomaly.adapters.anomalib.config import get_anomalib_config
from otx.algorithms.anomaly.configs.classification.draem import (
    DraemAnomalyClassificationConfig,
)
from otx.algorithms.anomaly.configs.classification.padim import (
    PadimAnomalyClassificationConfig,
)
from otx.algorithms.anomaly.configs.classification.stfpm import (
    STFPMAnomalyClassificationConfig,
)
from otx.algorithms.anomaly.configs.detection.draem import DraemAnomalyDetectionConfig
from otx.algorithms.anomaly.configs.detection.padim import PadimAnomalyDetectionConfig
from otx.algorithms.anomaly.configs.detection.stfpm import STFPMAnomalyDetectionConfig
from otx.algorithms.anomaly.configs.segmentation.draem import (
    DraemAnomalySegmentationConfig,
)
from otx.algorithms.anomaly.configs.segmentation.padim import (
    PadimAnomalySegmentationConfig,
)
from otx.algorithms.anomaly.configs.segmentation.stfpm import (
    STFPMAnomalySegmentationConfig,
)
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.helper import convert, create
from tests.unit.algorithms.anomaly.helpers.utils import get_model_template


@pytest.mark.parametrize(
    ["model_name", "configurable_parameters"],
    [
        ("padim", PadimAnomalyClassificationConfig),
        ("padim", PadimAnomalyDetectionConfig),
        ("padim", PadimAnomalySegmentationConfig),
        ("stfpm", STFPMAnomalyClassificationConfig),
        ("stfpm", STFPMAnomalyDetectionConfig),
        ("stfpm", STFPMAnomalySegmentationConfig),
        ("draem", DraemAnomalyClassificationConfig),
        ("draem", DraemAnomalyDetectionConfig),
        ("draem", DraemAnomalySegmentationConfig),
    ],
)
def test_model_template_loading(model_name, configurable_parameters):
    """Test that we can load the model template and create a config from it."""
    # Create from class
    configuration = configurable_parameters()
    configurable_parameters_yaml_str = convert(configuration, str)
    assert isinstance(configurable_parameters_yaml_str, str)

    # Check if it can be created from yaml
    configurable_parameters_yaml: ConfigurableParameters = create(configurable_parameters_yaml_str)

    model_template = get_model_template(model_name)
    hyper_parameters: dict = model_template.hyper_parameters.data
    configurable_parameters_loaded = create(hyper_parameters)

    assert configurable_parameters_yaml == configurable_parameters_loaded

    # Confirm that we can create an anomalib config from the loaded yaml
    get_anomalib_config(model_name, configurable_parameters_loaded)
