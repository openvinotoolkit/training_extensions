from pathlib import Path

import pytest

from otx.algorithms.anomaly.configs.base.draem import DraemAnomalyBaseConfig
from otx.algorithms.anomaly.configs.base.padim import PadimAnomalyBaseConfig
from otx.algorithms.anomaly.configs.base.stfpm import STFPMAnomalyBaseConfig
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.helper import convert, create
from otx.api.entities.model_template import ModelTemplate, parse_model_template


@pytest.mark.parametrize(
    ["model_name", "configurable_parameters"],
    [("padim", PadimAnomalyBaseConfig), ("stfpm", STFPMAnomalyBaseConfig), ("draem", DraemAnomalyBaseConfig)],
)
def test_model_template_loading(model_name, configurable_parameters):
    # Create from class
    configuration = configurable_parameters()
    configurable_parameters_yaml_str = convert(configuration, str)
    assert isinstance(configurable_parameters_yaml_str, str)

    # Check if it can be created from yaml
    configurable_parameters_yaml: ConfigurableParameters = create(configurable_parameters_yaml_str)

    template_file_root = Path("otx", "algorithms", "anomaly", "configs", "classification", model_name)
    template_file_path = (
        template_file_root / "template.yaml"
        if (template_file_root / "template.yaml").exists()
        else template_file_root / "template_experimental.yaml"
    )
    model_template: ModelTemplate = parse_model_template(str(template_file_path))
    hyper_parameters: dict = model_template.hyper_parameters.data
    configurable_parameters_loaded = create(hyper_parameters)

    assert configurable_parameters_yaml == configurable_parameters_loaded
