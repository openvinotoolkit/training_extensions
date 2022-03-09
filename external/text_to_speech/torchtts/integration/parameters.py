# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from attr import attrs
from sys import maxsize

from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group,
                                            boolean_attribute,
                                            configurable_integer,
                                            configurable_float,
                                            selectable,
                                            string_attribute,
                                            )
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.model_lifecycle import ModelLifecycle

@attrs
class OTETextToSpeechTaskParameters(ConfigurableParameters):
    header = string_attribute("Configuration for an speech-to-text task")
    description = header

    @attrs
    class __LearningParameters(ParameterGroup):
        header = string_attribute("Learning Parameters")
        description = header

        batch_size = configurable_integer(
            default_value=40,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        num_epochs = configurable_integer(
            default_value=350,
            min_value=1,
            max_value=9223372036854775807,
            header="Maximum number of training epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        learning_rate = configurable_float(
            default_value=0.0001,
            min_value=1.0e-7,
            max_value=1.0,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

    learning_parameters = add_parameter_group(__LearningParameters)


if __name__ == "__main__":
    import yaml
    from ote_sdk.configuration.helper import convert

    def remove_value_key(group: dict):
        """
        Recursively remove any reference to the key named "value" from the dictionary.
        """
        if "value" in group:
            del group["value"]

        for val in group.values():
            if isinstance(val, dict):
                remove_value_key(val)

    configuration = convert(OTETextToSpeechTaskParameters(), dict, enum_to_str=True, id_to_str=True)
    del configuration["id"]
    print(f"Writing to configuration.yaml")
    open("configuration.yaml", "w").write(yaml.dump(configuration))
