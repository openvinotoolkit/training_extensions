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
                                            configurable_boolean,
                                            selectable,
                                            string_attribute,
                                            )
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.model_lifecycle import ModelLifecycle


@attrs
class OTESpeechToTextTaskParameters(ConfigurableParameters):
    """
    Base OTE configurable parameters for speech to text task.
    """

    header = string_attribute("Configuration for an speech-to-text task")
    description = header

    @attrs
    class __LearningParameters(ParameterGroup):
        header = string_attribute("Learning Parameters")
        description = header

        batch_size = configurable_integer(
            default_value=32,
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
            default_value=200,
            min_value=1,
            max_value=9223372036854775807,
            header="Maximum number of training epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        val_check_interval = configurable_float(
            default_value=1.0,
            min_value=0.0,
            max_value=1.0,
            header="use to check every n steps (batches)",
            description="use to check every n steps (batches)",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        learning_rate = configurable_float(
            default_value=5.0e-4,
            min_value=1.0e-7,
            max_value=1.0,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        learning_rate_warmup_steps = configurable_integer(
            default_value=0,
            min_value=0,
            max_value=9223372036854775807,
            header="Learning rate warmup steps",
            description="Increasing this value will make training is more stable for high learning affects.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        lr_scheduler = configurable_boolean(
            default_value=True,
            header="Learning rate scheduler.",
            description="Use learning rate scheduler during training.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        num_workers = configurable_integer(
            default_value=4,
            min_value=0,
            max_value=9223372036854775807,
            header="Num workers.",
            description="Num workers.",
            affects_outcome_of=ModelLifecycle.NONE
        )

        vocab_size = configurable_integer(
            default_value=256,
            min_value=32,
            max_value=9223372036854775807,
            header="Vocab size.",
            description="Vocab size.",
            affects_outcome_of=ModelLifecycle.NONE
        )

        n_mels = configurable_integer(
            default_value=64,
            min_value=32,
            max_value=9223372036854775807,
            header="Num mels.",
            description="Num mels.",
            affects_outcome_of=ModelLifecycle.NONE
        )

    @attrs
    class __ExportParameters(ParameterGroup):
        header = string_attribute("Export Parameters")
        description = header

        sequence_length = configurable_integer(
            default_value=128,
            min_value=32,
            max_value=9223372036854775807,
            header="Length of the input sequence in the exported model.",
            description="Length of the input sequence in the exported model.",
            affects_outcome_of=ModelLifecycle.ARCHITECTURE
        )

    learning_parameters = add_parameter_group(__LearningParameters)
    export_parameters = add_parameter_group(__ExportParameters)


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

    configuration = convert(OTESpeechToTextTaskParameters(), dict, enum_to_str=True, id_to_str=True)
    del configuration["id"]
    print(f"Writing to configuration.yaml")
    open("configuration.yaml", "w").write(yaml.dump(configuration))
