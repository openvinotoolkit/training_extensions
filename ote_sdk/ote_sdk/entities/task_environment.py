"""This module implements the TaskEnvironment entity"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional, Type, TypeVar

from ote_sdk.configuration import ConfigurableParameters, ote_config_helper
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.entities.model_template import ModelTemplate
from ote_sdk.utils.argument_checks import (
    check_parameter_type,
    check_required_parameters_type,
)

TypeVariable = TypeVar("TypeVariable", bound=ConfigurableParameters)


# pylint: disable=too-many-instance-attributes; Requires refactor
class TaskEnvironment:
    """
    Defines the machine learning environment the task runs in.

    :param model_template: The model template used for this task
    :param model: Model to use; if not specified, the task must be either weight-less
        or use pre-trained or randomly initialised weights.
    :param hyper_parameters: Set of hyper parameters
    :param label_schema: Label schema associated to this task
    """

    def __init__(
        self,
        model_template: ModelTemplate,
        model: Optional[ModelEntity],
        hyper_parameters: ConfigurableParameters,
        label_schema: LabelSchemaEntity,
    ):
        # Initialization parameters validation
        check_required_parameters_type(
            [
                (model_template, "model_template", ModelTemplate),
                (hyper_parameters, "hyper_parameters", ConfigurableParameters),
                (label_schema, "label_schema", LabelSchemaEntity),
            ]
        )
        if model:
            check_parameter_type(
                parameter=model, parameter_name="model", expected_type=ModelEntity
            )

        self.model_template = model_template
        self.model = model
        self.__hyper_parameters = hyper_parameters
        self.label_schema = label_schema

    def __repr__(self):
        return (
            f"TaskEnvironment(model={self.model}, label_schema={self.label_schema}, "
            f"hyper_params={self.__hyper_parameters})"
        )

    def __eq__(self, other):
        if isinstance(other, TaskEnvironment):
            return (
                self.model == other.model
                and self.label_schema == other.label_schema
                and self.get_hyper_parameters(instance_of=None)
                == other.get_hyper_parameters(instance_of=None)
            )
        return False

    def get_labels(self, include_empty: bool = False) -> List[LabelEntity]:
        """
        Return the labels in this task environment (based on the label schema).

        :param include_empty: Include the empty label if ``True``
        """
        return self.label_schema.get_labels(include_empty)

    def get_hyper_parameters(
        self, instance_of: Optional[Type[TypeVariable]] = None
    ) -> TypeVariable:
        """
        Returns Configuration for the task, de-serialized as type specified in `instance_of`

        If the type of the configurable parameters is unknown, a generic
        ConfigurableParameters object with all available parameters can be obtained
        by calling method with instance_of = None.

        :example:
            >>> self.get_hyper_parameters(instance_of=TorchSegmentationConfig)
            TorchSegmentationConfig()

        :param instance_of: subtype of ModelConfig of the hyper paramters
        """
        if instance_of is None:
            # If the instance_of is None, the type variable is not defined so the
            # return type won't be deduced correctly
            return self.__hyper_parameters  # type: ignore

        # Otherwise, update the base config according to what is stored in the repo.
        base_config = instance_of(header=self.__hyper_parameters.header)

        ote_config_helper.substitute_values(
            base_config, value_input=self.__hyper_parameters, allow_missing_values=True
        )

        return base_config

    def set_hyper_parameters(self, hyper_parameters: ConfigurableParameters):
        """
        Sets the hyper parameters for the task

        :example:
            >>> self.set_hyper_parameters(hyper_parameters=TorchSegmentationParameters())
            None

        :param hyper_parameters: ConfigurableParameters entity to assign to task
        """
        if not isinstance(hyper_parameters, ConfigurableParameters):
            raise ValueError(
                f"Unable to set hyper parameters, invalid input: {hyper_parameters}"
            )
        self.__hyper_parameters = hyper_parameters

    def get_model_configuration(self) -> ModelConfiguration:
        """
        Get the configuration needed to use the current model.

        That is the current set of:

        * configurable parameters
        * labels
        * label schema

        :return: Model configuration
        """
        return ModelConfiguration(self.__hyper_parameters, self.label_schema)
