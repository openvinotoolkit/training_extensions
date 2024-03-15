"""This module implements the TaskEnvironment entity."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional, Type, TypeVar

from otx.api.configuration import ConfigurableParameters, cfg_helper
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.model_template import ModelTemplate

TypeVariable = TypeVar("TypeVariable", bound=ConfigurableParameters)


# pylint: disable=too-many-instance-attributes; Requires refactor
class TaskEnvironment:
    """Defines the machine learning environment the task runs in.

    Args:
    model_template (ModelTemplate): The model template used for this task
    model (Optional[ModelEntity]): Model to use; if not specified, the task must be either weight-less
        or use pre-trained or randomly initialised weights.
    hyper_parameters (ConfigurableParameters): Set of hyper parameters
    label_schema (LabelSchemaEntity): Label schema associated to this task
    """

    def __init__(
        self,
        model_template: ModelTemplate,
        model: Optional[ModelEntity],
        hyper_parameters: ConfigurableParameters,
        label_schema: LabelSchemaEntity,
    ):

        self.model_template = model_template
        self.model = model
        self.__hyper_parameters = hyper_parameters
        self.label_schema = label_schema

    def __repr__(self):
        """String representation of the TaskEnvironment object."""
        return (
            f"TaskEnvironment(model={self.model}, label_schema={self.label_schema}, "
            f"hyper_params={self.__hyper_parameters})"
        )

    def __eq__(self, other: object) -> bool:
        """Compares two TaskEnvironment objects.

        Args:
            other (TaskEnvironment): Object to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, TaskEnvironment):
            return (
                self.model == other.model
                and self.label_schema == other.label_schema
                # TODO get_hyperparameters should return Union rather than TypeVariable
                and self.get_hyper_parameters(instance_of=None)  # type: ignore
                == other.get_hyper_parameters(instance_of=None)
            )
        return False

    def get_labels(self, include_empty: bool = False) -> List[LabelEntity]:
        """Return the labels in this task environment (based on the label schema).

        Args:
            include_empty (bool): Include the empty label if ``True``. Defaults to False.

        Returns:
            List[LabelEntity]: List of labels
        """
        return self.label_schema.get_labels(include_empty)

    def get_hyper_parameters(self, instance_of: Optional[Type[TypeVariable]] = None) -> TypeVariable:
        """Returns Configuration for the task, de-serialized as type specified in `instance_of`.

        If the type of the configurable parameters is unknown, a generic
        ConfigurableParameters object with all available parameters can be obtained
        by calling method with instance_of = None.

        Example:
            >>> self.get_hyper_parameters(instance_of=TorchSegmentationConfig)
            TorchSegmentationConfig()

        Args:
            instance_of (Optional[Type[TypeVariable]]): subtype of ModelConfig of the hyperparamters. Defaults to None.

        Returns:
            TypeVariable: ConfigurableParameters entity
        """
        if instance_of is None:
            # If the instance_of is None, the type variable is not defined so the
            # return type won't be deduced correctly
            return self.__hyper_parameters  # type: ignore

        # Otherwise, update the base config according to what is stored in the repo.
        base_config = instance_of(header=self.__hyper_parameters.header)

        cfg_helper.substitute_values(base_config, value_input=self.__hyper_parameters, allow_missing_values=True)

        return base_config

    def set_hyper_parameters(self, hyper_parameters: ConfigurableParameters):
        """Sets the hyper parameters for the task.

        Example:
            >>> self.set_hyper_parameters(hyper_parameters=TorchSegmentationParameters())
            None
        Args:
            hyper_parameters (ConfigurationParameter): ConfigurableParameters entity to assign to task
        """
        if not isinstance(hyper_parameters, ConfigurableParameters):
            raise ValueError(f"Unable to set hyper parameters, invalid input: {hyper_parameters}")
        self.__hyper_parameters = hyper_parameters

    def get_model_configuration(self) -> ModelConfiguration:
        """Get the configuration needed to use the current model.

        That is the current set of:
            * configurable parameters
            * labels
            * label schema

        Returns:
            ModelConfiguration: Model configuration
        """
        return ModelConfiguration(self.__hyper_parameters, self.label_schema)
