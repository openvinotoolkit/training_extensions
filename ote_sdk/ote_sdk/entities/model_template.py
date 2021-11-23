"""This file defines the ModelConfiguration, ModelEntity and Model classes"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.
import copy
import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Dict, List, Optional, Sequence, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from ote_sdk.configuration.elements import metadata_keys
from ote_sdk.entities.label import Domain


class TargetDevice(IntEnum):
    """Represents the target device for a given model.

    This device might be used for instance be used for training or inference.
    """

    UNSPECIFIED = auto()
    CPU = auto()
    GPU = auto()
    VPU = auto()


class ModelOptimizationMethod(Enum):
    """
    Optimized model format.
    """

    TENSORRT = auto()
    OPENVINO = auto()

    def __str__(self) -> str:
        return str(self.name)


@dataclass
class DatasetRequirements:
    """
    Expected requirements for the dataset in order to use this algorithm.

    :var classes: Classes which must be present in the dataset
    """

    classes: Optional[List[str]] = None


@dataclass
class ExportableCodePaths:
    """
    The paths to the different versions of the exportable code for a given model template.
    """

    default: Optional[str] = None
    openvino: Optional[str] = None


class TaskFamily(Enum):
    """
    Overall task family.
    """

    VISION = auto()
    FLOW_CONTROL = auto()
    DATASET = auto()

    def __str__(self) -> str:
        return str(self.name)


class TaskType(Enum):
    """
    The type of algorithm within the task family.
    """

    NULL = auto()
    DATASET = auto()
    CLASSIFICATION = auto()
    SEGMENTATION = auto()
    DETECTION = auto()
    ANOMALY_DETECTION = auto()
    CROP = auto()
    TILE = auto()
    COUNTING = auto()
    ACTIVELEARNING = auto()
    ANOMALY_SEGMENTATION = auto()
    ANOMALY_CLASSIFICATION = auto()

    def __str__(self) -> str:
        return str(self.name)


def task_type_to_label_domain(task_type: TaskType) -> Domain:
    """
    Links the task type to the label domain enum.

    Note that not all task types have an associated domain (e.g. crop task).
    In this case, a ``ValueError`` is raised.
    """
    mapping = {
        TaskType.CLASSIFICATION: Domain.CLASSIFICATION,
        TaskType.DETECTION: Domain.DETECTION,
        TaskType.SEGMENTATION: Domain.SEGMENTATION,
        TaskType.COUNTING: Domain.DETECTION,
        TaskType.ANOMALY_CLASSIFICATION: Domain.ANOMALY_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION: Domain.ANOMALY_DETECTION,
        TaskType.ANOMALY_SEGMENTATION: Domain.ANOMALY_SEGMENTATION,
    }

    try:
        return mapping[task_type]
    except KeyError as exc:
        raise ValueError(
            f"Task type {task_type} does not have any associated label domain."
        ) from exc


@dataclass
class HyperParameterData:
    """
    Class that contains the raw hyper parameter data, for those hyper parameters for the model that are
    user-configurable.

    :var base_path: The path to the yaml file specifying the base configurable parameters to use in the model
    :var parameter_overrides: Nested dictionary that describes overrides for the metadata for the user-configurable
        hyper parameters that are used in the model. This allows multiple models to share the same base hyper
        parameters, while for each individual model the defaults, parameter ranges, descriptions, etc. can still
        be customized.
    """

    base_path: Optional[str] = None
    parameter_overrides: Dict = field(default_factory=dict)
    __data: Dict = field(default_factory=dict, repr=False)
    __has_valid_configurable_parameters: bool = field(default=False, repr=False)

    def load_parameters(self, model_template_path: str):
        """
        Loads the actual hyper parameters defined in the file at `base_path`, and performs any overrides specified in
        the `parameter_overrides`.

        :param model_template_path: file path to the model template file in which the HyperParameters live.
        """
        has_valid_configurable_parameters = False
        if self.base_path is not None and os.path.exists(model_template_path):
            model_template_dir = os.path.dirname(model_template_path)
            base_hyper_parameter_path = os.path.join(model_template_dir, self.base_path)

            config_dict = OmegaConf.load(base_hyper_parameter_path)
            data = OmegaConf.to_container(config_dict)
            if isinstance(data, dict):
                self.__remove_parameter_values_from_data(data)
                self.__data = data
                has_valid_configurable_parameters = True
            else:
                raise ValueError(
                    f"Unexpected configurable parameter file found at path {base_hyper_parameter_path}"
                    f", expected a dictionary-like format, got list-like instead."
                )
        if self.has_overrides and has_valid_configurable_parameters:
            self.substitute_parameter_overrides()
        self.__has_valid_configurable_parameters = has_valid_configurable_parameters

    @property
    def data(self) -> Dict:
        """
        Returns a dictionary containing the set of hyper parameters defined in the ModelTemplate. This does not
        contain the actual parameter values, but instead holds the parameter schema's in a structured manner. The
        actual values should be either loaded from the database, or will be initialized from the defaults upon
        creating a configurable parameter object out of this data
        """
        return self.__data

    @property
    def has_overrides(self) -> bool:
        """
        Returns True if any parameter overrides are defined by the HyperParameters instance, False otherwise
        """
        return self.parameter_overrides != {}

    @property
    def has_valid_configurable_parameters(self) -> bool:
        """
        Returns True if the HyperParameterData instance contains valid configurable
        parameters, extracted from the model template. False otherwise.
        """
        return self.__has_valid_configurable_parameters

    def substitute_parameter_overrides(self):
        """
        Carries out the parameter overrides specified in the `parameter_overrides` attribute. Validates whether the
        overridden parameters exist in the base set of configurable parameters, and whether the metadata values that
        should be overridden are valid metadata attributes.
        """
        self.__substitute_parameter_overrides(self.parameter_overrides, self.__data)

    def __substitute_parameter_overrides(
        self, override_dict: Dict, parameter_dict: Dict
    ):
        """
        Recursively substitutes overridden parameter values specified in `override_dict` into the base set of
        hyper parameters passed in as `parameter_dict`

        :param override_dict: dictionary containing the parameter overrides
        :param parameter_dict: dictionary that contains the base set of hyper parameters, in which the overridden
            values are substituted
        """
        for key, value in override_dict.items():
            if isinstance(value, dict) and not metadata_keys.allows_dictionary_values(
                key
            ):
                if key in parameter_dict.keys():
                    self.__substitute_parameter_overrides(value, parameter_dict[key])
                else:
                    raise ValueError(
                        f"Unable to perform parameter override. Parameter or parameter group named {key} "
                        f"is not valid for the base hyper parameters specified in {self.base_path}"
                    )
            else:
                if metadata_keys.allows_model_template_override(key):
                    parameter_dict[key] = value
                else:
                    raise KeyError(
                        f"{key} is not a valid keyword for hyper parameter overrides"
                    )

    @classmethod
    def __remove_parameter_values_from_data(cls, data: dict):
        """
        This method removes the actual parameter values from the input parameter data.
        These values should be removed because the parameters should be instantiated
        from the default_values, instead of their values.

        NOTE: This method modifies its input dictionary, it does not return a new copy

        :param data: Parameter dictionary to remove values from
        """
        data_copy = copy.deepcopy(data)
        for key, value in data_copy.items():
            if isinstance(value, dict):
                if key != metadata_keys.UI_RULES:
                    cls.__remove_parameter_values_from_data(data[key])
            elif key == "value":
                data.pop(key)

    def manually_set_data_and_validate(self, hyper_parameters: dict):
        """
        This function is used to manually set the hyper parameter data from a
        dictionary. It is meant to be used in testing only, in cases where the model
        template is not backed up by an actual yaml file.

        :param hyper_parameters: Dictionary containing the data to be set
        """
        self.__data = hyper_parameters
        self.__has_valid_configurable_parameters = True


class InstantiationType(Enum):
    """
    The method to instantiate a given task.
    """

    NONE = auto()
    CLASS = auto()
    GRPC = auto()

    def __str__(self) -> str:
        return str(self.name)


@dataclass
class Dependency:
    """
    Dependency required by the task.

    :var source: Source of the dependency
    :var destination: Destination folder to install the dependency
    :var size: Size of the dependency in bytes
    :var sha256: SHA-256 checksum of the dependency file
    """

    source: str
    destination: str
    size: Optional[int] = None
    sha256: Optional[str] = None


@dataclass
class EntryPoints:
    """
    Path of the Python classes implementing the task interface.

    :var base: Base interface implementing the functionality in a framework such as PyTorch or TensorFlow
    :var openvino: OpenVINO interface
    :var nncf: NNCF interface
    """

    base: str
    openvino: Optional[str] = None
    nncf: Optional[str] = None


# pylint: disable=too-many-instance-attributes
@dataclass
class ModelTemplate:
    """
    This class represents a Task in the Task database. It can be either a CLASS type,
    with the class path specified or a GRPC type with its address.
    The task chain uses this information to setup a `ChainLink` (A task in the chain)

    :var model_template_id: ID of the model template
    :var model_template_path: path to the original model template file
    :var name: user-friendly name for the algorithm used in the task
    :var summary: Summary of what the algorithm does
    :var application: Name of the application solved by this algorithm
    :var framework: The framework used by the algorithm
    :var max_nodes: Max number of nodes for training
    :var initial_weights: Optional URL to the initial weights used by the algorithm
    :var is_trainable: specify whether task is trainable
    :var training_targets: device used for training
    :var inference_targets: device used for inference
    :var dataset_requirements: list of dataset requirements
    :var capabilities: list of task capabilities
    :var instantiation: InstantiationType (CLASS or GRPC)
    :var hyper_parameters: HyperParameterData object containing the base path to the configurable parameter definition,
        as well as any overrides for the base parameters that are specific for the current template
    :var grpc_address: the grpc host address (for instantiation type == GRPC)
    :var entrypoints: Entrypoints implementing the Python task interface
    :var exportable_code_expression: if it exists, the path to the exportable code sources
    :var task_type_sort_priority: priority of order of how tasks are shown in the pipeline dropdown
        for a given task type. E.g. for classification Inception is default and has weight 0.
        Unassigned priority will have -1 as priority.
        mobilenet is less important, and has a higher value. Default is zero (the highest priority).
    :var model_optimization_methods: list of ModelOptimizationMethod. This lists all methods available
        to optimize the inference model for the task
    :var gigaflops: how many billions of operations are required to do inference on a single data item
    :var size: how much disk space the model will approximately take
    """

    model_template_id: str
    model_template_path: str
    name: str
    task_family: TaskFamily
    task_type: TaskType
    instantiation: InstantiationType
    summary: str = ""
    framework: Optional[str] = None
    max_nodes: int = 1
    application: Optional[str] = None
    dependencies: List[Dependency] = field(default_factory=list)
    initial_weights: Optional[str] = None
    training_targets: List[TargetDevice] = field(default_factory=list)
    inference_targets: List[TargetDevice] = field(default_factory=list)
    dataset_requirements: DatasetRequirements = field(
        default_factory=DatasetRequirements
    )
    model_optimization_methods: List[ModelOptimizationMethod] = field(
        default_factory=list
    )
    hyper_parameters: HyperParameterData = field(default_factory=HyperParameterData)
    is_trainable: bool = True
    capabilities: List[str] = field(default_factory=list)
    grpc_address: Optional[str] = None
    entrypoints: Optional[EntryPoints] = None
    exportable_code_paths: ExportableCodePaths = field(
        default_factory=ExportableCodePaths
    )
    task_type_sort_priority: int = -1
    gigaflops: float = 0
    size: float = 0

    def __post_init__(self):
        if self.instantiation == InstantiationType.GRPC and self.grpc_address == "":
            raise ValueError(
                "Task is registered as gRPC, but no gRPC address is specified"
            )
        if self.instantiation == InstantiationType.CLASS and self.entrypoints is None:
            raise ValueError(
                "Task is registered as CLASS, but entrypoints were not specified"
            )
        if (
            self.task_family == TaskFamily.VISION
            and self.hyper_parameters.base_path is None
        ):
            raise ValueError(
                "Task is registered as a VISION task but no hyper parameters were defined."
            )
        if (
            self.task_family != TaskFamily.VISION
            and self.hyper_parameters.base_path is not None
        ):
            raise ValueError(
                "Hyper parameters are currently not supported for non-VISION tasks."
            )

        # Load the full hyper parameters
        self.hyper_parameters.load_parameters(self.model_template_path)

    def computes_uncertainty_score(self) -> bool:
        """
        Returns true if "compute_uncertainty_score" is in capabilities

        :return: true if "compute_uncertainty_score" is in capabilities, false otherwise
        """
        return "compute_uncertainty_score" in self.capabilities

    def computes_representations(self) -> bool:
        """
        Returns true if "compute_representations" is in capabilities

        :return: true if "compute_representations" is in capabilities, false otherwise
        """
        return "compute_representations" in self.capabilities

    def is_task_global(self) -> bool:
        """
        Returns ``True`` if the task is global task i.e. if task produces global labels
        """
        return self.task_type in [TaskType.CLASSIFICATION]


class NullModelTemplate(ModelTemplate):
    """
    Represent an empty model template. Note that a task based on this model template cannot be instantiated.
    """

    def __init__(self) -> None:
        super().__init__(
            model_template_id="",
            model_template_path="",
            task_family=TaskFamily.FLOW_CONTROL,
            task_type=TaskType.NULL,
            name="Null algorithm",
            instantiation=InstantiationType.NONE,
            capabilities=[],
        )


ANOMALY_TASK_TYPES: Sequence[TaskType] = (
    TaskType.ANOMALY_DETECTION,
    TaskType.ANOMALY_CLASSIFICATION,
    TaskType.ANOMALY_SEGMENTATION,
)


TRAINABLE_TASK_TYPES: Sequence[TaskType] = (
    TaskType.CLASSIFICATION,
    TaskType.DETECTION,
    TaskType.SEGMENTATION,
    TaskType.COUNTING,
    TaskType.ANOMALY_DETECTION,
    TaskType.ANOMALY_CLASSIFICATION,
    TaskType.ANOMALY_SEGMENTATION,
)


def _parse_model_template_from_omegaconf(
    config: Union[DictConfig, ListConfig]
) -> ModelTemplate:
    """
    Parse an OmegaConf configuration into a model template.
    """
    schema = OmegaConf.structured(ModelTemplate)
    config = OmegaConf.merge(schema, config)
    return cast(ModelTemplate, OmegaConf.to_object(config))


def parse_model_template(model_template_path: str) -> ModelTemplate:
    """
    Read a model template from a file.

    :param model_template_path: Path to the model template template.yaml file
    """
    config = OmegaConf.load(model_template_path)
    if not isinstance(config, DictConfig):
        raise ValueError(
            "Expected the configuration file to contain a dictionary, not a list"
        )

    if "model_template_id" not in config:
        config["model_template_id"] = config["name"].replace(" ", "_")
    config["model_template_path"] = model_template_path
    return _parse_model_template_from_omegaconf(config)


def parse_model_template_from_dict(model_template_dict: dict) -> ModelTemplate:
    """
    Read a model template from a dictionary.

    Note that the model_template_id must be defined inside the dictionary.
    """
    config = OmegaConf.create(model_template_dict)
    return _parse_model_template_from_omegaconf(config)
