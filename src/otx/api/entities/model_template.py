"""This file defines the ModelConfiguration, ModelEntity and Model classes."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Dict, List, NamedTuple, Optional, Sequence, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from otx.api.configuration.elements import metadata_keys
from otx.api.configuration.enums import AutoHPOState
from otx.api.configuration.helper.utils import search_in_config_dict
from otx.api.entities.label import Domain


class TargetDevice(IntEnum):
    """Represents the target device for a given model.

    This device might be used for instance be used for training or inference.
    """

    UNSPECIFIED = auto()
    CPU = auto()
    GPU = auto()
    VPU = auto()


class ModelOptimizationMethod(Enum):
    """Optimized model format."""

    TENSORRT = auto()
    OPENVINO = auto()

    def __str__(self) -> str:
        """Returns ModelOptimizationMethod as string."""
        return str(self.name)


@dataclass
class DatasetRequirements:
    """Expected requirements for the dataset in order to use this algorithm.

    Attributes:
        classes (Optional[List[str]]): Classes which must be present in the dataset
    """

    classes: Optional[List[str]] = None


@dataclass
class ExportableCodePaths:
    """The paths to the different versions of the exportable code for a given model template."""

    default: Optional[str] = None
    openvino: Optional[str] = None


class TaskFamily(Enum):
    """Overall task family."""

    VISION = auto()
    FLOW_CONTROL = auto()
    DATASET = auto()

    def __str__(self) -> str:
        """Returns task family as a string."""
        return str(self.name)


class TaskInfo(NamedTuple):
    """Task information.

    NamedTuple to store information about the task type like label domain, if it is
    trainable, if it is an anomaly task and if it supports global or local labels.
    """

    domain: Domain
    is_trainable: bool
    is_anomaly: bool
    is_global: bool
    is_local: bool


class TaskType(Enum):
    """The type of algorithm within the task family.

    Also contains relevant information about the task type like label domain, if it is trainable,
    if it is an anomaly task or if it supports global or local labels.

    Args:
        value (int): (Unused) Unique integer for .value property of Enum (auto() does not work)
        task_info (TaskInfo): NamedTuple containing information about the task's capabilities
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        value: int,
        task_info: TaskInfo,
    ):
        self.domain = task_info.domain
        self.is_trainable = task_info.is_trainable
        self.is_anomaly = task_info.is_anomaly
        self.is_global = task_info.is_global
        self.is_local = task_info.is_local

    def __new__(cls, *args):
        """Returns new instance."""
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    NULL = 1, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    DATASET = 2, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    CLASSIFICATION = 3, TaskInfo(
        domain=Domain.CLASSIFICATION,
        is_trainable=True,
        is_anomaly=False,
        is_global=True,
        is_local=False,
    )
    SEGMENTATION = 4, TaskInfo(
        domain=Domain.SEGMENTATION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    DETECTION = 5, TaskInfo(
        domain=Domain.DETECTION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    ANOMALY_DETECTION = 6, TaskInfo(
        domain=Domain.ANOMALY_DETECTION,
        is_trainable=True,
        is_anomaly=True,
        is_global=False,
        is_local=True,
    )
    CROP = 7, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    TILE = 8, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    INSTANCE_SEGMENTATION = 9, TaskInfo(
        domain=Domain.INSTANCE_SEGMENTATION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    ACTIVELEARNING = 10, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    ANOMALY_SEGMENTATION = 11, TaskInfo(
        domain=Domain.ANOMALY_SEGMENTATION,
        is_trainable=True,
        is_anomaly=True,
        is_global=False,
        is_local=True,
    )
    ANOMALY_CLASSIFICATION = 12, TaskInfo(
        domain=Domain.ANOMALY_CLASSIFICATION,
        is_trainable=True,
        is_anomaly=True,
        is_global=True,
        is_local=False,
    )
    ROTATED_DETECTION = 13, TaskInfo(
        domain=Domain.ROTATED_DETECTION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
        ACTION_CLASSIFICATION = 14, TaskInfo(
            domain=Domain.ACTION_CLASSIFICATION,
            is_trainable=True,
            is_anomaly=False,
            is_global=False,
            is_local=True,
        )
        ACTION_DETECTION = 15, TaskInfo(
            domain=Domain.ACTION_DETECTION, is_trainable=True, is_anomaly=False, is_global=False, is_local=True
        )
    if os.getenv("FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS", "0") == "1":
        VISUAL_PROMPTING = 16, TaskInfo(  # TODO: Is 16 okay when action flag is False?
            domain=Domain.VISUAL_PROMPTING,
            is_trainable=True,
            is_anomaly=False,
            is_global=False,
            is_local=True,  # TODO: check whether is it local or not
        )

    def __str__(self) -> str:
        """Returns name."""
        return self.name

    def __repr__(self) -> str:
        """Returns name."""
        return self.name


def task_type_to_label_domain(task_type: TaskType) -> Domain:
    """Links the task type to the label domain enum.

    Note that not all task types have an associated domain (e.g. crop task).
    In this case, a ``ValueError`` is raised.

    Args:
        task_type (TaskType): The task type to get the label domain for.

    Returns:
        Domain: The label domain for the task type.
    """
    mapping = {
        TaskType.CLASSIFICATION: Domain.CLASSIFICATION,
        TaskType.DETECTION: Domain.DETECTION,
        TaskType.SEGMENTATION: Domain.SEGMENTATION,
        TaskType.INSTANCE_SEGMENTATION: Domain.INSTANCE_SEGMENTATION,
        TaskType.ANOMALY_CLASSIFICATION: Domain.ANOMALY_CLASSIFICATION,
        TaskType.ANOMALY_DETECTION: Domain.ANOMALY_DETECTION,
        TaskType.ANOMALY_SEGMENTATION: Domain.ANOMALY_SEGMENTATION,
        TaskType.ROTATED_DETECTION: Domain.ROTATED_DETECTION,
    }
    if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
        mapping = {
            **mapping,
            TaskType.ACTION_CLASSIFICATION: Domain.ACTION_CLASSIFICATION,
            TaskType.ACTION_DETECTION: Domain.ACTION_DETECTION,
        }
    if os.getenv("FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS", "0") == "1":
        mapping = {
            **mapping,
            TaskType.VISUAL_PROMPTING: Domain.VISUAL_PROMPTING,
        }

    try:
        return mapping[task_type]
    except KeyError as exc:
        raise ValueError(f"Task type {task_type} does not have any associated label domain.") from exc


@dataclass
class HyperParameterData:
    """HyperParameter Data.

    Class that contains the raw hyper parameter data, for those hyper parameters for the model that are
    user-configurable.

    Attributes:
        base_path (Optional[str]): The path to the yaml file specifying the base configurable parameters to use in the
            model. Defaults to None.
        parameter_overrides (Dict): Nested dictionary that describes overrides for the metadata for the
            user-configurable hyper parameters that are used in the model. This allows multiple models to share the
            same base hyper-parameters, while for each individual model the defaults, parameter ranges, descriptions,
            etc. can still be customized.
    """

    base_path: Optional[str] = None
    parameter_overrides: Dict = field(default_factory=dict)
    __data: Dict = field(default_factory=dict, repr=False)
    __has_valid_configurable_parameters: bool = field(default=False, repr=False)

    def load_parameters(self, model_template_path: str):
        """Load hyper parameters.

        Loads the actual hyper parameters defined in the file at `base_path`, and performs any overrides specified in
        the `parameter_overrides`.

        Args:
            model_template_path (str): file path to the model template file in which the HyperParameters live.
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
        """Returns a dictionary containing the set of hyper parameters defined in the ModelTemplate.

        This does not contain the actual parameter values, but instead holds the parameter schema's in
        a structured manner. The actual values should be either loaded from the database, or will be initialized from
        the defaults upon creating a configurable parameter object out of this data.
        """
        return self.__data

    @property
    def has_overrides(self) -> bool:
        """Returns True if any parameter overrides are defined by the HyperParameters instance, False otherwise."""
        return self.parameter_overrides != {}

    @property
    def has_valid_configurable_parameters(self) -> bool:
        """Check if configurable parameters are valid.

        Returns True if the HyperParameterData instance contains valid configurable parameters, extracted from the
        model template. False otherwise.
        """
        return self.__has_valid_configurable_parameters

    def substitute_parameter_overrides(self):
        """Carries out the parameter overrides specified in the `parameter_overrides` attribute.

        Validates whether the overridden parameters exist in the base set of configurable parameters,
        and whether the metadata values that should be overridden are valid metadata attributes.
        """
        self.__substitute_parameter_overrides(self.parameter_overrides, self.__data)

    def __substitute_parameter_overrides(self, override_dict: Dict, parameter_dict: Dict):
        """Substitutes parameters form override_dict into parameter_dict.

        Recursively substitutes overridden parameter values specified in `override_dict` into the base set of
        hyper parameters passed in as `parameter_dict`

        Args:
            override_dict (Dict): dictionary containing the parameter overrides
            parameter_dict (Dict): dictionary that contains the base set of hyper parameters, in which the overridden
                values are substituted
        """
        for key, value in override_dict.items():
            if isinstance(value, dict) and not metadata_keys.allows_dictionary_values(key):
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
                    raise KeyError(f"{key} is not a valid keyword for hyper parameter overrides")

    @classmethod
    def __remove_parameter_values_from_data(cls, data: dict):
        """This method removes the actual parameter values from the input parameter data.

        These values should be removed because the parameters should be instantiated
        from the default_values, instead of their values.

        NOTE: This method modifies its input dictionary, it does not return a new copy

        Args:
            data: Parameter dictionary to remove values from
        """
        data_copy = copy.deepcopy(data)
        for key, value in data_copy.items():
            if isinstance(value, dict):
                if key != metadata_keys.UI_RULES:
                    cls.__remove_parameter_values_from_data(data[key])
            elif key == "value":
                data.pop(key)

    def manually_set_data_and_validate(self, hyper_parameters: dict):
        """This function is used to manually set the hyper parameter data from a dictionary.

        It is meant to be used in testing only, in cases where the model
        template is not backed up by an actual yaml file.

        Args:
            hyper_parameters (Dict): Dictionary containing the data to be set
        """
        self.__data = hyper_parameters
        self.__has_valid_configurable_parameters = True


class InstantiationType(Enum):
    """The method to instantiate a given task."""

    NONE = auto()
    CLASS = auto()
    GRPC = auto()

    def __str__(self) -> str:
        """Returns the name of the instantiation type."""
        return str(self.name)


@dataclass
class Dependency:
    """Dependency required by the task.

    Attributes:
        source (str): Source of the dependency
        destination (str): Destination folder to install the dependency
        size (Optional[int]): Size of the dependency in bytes
        sha256 (Optional[str]): SHA-256 checksum of the dependency file
    """

    source: str
    destination: str
    size: Optional[int] = None
    sha256: Optional[str] = None


@dataclass
class EntryPoints:
    """Path of the Python classes implementing the task interface.

    Attributes:
        base (str): Base interface implementing the functionality in a framework such as PyTorch or TensorFlow
        openvino (Optional[str]): OpenVINO interface.
        nncf (Optional[str]): NNCF interface
    """

    base: str
    openvino: Optional[str] = None
    nncf: Optional[str] = None


class ModelCategory(Enum):
    """Represents model category regarding accuracy & speed trade-off."""

    SPEED = auto()
    BALANCE = auto()
    ACCURACY = auto()
    OTHER = auto()

    def __str__(self) -> str:
        """Returns the name of the model category."""
        return str(self.name)


class ModelStatus(Enum):
    """Represents model status regarding deprecation process."""

    ACTIVE = auto()
    DEPRECATED = auto()

    def __str__(self) -> str:
        """Returns the name of the model status."""
        return str(self.name)


# pylint: disable=too-many-instance-attributes
@dataclass
class ModelTemplate:
    """This class represents a Task in the Task database.

    It can be either a CLASS type, with the class path specified or a GRPC type with its address.
    The task chain uses this information to setup a `ChainLink` (A task in the chain)

    model_template_id (str): ID of the model template
    model_template_path (str): path to the original model template file
    name (str): user-friendly name for the algorithm used in the task
    task_family (TaskFamily): overall task family of the task. One of VISION, FLOW_CONTROL AND DATASET.
    task_type (TaskType): Type of algorithm within task family.
    instantiation (InstantiationType): InstantiationType (CLASS or GRPC)
    summary (str): Summary of what the algorithm does. Defaults to "".
    framework (Optional[str]): The framework used by the algorithm. Defaults to None.
    max_nodes (int): Max number of nodes for training. Defaults to 1.
    application (Optional[str]): Name of the application solved by this algorithm. Defaults to None.
    dependencies (Liar[Dependency]): List of dependencies required by the algorithm. Defaults to empty `field`.
    initial_weights (Optional[str]): Optional URL to the initial weights used by the algorithm. Defaults to None
    training_targets (List[TargetDevice]): device used for training. Defaults to empty `field`.
    inference_targets (List[TargetDevices]): device used for inference. Defaults to empty `field`.
    dataset_requirements (DatasetRequirements): list of dataset requirements. Defaults to empty `field`.
    model_optimization_methods (List[ModelOptimizationMethod]): list of ModelOptimizationMethod.
        This lists all methods available to optimize the inference model for the task
    hyper_parameters (HyperParameterData): HyperParameterData object containing the base path to the configurable
        parameter definition, as well as any overrides for the base parameters that are specific for the
        current template.
    is_trainable (bool): specify whether task is trainable
    capabilities (List[str]): list of task capabilities
    grpc_address (Optional[str]): the grpc host address (for instantiation type == GRPC)
    entrypoints (Optional[Entrypoints]): Entrypoints implementing the Python task interface
    base_model_path (str): Path to template file for the base model used for nncf compression.
    exportable_code_paths (ExportableCodePaths): if it exists, the path to the exportable code sources.
        Defaults to empty `field`.
    task_type_sort_priority (int): priority of order of how tasks are shown in the pipeline dropdown for a given task
        type. E.g. for classification Inception is default and has weight 0. Unassigned priority will have -1 as
        priority. mobilenet is less important, and has a higher value. Default is zero (the highest priority).
    gigaflops (float): how many billions of operations are required to do inference on a single data item.
    size (float): how much disk space the model will approximately take.
    model_category (ModelCategory): Represents model category regarding accuracy & speed trade-off. Default to OTHER.
    model_status (ModelStatus): Represents model status regarding deprecation process. Default to ACTIVE.
    is_default_for_task (bool): Whether this model is a default recommendation for the task
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
    dataset_requirements: DatasetRequirements = field(default_factory=DatasetRequirements)
    model_optimization_methods: List[ModelOptimizationMethod] = field(default_factory=list)
    hyper_parameters: HyperParameterData = field(default_factory=HyperParameterData)
    is_trainable: bool = True
    capabilities: List[str] = field(default_factory=list)
    grpc_address: Optional[str] = None
    entrypoints: Optional[EntryPoints] = None
    base_model_path: str = ""
    exportable_code_paths: ExportableCodePaths = field(default_factory=ExportableCodePaths)
    task_type_sort_priority: int = -1
    gigaflops: float = 0
    size: float = 0
    hpo: Optional[Dict] = None
    model_category: ModelCategory = ModelCategory.OTHER
    model_status: ModelStatus = ModelStatus.ACTIVE
    is_default_for_task: bool = False

    def __post_init__(self):
        """Do sanitation checks before loading the hyper-parameters."""
        if self.instantiation == InstantiationType.GRPC and self.grpc_address == "":
            raise ValueError("Task is registered as gRPC, but no gRPC address is specified")
        if self.instantiation == InstantiationType.CLASS and self.entrypoints is None:
            raise ValueError("Task is registered as CLASS, but entrypoints were not specified")
        if self.task_family == TaskFamily.VISION and self.hyper_parameters.base_path is None:
            raise ValueError("Task is registered as a VISION task but no hyper parameters were defined.")
        if self.task_family != TaskFamily.VISION and self.hyper_parameters.base_path is not None:
            raise ValueError("Hyper parameters are currently not supported for non-VISION tasks.")

        # Load the full hyper parameters
        self.hyper_parameters.load_parameters(self.model_template_path)

    def computes_uncertainty_score(self) -> bool:
        """Returns true if "compute_uncertainty_score" is in capabilities false otherwise."""
        return "compute_uncertainty_score" in self.capabilities

    def computes_representations(self) -> bool:
        """Returns true if "compute_representations" is in capabilities."""
        return "compute_representations" in self.capabilities

    def is_task_global(self) -> bool:
        """Returns ``True`` if the task is global task i.e. if task produces global labels."""
        return self.task_type.is_global

    def supports_auto_hpo(self) -> bool:
        """Returns `True` if the algorithm supports automatic hyper parameter optimization, `False` otherwise."""
        if not self.hyper_parameters.has_valid_configurable_parameters:
            return False
        auto_hpo_state_results = search_in_config_dict(
            self.hyper_parameters.data, key_to_search=metadata_keys.AUTO_HPO_STATE
        )
        for result in auto_hpo_state_results:
            if str(result[0]).lower() == str(AutoHPOState.POSSIBLE):
                return True
        return False


class NullModelTemplate(ModelTemplate):
    """Represent an empty model template. Note that a task based on this model template cannot be instantiated."""

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
    TaskType.INSTANCE_SEGMENTATION,
    TaskType.ANOMALY_DETECTION,
    TaskType.ANOMALY_CLASSIFICATION,
    TaskType.ANOMALY_SEGMENTATION,
    TaskType.ROTATED_DETECTION,
)
if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
    TRAINABLE_TASK_TYPES = (
        *TRAINABLE_TASK_TYPES,
        TaskType.ACTION_CLASSIFICATION,
        TaskType.ACTION_DETECTION,
    )
if os.getenv("FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS", "0") == "1":
    TRAINABLE_TASK_TYPES = (
        *TRAINABLE_TASK_TYPES,
        TaskType.VISUAL_PROMPTING,
    )


def _parse_model_template_from_omegaconf(config: Union[DictConfig, ListConfig]) -> ModelTemplate:
    """Parse an OmegaConf configuration into a model template.

    Args:
        config (Union[DictConfig, ListConfig]): The configuration to parse.

    Returns:
        ModelTemplate: The parsed model template.
    """
    schema = OmegaConf.structured(ModelTemplate)
    config = OmegaConf.merge(schema, config)
    return cast(ModelTemplate, OmegaConf.to_object(config))


def parse_model_template(model_template_path: str) -> ModelTemplate:
    """Read a model template from a file.

    Args:
        model_template_path (str): Path to the model template template.yaml file

    Returns:
        ModelTemplate: The model template parsed from the file.
    """
    config = OmegaConf.load(model_template_path)
    if not isinstance(config, DictConfig):
        raise ValueError("Expected the configuration file to contain a dictionary, not a list")

    if "model_template_id" not in config:
        config["model_template_id"] = config["name"].replace(" ", "_")
    config["model_template_path"] = model_template_path
    return _parse_model_template_from_omegaconf(config)


def parse_model_template_from_dict(model_template_dict: dict) -> ModelTemplate:
    """Read a model template from a dictionary.

    Note that the model_template_id must be defined inside the dictionary.

    Args:
        model_template_dict (dict): Dictionary containing the model template.

    Returns:
        ModelTemplate: The model template.
    """
    config = OmegaConf.create(model_template_dict)
    return _parse_model_template_from_omegaconf(config)
