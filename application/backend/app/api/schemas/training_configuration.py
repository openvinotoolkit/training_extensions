#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import types
from enum import StrEnum
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel, Discriminator, Field, Tag
from pydantic.fields import FieldInfo

from app.models.training_configuration import ParamValueType, TrainingConfiguration


class ConfigurableParameterViewElementType(StrEnum):
    PARAMETER = "parameter"
    PARAMETER_GROUP = "parameter_group"


class _BaseConfigurableParameterView(BaseModel):
    """Base fields shared by all parameter view variants."""

    type: Literal[ConfigurableParameterViewElementType.PARAMETER] = ConfigurableParameterViewElementType.PARAMETER
    key: str = Field(title="Key to identify the parameter")
    name: str = Field(title="User-friendly name of the parameter")
    description: str = Field(title="Extended description of the parameter", default="")
    depends_on: dict[str, Any] | None = Field(
        default=None,
        title="Dependency condition",
        description=(
            "If set, this parameter is only applicable when the specified sibling parameter has the given value. "
            "For example, {'type': 'cosine_annealing'} means this parameter only applies when the 'type' "
            "parameter in the same group is set to 'cosine_annealing'."
        ),
    )


class BoolParameterView(_BaseConfigurableParameterView):
    """Configurable boolean parameter."""

    value_type: Literal["bool"] = "bool"
    value: bool = Field(title="Actual value of the parameter")
    default_value: bool = Field(title="Default value of the parameter")


class StringParameterView(_BaseConfigurableParameterView):
    """Configurable string parameter."""

    value_type: Literal["str"] = "str"
    value: str = Field(title="Actual value of the parameter")
    default_value: str = Field(title="Default value of the parameter")
    allowed_values: list[str] | None = Field(
        default=None,
        title="List of allowed values for the parameter. None if it doesn't have a predefined set of valid values.",
    )


class IntParameterView(_BaseConfigurableParameterView):
    """Configurable integer parameter with optional min/max bounds."""

    value_type: Literal["int"] = "int"
    value: int = Field(title="Actual value of the parameter")
    default_value: int = Field(title="Default value of the parameter")
    min_value: int | float | None = Field(default=None, title="Minimum value for numeric parameters. None if unbounded")
    max_value: int | float | None = Field(default=None, title="Maximum value for numeric parameters. None if unbounded")
    allowed_values: list[int] | None = Field(
        default=None,
        title="List of allowed values for the parameter. None if it doesn't have a predefined set of valid values.",
    )


class FloatParameterView(_BaseConfigurableParameterView):
    """Configurable float parameter with optional min/max bounds."""

    value_type: Literal["float"] = "float"
    value: float = Field(title="Actual value of the parameter")
    default_value: float = Field(title="Default value of the parameter")
    min_value: int | float | None = Field(default=None, title="Minimum value for numeric parameters. None if unbounded")
    max_value: int | float | None = Field(default=None, title="Maximum value for numeric parameters. None if unbounded")
    allowed_values: list[float] | None = Field(
        default=None,
        title="List of allowed values for the parameter. None if it doesn't have a predefined set of valid values.",
    )


class FloatRangeParameterView(_BaseConfigurableParameterView):
    """Configurable float range parameter."""

    value_type: Literal["float_range"] = "float_range"
    value: tuple[float, float] = Field(title="Actual value of the parameter")
    default_value: tuple[float, float] = Field(title="Default value of the parameter")
    min_value: float = Field(title="Minimum value for range elements.")
    max_value: float = Field(title="Maximum value for range elements.")


def _parameter_view_discriminator(v: dict | _BaseConfigurableParameterView) -> str:
    if isinstance(v, dict):
        return v.get("value_type", "str")
    return getattr(v, "value_type", "str")


ConfigurableParameterView = Annotated[
    Annotated[BoolParameterView, Tag("bool")]
    | Annotated[StringParameterView, Tag("str")]
    | Annotated[IntParameterView, Tag("int")]
    | Annotated[FloatParameterView, Tag("float")]
    | Annotated[FloatRangeParameterView, Tag("float_range")],
    Discriminator(_parameter_view_discriminator),
]


class ConfigurableParameterGroupView(BaseModel):
    """A group of related parameters, which can contain both individual parameters and nested parameter groups."""

    type: Literal[ConfigurableParameterViewElementType.PARAMETER_GROUP] = (
        ConfigurableParameterViewElementType.PARAMETER_GROUP
    )
    key: str = Field(title="Key to identify the parameter group")
    name: str = Field(title="User-friendly name of the parameter group")
    description: str = Field(title="Extended description of the parameter group", default="")
    depends_on: dict[str, Any] | None = Field(
        default=None,
        title="Dependency condition",
        description=(
            "If set, this parameter group is only applicable when the specified sibling parameter has the given value."
        ),
    )
    parameters: list[ConfigurableParameterView | "ConfigurableParameterGroupView"] = Field(
        title="List of parameters in the group"
    )


class TrainingConfigurationView(BaseModel):
    """
    Configuration of training parameters, structured as a list of parameter groups and parameters that can be tuned by
    the user to control the training process.
    """

    parameters: list[ConfigurableParameterView | ConfigurableParameterGroupView] = Field(
        title="Training configuration parameters",
        description=(
            "List of configurable parameters and parameter groups for training. "
            "Parameters are organized hierarchically into groups based on their functions and relative similarities. "
            "Each parameter includes metadata such as type, default value, and constraints to guide user input."
        ),
    )

    @classmethod
    def _get_literal_strenum_class(cls, annotation: type) -> type[StrEnum] | None:
        """If annotation is Literal[<StrEnum member>, ...], return the StrEnum class. Otherwise return None."""
        if get_origin(annotation) is not Literal:
            return None
        args = get_args(annotation)
        if args and isinstance(args[0], StrEnum):
            return type(args[0])
        return None

    @classmethod
    def _extract_constraints(cls, field_info: FieldInfo) -> tuple[float | None, float | None]:
        """Extract min/max constraints from field metadata."""
        min_value = None
        max_value = None

        if hasattr(field_info, "metadata"):
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    min_value = constraint.ge
                elif hasattr(constraint, "gt"):
                    min_value = constraint.gt
                if hasattr(constraint, "le"):
                    max_value = constraint.le
                elif hasattr(constraint, "lt"):
                    max_value = constraint.lt

        return min_value, max_value

    @classmethod
    def _extract_range_bounds(cls, field_info: FieldInfo) -> tuple[float, float]:
        """Extract min_value/max_value bounds for float_range fields from json_schema_extra."""
        if not isinstance(field_info.json_schema_extra, dict):
            raise ValueError(
                f"Field '{field_info.title}' must have a json_schema_extra to define min and max range parameters"
            )
        min_value = field_info.json_schema_extra["min_value"]
        max_value = field_info.json_schema_extra["max_value"]
        if not isinstance(min_value, float) or not isinstance(max_value, float):
            raise ValueError(
                f"Field '{field_info.title}' must have float min and max in json_schema_extra for range parameters"
            )
        return min_value, max_value

    @classmethod
    def _get_value_type(cls, field_info: FieldInfo) -> Literal["bool", "int", "float", "str", "float_range"]:  # noqa: C901, PLR0911
        """Determine the value type from field annotation."""
        annotation = field_info.annotation

        # Handle Optional/Union types by extracting the non-None type
        origin = get_origin(annotation)
        if origin is Union or origin is types.UnionType:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if args:
                annotation = args[0]
                origin = get_origin(annotation)

        if hasattr(annotation, "__origin__") or origin is not None:
            # Detect Literal[StrEnum member] -> treat as str
            if cls._get_literal_strenum_class(annotation) is not None:  # pyrefly: ignore[bad-argument-type]
                return "str"

            # Detect tuple[float, float]
            if origin is tuple:
                args = get_args(annotation)
                if len(args) == 2 and all(a is float for a in args):
                    return "float_range"

        # Check if annotation is a StrEnum subclass -> treat as str
        if isinstance(annotation, type) and issubclass(annotation, StrEnum):
            return "str"

        # Map Python types to string representations
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is str:
            return "str"
        # Default to str for unknown types
        return "str"

    @classmethod
    def _extract_depends_on(cls, field_info: FieldInfo) -> dict[str, Any] | None:
        """Extract depends_on condition from field's json_schema_extra."""
        if isinstance(field_info.json_schema_extra, dict):
            depends_on = field_info.json_schema_extra.get("depends_on")
            if isinstance(depends_on, dict):
                return depends_on  # pyrefly: ignore[bad-return]
        return None

    @classmethod
    def _field_to_configurable_parameter(
        cls,
        key: str,
        value: ParamValueType | None,
        default_value: ParamValueType | None,
        field_info: FieldInfo,
        allowed_values: list | None = None,
    ) -> ConfigurableParameterView:
        """Convert a single field to the appropriate ConfigurableParameterView variant."""
        value_type = cls._get_value_type(field_info)

        # For StrEnum fields, derive title and description from the enum class
        annotation = field_info.annotation
        enum_class = None

        # Check Literal[StrEnum member]
        if annotation is not None:
            enum_class = cls._get_literal_strenum_class(annotation)

        # Check plain StrEnum subclass
        if enum_class is None and isinstance(annotation, type) and issubclass(annotation, StrEnum):
            enum_class = annotation

        if enum_class is not None:
            # Try to get title/description from enum member instance
            literal_args = get_args(annotation) if get_origin(annotation) is Literal else None
            if literal_args and isinstance(literal_args[0], StrEnum):
                member = literal_args[0]
                title = getattr(member, "title", None) or field_info.title
                description = getattr(member, "description", None) or field_info.description or ""
            else:
                # Plain StrEnum annotation — try class-level attributes or field info
                title = field_info.title
                description = field_info.description or ""
        else:
            title = field_info.title
            description = field_info.description or ""

        if title is None:
            raise ValueError(
                f"Field '{key}' is missing a title in its FieldInfo, "
                f"which is required to associate a user-friendly name to the parameter."
            )

        depends_on = cls._extract_depends_on(field_info)

        common_kwargs = {
            "key": key,
            "name": title,
            "description": description,
            "value": value,
            "default_value": default_value,
            "depends_on": depends_on,
        }

        if value_type == "int":
            min_value, max_value = cls._extract_constraints(field_info)
            return IntParameterView(
                **common_kwargs,  # type: ignore
                min_value=min_value,
                max_value=max_value,
                allowed_values=allowed_values,
            )
        if value_type == "float":
            min_value, max_value = cls._extract_constraints(field_info)
            return FloatParameterView(
                **common_kwargs,  # type: ignore
                min_value=min_value,
                max_value=max_value,
                allowed_values=allowed_values,
            )
        if value_type == "bool":
            return BoolParameterView(**common_kwargs)  # type: ignore
        if value_type == "float_range":
            min_value, max_value = cls._extract_range_bounds(field_info)
            return FloatRangeParameterView(**common_kwargs, min_value=min_value, max_value=max_value)  # type: ignore
        return StringParameterView(**common_kwargs, allowed_values=allowed_values)  # type: ignore

    @classmethod
    def _resolve_allowed_values(cls, model: BaseModel, field_info: FieldInfo) -> list | None:
        """Resolve allowed_values for a field from json_schema_extra or StrEnum annotation."""
        allowed_values = None

        # Check json_schema_extra for allowed_values_from
        if isinstance(field_info.json_schema_extra, dict):
            allowed_values_from = cast(str | None, field_info.json_schema_extra.get("allowed_values_from"))
            if allowed_values_from is not None:
                allowed_values = getattr(model, allowed_values_from, None)
            read_only = cast(bool | None, field_info.json_schema_extra.get("read_only"))
            if read_only:
                allowed_values = [str(field_info.default)]

        if allowed_values is None:
            annotation = field_info.annotation

            # Check if annotation is a plain StrEnum subclass
            if isinstance(annotation, type) and issubclass(annotation, StrEnum):
                allowed_values = [member.value for member in annotation]

            # Check if annotation is Literal[<StrEnum member>, ...]
            elif annotation is not None:
                enum_class = cls._get_literal_strenum_class(annotation)
                if enum_class is not None:
                    allowed_values = [member.value for member in enum_class]

        return allowed_values

    @classmethod
    def _model_to_parameter_group(
        cls, key: str, model: BaseModel, default_model: BaseModel | None, field_info: FieldInfo
    ) -> ConfigurableParameterGroupView:
        """Convert a Pydantic model to a ConfigurableParameterGroupView with nested parameters."""
        parameters: list[ConfigurableParameterView | ConfigurableParameterGroupView] = []

        for field_name, child_field_info in type(model).model_fields.items():
            # Skip validation-only fields
            if isinstance(child_field_info.json_schema_extra, dict) and child_field_info.json_schema_extra.get(
                "validation_only"
            ):
                continue

            value = getattr(model, field_name)

            # Skip parameters with None values
            if value is None:
                continue

            if isinstance(value, BaseModel):
                # Nested model -> create a nested parameter group
                default_child = getattr(default_model, field_name) if default_model is not None else None
                nested_group = cls._model_to_parameter_group(field_name, value, default_child, child_field_info)
                # Only add the group if it has parameters (not empty due to None filtering)
                if nested_group.parameters:
                    parameters.append(nested_group)
            else:
                # Resolve allowed_values from another field or StrEnum annotation
                allowed_values = cls._resolve_allowed_values(model, child_field_info)

                # Get the default value from the default model, falling back to field_info.default
                if default_model is not None:
                    default_value = getattr(default_model, field_name, child_field_info.default)
                else:
                    default_value = child_field_info.default

                # Scalar value -> create a parameter
                parameters.append(
                    cls._field_to_configurable_parameter(
                        field_name, value, default_value, child_field_info, allowed_values
                    )
                )

        if field_info.title is None:
            raise ValueError(
                f"Field '{key}' is missing a title in its FieldInfo, "
                f"which is required to associate a user-friendly name to the parameter group."
            )

        return ConfigurableParameterGroupView(
            key=key,
            name=field_info.title,
            description=field_info.description if field_info.description else "",
            depends_on=cls._extract_depends_on(field_info),
            parameters=parameters,
        )

    @classmethod
    def _merge_parameter_groups(cls, *groups: ConfigurableParameterGroupView) -> ConfigurableParameterGroupView:
        """Merge multiple parameter groups with the same key into one."""
        if not groups:
            raise ValueError("At least one group is required")

        # Use the first group as base
        base = groups[0]
        merged_parameters: list[ConfigurableParameterView | ConfigurableParameterGroupView] = []

        # Collect all parameters by key for merging
        params_by_key: dict[str, list[ConfigurableParameterView | ConfigurableParameterGroupView]] = {}
        for group in groups:
            for param in group.parameters:
                if param.key not in params_by_key:
                    params_by_key[param.key] = []
                params_by_key[param.key].append(param)

        # Merge parameters with same key, or just add unique ones
        for key, params in params_by_key.items():
            if len(params) == 1:
                merged_parameters.append(params[0])
            # Multiple params with same key - they should all be groups to merge
            elif all(isinstance(p, ConfigurableParameterGroupView) for p in params):
                merged_parameters.append(cls._merge_parameter_groups(*params))  # type: ignore
            else:
                # If not all groups, just take the first one
                merged_parameters.append(params[0])

        return ConfigurableParameterGroupView(
            key=base.key,
            name=base.name,
            description=base.description,
            parameters=merged_parameters,
        )

    @classmethod
    def from_training_configuration(
        cls, config: TrainingConfiguration, default_config: TrainingConfiguration
    ) -> "TrainingConfigurationView":
        """Convert TrainingConfiguration to TrainingConfigurationView.

        Args:
            config: The current training configuration with actual parameter values.
            default_config: The default training configuration, used to populate default_value fields.
        """

        # Convert task-level dataset_preparation to a parameter group
        task_dataset_prep = cls._model_to_parameter_group(
            "dataset_preparation",
            config.task_level_parameters.dataset_preparation,
            default_config.task_level_parameters.dataset_preparation,
            type(config.task_level_parameters).model_fields["dataset_preparation"],
        )

        # Convert model-level dataset_preparation to a parameter group
        model_dataset_prep = cls._model_to_parameter_group(
            "dataset_preparation",
            config.algo_level_parameters.dataset_preparation,
            default_config.algo_level_parameters.dataset_preparation,
            type(config.algo_level_parameters).model_fields["dataset_preparation"],
        )

        # Merge both dataset_preparation groups
        merged_dataset_prep = cls._merge_parameter_groups(task_dataset_prep, model_dataset_prep)

        # Convert training parameters to a parameter group
        training_group = cls._model_to_parameter_group(
            "training",
            config.algo_level_parameters.training,
            default_config.algo_level_parameters.training,
            type(config.algo_level_parameters).model_fields["training"],
        )

        # Convert evaluation parameters to a parameter group
        evaluation_group = cls._model_to_parameter_group(
            "evaluation",
            config.task_level_parameters.evaluation,
            default_config.task_level_parameters.evaluation,
            type(config.task_level_parameters).model_fields["evaluation"],
        )

        # Return with direct list of parameter groups
        return cls(parameters=[merged_dataset_prep, training_group, evaluation_group])

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameters": [
                    {
                        "type": "parameter_group",
                        "key": "dataset_preparation",
                        "name": "Dataset preparation",
                        "description": (
                            "Configurable parameters related to the training data, such as augmentations and filters."
                        ),
                        "parameters": [
                            {
                                "type": "parameter_group",
                                "key": "subset_split",
                                "name": "Subset split",
                                "description": (
                                    "Subset split parameters define how the dataset is divided into training, "
                                    "validation, and test subsets. The training subset is used to fit the model, "
                                    "the validation subset is used to estimate the prediction error during training "
                                    "and the test subset is used to evaluate the final performance of the model. "
                                    "The percentages for training, validation, and test subsets must sum to 100."
                                ),
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "training",
                                        "name": "Training percentage",
                                        "description": "Percentage of data to use for training",
                                        "value": 70,
                                        "default_value": 70,
                                        "value_type": "int",
                                        "min_value": 1,
                                        "max_value": 100,
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "validation",
                                        "name": "Validation percentage",
                                        "description": "Percentage of data to use for validation",
                                        "value": 20,
                                        "default_value": 20,
                                        "value_type": "int",
                                        "min_value": 1,
                                        "max_value": 100,
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "test",
                                        "name": "Test percentage",
                                        "description": "Percentage of data to use for testing",
                                        "value": 10,
                                        "default_value": 10,
                                        "value_type": "int",
                                        "min_value": 1,
                                        "max_value": 100,
                                        "allowed_values": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "filtering",
                                "name": "Filtering",
                                "description": (
                                    "Filtering parameters define criteria for including or excluding annotations "
                                    "from the dataset. Depending on the scenario, an appropriate filter configuration "
                                    "can speed up the training process and/or improve the model performance by "
                                    "removing noisy annotations."
                                ),
                                "parameters": [
                                    {
                                        "type": "parameter_group",
                                        "key": "min_annotation_pixels",
                                        "name": "Minimum annotation pixels",
                                        "description": "Minimum number of pixels in an annotation",
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable minimum annotation pixels filtering",
                                                "description": "Whether to apply minimum annotation pixels filtering",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "value",
                                                "name": "Minimum annotation pixels",
                                                "description": "Minimum number of pixels in an annotation",
                                                "value": 1,
                                                "default_value": 1,
                                                "value_type": "int",
                                                "min_value": 0,
                                                "max_value": 200000000,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "min_annotation_objects",
                                        "name": "Minimum annotation objects",
                                        "description": "Minimum number of objects in an annotation",
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable minimum annotation objects filtering",
                                                "description": "Whether to apply minimum annotation objects filtering",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "value",
                                                "name": "Minimum annotation objects",
                                                "description": "Minimum number of objects in an annotation",
                                                "value": 1,
                                                "default_value": 1,
                                                "value_type": "int",
                                                "min_value": 0,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "max_annotation_objects",
                                        "name": "Maximum annotation objects",
                                        "description": "Maximum number of objects in an annotation",
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable maximum annotation objects filtering",
                                                "description": "Whether to apply maximum annotation objects filtering",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "value",
                                                "name": "Maximum annotation objects",
                                                "description": "Maximum number of objects in an annotation",
                                                "value": 10000,
                                                "default_value": 10000,
                                                "value_type": "int",
                                                "min_value": 0,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "intensity_mapping",
                                "name": "Intensity mapping",
                                "description": (
                                    "Intensity mapping parameters control how raw pixel values are normalised to [0, 1]"
                                    " range before training. This is especially important for images with non-standard "
                                    "bit depths (e.g. 16-bit), where the default [0, 255] assumption does not hold."
                                ),
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "mode",
                                        "name": "Intensity mapping mode",
                                        "description": (
                                            "Strategy used to transform pixel intensities. "
                                            "'Unit interval scaling' divides by max_value, thus mapping the range "
                                            "[0, max_value] to [0, 1]. 'Windowing' isolates a specific intensity "
                                            "range, mapping a specific window (specified with center and "
                                            "width) to [0, 1] and clipping values outside the window. "
                                            "'Range scaling with clipping' multiplies pixel values by a scale factor, "
                                            "clips the result to a specified range (clip_min_value, clip_max_value) "
                                            "and finally normalizes to [0, 1]."
                                        ),
                                        "value": "Unit interval scaling",
                                        "default_value": "Unit interval scaling",
                                        "value_type": "str",
                                        "allowed_values": [
                                            "Unit interval scaling",
                                            "Windowing",
                                            "Range scaling with clipping",
                                        ],
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "max_value",
                                        "name": "Maximum pixel value",
                                        "description": (
                                            "Maximum possible pixel value in the raw image. "
                                            "For 8-bit images use 255, for 16-bit images use 65535."
                                        ),
                                        "value": 255.0,
                                        "default_value": 255.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": None,
                                        "allowed_values": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "min_value",
                                        "name": "Minimum output value",
                                        "description": (
                                            "Minimum output value after rescaling the image; "
                                            "Pixel values below this threshold are clipped."
                                        ),
                                        "value": 0.0,
                                        "default_value": 0.0,
                                        "value_type": "float",
                                        "min_value": None,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": {"mode": "Range scaling with clipping"},
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "window_center",
                                        "name": "Window center",
                                        "description": (
                                            "Center of the intensity window for 'windowing' mode. Together with width, "
                                            "it defines the intensity range that is mapped to [0, 1]."
                                        ),
                                        "value": 127.5,
                                        "default_value": 127.5,
                                        "value_type": "float",
                                        "min_value": None,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": {"mode": "Windowing"},
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "window_width",
                                        "name": "Window width",
                                        "description": (
                                            "Width of the intensity window for 'windowing' mode. "
                                            "The effective range is [center - width/2, center + width/2], "
                                            "mapped linearly to [0, 1]."
                                        ),
                                        "value": 255.0,
                                        "default_value": 255.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": {"mode": "Windowing"},
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "scale_factor",
                                        "name": "Scale factor",
                                        "description": (
                                            "Multiplicative factor applied to pixel values, before clipping the result "
                                            "to [min_value, max_value]."
                                        ),
                                        "value": 1.0,
                                        "default_value": 1.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": {"mode": "Range scaling with clipping"},
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "augmentation",
                                "name": "Data augmentation",
                                "description": (
                                    "Data augmentation is a technique used in machine learning to artificially expand "
                                    "a training dataset by applying transformations (e.g., rotation, scaling, noise) "
                                    "to existing data. It improves model generalization and reduces overfitting "
                                    "by increasing data variability without collecting new samples."
                                ),
                                "parameters": [
                                    {
                                        "type": "parameter_group",
                                        "key": "iou_random_crop",
                                        "name": "IoU random crop",
                                        "description": (
                                            "Randomly crop images based on Intersection over Union (IoU) criteria. "
                                            "Note: this augmentation is not supported when tiling is enabled."
                                        ),
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": True,
                                                "default_value": True,
                                                "value_type": "bool",
                                            }
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "random_affine",
                                        "name": "Random affine",
                                        "description": (
                                            "Apply random affine transformations (rotation, translation, scaling, "
                                            "shear) to the image."
                                        ),
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "max_rotate_degree",
                                                "name": "Rotation degrees",
                                                "description": (
                                                    "Maximum rotation angle in degrees for affine transformation. "
                                                    "A random angle in the range [-max_rotate_degree, "
                                                    "max_rotate_degree] will be applied. For example, "
                                                    "max_rotate_degree=10 allows up to ±10 degrees rotation."
                                                ),
                                                "value": 10.0,
                                                "default_value": 10.0,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "max_translate_ratio",
                                                "name": "Horizontal translation",
                                                "description": (
                                                    "Maximum translation as a fraction of image width or height. "
                                                    "A random translation in the range [-max_translate_ratio, "
                                                    "max_translate_ratio] will be applied along both axes. "
                                                    "For example, 0.1 allows up to ±10% translation."
                                                ),
                                                "value": 0.1,
                                                "default_value": 0.1,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": 1.0,
                                                "allowed_values": None,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "scaling_ratio_range",
                                                "name": "Scaling ratio range",
                                                "description": (
                                                    "Range (min, max) of scaling factors to apply during affine "
                                                    "transformation. Both values should be > 0.0. For example, "
                                                    "(0.8, 1.2) will randomly scale the image between 80% and 120% "
                                                    "of its original size."
                                                ),
                                                "value": [0.5, 1.5],
                                                "default_value": [0.5, 1.5],
                                                "value_type": "float_range",
                                                "min_value": 0.0,
                                                "max_value": 10.0,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "max_shear_degree",
                                                "name": "Maximum shear degree",
                                                "description": (
                                                    "Maximum absolute shear angle in degrees to apply "
                                                    "during affine transformation. A random shear in the range "
                                                    "[-max_shear_degree, max_shear_degree] will be applied."
                                                ),
                                                "value": 2.0,
                                                "default_value": 2.0,
                                                "value_type": "float",
                                                "min_value": None,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "random_horizontal_flip",
                                        "name": "Random horizontal flip",
                                        "description": (
                                            "Randomly flip images horizontally along the vertical axis "
                                            "(swap left and right)."
                                        ),
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": True,
                                                "default_value": True,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "probability",
                                                "name": "Probability",
                                                "description": (
                                                    "Probability of applying horizontal flip. "
                                                    "A value of 0.5 means each image has a 50% chance to be "
                                                    "flipped horizontally."
                                                ),
                                                "value": 0.5,
                                                "default_value": 0.5,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": 1.0,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "random_vertical_flip",
                                        "name": "Random vertical flip",
                                        "description": (
                                            "Randomly flip images vertically along the "
                                            "horizontal axis (swap top and bottom)."
                                        ),
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "probability",
                                                "name": "Probability",
                                                "description": (
                                                    "Probability of applying vertical flip. A value of 0.5 means "
                                                    "each image has a 50% chance to be flipped vertically."
                                                ),
                                                "value": 0.5,
                                                "default_value": 0.5,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": 1.0,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "color_jitter",
                                        "name": "Color jitter",
                                        "description": (
                                            "Randomly adjust brightness, contrast, saturation, and hue of the image."
                                        ),
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "brightness",
                                                "name": "Brightness range",
                                                "description": (
                                                    "Range (min, max) of brightness adjustment factors. "
                                                    "A random factor from this range will be multiplied with the "
                                                    "image brightness. For example, (0.8, 1.2) means "
                                                    "brightness can be reduced by 20% or increased by 20%."
                                                ),
                                                "value": [0.875, 1.125],
                                                "default_value": [0.875, 1.125],
                                                "value_type": "float_range",
                                                "min_value": 0.0,
                                                "max_value": 5.0,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "contrast",
                                                "name": "Contrast range",
                                                "description": (
                                                    "Range (min, max) of contrast adjustment factors. "
                                                    "A random factor from this range will be multiplied with the "
                                                    "image contrast. For example, (0.5, 1.5) means contrast "
                                                    "can be halved or increased by up to 50%."
                                                ),
                                                "value": [0.5, 1.5],
                                                "default_value": [0.5, 1.5],
                                                "value_type": "float_range",
                                                "min_value": 0.0,
                                                "max_value": 5.0,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "saturation",
                                                "name": "Saturation range",
                                                "description": (
                                                    "Range (min, max) of saturation adjustment factors. "
                                                    "A random factor from this range will be multiplied with the "
                                                    "image saturation. For example, (0.5, 1.5) means saturation "
                                                    "can be halved or increased by up to 50%."
                                                ),
                                                "value": [0.5, 1.5],
                                                "default_value": [0.5, 1.5],
                                                "value_type": "float_range",
                                                "min_value": 0.0,
                                                "max_value": 5.0,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "hue",
                                                "name": "Hue range",
                                                "description": (
                                                    "Range (min, max) of hue adjustment values. "
                                                    "A random value from this range will be added to the image hue. "
                                                    "For example, (-0.05, 0.05) means hue can be shifted "
                                                    "by up to ±0.05."
                                                ),
                                                "value": [-0.05, 0.05],
                                                "default_value": [-0.05, 0.05],
                                                "value_type": "float_range",
                                                "min_value": -0.5,
                                                "max_value": 0.5,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "probability",
                                                "name": "Probability",
                                                "description": (
                                                    "Probability of applying color jitter. A value of 0.5 means each "
                                                    "image has a 50% chance to be color jittered."
                                                ),
                                                "value": 0.5,
                                                "default_value": 0.5,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": 1.0,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "gaussian_blur",
                                        "name": "Gaussian blur",
                                        "description": "Apply Gaussian blur to the image.",
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "kernel_size",
                                                "name": "Kernel size",
                                                "description": (
                                                    "Size of the Gaussian kernel. Larger kernel sizes result in "
                                                    "stronger blurring. Must be a positive odd integer."
                                                ),
                                                "value": 5,
                                                "default_value": 5,
                                                "value_type": "int",
                                                "min_value": 0,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "sigma",
                                                "name": "Sigma range",
                                                "description": (
                                                    "Range (min, max) of sigma values for Gaussian blur. "
                                                    "Sigma controls the amount of blurring. "
                                                    "A random value from this range will be used for each image."
                                                ),
                                                "value": [0.1, 2.0],
                                                "default_value": [0.1, 2.0],
                                                "value_type": "float_range",
                                                "min_value": 0.0,
                                                "max_value": 10.0,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "probability",
                                                "name": "Probability",
                                                "description": (
                                                    "Probability of applying Gaussian blur. A value of 0.5 means "
                                                    "each image has a 50% chance to be blurred."
                                                ),
                                                "value": 0.5,
                                                "default_value": 0.5,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": 1.0,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "gaussian_noise",
                                        "name": "Gaussian noise",
                                        "description": "Add Gaussian noise to the image.",
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "mean",
                                                "name": "Mean",
                                                "description": (
                                                    "Mean of the Gaussian noise to be added to the image. "
                                                    "Typically set to 0.0 for zero-mean noise."
                                                ),
                                                "value": 0.0,
                                                "default_value": 0.0,
                                                "value_type": "float",
                                                "min_value": None,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "sigma",
                                                "name": "Standard deviation",
                                                "description": (
                                                    "Standard deviation of the Gaussian noise. Controls the intensity "
                                                    "of the noise. Higher values result in noisier images."
                                                ),
                                                "value": 0.1,
                                                "default_value": 0.1,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "probability",
                                                "name": "Probability",
                                                "description": (
                                                    "Probability of applying Gaussian noise. A value of 0.5 means "
                                                    "each image has a 50% chance to have noise added."
                                                ),
                                                "value": 0.5,
                                                "default_value": 0.5,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": 1.0,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                    {
                                        "type": "parameter_group",
                                        "key": "tiling",
                                        "name": "Tiling",
                                        "description": (
                                            "Split images into overlapping tiles for processing, "
                                            "useful for detecting small objects."
                                        ),
                                        "parameters": [
                                            {
                                                "type": "parameter",
                                                "key": "enable",
                                                "name": "Enable",
                                                "description": "Toggle to apply this augmentation.",
                                                "value": False,
                                                "default_value": False,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "enable_adaptive_tiling",
                                                "name": "Adaptive tiling",
                                                "description": "Whether to use adaptive tiling based on image content",
                                                "value": True,
                                                "default_value": True,
                                                "value_type": "bool",
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "tile_size",
                                                "name": "Tile size",
                                                "description": (
                                                    "Size of each tile in pixels. Decreasing the tile size typically "
                                                    "results in higher accuracy, but it is also more computationally "
                                                    "expensive due to the higher number of tiles. In any case, the "
                                                    "tile must be large enough to capture the entire object and its "
                                                    "surrounding context, so choose a value larger than the size "
                                                    "of most annotations."
                                                ),
                                                "value": 400,
                                                "default_value": 128,
                                                "value_type": "int",
                                                "min_value": 64,
                                                "max_value": None,
                                                "allowed_values": None,
                                            },
                                            {
                                                "type": "parameter",
                                                "key": "tile_overlap",
                                                "name": "Tile overlap",
                                                "description": (
                                                    "Overlap between adjacent tiles as a fraction of tile size"
                                                ),
                                                "value": 0.2,
                                                "default_value": 0.2,
                                                "value_type": "float",
                                                "min_value": 0.0,
                                                "max_value": 1.0,
                                                "allowed_values": None,
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "type": "parameter_group",
                        "key": "training",
                        "name": "Training",
                        "description": "Configurable parameters related to the learning phase (hyperparameters).",
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "max_epochs",
                                "name": "Maximum epochs",
                                "description": (
                                    "Maximum number of epochs to train the model. "
                                    "An epoch is one complete pass through the training dataset."
                                ),
                                "value": 200,
                                "default_value": 200,
                                "value_type": "int",
                                "min_value": 0,
                                "max_value": None,
                                "allowed_values": None,
                            },
                            {
                                "type": "parameter_group",
                                "key": "early_stopping",
                                "name": "Early stopping",
                                "description": (
                                    "Early stopping is a technique to prevent overfitting by stopping training "
                                    "when performance on a validation set stops improving."
                                ),
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Toggle early stopping",
                                        "description": (
                                            "Whether to stop training early when performance stops improving"
                                        ),
                                        "value": True,
                                        "default_value": True,
                                        "value_type": "bool",
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "patience",
                                        "name": "Patience",
                                        "description": (
                                            "Number of epochs with no improvement after which training will be stopped"
                                        ),
                                        "value": 10,
                                        "default_value": 10,
                                        "value_type": "int",
                                        "min_value": 0,
                                        "max_value": None,
                                        "allowed_values": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter",
                                "key": "learning_rate",
                                "name": "Learning rate",
                                "description": (
                                    "Learning rate for the optimizer, controlling the step size during model weight "
                                    "updates. A smaller learning rate may lead to more stable convergence, while a "
                                    "larger learning rate may speed up training but risk overshooting minima in "
                                    "the loss landscape."
                                ),
                                "value": 0.004,
                                "default_value": 0.004,
                                "value_type": "float",
                                "min_value": 0,
                                "max_value": 1,
                                "allowed_values": None,
                            },
                            {
                                "type": "parameter",
                                "key": "input_size_width",
                                "name": "Input size width",
                                "description": (
                                    "Width size in pixels for model input images. "
                                    "Determines the horizontal resolution at which images are processed."
                                ),
                                "value": 992,
                                "default_value": 992,
                                "value_type": "int",
                                "min_value": 0,
                                "max_value": None,
                                "allowed_values": [128, 256, 384, 512, 640, 800, 992, 1024],
                            },
                            {
                                "type": "parameter",
                                "key": "input_size_height",
                                "name": "Input size height",
                                "description": (
                                    "Height size in pixels for model input images. "
                                    "Determines the vertical resolution at which images are processed."
                                ),
                                "value": 800,
                                "default_value": 800,
                                "value_type": "int",
                                "min_value": 0,
                                "max_value": None,
                                "allowed_values": [128, 256, 384, 512, 640, 800, 992, 1024],
                            },
                        ],
                    },
                    {
                        "type": "parameter_group",
                        "key": "evaluation",
                        "name": "Evaluation parameters",
                        "description": "Configurable parameters related to the model evaluation.",
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "validation_metric",
                                "name": "Validation metric",
                                "description": "Metric used to evaluate model performance during validation.",
                                "value": "default",
                                "default_value": "default",
                                "value_type": "str",
                                "allowed_values": ["default"],
                            }
                        ],
                    },
                ]
            }
        }
    }
