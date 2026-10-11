# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator


class EarlyStopping(BaseModel):
    enable: bool = Field(
        default=False,
        title="Enable",
        description="Toggle to enable or disable early stopping during training.",
    )
    patience: int = Field(
        ge=1,
        default=1,
        title="Patience",
        description="Number of epochs with no improvement after which training will be stopped.",
    )


class SchedulerType(StrEnum):
    """
    Learning rate scheduler type

    Important: the enum keys must match the lowercase strings expected by the training library (getitune),
    whereas the values correspond to the user-friendly names shown in the UI.
    """

    REDUCE_LR_ON_PLATEAU = "Reduce LR on loss plateau"
    COSINE_ANNEALING = "Cosine annealing"


class LrLinearWarmupParameters(BaseModel):
    enable: bool = Field(
        default=False,
        title="Enable",
        description="Toggle to enable or disable the LR linear warmup phase at the beginning of training.",
    )
    epochs: int = Field(
        default=5,
        ge=1,
        title="Warmup epochs",
        description="Number of epochs for the LR linear warmup phase.",
    )


class SchedulerParameters(BaseModel):
    type: SchedulerType = Field(
        default=SchedulerType.REDUCE_LR_ON_PLATEAU,
        title="Scheduler type",
        description=(
            "Type of learning rate scheduler to use during training. "
            "With ReduceLROnPlateau, the learning rate will be reduced by a predetermined factor when the validation "
            "metric stops improving. With CosineAnnealing, the learning rate will follow a cosine decay schedule, "
            "gradually decreasing over the course of training."
        ),
        json_schema_extra={"read_only": True},
    )
    warmup: LrLinearWarmupParameters = Field(
        default_factory=LrLinearWarmupParameters,
        title="Learning rate linear warmup",
        description=(
            "Learning rate warmup is a technique where the learning rate starts at a lower value and gradually "
            "increases to the initial learning rate over a specified number of epochs at the beginning of training. "
            "This can help stabilize training and improve convergence, especially when using large learning rates "
            "or training on complex datasets."
        ),
    )
    factor: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        title="Factor",
        description="Factor by which the learning rate will be reduced. new_lr = lr * factor.",
        json_schema_extra={"depends_on": {"type": "reduce_lr_on_plateau"}},
    )
    patience: int = Field(
        default=5,
        ge=1,
        title="Patience",
        description="Number of epochs with no improvement after which learning rate will be reduced.",
        json_schema_extra={"depends_on": {"type": "reduce_lr_on_plateau"}},
    )
    min_lr: float = Field(
        default=0,
        ge=0,
        lt=1,
        title="Minimum learning rate",
        description="Minimum learning rate after annealing.",
        json_schema_extra={"depends_on": {"type": "cosine_annealing"}},
    )

    @field_validator("type", mode="before")
    @classmethod
    def validate_mode(cls, value: str) -> SchedulerType:
        """Accept both the user-friendly enum value and the lowercase key name."""
        try:
            return SchedulerType(value)
        except ValueError:
            # Try matching by uppercase key name (e.g. "reduce_lr_on_plateau" -> "REDUCE_LR_ON_PLATEAU")
            try:
                return SchedulerType[value.upper()]
            except KeyError:
                raise ValueError(f"Invalid intensity mapping mode: {value!r}") from None

    @field_serializer("type")
    @staticmethod
    def serialize_mode(value: SchedulerType | str) -> str:
        """Serialize mode as the lowercase enum key name expected by the training library."""
        if not isinstance(value, SchedulerType):
            value = SchedulerType(value)
        return value.name.lower()


class GradientAccumulationParameters(BaseModel):
    enable: bool = Field(
        default=False, title="Enable", description="Toggle to enable or disable gradient accumulation during training."
    )
    batches: int = Field(
        ge=1,
        default=1,
        title="Gradient accumulation batches",
        description=(
            "Number of steps (batches) to accumulate gradients before performing gradient descent step. "
            "Effective batch size during training: batch_size * accumulate_grad_batches."
        ),
    )


class GradientClipParameters(BaseModel):
    enable: bool = Field(
        default=False,
        title="Enable",
        description="Toggle to enable or disable gradient clipping during training.",
    )
    max_grad_norm: float = Field(
        gt=0,
        default=1.0,
        title="Maximum gradient L2 norm",
        description="Maximum L2 norm of the gradients. Gradients with norm larger than this value will be clipped.",
    )


class AlgoLevelTrainingParameters(BaseModel):
    """Hyperparameters for model training process."""

    max_epochs: int = Field(
        ge=1,
        default=200,
        title="Maximum epochs",
        description=(
            "Maximum number of epochs to train the model. An epoch is one complete pass through the training dataset."
        ),
    )
    batch_size: int = Field(
        ge=1,
        default=4,
        title="Batch size",
        description=(
            "Number of training samples processed before the model's internal parameters are updated. "
            "A larger batch size can speed up training but may require more memory, while a smaller batch size "
            "can help avoid OOM (Out of Memory) errors at the cost of longer training times and potentially noisier "
            "gradient estimates."
        ),
    )
    num_workers: int = Field(
        ge=0,
        le=64,
        default=4,
        title="Number of workers",
        description=(
            "Number of subprocesses used for data loading. 0 means that the data will be loaded in the main "
            "process. More workers can speed up data loading and augmentation, at the cost of higher CPU and "
            "memory usage."
        ),
    )
    early_stopping: EarlyStopping = Field(
        default_factory=EarlyStopping,
        title="Early stopping",
        description=(
            "Early stopping is a technique to prevent overfitting by stopping training when performance "
            "on a validation set stops improving."
        ),
    )
    learning_rate: float = Field(
        gt=0,
        lt=1,
        title="Learning rate",
        description=(
            "Learning rate for the optimizer, controlling the step size during model weight updates. "
            "A smaller learning rate may lead to more stable convergence, while a larger learning rate may speed up "
            "training but risk overshooting minima in the loss landscape."
        ),
    )
    weight_decay: float = Field(
        ge=0,
        lt=1,
        default=1e-4,
        title="Weight decay",
        description=(
            "Weight decay is a regularization technique that adds a penalty to the loss function based on the "
            "squared magnitude of the model weights (L2 regularization). "
            "It helps prevent overfitting by discouraging large weight values."
        ),
    )
    scheduler: SchedulerParameters = Field(
        default_factory=SchedulerParameters,
        title="Learning rate scheduler",
        description=(
            "The learning rate scheduler adjusts the learning rate during training according to a predefined schedule "
            "or based on validation performance, helping to improve convergence and training stability."
        ),
    )
    gradient_accumulation: GradientAccumulationParameters = Field(
        default_factory=GradientAccumulationParameters,
        title="Gradient accumulation",
        description=(
            "Gradient accumulation allows simulating larger batch sizes by accumulating gradients "
            "over multiple forward/backward passes before updating the model weights."
        ),
    )
    gradient_clip: GradientClipParameters = Field(
        default_factory=GradientClipParameters,
        title="Gradient clipping",
        description=(
            "Gradient clipping prevents exploding gradients by capping gradient norms during backpropagation."
        ),
    )
    input_size_width: int = Field(
        gt=0,
        title="Input size width",
        description=(
            "Width size in pixels for model input images. "
            "Determines the horizontal resolution at which images are processed."
        ),
        json_schema_extra={"allowed_values_from": "allowed_values_input_size"},
    )
    input_size_height: int = Field(
        gt=0,
        title="Input size height",
        description=(
            "Height size in pixels for model input images. "
            "Determines the vertical resolution at which images are processed."
        ),
        json_schema_extra={"allowed_values_from": "allowed_values_input_size"},
    )
    allowed_values_input_size: list[int] = Field(
        title="Supported input size (width and height)",
        description="List of supported values for input width and height.",
        json_schema_extra={"validation_only": True},
    )

    @model_validator(mode="after")
    def validate_input_size(self) -> "AlgoLevelTrainingParameters":
        # Validate that allowed_values_input_size is provided and not empty
        if len(self.allowed_values_input_size) == 0:
            raise ValueError("'allowed_values_input_size' must contain at least one value")

        # Validate that input_size_width and input_size_height are specified
        w, h = self.input_size_width, self.input_size_height
        if w is None or h is None:
            raise ValueError("Both 'input_size_width' and 'input_size_height' must be specified")

        # Validate against allowed input sizes if available
        if allowed_values := self.allowed_values_input_size:
            if w and w not in allowed_values:
                raise ValueError(
                    f"Input size width '{w}' is not in the list of supported input sizes: {allowed_values}"
                )
            if h and h not in allowed_values:
                raise ValueError(
                    f"Input size height '{h}' is not in the list of supported input sizes: {allowed_values}"
                )
        return self
