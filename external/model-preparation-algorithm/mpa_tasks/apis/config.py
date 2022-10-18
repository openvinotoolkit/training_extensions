# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from sys import maxsize

from attr import attrs
from ote_sdk.configuration import ConfigurableEnum, ConfigurableParameters
from ote_sdk.configuration.elements import (
    ParameterGroup,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    selectable,
)
from ote_sdk.configuration.model_lifecycle import ModelLifecycle


class TrainType(ConfigurableEnum):
    FineTune = "FineTune"
    SemiSupervised = "SemiSupervised"
    SelfSupervised = "SelfSupervised"
    Incremental = "Incremental"
    FutureWork = "FutureWork"


class LearningRateSchedule(ConfigurableEnum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP_WISE = "step_wise"
    CYCLIC = "cyclic"
    CUSTOM = "custom"


@attrs
class BaseConfig(ConfigurableParameters):
    @attrs
    class BaseLearningParameters(ParameterGroup):
        batch_size = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing thisvalue "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        num_iters = configurable_integer(
            default_value=1,
            min_value=1,
            max_value=100000,
            header="Number of training iterations",
            description="Increasing this value causes the results to be more robust but training time will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        learning_rate = configurable_float(
            default_value=0.01,
            min_value=1e-07,
            max_value=1e-01,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        learning_rate_warmup_iters = configurable_integer(
            default_value=100,
            min_value=0,
            max_value=10000,
            header="Number of iterations for learning rate warmup",
            description="",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        use_adaptive_interval = configurable_boolean(
            default_value=False,
            header="Use adaptive validation interval",
            description="Depending on the size of iteration per epoch, \
                         adaptively update the validation interval and related values.",
            warning="This will automatically control the patience and interval when early stopping is enabled.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        enable_early_stopping = configurable_boolean(
            default_value=True,
            header="Enable early stopping of the training",
            description="Early exit from training when validation accuracy isn't \
                         changed or decreased for several epochs.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        early_stop_start = configurable_integer(
            default_value=3,
            min_value=0,
            max_value=1000,
            header="Start epoch for early stopping",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        early_stop_patience = configurable_integer(
            default_value=5,
            min_value=0,
            max_value=50,
            header="Patience for early stopping",
            description="Training will stop if the model does not improve within the number of epochs of patience.",
            warning="This is applied exclusively when early stopping is enabled.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        early_stop_iteration_patience = configurable_integer(
            default_value=0,
            min_value=0,
            max_value=1000,
            header="Iteration patience for early stopping",
            description="Training will stop if the model does not improve within the number of iterations of patience. \
                        the model is trained enough with the number of iterations of patience before early stopping.",
            warning="This is applied exclusively when early stopping is enabled.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        num_workers = configurable_integer(
            default_value=0,
            min_value=0,
            max_value=8,
            header="Number of cpu threads to use during batch generation",
            description="Increasing this value might improve training speed however it might cause out of memory "
            "errors. If the number of workers is set to zero, data loading will happen in the main "
            "training thread.",
            affects_outcome_of=ModelLifecycle.NONE,
        )

        num_checkpoints = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=100,
            header="Number of checkpoints that is done during the single training round",
            description="",
            affects_outcome_of=ModelLifecycle.NONE,
        )

    @attrs
    class BasePostprocessing(ParameterGroup):
        result_based_confidence_threshold = configurable_boolean(
            default_value=True,
            header="Result based confidence threshold",
            description="Confidence threshold is derived from the results",
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

        confidence_threshold = configurable_float(
            default_value=0.35,
            min_value=0,
            max_value=1,
            header="Confidence threshold",
            description="This threshold only takes effect if the threshold is not set based on the result.",
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

    @attrs
    class BaseNNCFOptimization(ParameterGroup):
        enable_quantization = configurable_boolean(
            default_value=True,
            header="Enable quantization algorithm",
            description="Enable quantization algorithm",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        enable_pruning = configurable_boolean(
            default_value=False,
            header="Enable filter pruning algorithm",
            description="Enable filter pruning algorithm",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        pruning_supported = configurable_boolean(
            default_value=False,
            header="Whether filter pruning is supported",
            description="Whether filter pruning is supported",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        maximal_accuracy_degradation = configurable_float(
            default_value=1.0,
            min_value=0.0,
            max_value=100.0,
            header="Maximum accuracy degradation",
            description="The maximal allowed accuracy metric drop",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    @attrs
    class BasePOTParameter(ParameterGroup):
        stat_subset_size = configurable_integer(
            header="Number of data samples",
            description="Number of data samples used for post-training optimization",
            default_value=300,
            min_value=1,
            max_value=maxsize,
        )

        stat_requests_number = configurable_integer(
            header="Number of requests",
            description="Number of requests during statistics collection",
            default_value=0,
            min_value=0,
            max_value=maxsize
        )

    @attrs
    class BaseAlgoBackendParameters(ParameterGroup):
        train_type = selectable(
            default_value=TrainType.Incremental,
            header="train type",
            description="training schema for the MPA task",
            editable=False,
            visible_in_ui=True,
        )
