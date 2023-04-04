"""Base Configuration of OTX Common Algorithms."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from sys import maxsize

from attr import attrs

from otx.api.configuration import ConfigurableEnum, ConfigurableParameters
from otx.api.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    selectable,
    string_attribute,
)
from otx.api.configuration.model_lifecycle import ModelLifecycle

from .configuration_enums import POTQuantizationPreset

# pylint: disable=invalid-name


class TrainType(ConfigurableEnum):
    """TrainType for OTX Algorithms."""

    Finetune = "Finetune"
    Semisupervised = "Semisupervised"
    Selfsupervised = "Selfsupervised"
    Incremental = "Incremental"
    Futurework = "Futurework"


class LearningRateSchedule(ConfigurableEnum):
    """LearningRateSchedule for OTX Algorithms."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP_WISE = "step_wise"
    CYCLIC = "cyclic"
    CUSTOM = "custom"


@attrs
class BaseConfig(ConfigurableParameters):
    """BaseConfig Class for OTX Algorithms."""

    @attrs
    class BaseLearningParameters(ParameterGroup):
        """BaseLearningParameters for OTX Algorithms."""

        batch_size = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=2048,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing thisvalue "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        unlabeled_batch_size = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=512,
            header="Unlabeled batch size",
            description="The number of unlabeled training samples seen in each iteration of semi-supervised learning."
            "Increasing this value improves training time and may make the training more stable."
            "A larger batch size has higher memory requirements.",
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
            max_value=1.0,
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

        enable_supcon = configurable_boolean(
            default_value=False,
            header="Enable Supervised Contrastive helper loss",
            description="This auxiliar loss might increase robustness and accuracy for small datasets",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    @attrs
    class BasePostprocessing(ParameterGroup):
        """BasePostprocessing for OTX Algorithms."""

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
        """BaseNNCFOptimization for OTX Algorithms."""

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
        """BasePOTParameter for OTX Algorithms."""

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
            max_value=maxsize,
        )

        preset = selectable(
            default_value=POTQuantizationPreset.PERFORMANCE,
            header="Preset",
            description="Quantization preset that defines quantization scheme",
            editable=True,
            visible_in_ui=True,
        )

    @attrs
    class BaseAlgoBackendParameters(ParameterGroup):
        """BaseAlgoBackendParameters for OTX Algorithms."""

        train_type = selectable(
            default_value=TrainType.Incremental,
            header="train type",
            description="Training scheme option that determines how to train the model",
            editable=False,
            visible_in_ui=True,
        )

        mem_cache_size = configurable_integer(
            header="Size of memory pool for caching decoded data to load data faster",
            description="Size of memory pool for caching decoded data to load data faster",
            default_value=0,
            min_value=0,
            max_value=maxsize,
            visible_in_ui=False,
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    @attrs
    class BaseTilingParameters(ParameterGroup):
        """BaseTilingParameters for OTX Algorithms."""

        header = string_attribute("Tiling Parameters")
        enable_tiling = configurable_boolean(
            default_value=False,
            header="Enable tiling",
            description="Set to True to allow tiny objects to be better detected.",
            warning="Tiling trades off speed for accuracy as it increases the number of images to be processed.",
            affects_outcome_of=ModelLifecycle.NONE,
        )

        enable_adaptive_params = configurable_boolean(
            default_value=True,
            header="Enable adaptive tiling parameters",
            description="Config tile size and tile overlap adaptively based on annotated dataset statistic",
            warning="",
            affects_outcome_of=ModelLifecycle.NONE,
        )

        tile_size = configurable_integer(
            header="Tile Image Size",
            description="Tile Image Size",
            default_value=400,
            min_value=100,
            max_value=1024,
            affects_outcome_of=ModelLifecycle.NONE,
        )

        tile_overlap = configurable_float(
            header="Tile Overlap",
            description="Overlap between each two neighboring tiles.",
            default_value=0.2,
            min_value=0.0,
            max_value=1.0,
            affects_outcome_of=ModelLifecycle.NONE,
        )

        tile_max_number = configurable_integer(
            header="Max object per image",
            description="Max object per image",
            default_value=1500,
            min_value=1,
            max_value=10000,
            affects_outcome_of=ModelLifecycle.NONE,
        )

    tiling_parameters = add_parameter_group(BaseTilingParameters)
