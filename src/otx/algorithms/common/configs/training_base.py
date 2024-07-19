"""Base Configuration of OTX Common Algorithms."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

from .configuration_enums import BatchSizeAdaptType, InputSizePreset, POTQuantizationPreset, StorageCacheScheme

# pylint: disable=invalid-name


class TrainType(ConfigurableEnum):
    """TrainType for OTX Algorithms."""

    Finetune = "Finetune"
    Semisupervised = "Semisupervised"
    Selfsupervised = "Selfsupervised"
    Incremental = "Incremental"
    Zeroshot = "Zeroshot"
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
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        inference_batch_size = configurable_integer(
            default_value=1,
            min_value=1,
            max_value=512,
            header="Inference batch size",
            description="The number of samples seen in each iteration of inference. Increasing this value "
            "improves inference time. A larger batch size has higher memory requirements.",
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
            max_value=1000,
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

        enable_supcon = configurable_boolean(
            default_value=False,
            header="Enable Supervised Contrastive helper loss",
            description="This auxiliar loss might increase robustness and accuracy for small datasets",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        auto_adapt_batch_size = selectable(
            default_value=BatchSizeAdaptType.NONE,
            header="Adapt batch size according to current GPU memory.",
            description="Safe => Prevent GPU out of memory. Full => Find a batch size using most of GPU memory.",
            warning="Enabling this could change the actual batch size depending on the current GPU status. "
            "The learning rate also could be adjusted according to the adapted batch size. This process "
            "might change a model performance and take some extra computation time to try a few batch size candidates.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        auto_num_workers = configurable_boolean(
            default_value=False,
            header="Enable auto adaptive num_workers",
            description="Adapt num_workers according to current hardware status automatically.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        input_size = selectable(
            default_value=InputSizePreset.DEFAULT,
            header="Configure model input size.",
            description="The input size of the given model could be configured to one of the predefined resolutions."
            "Reduced training and inference time could be expected by using smaller input size."
            "Defaults to per-model default resolution.",
            warning="Modifying input size may decrease model performance.",
            affects_outcome_of=ModelLifecycle.NONE,
            visible_in_ui=False,
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

        nms_iou_threshold = configurable_float(
            default_value=0.5,
            min_value=0,
            max_value=1,
            header="NMS IoU Threshold",
            description="IoU Threshold for NMS Postprocessing."
            "Intersection over Union (IoU) threshold is set to remove overlapping predictions."
            "If the IoU between two predictions is greater than or equal to the IoU threshold, "
            "they are considered overlapping and will be discarded.",
            affects_outcome_of=ModelLifecycle.INFERENCE,
            warning="If you want to chage the value of IoU Threshold of model, "
            "then you need to re-train model with new IoU threshold.",
        )

        max_num_detections = configurable_integer(
            header="Maximum number of detection per image",
            description="Extra detection outputs will be discared in non-maximum suppression process. "
            "Defaults to 0, which means per-model default value.",
            default_value=0,
            min_value=0,
            max_value=10000,
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

        use_ellipse_shapes = configurable_boolean(
            default_value=False,
            header="Use ellipse shapes",
            description="Use direct ellipse shape in inference instead of polygon from mask",
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
            max_value=1000,
        )

        stat_requests_number = configurable_integer(
            header="Number of requests",
            description="Number of requests during statistics collection",
            default_value=0,
            min_value=0,
            max_value=200,
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
            visible_in_ui=False,
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

        storage_cache_scheme = selectable(
            default_value=StorageCacheScheme.NONE,
            header="Scheme for storage cache",
            description="Scheme for storage cache",
            editable=False,
            visible_in_ui=False,
        )

    @attrs
    class BaseTilingParameters(ParameterGroup):
        """BaseTilingParameters for OTX Algorithms."""

        header = string_attribute("Tiling Parameters")
        enable_tiling = configurable_boolean(
            default_value=False,
            header="Enable tiling",
            description="Set to True to allow tiny objects to be better detected.",
            warning="Tiling trades off speed for accuracy as it increases the number of images to be processed. "
            "Important: In the current version, depending on the dataset size and the available hardware resources, "
            "a model may not train successfully when tiling is enabled.",
            affects_outcome_of=ModelLifecycle.NONE,
        )

        enable_tile_classifier = configurable_boolean(
            default_value=False,
            header="Enable tile classifier",
            description="Enabling tile classifier enhances the speed of tiling inference by incorporating a tile "
            "classifier into the instance segmentation model. This feature prevents the detector from "
            "making predictions on tiles that do not contain any objects, thus optimizing its "
            "speed performance.",
            warning="The tile classifier prioritizes inference speed over training speed, it requires more training "
            "in order to achieve its optimized performance.",
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
            max_value=4096,
            affects_outcome_of=ModelLifecycle.NONE,
        )

        tile_overlap = configurable_float(
            header="Tile Overlap",
            description="Overlap between each two neighboring tiles.",
            default_value=0.2,
            min_value=0.0,
            max_value=0.9,
            affects_outcome_of=ModelLifecycle.NONE,
        )

        tile_max_number = configurable_integer(
            header="Max object per image",
            description="Maximum number of objects per tile. If set to 1500, the tile adaptor "
            "will automatically determine the value. Otherwise, the manually set value will be used.",
            default_value=1500,
            min_value=1,
            max_value=5000,
            affects_outcome_of=ModelLifecycle.NONE,
        )

        tile_ir_scale_factor = configurable_float(
            header="OpenVINO IR Scale Factor",
            description="The purpose of the scale parameter is to optimize the performance and "
            "efficiency of tiling in OpenVINO IR during inference. By controlling the increase in tile size and "
            "input size, the scale parameter allows for more efficient parallelization of the workload and "
            "improve the overall performance and efficiency of the inference process on OpenVINO.",
            warning="Setting the scale factor value too high may cause the application "
            "to crash or result in out-of-memory errors. It is recommended to "
            "adjust the scale factor value carefully based on the available "
            "hardware resources and the needs of the application.",
            default_value=1.0,
            min_value=1.0,
            max_value=4.0,
            affects_outcome_of=ModelLifecycle.NONE,
        )

        tile_sampling_ratio = configurable_float(
            header="Sampling Ratio for entire tiling",
            description="Since tiling train and validation to all tile from large image, "
            "usually it takes lots of time than normal training."
            "The tile_sampling_ratio is ratio for sampling entire tile dataset."
            "Sampling tile dataset would save lots of time for training and validation time."
            "Note that sampling will be applied to training and validation dataset, not test dataset.",
            default_value=1.0,
            min_value=0.000001,
            max_value=1.0,
            affects_outcome_of=ModelLifecycle.NONE,
        )

        object_tile_ratio = configurable_float(
            header="Object tile ratio",
            description="The desired ratio of min object size and tile size.",
            default_value=0.03,
            min_value=0.00,
            max_value=1.00,
            affects_outcome_of=ModelLifecycle.NONE,
        )

    tiling_parameters = add_parameter_group(BaseTilingParameters)
