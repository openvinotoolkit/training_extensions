#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import pytest

from app.api.schemas import TrainingConfigurationView
from app.models.training_configuration import (
    AlgoLevelDatasetPreparationParameters,
    AlgoLevelParameters,
    AlgoLevelTrainingParameters,
    TaskLevelDatasetPreparationParameters,
)
from app.models.training_configuration.augmentation import (
    AugmentationParameters,
    ColorJitter,
    Mixup,
    Mosaic,
    RandomAffine,
    RandomErasing,
    RandomGrayscale,
    RandomIOUCrop,
    RandomSharpness,
    RandomZoomOut,
)
from app.models.training_configuration.configuration import TaskLevelParameters, TrainingConfiguration
from app.models.training_configuration.dataset_preparation import (
    Filtering,
    IntensityMapping,
    IntensityMappingMode,
    MaxAnnotationObjects,
    MinAnnotationObjects,
    MinAnnotationPixels,
    SubsetSplit,
)
from app.models.training_configuration.training import (
    EarlyStopping,
    GradientAccumulationParameters,
    GradientClipParameters,
    LrLinearWarmupParameters,
    SchedulerParameters,
    SchedulerType,
)


@pytest.fixture
def fxt_training_configuration() -> TrainingConfiguration:
    """Create a mock training configuration."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(
            dataset_preparation=TaskLevelDatasetPreparationParameters(
                subset_split=SubsetSplit(training=75, validation=15, test=10),
                filtering=Filtering(
                    min_annotation_pixels=MinAnnotationPixels(enable=True, value=20),
                    min_annotation_objects=MinAnnotationObjects(enable=False, value=1),
                    max_annotation_objects=MaxAnnotationObjects(enable=False, value=50),
                ),
                intensity_mapping=IntensityMapping(
                    mode=IntensityMappingMode.WINDOW,
                    max_intensity_value=255.0,
                    clip_min_value=0.0,
                    clip_max_value=255.0,
                    window_center=200.0,
                    window_width=400.0,
                    scale_factor=1.0,
                ),
            ),
        ),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(
                augmentation=AugmentationParameters(
                    color_jitter=ColorJitter(
                        enable=True,
                        brightness=(0.9, 1.1),
                        contrast=(0.85, 1.15),
                        saturation=(0.8, 1.2),
                        hue=(-0.05, 0.05),
                        probability=0.6,
                    ),
                    random_erasing=RandomErasing(
                        enable=True,
                        scale=(0.03, 0.25),
                        ratio=(0.5, 2.0),
                        probability=0.4,
                        value=0.1,
                    ),
                    random_grayscale=RandomGrayscale(
                        enable=True,
                        probability=0.2,
                    ),
                    random_sharpness=RandomSharpness(
                        enable=True,
                        sharpness=0.8,
                        probability=0.3,
                    ),
                    random_affine=RandomAffine(
                        enable=True,
                        max_rotate_degree=15.0,
                        max_translate_ratio=0.2,
                        scaling_ratio_range=(0.6, 1.4),
                        max_shear_degree=5.0,
                        probability=0.7,
                    ),
                    random_zoom_out=RandomZoomOut(
                        enable=True,
                        fill=128,
                        side_range=(1.0, 3.0),
                        probability=0.4,
                    ),
                    iou_random_crop=RandomIOUCrop(
                        enable=True,
                        probability=0.9,
                        min_scale=0.4,
                        max_scale=0.9,
                    ),
                    mosaic=Mosaic(
                        enable=True,
                        probability=0.8,
                        scale=0.4,
                        translate=0.2,
                    ),
                    mixup=Mixup(
                        enable=True,
                        probability=0.6,
                        alpha=2.0,
                    ),
                )
            ),
            training=AlgoLevelTrainingParameters(
                max_epochs=120,
                batch_size=8,
                early_stopping=EarlyStopping(enable=True, patience=5),
                learning_rate=0.001,
                weight_decay=0.01,
                scheduler=SchedulerParameters(
                    type=SchedulerType.REDUCE_LR_ON_PLATEAU,
                    warmup=LrLinearWarmupParameters(enable=True, epochs=3),
                    factor=0.5,
                    patience=7,
                    min_lr=1e-5,
                ),
                gradient_accumulation=GradientAccumulationParameters(enable=True, batches=4),
                gradient_clip=GradientClipParameters(enable=True, max_grad_norm=2.0),
                input_size_width=256,
                input_size_height=256,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


@pytest.fixture
def fxt_default_training_configuration() -> TrainingConfiguration:
    """Create a default training configuration with default values."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(
                augmentation=AugmentationParameters(
                    color_jitter=ColorJitter(
                        enable=False,
                        brightness=(0.8, 1.2),
                        contrast=(0.75, 1.25),
                        saturation=(0.9, 1.1),
                        hue=(-0.1, 0.1),
                        probability=0.5,
                    ),
                    random_erasing=RandomErasing(
                        enable=False,
                        scale=(0.02, 0.33),
                        ratio=(0.3, 3.3),
                        probability=0.5,
                        value=0.0,
                    ),
                    random_grayscale=RandomGrayscale(
                        enable=False,
                        probability=0.1,
                    ),
                    random_sharpness=RandomSharpness(
                        enable=False,
                        sharpness=0.5,
                        probability=0.5,
                    ),
                    random_affine=RandomAffine(
                        enable=False,
                        max_rotate_degree=10.0,
                        max_translate_ratio=0.1,
                        scaling_ratio_range=(0.5, 1.5),
                        max_shear_degree=2.0,
                        probability=0.5,
                    ),
                    random_zoom_out=RandomZoomOut(
                        enable=False,
                        fill=0,
                        side_range=(1.0, 4.0),
                        probability=0.5,
                    ),
                    iou_random_crop=RandomIOUCrop(
                        enable=False,
                        probability=1.0,
                        min_scale=0.3,
                        max_scale=1.0,
                    ),
                    mosaic=Mosaic(
                        enable=False,
                        probability=1.0,
                    ),
                    mixup=Mixup(
                        enable=False,
                        probability=1.0,
                        alpha=1.5,
                    ),
                )
            ),
            training=AlgoLevelTrainingParameters(
                max_epochs=250,
                batch_size=4,
                early_stopping=EarlyStopping(enable=False, patience=1),
                learning_rate=0.0015,
                weight_decay=1e-4,
                scheduler=SchedulerParameters(
                    type=SchedulerType.REDUCE_LR_ON_PLATEAU,
                    warmup=LrLinearWarmupParameters(enable=False, epochs=5),
                    factor=0.1,
                    patience=10,
                    min_lr=1e-6,
                ),
                gradient_accumulation=GradientAccumulationParameters(enable=False, batches=1),
                gradient_clip=GradientClipParameters(enable=False, max_grad_norm=1.0),
                input_size_width=512,
                input_size_height=512,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


@pytest.fixture
def fxt_training_configuration_view_json() -> dict:
    return {
        "parameters": [
            {
                "type": "parameter_group",
                "key": "dataset_preparation",
                "name": "Dataset preparation",
                "description": (
                    "Configurable parameters related to the training data, such as augmentations and filters."
                ),
                "depends_on": None,
                "parameters": [
                    {
                        "type": "parameter_group",
                        "key": "subset_split",
                        "name": "Subset split",
                        "description": (
                            "Subset split parameters define how the dataset is divided into training, validation, "
                            "and test subsets. The training subset is used to fit the model, the validation subset "
                            "is used to estimate the prediction error during training and the test subset is used "
                            "to evaluate the final performance of the model. The percentages for training, validation, "
                            "and test subsets must sum to 100."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "training",
                                "name": "Training percentage",
                                "description": "Percentage of data to use for training",
                                "value": 75,
                                "default_value": 70,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": 100,
                                "allowed_values": None,
                                "depends_on": None,
                            },
                            {
                                "type": "parameter",
                                "key": "validation",
                                "name": "Validation percentage",
                                "description": "Percentage of data to use for validation",
                                "value": 15,
                                "default_value": 20,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": 100,
                                "allowed_values": None,
                                "depends_on": None,
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
                                "depends_on": None,
                            },
                        ],
                    },
                    {
                        "type": "parameter_group",
                        "key": "filtering",
                        "name": "Filtering",
                        "description": (
                            "Filtering parameters define criteria for including or excluding annotations from "
                            "the dataset. Depending on the scenario, an appropriate filter configuration can speed up "
                            "the training process and/or improve the model performance by removing noisy annotations."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter_group",
                                "key": "min_annotation_pixels",
                                "name": "Minimum annotation pixels",
                                "description": "Minimum number of pixels in an annotation",
                                "depends_on": None,
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable minimum annotation pixels filtering",
                                        "description": "Whether to apply minimum annotation pixels filtering",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "value",
                                        "name": "Minimum annotation pixels",
                                        "description": "Minimum number of pixels in an annotation",
                                        "value": 20,
                                        "default_value": 1,
                                        "value_type": "int",
                                        "min_value": 0,
                                        "max_value": 200000000,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "min_annotation_objects",
                                "name": "Minimum annotation objects",
                                "description": "Minimum number of objects in an annotation",
                                "depends_on": None,
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable minimum annotation objects filtering",
                                        "description": "Whether to apply minimum annotation objects filtering",
                                        "value": False,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
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
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "max_annotation_objects",
                                "name": "Maximum annotation objects",
                                "description": "Maximum number of objects in an annotation",
                                "depends_on": None,
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable maximum annotation objects filtering",
                                        "description": "Whether to apply maximum annotation objects filtering",
                                        "value": False,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "value",
                                        "name": "Maximum annotation objects",
                                        "description": "Maximum number of objects in an annotation",
                                        "value": 50,
                                        "default_value": 10000,
                                        "value_type": "int",
                                        "min_value": 0,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": None,
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
                            "Intensity mapping parameters control how raw pixel values are normalised to [0, 1] range "
                            "before training. This is especially important for images with non-standard bit depths "
                            "(e.g. 16-bit), where the default [0, 255] assumption does not hold."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "mode",
                                "name": "Intensity mapping mode",
                                "description": (
                                    "Strategy used to transform pixel intensities. "
                                    "'Unit interval scaling' divides by max_intensity_value, thus mapping the range "
                                    "[0, max_intensity_value] to [0, 1]. 'Windowing' isolates a specific intensity "
                                    "range, mapping a specific window (specified with center and width) to [0, 1] and "
                                    "clipping values outside the window. 'Range scaling with clipping' multiplies "
                                    "pixel values by a scale factor, clips the result to a specified range "
                                    "(clip_min_value, clip_max_value) and finally normalizes to [0, 1]."
                                ),
                                "value": "Windowing",
                                "default_value": "Unit interval scaling",
                                "value_type": "str",
                                "allowed_values": ["Unit interval scaling", "Windowing", "Range scaling with clipping"],
                                "depends_on": None,
                            },
                            {
                                "type": "parameter",
                                "key": "max_intensity_value",
                                "name": "Maximum pixel intensity",
                                "description": (
                                    "Maximum possible pixel value in the raw image, used as the divisor for "
                                    "'Unit interval scaling'. "
                                    "For 8-bit images use 255, for 16-bit images use 65535."
                                ),
                                "value": 255.0,
                                "default_value": 255.0,
                                "value_type": "float",
                                "min_value": 0.0,
                                "max_value": None,
                                "allowed_values": None,
                                "depends_on": {"mode": "Unit interval scaling"},
                            },
                            {
                                "type": "parameter",
                                "key": "clip_min_value",
                                "name": "Clip minimum value",
                                "description": (
                                    "Minimum output value after rescaling the image in 'Range scaling with clipping' "
                                    "mode; pixel values below this threshold are clipped."
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
                                "key": "clip_max_value",
                                "name": "Clip maximum value",
                                "description": (
                                    "Maximum output value after rescaling the image in 'Range scaling with clipping' "
                                    "mode; pixel values above this threshold are clipped."
                                ),
                                "value": 255.0,
                                "default_value": 255.0,
                                "value_type": "float",
                                "min_value": 0.0,
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
                                "value": 200.0,
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
                                "value": 400.0,
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
                                    "Multiplicative factor applied to pixel values, "
                                    "before clipping the result to [clip_min_value, clip_max_value]."
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
                            "Data augmentation is a technique used in machine learning to artificially expand a "
                            "training dataset by applying transformations (e.g., rotation, scaling, noise) to "
                            "existing data. It improves model generalization and reduces overfitting by "
                            "increasing data variability without collecting new samples."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter_group",
                                "key": "random_zoom_out",
                                "name": "Random zoom out",
                                "description": (
                                    "Randomly zoom out the image by placing it on a larger canvas with "
                                    "padding. Applied before resize."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "fill",
                                        "name": "Fill value",
                                        "description": (
                                            "Fill value for the area outside the image when zooming out. "
                                            "Typically 0 for black padding. Value should be between 0 and 255."
                                        ),
                                        "value": 128,
                                        "default_value": 0,
                                        "value_type": "int",
                                        "min_value": 0,
                                        "max_value": 255,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "side_range",
                                        "name": "Side range",
                                        "description": (
                                            "Range (min, max) of the zoom-out scale factor for each side. "
                                            "The image will be placed on a canvas scaled by a random factor "
                                            "from this range. For example, (1.0, 4.0) means the canvas can "
                                            "be up to 4x the original image size. The minimum value must "
                                            "be >= 1.0."
                                        ),
                                        "value": [1.0, 3.0],
                                        "default_value": [1.0, 4.0],
                                        "value_type": "float_range",
                                        "min_value": 1.0,
                                        "max_value": 16.0,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying random zoom out. "
                                            "A value of 0.5 means each image has a 50% chance to be zoomed out."
                                        ),
                                        "value": 0.4,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "iou_random_crop",
                                "name": "IoU random crop",
                                "description": (
                                    "Randomly crop images based on Intersection over Union (IoU) criteria. "
                                    "Applied before resize. Note: this augmentation is not supported when "
                                    "tiling is enabled."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying IoU random crop. "
                                            "A value of 1.0 means the crop is always applied when enabled."
                                        ),
                                        "value": 0.9,
                                        "default_value": 1.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "min_scale",
                                        "name": "Minimum scale",
                                        "description": (
                                            "Minimum fraction of the original image area to retain after "
                                            "cropping. For example, 0.3 means the crop will be at least 30% "
                                            "of the original area."
                                        ),
                                        "value": 0.4,
                                        "default_value": 0.3,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "max_scale",
                                        "name": "Maximum scale",
                                        "description": (
                                            "Maximum fraction of the original image area to retain after "
                                            "cropping. A value of 1.0 means the crop can be as large as the "
                                            "entire image."
                                        ),
                                        "value": 0.9,
                                        "default_value": 1.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "mosaic",
                                "name": "Mosaic",
                                "description": (
                                    "Combines 4 images into one mosaic for augmentation. Applied before resize. "
                                    "Note: this augmentation is not supported when tiling is enabled."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying mosaic augmentation. "
                                            "A value of 1.0 means mosaic is always applied when enabled."
                                        ),
                                        "value": 0.8,
                                        "default_value": 1.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "scale",
                                        "name": "Scale",
                                        "description": (
                                            "Scale factor for the random perspective crop applied after "
                                            "mosaic assembly. "
                                            "The random scale is sampled from [1 - scale, 1 + scale]. "
                                            "Higher values produce more aggressive zoom variation."
                                        ),
                                        "value": 0.4,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "translate",
                                        "name": "Translate",
                                        "description": (
                                            "Translate fraction for the random perspective crop. "
                                            "The crop centre shifts by up to +/- translate * image_size pixels. "
                                            "Higher values allow more positional variation."
                                        ),
                                        "value": 0.2,
                                        "default_value": 0.1,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "random_affine",
                                "name": "Random affine",
                                "description": (
                                    "Apply random affine transformations (rotation, translation, scaling, shear) "
                                    "to the image. Applied after resize."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "max_rotate_degree",
                                        "name": "Rotation degrees",
                                        "description": (
                                            "Maximum rotation angle in degrees for affine transformation. "
                                            "A random angle in the range [-max_rotate_degree, max_rotate_degree] "
                                            "will be applied. For example, max_rotate_degree=10 allows up to ±10 "
                                            "degrees rotation."
                                        ),
                                        "value": 15.0,
                                        "default_value": 10.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "max_translate_ratio",
                                        "name": "Horizontal translation",
                                        "description": (
                                            "Maximum translation as a fraction of image width or height. "
                                            "A random translation in the range [-max_translate_ratio, "
                                            "max_translate_ratio] will be applied along both axes. For example, "
                                            "0.1 allows up to ±10% translation."
                                        ),
                                        "value": 0.2,
                                        "default_value": 0.1,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
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
                                        "value": [0.6, 1.4],
                                        "default_value": [0.5, 1.5],
                                        "value_type": "float_range",
                                        "min_value": 0.0,
                                        "max_value": 10.0,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "max_shear_degree",
                                        "name": "Maximum shear degree",
                                        "description": (
                                            "Maximum absolute shear angle in degrees to apply during affine "
                                            "transformation. A random shear in the range [-max_shear_degree, "
                                            "max_shear_degree] will be applied."
                                        ),
                                        "value": 5.0,
                                        "default_value": 2.0,
                                        "value_type": "float",
                                        "min_value": None,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying the affine transformation. "
                                            "A value of 0.5 means each image has a 50% chance to be transformed."
                                        ),
                                        "value": 0.7,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "mixup",
                                "name": "Mixup",
                                "description": (
                                    "Blends two images and their labels for augmentation. Applied before resize. "
                                    "Note: this augmentation is not supported when tiling is enabled."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": "Probability of applying mixup augmentation",
                                        "value": 0.6,
                                        "default_value": 1.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "alpha",
                                        "name": "Alpha",
                                        "description": (
                                            "Controls how two images are blended together. "
                                            "Low values (e.g. 0.5) produce uneven blending where one image dominates. "
                                            "A value of 1.0 gives any blend ratio equal chance. "
                                            "Higher values (e.g. 1.5-3.0) favour an equal 50/50 mix of both images."
                                        ),
                                        "value": 2.0,
                                        "default_value": 1.5,
                                        "value_type": "float",
                                        "min_value": 0.1,
                                        "max_value": 10.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "color_jitter",
                                "name": "Color jitter",
                                "description": (
                                    "Randomly adjust brightness, contrast, saturation, and hue of the image. "
                                    "Applied after resize."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "brightness",
                                        "name": "Brightness range",
                                        "description": (
                                            "Range (min, max) of brightness adjustment factors. "
                                            "A random factor from this range will be multiplied with the image "
                                            "brightness. For example, (0.8, 1.2) means brightness can be reduced "
                                            "by 20% or increased by 20%."
                                        ),
                                        "value": [0.9, 1.1],
                                        "default_value": [0.8, 1.2],
                                        "value_type": "float_range",
                                        "min_value": 0.0,
                                        "max_value": 5.0,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "contrast",
                                        "name": "Contrast range",
                                        "description": (
                                            "Range (min, max) of contrast adjustment factors. "
                                            "A random factor from this range will be multiplied with the image "
                                            "contrast. For example, (0.5, 1.5) means contrast can be halved or "
                                            "increased by up to 50%."
                                        ),
                                        "value": [0.85, 1.15],
                                        "default_value": [0.75, 1.25],
                                        "value_type": "float_range",
                                        "min_value": 0.0,
                                        "max_value": 5.0,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "saturation",
                                        "name": "Saturation range",
                                        "description": (
                                            "Range (min, max) of saturation adjustment factors. "
                                            "A random factor from this range will be multiplied with the image "
                                            "saturation. For example, (0.5, 1.5) means saturation can be halved "
                                            "or increased by up to 50%."
                                        ),
                                        "value": [0.8, 1.2],
                                        "default_value": [0.9, 1.1],
                                        "value_type": "float_range",
                                        "min_value": 0.0,
                                        "max_value": 5.0,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "hue",
                                        "name": "Hue range",
                                        "description": (
                                            "Range (min, max) of hue adjustment values. "
                                            "A random value from this range will be added to the image hue. "
                                            "For example, (-0.05, 0.05) means hue can be shifted by up to ±0.05."
                                        ),
                                        "value": [-0.05, 0.05],
                                        "default_value": [-0.1, 0.1],
                                        "value_type": "float_range",
                                        "min_value": -0.5,
                                        "max_value": 0.5,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying color jitter. "
                                            "A value of 0.5 means each image has a 50% chance to be color jittered."
                                        ),
                                        "value": 0.6,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "random_erasing",
                                "name": "Random erasing",
                                "description": (
                                    "Randomly erase a rectangular region in the image and fill it with a "
                                    "constant value. "
                                    "Also known as Cutout. Helps the model learn to rely on broader context "
                                    "rather than "
                                    "specific local features. Applied after resize."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "scale",
                                        "name": "Erasing area scale range",
                                        "description": (
                                            "Range (min, max) of the proportion of the image area to erase. "
                                            "For example, (0.02, 0.33) means erasing between 2% and 33% of the "
                                            "image area."
                                        ),
                                        "value": [0.03, 0.25],
                                        "default_value": [0.02, 0.33],
                                        "value_type": "float_range",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "ratio",
                                        "name": "Erasing aspect ratio range",
                                        "description": (
                                            "Range (min, max) of the aspect ratio of the erased area. "
                                            "For example, (0.3, 3.3) allows the erased rectangle to have "
                                            "varying proportions."
                                        ),
                                        "value": [0.5, 2.0],
                                        "default_value": [0.3, 3.3],
                                        "value_type": "float_range",
                                        "min_value": 0.0,
                                        "max_value": 10.0,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying random erasing. "
                                            "A value of 0.5 means each image has a 50% chance to have a "
                                            "region erased."
                                        ),
                                        "value": 0.4,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "value",
                                        "name": "Fill value",
                                        "description": (
                                            "Fill value for the erased region, normalized to [0, 1]. "
                                            "A value of 0.0 fills with black. A value of 1.0 fills with white."
                                        ),
                                        "value": 0.1,
                                        "default_value": 0.0,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "random_grayscale",
                                "name": "Random grayscale",
                                "description": (
                                    "Randomly convert the image to grayscale. Forces the model to learn "
                                    "shape and texture "
                                    "features rather than relying solely on color information. "
                                    "Applied after resize."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of converting the image to grayscale. "
                                            "A value of 0.1 means each image has a 10% chance to be "
                                            "converted to grayscale."
                                        ),
                                        "value": 0.2,
                                        "default_value": 0.1,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter_group",
                                "key": "random_sharpness",
                                "name": "Random sharpness",
                                "description": (
                                    "Randomly adjust the sharpness of the image. Complements Gaussian "
                                    "blur by also "
                                    "allowing images to become sharper, improving robustness to varying "
                                    "image quality. "
                                    "Applied after resize."
                                ),
                                "depends_on": {"deim_framework": [False, None]},
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": "Toggle to apply this augmentation.",
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "sharpness",
                                        "name": "Sharpness factor",
                                        "description": (
                                            "Factor controlling the strength of the sharpness adjustment. "
                                            "A value of 0.0 means no sharpening, higher values increase "
                                            "the effect. "
                                            "Typical values are between 0.0 and 1.0."
                                        ),
                                        "value": 0.8,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "probability",
                                        "name": "Probability",
                                        "description": (
                                            "Probability of applying sharpness adjustment. "
                                            "A value of 0.5 means each image has a 50% chance to be "
                                            "sharpened."
                                        ),
                                        "value": 0.3,
                                        "default_value": 0.5,
                                        "value_type": "float",
                                        "min_value": 0.0,
                                        "max_value": 1.0,
                                        "allowed_values": None,
                                        "depends_on": None,
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
                "depends_on": None,
                "parameters": [
                    {
                        "type": "parameter",
                        "key": "max_epochs",
                        "name": "Maximum epochs",
                        "description": (
                            "Maximum number of epochs to train the model. An epoch is one complete pass through the "
                            "training dataset."
                        ),
                        "value": 120,
                        "default_value": 250,
                        "value_type": "int",
                        "min_value": 1,
                        "max_value": None,
                        "allowed_values": None,
                        "depends_on": None,
                    },
                    {
                        "type": "parameter",
                        "key": "batch_size",
                        "name": "Batch size",
                        "description": (
                            "Number of training samples processed before the model's internal parameters are updated. "
                            "A larger batch size can speed up training but may require more memory, while a smaller "
                            "batch size can help avoid OOM (Out of Memory) errors at the cost of longer training times "
                            "and potentially noisier gradient estimates."
                        ),
                        "value": 8,
                        "default_value": 4,
                        "value_type": "int",
                        "min_value": 1,
                        "max_value": None,
                        "allowed_values": None,
                        "depends_on": None,
                    },
                    {
                        "type": "parameter",
                        "key": "num_workers",
                        "name": "Number of workers",
                        "description": (
                            "Number of subprocesses used for data loading. 0 means that the data will be loaded in "
                            "the main process. More workers can speed up data loading and augmentation, at the cost "
                            "of higher CPU and memory usage."
                        ),
                        "value": 4,
                        "default_value": 4,
                        "value_type": "int",
                        "min_value": 0,
                        "max_value": 64,
                        "allowed_values": None,
                        "depends_on": None,
                    },
                    {
                        "type": "parameter_group",
                        "key": "early_stopping",
                        "name": "Early stopping",
                        "description": (
                            "Early stopping is a technique to prevent overfitting by stopping training when "
                            "performance on a validation set stops improving."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "enable",
                                "name": "Enable",
                                "description": "Toggle to enable or disable early stopping during training.",
                                "value": True,
                                "default_value": False,
                                "value_type": "bool",
                                "depends_on": None,
                            },
                            {
                                "type": "parameter",
                                "key": "patience",
                                "name": "Patience",
                                "description": (
                                    "Number of epochs with no improvement after which training will be stopped."
                                ),
                                "value": 5,
                                "default_value": 1,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": None,
                                "allowed_values": None,
                                "depends_on": None,
                            },
                        ],
                    },
                    {
                        "type": "parameter",
                        "key": "learning_rate",
                        "name": "Learning rate",
                        "description": (
                            "Learning rate for the optimizer, controlling the step size during model weight updates. "
                            "A smaller learning rate may lead to more stable convergence, while a larger learning rate "
                            "may speed up training but risk overshooting minima in the loss landscape."
                        ),
                        "value": 0.001,
                        "default_value": 0.0015,
                        "value_type": "float",
                        "min_value": 0,
                        "max_value": 1,
                        "allowed_values": None,
                        "depends_on": None,
                    },
                    {
                        "type": "parameter",
                        "key": "weight_decay",
                        "name": "Weight decay",
                        "description": (
                            "Weight decay is a regularization technique that adds a penalty to the loss function "
                            "based on the squared magnitude of the model weights (L2 regularization). "
                            "It helps prevent overfitting by discouraging large weight values."
                        ),
                        "value": 0.01,
                        "default_value": 1e-4,
                        "value_type": "float",
                        "min_value": 0,
                        "max_value": 1,
                        "allowed_values": None,
                        "depends_on": None,
                    },
                    {
                        "type": "parameter_group",
                        "key": "scheduler",
                        "name": "Learning rate scheduler",
                        "description": (
                            "The learning rate scheduler adjusts the learning rate during training according to a "
                            "predefined schedule or based on validation performance, helping to improve convergence "
                            "and training stability."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "type",
                                "name": "Scheduler type",
                                "description": (
                                    "Type of learning rate scheduler to use during training. With ReduceLROnPlateau, "
                                    "the learning rate will be reduced by a predetermined factor when the validation "
                                    "metric stops improving. With CosineAnnealing, the learning rate will follow a "
                                    "cosine decay schedule, gradually decreasing over the course of training."
                                ),
                                "value": "Reduce LR on loss plateau",
                                "default_value": "Reduce LR on loss plateau",
                                "value_type": "str",
                                "allowed_values": ["Reduce LR on loss plateau"],
                                "depends_on": None,
                            },
                            {
                                "type": "parameter_group",
                                "key": "warmup",
                                "name": "Learning rate linear warmup",
                                "description": (
                                    "Learning rate warmup is a technique where the learning rate starts at a lower "
                                    "value and gradually increases to the initial learning rate over a specified "
                                    "number of epochs at the beginning of training. This can help stabilize training "
                                    "and improve convergence, especially when using large learning rates or training "
                                    "on complex datasets."
                                ),
                                "depends_on": None,
                                "parameters": [
                                    {
                                        "type": "parameter",
                                        "key": "enable",
                                        "name": "Enable",
                                        "description": (
                                            "Toggle to enable or disable the LR linear warmup phase at the "
                                            "beginning of training."
                                        ),
                                        "value": True,
                                        "default_value": False,
                                        "value_type": "bool",
                                        "depends_on": None,
                                    },
                                    {
                                        "type": "parameter",
                                        "key": "epochs",
                                        "name": "Warmup epochs",
                                        "description": "Number of epochs for the LR linear warmup phase.",
                                        "value": 3,
                                        "default_value": 5,
                                        "value_type": "int",
                                        "min_value": 1,
                                        "max_value": None,
                                        "allowed_values": None,
                                        "depends_on": None,
                                    },
                                ],
                            },
                            {
                                "type": "parameter",
                                "key": "factor",
                                "name": "Factor",
                                "description": (
                                    "Factor by which the learning rate will be reduced. new_lr = lr * factor."
                                ),
                                "value": 0.5,
                                "default_value": 0.1,
                                "value_type": "float",
                                "min_value": 0,
                                "max_value": 1,
                                "allowed_values": None,
                                "depends_on": {"type": "reduce_lr_on_plateau"},
                            },
                            {
                                "type": "parameter",
                                "key": "patience",
                                "name": "Patience",
                                "description": (
                                    "Number of epochs with no improvement after which learning rate will be reduced."
                                ),
                                "value": 7,
                                "default_value": 10,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": None,
                                "allowed_values": None,
                                "depends_on": {"type": "reduce_lr_on_plateau"},
                            },
                            {
                                "type": "parameter",
                                "key": "min_lr",
                                "name": "Minimum learning rate",
                                "description": "Minimum learning rate after annealing.",
                                "value": 1e-5,
                                "default_value": 1e-6,
                                "value_type": "float",
                                "min_value": 0,
                                "max_value": 1,
                                "allowed_values": None,
                                "depends_on": {"type": "cosine_annealing"},
                            },
                        ],
                    },
                    {
                        "type": "parameter_group",
                        "key": "gradient_accumulation",
                        "name": "Gradient accumulation",
                        "description": (
                            "Gradient accumulation allows simulating larger batch sizes by accumulating gradients "
                            "over multiple forward/backward passes before updating the model weights."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "enable",
                                "name": "Enable",
                                "description": "Toggle to enable or disable gradient accumulation during training.",
                                "value": True,
                                "default_value": False,
                                "value_type": "bool",
                                "depends_on": None,
                            },
                            {
                                "type": "parameter",
                                "key": "batches",
                                "name": "Gradient accumulation batches",
                                "description": (
                                    "Number of steps (batches) to accumulate gradients before performing gradient "
                                    "descent step. Effective batch size during training: "
                                    "batch_size * accumulate_grad_batches."
                                ),
                                "value": 4,
                                "default_value": 1,
                                "value_type": "int",
                                "min_value": 1,
                                "max_value": None,
                                "allowed_values": None,
                                "depends_on": None,
                            },
                        ],
                    },
                    {
                        "type": "parameter_group",
                        "key": "gradient_clip",
                        "name": "Gradient clipping",
                        "description": (
                            "Gradient clipping prevents exploding gradients by capping gradient norms during "
                            "backpropagation."
                        ),
                        "depends_on": None,
                        "parameters": [
                            {
                                "type": "parameter",
                                "key": "enable",
                                "name": "Enable",
                                "description": "Toggle to enable or disable gradient clipping during training.",
                                "value": True,
                                "default_value": False,
                                "value_type": "bool",
                                "depends_on": None,
                            },
                            {
                                "type": "parameter",
                                "key": "max_grad_norm",
                                "name": "Maximum gradient L2 norm",
                                "description": (
                                    "Maximum L2 norm of the gradients. Gradients with norm larger than this value will "
                                    "be clipped."
                                ),
                                "value": 2.0,
                                "default_value": 1.0,
                                "value_type": "float",
                                "min_value": 0,
                                "max_value": None,
                                "allowed_values": None,
                                "depends_on": None,
                            },
                        ],
                    },
                    {
                        "type": "parameter",
                        "key": "input_size_width",
                        "name": "Input size width",
                        "description": (
                            "Width size in pixels for model input images. Determines the horizontal resolution at "
                            "which images are processed."
                        ),
                        "value": 256,
                        "default_value": 512,
                        "value_type": "int",
                        "min_value": 0,
                        "max_value": None,
                        "allowed_values": [128, 256, 512],
                        "depends_on": None,
                    },
                    {
                        "type": "parameter",
                        "key": "input_size_height",
                        "name": "Input size height",
                        "description": (
                            "Height size in pixels for model input images. Determines the vertical resolution at "
                            "which images are processed."
                        ),
                        "value": 256,
                        "default_value": 512,
                        "value_type": "int",
                        "min_value": 0,
                        "max_value": None,
                        "allowed_values": [128, 256, 512],
                        "depends_on": None,
                    },
                ],
            },
            {
                "type": "parameter_group",
                "key": "evaluation",
                "name": "Evaluation parameters",
                "description": "Configurable parameters related to the model evaluation.",
                "depends_on": None,
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
                        "depends_on": None,
                    },
                ],
            },
        ]
    }


@pytest.fixture
def fxt_training_configuration_with_deim() -> TrainingConfiguration:
    """Create a training configuration with deim_framework=True."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(
            dataset_preparation=TaskLevelDatasetPreparationParameters(
                subset_split=SubsetSplit(training=75, validation=15, test=10),
                filtering=Filtering(
                    min_annotation_pixels=MinAnnotationPixels(enable=True, value=20),
                    min_annotation_objects=MinAnnotationObjects(enable=False, value=1),
                    max_annotation_objects=MaxAnnotationObjects(enable=False, value=50),
                ),
            ),
        ),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(
                augmentation=AugmentationParameters(
                    deim_framework=True,
                    random_zoom_out=RandomZoomOut(
                        enable=True,
                        fill=128,
                        side_range=(1.0, 3.0),
                        probability=0.4,
                    ),
                    iou_random_crop=RandomIOUCrop(
                        enable=True,
                        probability=0.9,
                        min_scale=0.4,
                        max_scale=0.9,
                    ),
                )
            ),
            training=AlgoLevelTrainingParameters(
                max_epochs=120,
                batch_size=8,
                early_stopping=EarlyStopping(enable=True, patience=5),
                learning_rate=0.001,
                weight_decay=0.01,
                scheduler=SchedulerParameters(
                    type=SchedulerType.COSINE_ANNEALING,
                    warmup=LrLinearWarmupParameters(enable=True, epochs=3),
                    factor=0.5,
                    patience=7,
                    min_lr=1e-5,
                ),
                gradient_accumulation=GradientAccumulationParameters(enable=True, batches=4),
                gradient_clip=GradientClipParameters(enable=True, max_grad_norm=2.0),
                input_size_width=256,
                input_size_height=256,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


@pytest.fixture
def fxt_default_training_configuration_with_deim() -> TrainingConfiguration:
    """Create a default training configuration with deim_framework=False."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(
                augmentation=AugmentationParameters(
                    deim_framework=False,
                    random_zoom_out=RandomZoomOut(
                        enable=False,
                        fill=0,
                        side_range=(1.0, 4.0),
                        probability=0.5,
                    ),
                    iou_random_crop=RandomIOUCrop(
                        enable=False,
                        probability=1.0,
                        min_scale=0.3,
                        max_scale=1.0,
                    ),
                )
            ),
            training=AlgoLevelTrainingParameters(
                max_epochs=250,
                batch_size=4,
                early_stopping=EarlyStopping(enable=False, patience=1),
                learning_rate=0.0015,
                weight_decay=1e-4,
                scheduler=SchedulerParameters(
                    type=SchedulerType.REDUCE_LR_ON_PLATEAU,
                    warmup=LrLinearWarmupParameters(enable=False, epochs=5),
                    factor=0.1,
                    patience=10,
                    min_lr=1e-6,
                ),
                gradient_accumulation=GradientAccumulationParameters(enable=False, batches=1),
                gradient_clip=GradientClipParameters(enable=False, max_grad_norm=1.0),
                input_size_width=512,
                input_size_height=512,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


class TestTrainingConfigurationView:
    def test_from_training_configuration(
        self, fxt_training_configuration, fxt_default_training_configuration, fxt_training_configuration_view_json
    ):
        view = TrainingConfigurationView.from_training_configuration(
            fxt_training_configuration, fxt_default_training_configuration
        )

        assert view.model_dump(mode="json") == fxt_training_configuration_view_json

    def test_from_training_configuration_with_deim_framework(
        self,
        fxt_training_configuration_with_deim,
        fxt_default_training_configuration_with_deim,
    ):
        """Test that bool | None fields like deim_framework are correctly handled."""
        view = TrainingConfigurationView.from_training_configuration(
            fxt_training_configuration_with_deim, fxt_default_training_configuration_with_deim
        )
        result = view.model_dump(mode="json")

        # Find the augmentation group within dataset_preparation
        dataset_prep = result["parameters"][0]
        assert dataset_prep["key"] == "dataset_preparation"

        augmentation_group = None
        for param in dataset_prep["parameters"]:
            if param["key"] == "augmentation":
                augmentation_group = param
                break
        assert augmentation_group is not None

        # Find the deim_framework parameter in the augmentation group
        deim_param = None
        for param in augmentation_group["parameters"]:
            if param.get("key") == "deim_framework":
                deim_param = param
                break
        assert deim_param is not None, "deim_framework parameter should be present when set to True"

        # Verify it is correctly typed as bool, not str
        assert deim_param["value_type"] == "bool"
        assert deim_param["value"] is True
        assert deim_param["default_value"] is False
        assert deim_param["type"] == "parameter"
        assert deim_param["name"] == "DEIM framework"

    def test_from_training_configuration_with_deim_framework_false(
        self,
        fxt_default_training_configuration_with_deim,
    ):
        """Test that deim_framework=False is also correctly handled as bool."""
        # Use the default config (deim_framework=False) as both config and default
        view = TrainingConfigurationView.from_training_configuration(
            fxt_default_training_configuration_with_deim, fxt_default_training_configuration_with_deim
        )
        result = view.model_dump(mode="json")

        dataset_prep = result["parameters"][0]
        augmentation_group = None
        for param in dataset_prep["parameters"]:
            if param["key"] == "augmentation":
                augmentation_group = param
                break
        assert augmentation_group is not None

        deim_param = None
        for param in augmentation_group["parameters"]:
            if param.get("key") == "deim_framework":
                deim_param = param
                break
        assert deim_param is not None, "deim_framework parameter should be present when set to False"

        assert deim_param["value_type"] == "bool"
        assert deim_param["value"] is False
        assert deim_param["default_value"] is False

    def test_from_training_configuration_deim_framework_none_is_excluded(
        self,
        fxt_training_configuration,
        fxt_default_training_configuration,
    ):
        """Test that deim_framework=None (default) is excluded from the view."""
        view = TrainingConfigurationView.from_training_configuration(
            fxt_training_configuration, fxt_default_training_configuration
        )
        result = view.model_dump(mode="json")

        dataset_prep = result["parameters"][0]
        augmentation_group = None
        for param in dataset_prep["parameters"]:
            if param["key"] == "augmentation":
                augmentation_group = param
                break
        assert augmentation_group is not None

        # deim_framework should NOT be present when it's None
        deim_param = None
        for param in augmentation_group["parameters"]:
            if param.get("key") == "deim_framework":
                deim_param = param
                break
        assert deim_param is None, "deim_framework parameter should be excluded when value is None"
