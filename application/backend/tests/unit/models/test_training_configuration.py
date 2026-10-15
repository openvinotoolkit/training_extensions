# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.models.training_configuration.augmentation import GaussianBlur
from app.models.training_configuration.configuration import (
    AlgoLevelParameters,
    TaskLevelParameters,
    TrainingConfiguration,
)
from app.models.training_configuration.dataset_preparation import (
    AlgoLevelDatasetPreparationParameters,
    AugmentationParameters,
    IntensityMapping,
    IntensityMappingMode,
    TaskLevelDatasetPreparationParameters,
)
from app.models.training_configuration.training import AlgoLevelTrainingParameters, EarlyStopping


def make_config(
    *,
    training_pct: int = 70,
    validation_pct: int = 20,
    test_pct: int = 10,
    max_epochs: int = 50,
    learning_rate: float = 0.001,
    gaussian_blur: GaussianBlur | None = None,
) -> TrainingConfiguration:
    """Helper to build a TrainingConfiguration with sensible defaults."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(
            dataset_preparation=TaskLevelDatasetPreparationParameters.model_validate(
                {
                    "subset_split": {
                        "training": training_pct,
                        "validation": validation_pct,
                        "test": test_pct,
                    }
                }
            )
        ),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(
                augmentation=AugmentationParameters(gaussian_blur=gaussian_blur)
            ),
            training=AlgoLevelTrainingParameters(
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                early_stopping=EarlyStopping(enable=False, patience=1),
                input_size_width=224,
                input_size_height=224,
                allowed_values_input_size=[128, 224, 256],
            ),
        ),
    )


class TestTrainingConfiguration:
    def test_apply_updates_task_level_scalar(self) -> None:
        config = make_config(training_pct=70, validation_pct=20, test_pct=10)
        config.apply_updates(
            {
                "dataset_preparation.subset_split.training": 60,
                "dataset_preparation.subset_split.validation": 30,
                "dataset_preparation.subset_split.test": 10,
            }
        )
        split = config.task_level_parameters.dataset_preparation.subset_split
        assert split.training == 60
        assert split.validation == 30
        assert split.test == 10

    def test_apply_updates_algo_level_scalar(self) -> None:
        config = make_config(max_epochs=50)
        config.apply_updates({"training.max_epochs": 123})
        assert config.algo_level_parameters.training.max_epochs == 123

    def test_apply_updates_float_field(self) -> None:
        config = make_config(learning_rate=0.001)
        config.apply_updates({"training.learning_rate": 0.01})
        assert config.algo_level_parameters.training.learning_rate == pytest.approx(0.01)

    def test_apply_updates_bool_field(self) -> None:
        blur = GaussianBlur(enable=True, kernel_size=3, sigma=(0.4, 0.6), probability=0.5)
        config = make_config(gaussian_blur=blur)
        config.apply_updates({"dataset_preparation.augmentation.gaussian_blur.enable": False})
        assert config.algo_level_parameters.dataset_preparation.augmentation.gaussian_blur is not None
        assert config.algo_level_parameters.dataset_preparation.augmentation.gaussian_blur.enable is False

    def test_apply_updates_nested_model_field(self) -> None:
        blur = GaussianBlur(enable=True, kernel_size=3, sigma=(0.4, 0.6), probability=0.5)
        config = make_config(gaussian_blur=blur)
        config.apply_updates({"dataset_preparation.augmentation.gaussian_blur.kernel_size": 7})
        assert config.algo_level_parameters.dataset_preparation.augmentation.gaussian_blur is not None
        assert config.algo_level_parameters.dataset_preparation.augmentation.gaussian_blur.kernel_size == 7

    def test_apply_updates_returns_self(self) -> None:
        config = make_config()
        result = config.apply_updates({"training.max_epochs": 10})
        assert result is config

    def test_apply_updates_empty(self) -> None:
        config = make_config(max_epochs=50)
        config.apply_updates({})
        assert config.algo_level_parameters.training.max_epochs == 50

    def test_apply_updates_atomicity_on_invalid(self) -> None:
        """All-or-nothing: no updates should be applied when any path is invalid."""
        config = make_config(max_epochs=50)
        with pytest.raises(ValueError):
            config.apply_updates(
                {
                    "training.max_epochs": 99,
                    "training.nonexistent_field": 1,
                }
            )
        assert config.algo_level_parameters.training.max_epochs == 50

    def test_apply_updates_invalid_nonexistent_leaf(self) -> None:
        config = make_config()
        with pytest.raises(ValueError, match="nonexistent_field"):
            config.apply_updates({"dataset_preparation.subset_split.nonexistent_field": 5})

    def test_apply_updates_invalid_nonexistent_section(self) -> None:
        config = make_config()
        with pytest.raises(ValueError):
            config.apply_updates({"nonexistent_section.some_field": 5})

    def test_apply_updates_invalid_path_too_short(self) -> None:
        config = make_config()
        with pytest.raises(ValueError, match="at least one dot separator"):
            config.apply_updates({"max_epochs": 10})

    def test_apply_updates_intensity_mapping_mode(self) -> None:
        config = make_config()
        config.apply_updates({"dataset_preparation.intensity_mapping.mode": "Windowing"})
        assert config.task_level_parameters.dataset_preparation.intensity_mapping.mode == IntensityMappingMode.WINDOW

    def test_apply_updates_intensity_mapping_window_params(self) -> None:
        config = make_config()
        config.apply_updates(
            {
                "dataset_preparation.intensity_mapping.window_center": 500.0,
                "dataset_preparation.intensity_mapping.window_width": 1000.0,
            }
        )
        im = config.task_level_parameters.dataset_preparation.intensity_mapping
        assert im.window_center == pytest.approx(500.0)
        assert im.window_width == pytest.approx(1000.0)

    def test_intensity_mapping_defaults(self) -> None:
        im = IntensityMapping()
        assert im.mode == IntensityMappingMode.SCALE_TO_UNIT
        assert im.max_intensity_value == 255.0
        assert im.clip_min_value == 0.0
        assert im.clip_max_value == 255.0
        assert im.window_center == 127.5
        assert im.window_width == 255.0
        assert im.scale_factor == 1.0
