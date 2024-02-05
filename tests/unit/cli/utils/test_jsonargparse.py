# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from jsonargparse import Namespace
from otx.cli.utils.jsonargparse import (
    flatten_dict,
    get_configuration,
    get_short_docstring,
    list_override,
    patch_update_configs,
)


@pytest.fixture()
def fxt_configs() -> Namespace:
    return Namespace(
        data=Namespace(batch_size=32, num_workers=4),
        callbacks=[
            Namespace(
                class_path="otx.algo.callbacks.iteration_timer.IterationTimer",
                init_args=Namespace(prog_bar=True),
            ),
            Namespace(
                class_path="lightning.pytorch.callbacks.EarlyStopping",
                init_args=Namespace(patience=10),
            ),
            Namespace(
                class_path="lightning.pytorch.callbacks.RichModelSummary",
                init_args=Namespace(max_depth=1),
            ),
        ],
        logger=[
            Namespace(
                class_path="lightning.pytorch.loggers.csv_logs.CSVLogger",
                init_args=Namespace(name="csv/"),
            ),
            Namespace(
                class_path="lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
                init_args=Namespace(name="tensorboard/"),
            ),
        ],
    )


def test_list_override(fxt_configs) -> None:
    with patch_update_configs():
        list_override(fxt_configs, "callbacks", [])
        assert fxt_configs.callbacks[0].init_args.prog_bar
        assert fxt_configs.callbacks[1].init_args.patience == 10
        assert fxt_configs.callbacks[2].init_args.max_depth == 1

        # Wrong Config overriding
        wrong_override = [
            {
                "init_args": {"patience": 3},
            },
        ]
        with pytest.raises(ValueError, match="class_path is required in the override list."):
            list_override(fxt_configs, "callbacks", wrong_override)

        callbacks_override = [
            {
                "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                "init_args": {"patience": 3},
            },
        ]
        list_override(fxt_configs, "callbacks", callbacks_override)
        assert fxt_configs.callbacks[1].init_args.patience == 3

        logger_override = [
            {
                "class_path": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
                "init_args": {"name": "workspace/"},
            },
        ]
        list_override(fxt_configs, "logger", logger_override)
        assert fxt_configs.logger[1].init_args.name == "workspace/"

        new_callbacks_override = [
            {
                "class_path": "lightning.pytorch.callbacks.NewCallBack",
                "init_args": {"patience": 100},
            },
        ]
        list_override(fxt_configs, "callbacks", new_callbacks_override)
        assert len(fxt_configs.callbacks) == 4
        assert fxt_configs.callbacks[3].class_path == "lightning.pytorch.callbacks.NewCallBack"
        assert fxt_configs.callbacks[3].init_args.patience == 100


def test_update(fxt_configs) -> None:
    with patch_update_configs():
        # Test KeyError
        with pytest.raises(KeyError):
            fxt_configs.update(value=8, key=None)

        # Test updating a single key
        updated_configs = fxt_configs.update(8, "data.batch_size")
        assert updated_configs.data.batch_size == 8

        updated_configs = fxt_configs.update("8", "data.batch_size")
        assert updated_configs.data.batch_size == 8

        updated_configs = fxt_configs.update(None, "data.num_workers")
        assert "num_workers" not in updated_configs.data

        # Test updating multiple values using a namespace
        new_values = Namespace(
            callbacks=[
                Namespace(
                    class_path="new_callback",
                    init_args=Namespace(prog_bar=False),
                ),
            ],
            logger=[
                Namespace(
                    class_path="new_logger",
                    init_args=Namespace(name="new_name"),
                ),
            ],
        )
        updated_configs = fxt_configs.update(new_values)
        assert updated_configs.callbacks[0].class_path == "new_callback"
        assert updated_configs.callbacks[0].init_args.prog_bar is False
        assert updated_configs.logger[0].class_path == "new_logger"
        assert updated_configs.logger[0].init_args.name == "new_name"

        # Test updating multiple values using a dictionary
        new_values_dict = {
            "callbacks": [
                {
                    "class_path": "updated_callback",
                    "init_args": {"prog_bar": True},
                },
            ],
            "logger": [
                {
                    "class_path": "updated_logger",
                    "init_args": {"name": "updated_name"},
                },
            ],
        }
        updated_configs = fxt_configs.update(new_values_dict)
        assert updated_configs.callbacks[0].class_path == "updated_callback"
        assert updated_configs.callbacks[0].init_args.prog_bar is True
        assert updated_configs.logger[0].class_path == "updated_logger"
        assert updated_configs.logger[0].init_args.name == "updated_name"


def test_get_short_docstring() -> None:
    class Component:
        """This is a component."""

        def method(self) -> None:
            """This is a method."""

    class WithoutDocstring:
        pass

    assert get_short_docstring(Component) == "This is a component."
    assert get_short_docstring(Component.method) == "This is a method."
    assert get_short_docstring(WithoutDocstring) == ""


def test_flatten_dict() -> None:
    # Test case 1: Flattening an empty dictionary
    config: dict = {}
    flattened = flatten_dict(config)
    assert flattened == {}

    # Test case 2: Flattening a dictionary with nested keys
    config_1 = {
        "a": {
            "b": {
                "c": 1,
                "d": 2,
            },
            "e": 3,
        },
        "f": 4,
    }
    flattened = flatten_dict(config_1)
    expected_1 = {
        "a.b.c": 1,
        "a.b.d": 2,
        "a.e": 3,
        "f": 4,
    }
    assert flattened == expected_1

    # Test case 3: Flattening a dictionary with custom separator
    config_2 = {
        "a": {
            "b": {
                "c": 1,
                "d": 2,
            },
            "e": 3,
        },
        "f": 4,
    }
    flattened = flatten_dict(config_2, sep="-")
    expected_2 = {
        "a-b-c": 1,
        "a-b-d": 2,
        "a-e": 3,
        "f": 4,
    }
    assert flattened == expected_2

    # Test case 4: Flattening a dictionary with non-string keys
    config_3 = {
        1: {
            2: {
                3: "value",
            },
        },
    }
    flattened = flatten_dict(config_3)
    expected_3 = {
        "1.2.3": "value",
    }
    assert flattened == expected_3


def test_get_configuration(tmp_path):
    # Create a temporary configuration file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
        data:
            task: SEMANTIC_SEGMENTATION
        """,
    )

    # Call the get_configuration function
    config = get_configuration(config_file)
    assert "config" in config
    assert config["config"] == [config_file]
    assert "engine" in config
    assert "data" in config
    assert config["data"]["task"] == "SEMANTIC_SEGMENTATION"

    cli_args = ["verbose", "data_root", "task", "seed", "callback_monitor", "resume", "disable_infer_num_classes"]
    for arg in cli_args:
        assert arg not in config
