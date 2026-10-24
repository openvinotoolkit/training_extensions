# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock, patch

import pytest
from jsonargparse import ArgumentParser, Namespace

from getitune.cli.utils.jsonargparse import (
    apply_config,
    flatten_dict,
    get_configuration,
    get_short_docstring,
    list_override,
    namespace_override,
    patch_update_configs,
)


@pytest.fixture
def fxt_configs() -> Namespace:
    return Namespace(
        data=Namespace(
            train_subset=Namespace(
                batch_size=32,
                num_workers=4,
                transforms=[
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {
                            "keep_ratio": True,
                            "transform_bbox": True,
                            "transform_mask": True,
                            "scale": [1024, 1024],
                        },
                    },
                    {
                        "class_path": "torchvision.transforms.v2.Pad",
                        "init_args": {"pad_to_square": True, "transform_mask": True},
                    },
                    {
                        "class_path": "torchvision.transforms.v2.RandomHorizontalFlip",
                        "init_args": {"p": 0.5},
                    },
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.float32"}},
                    {
                        "class_path": "torchvision.transforms.v2.Normalize",
                        "init_args": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]},
                    },
                ],
                sampler=Namespace(class_path="torch.utils.data.RandomSampler", init_args={}),
            ),
        ),
        callbacks=[
            Namespace(
                class_path="getitune.backend.lightning.callbacks.iteration_timer.IterationTimer",
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


@pytest.mark.parametrize(
    "reset",
    [
        "data.train_subset.transforms",
        ["data.train_subset.transforms"],
        ["data.train_subset.transforms", "callbacks"],
    ],
)
def test_apply_config_with_reset(fxt_configs: Namespace, reset: str | list[str]) -> None:
    cfg = deepcopy(fxt_configs)
    with (
        patch_update_configs(),
        patch("jsonargparse.Path.__init__", return_value=None),
    ):
        # test for reset
        overrides = Namespace(
            overrides=Namespace(
                reset=reset,
                callbacks=[{"class_path": "new_callbacks"}],
                data=Namespace(
                    train_subset=Namespace(
                        transforms=[
                            {
                                "class_path": "torchvision.transforms.v2.ToImage",
                            },
                        ],
                    ),
                ),
            ),
        )

        mock_parser = Mock(spec=ArgumentParser)
        mock_parser.parse_path.return_value = overrides
        mock_parser.merge_config = ArgumentParser().merge_config

        apply_config(None, mock_parser, cfg, "dest", "value")

        # check values that should not to be changed
        assert cfg.data.train_subset.batch_size == fxt_configs.data.train_subset.batch_size
        assert cfg.data.train_subset.num_workers == fxt_configs.data.train_subset.num_workers
        assert cfg.data.train_subset.sampler == fxt_configs.data.train_subset.sampler
        assert cfg.logger == fxt_configs.logger

        # check values that should be changed
        assert len(cfg.data.train_subset.transforms) == len(overrides.overrides.data.train_subset.transforms)
        assert cfg.data.train_subset.transforms[0]["class_path"] == "torchvision.transforms.v2.ToImage"
        if isinstance(reset, list) and "callbacks" in reset:
            assert len(cfg.callbacks) == len(overrides.overrides.callbacks)
            assert cfg.callbacks[0].class_path == "new_callbacks"
        else:
            assert len(cfg.callbacks) == len(fxt_configs.callbacks) + 1
            assert cfg.callbacks[:-1] == fxt_configs.callbacks
            assert cfg.callbacks[-1].class_path == "new_callbacks"


def test_namespace_override(fxt_configs) -> None:
    cfg = deepcopy(fxt_configs)
    with patch_update_configs():
        # test for empty override
        overrides = Namespace()

        namespace_override(configs=cfg, key="data", overrides=overrides, convert_dict_to_namespace=False)

        assert cfg.data.train_subset.batch_size == fxt_configs.data.train_subset.batch_size
        assert cfg.data.train_subset.num_workers == fxt_configs.data.train_subset.num_workers
        assert cfg.data.train_subset.transforms == fxt_configs.data.train_subset.transforms
        assert cfg.data.train_subset.sampler == fxt_configs.data.train_subset.sampler
        assert cfg.callbacks == fxt_configs.callbacks
        assert cfg.logger == fxt_configs.logger

        # test for single key override
        overrides = Namespace(
            mem_cache_img_max_size=[100, 100],
            train_subset=Namespace(batch_size=64, num_workers=8),
        )

        namespace_override(configs=cfg, key="data", overrides=overrides, convert_dict_to_namespace=False)

        assert cfg.data.mem_cache_img_max_size == overrides.mem_cache_img_max_size
        assert cfg.data.train_subset.batch_size == overrides.train_subset.batch_size
        assert cfg.data.train_subset.num_workers == overrides.train_subset.num_workers

        # test for dict of list by using list_override
        overrides = Namespace(
            train_subset=Namespace(
                transforms=[
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {
                            "keep_ratio": False,  # for boolean
                            "scale": [512, 512],  # for tuple
                        },
                    },
                    {
                        "class_path": "torchvision.transforms.v2.Pad",
                        "init_args": {"size_divisor": 32},  # add new key
                    },
                    {
                        "class_path": "torchvision.transforms.v2.Normalize",
                        "init_args": {"std": [1.0, 1.0, 1.0]},  # for the last component
                    },
                ],
            ),
        )

        # to check before adding key
        pad_transform = next(
            t for t in cfg.data.train_subset.transforms if t["class_path"] == "torchvision.transforms.v2.Pad"
        )
        assert "size_divisor" not in pad_transform["init_args"]

        namespace_override(configs=cfg, key="data", overrides=overrides, convert_dict_to_namespace=False)

        # Find transforms by class_path since order may change
        resize_transform = next(
            t
            for t in cfg.data.train_subset.transforms
            if t["class_path"] == "getitune.data.augmentation.transforms.Resize"
        )
        pad_transform = next(
            t for t in cfg.data.train_subset.transforms if t["class_path"] == "torchvision.transforms.v2.Pad"
        )
        normalize_transform = next(
            t for t in cfg.data.train_subset.transforms if t["class_path"] == "torchvision.transforms.v2.Normalize"
        )

        # getitune.data.augmentation.transforms.Resize
        assert (
            resize_transform["init_args"]["keep_ratio"]
            == overrides.train_subset.transforms[0]["init_args"]["keep_ratio"]
        )
        assert resize_transform["init_args"]["scale"] == overrides.train_subset.transforms[0]["init_args"]["scale"]
        # torchvision.transforms.v2.Pad
        assert "size_divisor" in pad_transform["init_args"]
        assert (
            pad_transform["init_args"]["size_divisor"]
            == overrides.train_subset.transforms[1]["init_args"]["size_divisor"]
        )
        # torchvision.transforms.v2.Normalize
        assert normalize_transform["init_args"]["std"] == overrides.train_subset.transforms[-1]["init_args"]["std"]

        # test for appending new transform
        overrides = Namespace(
            train_subset=Namespace(
                transforms=[
                    {
                        "class_path": "torchvision.transforms.v2.ToImage",
                    },
                ],
            ),
        )

        namespace_override(configs=cfg, key="data", overrides=overrides, convert_dict_to_namespace=False)

        assert all(isinstance(transform, dict) for transform in cfg.data.train_subset.transforms)
        assert cfg.data.train_subset.transforms[-1]["class_path"] == "torchvision.transforms.v2.ToImage"

        # test for namespace override to update init_args
        overrides = Namespace(
            train_subset=Namespace(
                sampler=Namespace(
                    class_path="torch.utils.data.RandomSampler",
                    init_args={"efficient_mode": True},
                ),
            ),
        )

        namespace_override(configs=cfg, key="data", overrides=overrides, convert_dict_to_namespace=False)

        assert (
            cfg.data.train_subset.sampler.init_args["efficient_mode"]
            == overrides.train_subset.sampler.init_args["efficient_mode"]
        )

        # test for namespace override to update class_path
        overrides = Namespace(
            train_subset=Namespace(
                sampler=Namespace(
                    class_path="getitune.algo.samplers.balanced_sampler.BalancedSampler",
                ),
            ),
        )

        namespace_override(configs=cfg, key="data", overrides=overrides, convert_dict_to_namespace=False)

        assert cfg.data.train_subset.sampler.class_path == overrides.train_subset.sampler.class_path


def test_list_override(fxt_configs) -> None:
    with patch_update_configs():
        list_override(fxt_configs, "callbacks", [])
        # Find callbacks by class_path since order may change with the new list_override behavior
        iter_timer = next(
            c
            for c in fxt_configs.callbacks
            if c.class_path == "getitune.backend.lightning.callbacks.iteration_timer.IterationTimer"
        )
        early_stop = next(
            c for c in fxt_configs.callbacks if c.class_path == "lightning.pytorch.callbacks.EarlyStopping"
        )
        model_summary = next(
            c for c in fxt_configs.callbacks if c.class_path == "lightning.pytorch.callbacks.RichModelSummary"
        )
        assert iter_timer.init_args.prog_bar
        assert early_stop.init_args.patience == 10
        assert model_summary.init_args.max_depth == 1

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
        early_stop = next(
            c for c in fxt_configs.callbacks if c.class_path == "lightning.pytorch.callbacks.EarlyStopping"
        )
        assert early_stop.init_args.patience == 3

        logger_override = [
            {
                "class_path": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
                "init_args": {"name": "workspace/"},
            },
        ]
        list_override(fxt_configs, "logger", logger_override)
        tb_logger = next(
            lg
            for lg in fxt_configs.logger
            if lg.class_path == "lightning.pytorch.loggers.tensorboard.TensorBoardLogger"
        )
        assert tb_logger.init_args.name == "workspace/"

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
        updated_configs = fxt_configs.update(8, "data.train_subset.batch_size")
        assert updated_configs.data.train_subset.batch_size == 8

        updated_configs = fxt_configs.update("8", "data.train_subset.batch_size")
        assert updated_configs.data.train_subset.batch_size == 8

        updated_configs = fxt_configs.update(None, "data.train_subset.num_workers")
        assert "num_workers" not in updated_configs.data.train_subset

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
        callback_monitor: test/f1
        """,
    )

    # Call the get_configuration function
    config = get_configuration(config_file)
    assert "config" in config
    assert str(config["config"][0]) == str(config_file)
    assert "engine" in config
    assert "data" in config
    assert config["data"]["task"] == "SEMANTIC_SEGMENTATION"

    cli_args = ["verbose", "data_root", "seed", "callback_monitor", "resume", "disable_infer_num_classes"]
    for arg in cli_args:
        assert arg not in config
