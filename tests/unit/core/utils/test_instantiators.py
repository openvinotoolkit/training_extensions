# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from otx.core.config.data import SamplerConfig
from otx.core.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_sampler,
    partial_instantiate_class,
)


def test_instantiate_callbacks() -> None:
    callbacks_cfg = [
        {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "save_last": True,
                "save_top_k": 1,
                "monitor": "val_loss",
                "mode": "min",
            },
        },
        {
            "class_path": "lightning.pytorch.callbacks.EarlyStopping",
            "init_args": {
                "monitor": "val_loss",
                "mode": "min",
                "patience": 3,
            },
        },
    ]

    callbacks = instantiate_callbacks(callbacks_cfg=callbacks_cfg)
    assert len(callbacks) == 2
    assert callbacks[0].__class__.__name__ == "ModelCheckpoint"
    assert callbacks[1].__class__.__name__ == "EarlyStopping"

    callbacks = instantiate_callbacks(callbacks_cfg=[])
    assert len(callbacks) == 0


def test_instantiate_loggers() -> None:
    logger_cfg = [
        {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": "logs",
                "name": "tb_logs",
            },
        },
    ]

    loggers = instantiate_loggers(logger_cfg=logger_cfg)
    assert len(loggers) == 1
    assert loggers[0].__class__.__name__ == "TensorBoardLogger"

    loggers = instantiate_loggers(logger_cfg=None)
    assert len(loggers) == 0


def test_partial_instantiate_class() -> None:
    init = {
        "class_path": "torch.optim.SGD",
        "init_args": {
            "lr": 0.0049,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        },
    }

    partial = partial_instantiate_class(init=init)
    assert len(partial) == 1
    assert partial[0].__class__.__name__ == "partial"
    assert partial[0].func.__name__ == "SGD"
    assert partial[0].keywords == init["init_args"]
    assert partial[0].args == ()

    partial = partial_instantiate_class(init=None)
    assert partial is None


def test_instantiate_sampler(mocker) -> None:
    sampler_cfg = SamplerConfig(
        class_path="torch.utils.data.WeightedRandomSampler",
        init_args={
            "num_samples": 10,
            "replacement": True,
        },
    )

    mock_dataset = mocker.MagicMock()
    sampler = instantiate_sampler(sampler_config=sampler_cfg, dataset=mock_dataset)
    assert sampler.__class__.__name__ == "WeightedRandomSampler"
    assert sampler.num_samples == sampler_cfg.init_args["num_samples"]
    assert sampler.replacement == sampler_cfg.init_args["replacement"]
