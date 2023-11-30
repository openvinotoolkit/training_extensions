# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import DictConfig
from otx.cli.utils.hydra import configure_hydra_outputs


def test_configure_hydra_outputs(mocker) -> None:
    # Mock hydra functions
    mock_configure_log = mocker.patch("otx.cli.utils.hydra.configure_log")
    mock_save_config = mocker.patch("otx.cli.utils.hydra._save_config")
    mock_set_config = mocker.patch("hydra.core.hydra_config.HydraConfig.set_config")

    cfg = DictConfig({
        "hydra": {
            "job": {"name": "test"},
            "output_subdir": "outputs/.hydra",
            "runtime": {"output_dir": "outputs"},
            "overrides": {"task": "test"},
            "verbose": False,
            "job_logging": {
                'handlers': {
                    'file': {
                        'class': 'logging.FileHandler',
                        'formatter': 'simple',
                        'filename': '${hydra.runtime.output_dir}/${hydra.job.name}.log',
                    },
                },
            },
        },
        "base": {
            'work_dir': 'outputs/work_dir',
            'data_dir': 'inputs/dataset_dir',
            'log_dir': '${base.work_dir}/logs/',
            'output_dir': 'outputs/output_dir',
        },
    })
    configure_hydra_outputs(cfg)
    mock_configure_log.assert_called_once_with(cfg.hydra.job_logging, False)

    updated_cfg = cfg.copy()
    output_dir = Path(cfg.base.output_dir).resolve()
    updated_cfg.hydra.job_logging.handlers.file.filename = output_dir / f"{cfg.hydra.job.name}.log"
    updated_cfg.hydra.runtime.output_dir = output_dir
    mock_set_config.assert_called_once_with(updated_cfg)
    assert mock_save_config.call_count == 2
