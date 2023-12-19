# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from omegaconf import DictConfig
from otx.cli.utils.hydra import configure_hydra_outputs


def test_configure_hydra_outputs(mocker, tmp_path) -> None:
    # Mock hydra functions
    mock_configure_log = mocker.patch("otx.cli.utils.hydra.configure_log")
    mock_save_config = mocker.patch("otx.cli.utils.hydra._save_config")

    cfg = DictConfig(
        {
            "hydra": {
                "job": {"name": "test"},
                "output_subdir": "${base.output_dir}",
                "runtime": {"output_dir": "${base.output_dir}"},
                "overrides": {"task": "test"},
                "verbose": False,
                "job_logging": {
                    "handlers": {
                        "file": {
                            "class": "logging.FileHandler",
                            "formatter": "simple",
                            "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",
                        },
                    },
                },
            },
            "base": {
                "work_dir": str(tmp_path / "work_dir"),
                "data_dir": "inputs/dataset_dir",
                "log_dir": "${base.work_dir}/logs/",
                "output_dir": str(tmp_path / "outputs"),
            },
        },
    )
    configure_hydra_outputs(cfg)
    mock_configure_log.assert_called_once_with(cfg.hydra.job_logging, False)

    assert mock_save_config.call_count == 2
