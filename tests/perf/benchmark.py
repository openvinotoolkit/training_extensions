"""OTX Benchmark runner."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import logging
import os
import gc
import glob
import pandas as pd
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from otx.cli.cli import OTXCLI
from unittest.mock import patch

log = logging.getLogger(__name__)


class Benchmark:
    """Benchmark runner for OTX2.x.

    Args:
        data_root (str): Path to the root of dataset directories. Defaults to './data'.
        output_root (str): Output root dirctory for logs and results. Defaults to './otx-benchmark'.
        metrics (list[Metric]): Benchmark metric settings
        num_epoch (int): Overrides the per-model default number of epoch settings.
            Defaults to 0, which means no overriding.
        num_repeat (int): Number for trials with different random seed, which would be set
            as range(0, num_repeat). Defaults to 1.
        eval_upto (str): The last serial operation to evaluate. Choose one of ('train', 'export', 'optimize').
            Operations include the preceeding ones.
            e.x) Eval up to 'optimize': train -> eval -> export -> eval -> optimize -> eval
            Default to 'train'.
        tags (dict, optional): Key-values pair metadata for the experiment.
        dry_run (bool): Whether to just print the OTX command without execution. Defaults to False.
    """

    @dataclass
    class Model:
        """Benchmark model."""
        task: str
        name: str
        type: str

    @dataclass
    class Dataset:
        """Benchmark dataset."""
        name: str
        path: Path
        size: str
        data_format: str
        num_classes: int
        num_repeat: int = 1
        extra_overrides: dict | None = None

    @dataclass
    class Metric:
        """Benchmark metric."""
        name: str
        op: str
        margin: float

    def __init__(
        self,
        data_root: Path = Path("data"),
        output_root: Path = Path("otx-benchmark"),
        metrics: list[Metric] | None = None,
        num_epoch: int = 0,
        num_repeat: int = 1,
        eval_upto: str = "train",
        tags: dict[str,str] | None = None,
        dry_run: bool = False,
        deterministic: bool = False,
        accelerator: str = "gpu",
    ):
        self.data_root = data_root
        self.output_root = output_root
        self.metrics = metrics
        self.num_epoch = num_epoch
        self.num_repeat = num_repeat
        self.eval_upto = eval_upto
        self.tags = tags or {}
        self.dry_run = dry_run
        self.deterministic = deterministic
        self.accelerator = accelerator

    def run(
        self,
        model: Model,
        dataset: Dataset,
    ) -> pd.DataFrame | None:
        """Run configured benchmark with given dataset and model and return the result.

        Args:
            model (Model): Target model settings
            dataset (Dataset): Target dataset settings

        Retruns:
            pd.DataFrame | None: Table with benchmark metrics
        """

        tags = {
            "task": model.task,
            "model": model.name,
            "dataset": dataset.name,
            **self.tags,
        }

        num_repeat = dataset.num_repeat
        if self.num_repeat > 0:
            num_repeat = self.num_repeat  # Override by global setting

        for seed in range(num_repeat):
            run_name = f"{model.task}/{model.name}/{dataset.name}/{seed}"
            log.info(f"{run_name = }")
            work_dir = self.output_root / run_name
            tags["seed"] = str(seed)
            data_root = self.data_root / dataset.path

            # Train & test
            command = [
                "otx", "train",
                "--config", f"src/otx/recipe/{model.task}/{model.name}.yaml",
                "--data_root", str(data_root),
                "--work_dir", str(work_dir),
                "--model.num_classes", str(dataset.num_classes),
                "--data.config.data_format", dataset.data_format,
                "--engine.device", self.accelerator,
            ]
            for key, value in dataset.extra_overrides.items():
                command.append(f"--{key}")
                command.append(str(value))
            command.extend(["--seed", str(seed)])
            command.extend(["--deterministic", str(self.deterministic)])
            if self.num_epoch > 0:
                command.extend(["--max_epochs", str(self.num_epoch)])
            train_metrics = self._run_command(command)

            command = [
                "otx", "test",
                "--work_dir", str(work_dir),
            ]
            test_metrics = self._run_command(command)

            self._log_raw_csv(
                path=work_dir / "perf.train.csv",
                data={**tags, **train_metrics, **test_metrics}
            )

            # TODO: Export & test
            # TODO: Optimize & test

            # Force memory clean up
            gc.collect()

        return None

    def _run_command(self, command: list[str]) -> dict[str, Any]:
        if self.dry_run:
            print(" ".join(command))
            return {}
        with patch("sys.argv", command):
            log.info(f"{command = }")
            cli = OTXCLI()
        return cli.engine.trainer.callback_metrics

    def _log_raw_csv(self, path: Path, data: dict[str, Any]):
        data = {
            k: v if isinstance(v, str) else float(v)
            for k, v in data.items()
        }
        with open(path, "w") as f:
            writer = csv.DictWriter(f, data.keys())
            writer.writeheader()
            writer.writerow(data)

    @staticmethod
    def load_result(result_path: str) -> pd.DataFrame | None:
        """Load benchmark results recursively and merge as pd.DataFrame.

        Args:
            result_path (str): Result directory or speicific file.

        Retruns:
            pd.DataFrame: Table with benchmark metrics & options
        """
        # Search csv files
        if os.path.isdir(result_path):
            csv_file_paths = glob.glob(f"{result_path}/**/exp_summary.csv", recursive=True)
        else:
            csv_file_paths = [result_path]
        results = []
        # Load csv data
        for csv_file_path in csv_file_paths:
            result = pd.read_csv(csv_file_path)
            # Append metadata if any
            cfg_file_path = Path(csv_file_path).parent / "cfg.yaml"
            if cfg_file_path.exists():
                with cfg_file_path.open("r") as cfg_file:
                    tags = yaml.safe_load(cfg_file).get("tags", {})
                    for k, v in tags.items():
                        result[k] = v
            results.append(result)
        if len(results) > 0:
            # Merge experiments
            data = pd.concat(results, ignore_index=True)
            data["train_e2e_time"] = pd.to_timedelta(data["train_e2e_time"]).dt.total_seconds()  # H:M:S str -> seconds
            # Average by unique group
            grouped = data.groupby(["benchmark", "task", "data_size", "model"])
            aggregated = grouped.mean(numeric_only=True)
            # ["data/1", "data/2", "data/3"] -> "data/"
            aggregated["data"] = grouped["data"].agg(lambda x: os.path.commonprefix(x.tolist()))
            return aggregated
        else:
            return None
