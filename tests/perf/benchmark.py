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
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from otx.cli.cli import OTXCLI

log = logging.getLogger(__name__)


class Benchmark:
    """Benchmark runner for OTX2.x.

    Args:
        benchmark_type (str): 'accuracy' or 'efficiency'
        data_root (str): Path to the root of dataset directories. Defaults to './data'.
        output_root (str): Output root dirctory for logs and results. Defaults to './otx-benchmark'.
        criteria (list[Criterion]): Benchmark criteria settings
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
        deterministic (bool): Whether to turn on deterministic training mode. Defaults to False.
        accelerator (str): Accelerator device on which to run benchmark. Defaults to gpu.
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
    class Criterion:
        """Benchmark criterion."""
        name: str
        summary: str
        compare: str
        margin: float

    def __init__(
        self,
        benchmark_type: str = "accuracy",
        data_root: Path = Path("data"),
        output_root: Path = Path("otx-benchmark"),
        criteria: list[Criterion] | None = None,
        num_epoch: int = 0,
        num_repeat: int = 1,
        eval_upto: str = "train",
        tags: dict[str,str] | None = None,
        dry_run: bool = False,
        deterministic: bool = False,
        accelerator: str = "gpu",
    ):
        self.benchmark_type = benchmark_type
        self.data_root = data_root
        self.output_root = output_root
        self.criteria = criteria
        self.num_epoch = num_epoch
        self.num_repeat = num_repeat
        self.eval_upto = eval_upto
        self.tags = tags or {}
        self.dry_run = dry_run
        self.deterministic = deterministic
        self.accelerator = accelerator

        if num_epoch == 0:  # 0: use default
            if benchmark_type == "efficiency":
                self.num_epoch = 2

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

        run_name = f"{self.benchmark_type}/{model.task}/{model.name}/{dataset.name}"
        log.info(f"{run_name = }")
        work_dir = self.output_root / run_name
        data_root = self.data_root / dataset.path

        tags = {
            "benchmark": self.benchmark_type,
            "task": model.task,
            "data_size": dataset.size,
            "model": model.name,
            "dataset": dataset.name,
            **self.tags,
        }

        num_repeat = dataset.num_repeat
        if self.num_repeat > 0:
            num_repeat = self.num_repeat  # Override by global setting

        for seed in range(num_repeat):
            sub_work_dir = work_dir / str(seed)
            tags["seed"] = str(seed)

            # Train & test
            command = [
                "otx", "train",
                "--config", f"src/otx/recipe/{model.task}/{model.name}.yaml",
                "--data_root", str(data_root),
                "--work_dir", str(sub_work_dir),
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
            self._run_command(command)

            command = [
                "otx", "test",
                "--work_dir", str(sub_work_dir),
            ]
            self._run_command(command)

            # TODO: Export & test
            # TODO: Optimize & test

            self._log_metrics(work_dir=sub_work_dir, tags=tags)

            # Force memory clean up
            gc.collect()

        return self.load_result(work_dir)

    def _run_command(self, command: list[str]):
        if self.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)

    def _log_metrics(self, work_dir: Path, tags: dict[str, str]):
        if not work_dir.exists():
            return
        # Load raw metrics
        csv_files = glob.glob(f"{work_dir}/**/metrics.csv", recursive=True)
        raw_data = []
        for csv_file in csv_files:
            raw_data.append(pd.read_csv(csv_file))
        raw_data = pd.concat(raw_data, ignore_index=True)
        # Summarize
        metrics = []
        for criterion in self.criteria:
            if criterion.name not in raw_data:
                continue
            column = raw_data[criterion.name].dropna()
            if len(column) == 0:
                continue
            if criterion.summary == "mean":
                value = column[(len(column)-1):].mean()  # Drop 1st epoch if possible
            elif criterion.summary == "max":
                value = column.max()
            elif criterion.summary == "min":
                value = column.min()
            else:
                value = 0.0
            metrics.append(pd.Series([value], name=criterion.name))
        if len(metrics) == 0:
            return
        metrics = pd.concat(metrics, axis=1)
        # Write csv w/ tags
        for k, v in tags.items():
            metrics[k] = v
        metrics.to_csv(work_dir / "benchmark.raw.csv", index=False)

    @staticmethod
    def load_result(result_path: Path) -> pd.DataFrame | None:
        """Load benchmark results recursively and merge as pd.DataFrame.

        Args:
            result_path (Path): Result directory or speicific file.

        Retruns:
            pd.DataFrame: Table with benchmark metrics & options
        """
        # Search csv files
        if not result_path.exists():
            return None
        if result_path.is_dir():
            csv_files = glob.glob(f"{result_path}/**/benchmark.raw.csv", recursive=True)
        else:
            csv_files = [result_path]
        results = []
        # Load csv data
        for csv_file in csv_files:
            result = pd.read_csv(csv_file)
            results.append(result)
        # Merge data
        if len(results) > 0:
            data = pd.concat(results, ignore_index=True)
            # Average by unique group
            grouped = data.groupby(["benchmark", "task", "data_size", "model"])
            aggregated = grouped.mean(numeric_only=True)
            # Merge tag columns (non-numeric & non-index)
            tag_columns = set(data.columns) - set(aggregated.columns) - set(grouped.keys)
            for col in tag_columns:
                # Take common string prefix such as: ["data/1", "data/2", "data/3"] -> "data/"
                aggregated[col] = grouped[col].agg(lambda x: os.path.commonprefix(x.tolist()))
            return aggregated
        else:
            return None
