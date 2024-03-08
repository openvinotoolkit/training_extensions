# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX benchmark runner."""

from __future__ import annotations

import gc
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class Benchmark:
    """Benchmark runner for OTX2.x.

    Args:
        data_root (str): Path to the root of dataset directories. Defaults to './data'.
        output_root (str): Output root dirctory for logs and results. Defaults to './otx-benchmark'.
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
        category: str

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
        data_root: Path = Path("data"),
        output_root: Path = Path("otx-benchmark"),
        num_epoch: int = 0,
        num_repeat: int = 1,
        eval_upto: str = "train",
        tags: dict[str, str] | None = None,
        dry_run: bool = False,
        deterministic: bool = False,
        accelerator: str = "gpu",
    ):
        self.data_root = data_root
        self.output_root = output_root
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
        criteria: list[Criterion],
    ) -> pd.DataFrame | None:
        """Run configured benchmark with given dataset and model and return the result.

        Args:
            model (Model): Target model settings
            dataset (Dataset): Target dataset settings
            criteria (list[Criterion]): Target criteria settings

        Retruns:
            pd.DataFrame | None: Table with benchmark metrics
        """

        run_name = f"{model.task}/{model.name}/{dataset.name}"
        log.info(f"{run_name = }")
        work_dir = self.output_root / run_name
        data_root = self.data_root / dataset.path

        tags = {
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
                "otx",
                "train",
                "--config",
                f"src/otx/recipe/{model.task}/{model.name}.yaml",
                "--data_root",
                str(data_root),
                "--work_dir",
                str(sub_work_dir),
                "--model.num_classes",
                str(dataset.num_classes),
                "--data.config.data_format",
                dataset.data_format,
                "--engine.device",
                self.accelerator,
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
                "otx",
                "test",
                "--work_dir",
                str(sub_work_dir),
            ]
            self._run_command(command)

            # Export & test
            if self.eval_upto in ["export", "optimize"]:
                command = [
                    "otx",
                    "export",
                    "--work_dir",
                    str(sub_work_dir),
                ]
                self._run_command(command)

                command = [  # NOTE: not working for h_label_cls. to be fixed
                    "otx",
                    "test",
                    "--config",
                    str(sub_work_dir / ".latest" / "export" / "configs.yaml"),
                    "--checkpoint",
                    str(sub_work_dir / ".latest" / "export" / "exported_model.xml"),
                    "--work_dir",
                    str(sub_work_dir),
                ]
                self._run_command(command)

                self._rename_raw_data(work_dir=sub_work_dir / ".latest" / "test", replaces={"test": "export"})

            # Optimize & test
            if self.eval_upto == "optimize":
                command = [
                    "otx",
                    "optimize",
                    # NOTE: auto config should be implemented
                    "--config",
                    f"src/otx/recipe/{model.task}/openvino_model.yaml",
                    "--checkpoint",
                    str(sub_work_dir / ".latest" / "export" / "exported_model.xml"),
                    "--work_dir",
                    str(sub_work_dir),
                ]
                self._run_command(command)

                command = [
                    "otx",
                    "test",
                    # NOTE: auto config should be implemented
                    "--config",
                    f"src/otx/recipe/{model.task}/openvino_model.yaml",
                    "--checkpoint",
                    str(sub_work_dir / ".latest" / "optimize" / "optimized_model.xml"),
                    "--work_dir",
                    str(sub_work_dir),
                ]
                self._run_command(command)

                self._rename_raw_data(work_dir=sub_work_dir / ".latest" / "test", replaces={"test": "optimize"})

            # Parse raw data into raw metrics
            self._log_metrics(work_dir=sub_work_dir, tags=tags, criteria=criteria)

            # Force memory clean up
            gc.collect()

        return self.load_result(work_dir)

    def _run_command(self, command: list[str]) -> None:
        if self.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)  # noqa: S603

    def _log_metrics(self, work_dir: Path, tags: dict[str, str], criteria: list[Benchmark.Criterion]) -> None:
        if not work_dir.exists():
            return

        # Load raw metrics
        csv_files = work_dir.glob("**/metrics.csv")
        raw_data = [pd.read_csv(csv_file) for csv_file in csv_files]
        raw_data = pd.concat(raw_data, ignore_index=True)

        # Summarize
        metrics = []
        for criterion in criteria:
            if criterion.name not in raw_data:
                continue
            column = raw_data[criterion.name].dropna()
            if len(column) == 0:
                continue
            if criterion.summary == "mean":
                value = column[min(1, len(column) - 1) :].mean()  # Drop 1st epoch if possible
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

    def _rename_raw_data(self, work_dir: Path, replaces: dict[str, str]) -> None:
        csv_files = work_dir.glob("**/metrics.csv")
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            for src_str, dst_str in replaces.items():
                data.columns = data.columns.str.replace(src_str, dst_str)
            data.to_csv(csv_file, index=False)

    @staticmethod
    def load_result(result_path: Path) -> pd.DataFrame | None:
        """Load benchmark results recursively and merge as pd.DataFrame.

        Args:
            result_path (Path): Result directory or speicific file.

        Retruns:
            pd.DataFrame: Table with benchmark metrics & options
        """
        if not result_path.exists():
            return None
        # Load csv data
        csv_files = result_path.glob("**/benchmark.raw.csv") if result_path.is_dir() else [result_path]
        results = [pd.read_csv(csv_file) for csv_file in csv_files]
        if len(results) == 0:
            return None
        # Merge data
        data = pd.concat(results, ignore_index=True)
        # Average by unique group
        grouped = data.groupby(["task", "data_size", "model"])
        aggregated = grouped.mean(numeric_only=True)
        # Merge tag columns (non-numeric & non-index)
        tag_columns = set(data.columns) - set(aggregated.columns) - set(grouped.keys)
        for col in tag_columns:
            # Take common string prefix such as: ["data/1", "data/2", "data/3"] -> "data/"
            aggregated[col] = grouped[col].agg(lambda x: os.path.commonprefix(x.tolist()))
         # Average by task
        task_grouped = data.groupby(["task"], as_index=False)
        task_aggregated = task_grouped.mean(numeric_only=True)
        task_aggregated["data_size"] = "all"
        task_aggregated["model"] = "all"
        task_aggregated.set_index(["task", "data_size", "model"], inplace=True)
        return pd.concat([aggregated, task_aggregated])
