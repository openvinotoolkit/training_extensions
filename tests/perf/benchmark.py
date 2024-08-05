# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX benchmark runner."""

from __future__ import annotations

import gc
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Literal

import numpy as np
import pandas as pd

from .history import summary

log = logging.getLogger(__name__)


class AggregateError(Exception):
    def __init__(self, errors):
        error_messages = []
        for seed, error in errors:
            error_messages.append(f"Seed {seed}: {error}")
        error_message = "\n".join(error_messages)

        super().__init__(f"Exceptions occurred in the following seeds:\n{error_message}")


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
        reference_results (pd.DataFrame): Reference benchmark results for performance checking.
        resume_from (Path | None, optional):
            Previous performance directory to load. If training was already done in previous performance test,
            training is skipped and refer previous result.
        test_only (Literal["all", "train", "export", "optimize"] | None):
            Execute test only when resume argument is given. If necessary files are not found in resume directory,
            necessary operations can be executed. Defaults to None.
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
        group: str
        num_repeat: int = 1
        extra_overrides: dict | None = None
        unlabeled_data_path: Path | None = None

    @dataclass
    class Criterion:
        """Benchmark criterion."""

        name: str
        summary: str
        compare: str
        margin: float

        def __call__(self, result_entry: pd.Series, target_entry: pd.Series) -> None:
            """Check result against given target."""
            if self.name not in result_entry or result_entry[self.name] is None or np.isnan(result_entry[self.name]):
                print(f"[Check] {self.name} not in result")
                return
            if self.name not in target_entry or target_entry[self.name] is None or np.isnan(target_entry[self.name]):
                print(f"[Check] {self.name} not in target")
                return
            if self.compare == "==":
                print(
                    f"[Check] abs({self.name}:{result_entry[self.name]} - {self.name}:{target_entry[self.name]}) < {self.name}:{target_entry[self.name]} * {self.margin}",
                )
                assert abs(result_entry[self.name] - target_entry[self.name]) < target_entry[self.name] * self.margin
            elif self.compare == "<":
                print(
                    f"[Check] {self.name}:{result_entry[self.name]} < {self.name}:{target_entry[self.name]} * (1.0 + {self.margin})",
                )
                assert result_entry[self.name] < target_entry[self.name] * (1.0 + self.margin)
            elif self.compare == ">":
                print(
                    f"[Check] {self.name}:{result_entry[self.name]} > {self.name}:{target_entry[self.name]} * (1.0 - {self.margin})",
                )
                assert result_entry[self.name] > target_entry[self.name] * (1.0 - self.margin)

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
        reference_results: pd.DataFrame | None = None,
        resume_from: Path | None = None,
        test_only: Literal["all", "train", "export", "optimize"] | None = None,
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
        self.reference_results = reference_results
        self.resume_from = resume_from
        if (test_only == "export" and eval_upto == "train") or (
            test_only == "optimize" and eval_upto in ["train", "export"]
        ):
            msg = "test_only should be set to previous otx command than eval_upto."
            raise ValueError(msg)
        self.test_only = test_only

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
            "data_group": dataset.group,
            "model": model.name,
            "data": dataset.name,
            **self.tags,
        }

        num_repeat = dataset.num_repeat
        if self.num_repeat > 0:
            num_repeat = self.num_repeat  # Override by global setting

        exceptions = []
        for seed in range(num_repeat):
            try:
                sub_work_dir = work_dir / str(seed)
                tags["seed"] = str(seed)

                # Train & test
                copied_ops_dir = self._prepare_resume(tags, sub_work_dir)
                command = [
                    "otx",
                    "train",
                    "--config",
                    f"src/otx/recipe/{model.task}/{model.name}.yaml",
                    "--data_root",
                    str(data_root),
                    "--work_dir",
                    str(sub_work_dir),
                    "--engine.device",
                    self.accelerator,
                ]

                # Add unlabeled data path if exists
                if dataset.unlabeled_data_path is not None:
                    command.extend(
                        ["--data.unlabeled_subset.data_root", str(self.data_root / dataset.unlabeled_data_path)],
                    )

                for key, value in dataset.extra_overrides.get("train", {}).items():
                    command.append(f"--{key}")
                    command.append(str(value))
                command.extend(["--seed", str(seed)])
                # TODO(someone): Disable deterministic for instance segmentation as it causes OOM.
                # https://github.com/pytorch/vision/issues/8168#issuecomment-1890599205
                command.extend(["--deterministic", str(self.deterministic)])
                if self.num_epoch > 0:
                    command.extend(["--max_epochs", str(self.num_epoch)])
                extra_metrics = {}
                if "train" in copied_ops_dir:
                    command.append("--print_config")
                    with (sub_work_dir / ".latest" / "train" / "configs.yaml").open("w") as f:
                        self._run_command(command, stdout=f)  # replace previuos configs.yaml to new one
                else:
                    start_time = time()
                    self._run_command(command)
                    extra_metrics["train/e2e_time"] = time() - start_time
                    self._rename_raw_data(
                        work_dir=sub_work_dir / ".latest" / "train",
                        replaces={"train_": "train/", "{pre}": "train/"},
                    )
                self._log_metrics(
                    work_dir=sub_work_dir / ".latest" / "train",
                    tags=tags,
                    criteria=criteria,
                    extra_metrics=extra_metrics,
                )

                self._run_test(sub_work_dir, dataset, tags, criteria, what2test="train")

                # Export & test
                if self.eval_upto in ["export", "optimize"]:
                    if "export" not in copied_ops_dir:
                        command = [
                            "otx",
                            "export",
                            "--work_dir",
                            str(sub_work_dir),
                        ]
                        for key, value in dataset.extra_overrides.get("export", {}).items():
                            command.append(f"--{key}")
                            command.append(str(value))
                        self._run_command(command)

                    exported_model_path = sub_work_dir / ".latest" / "export" / "exported_model.xml"
                    if not exported_model_path.exists():
                        exported_model_path = sub_work_dir / ".latest" / "export" / "exported_model_decoder.xml"

                    self._run_test(
                        sub_work_dir,
                        dataset,
                        tags,
                        criteria,
                        checkpoint=exported_model_path,
                        what2test="export",
                    )

                # Optimize & test
                if self.eval_upto == "optimize":
                    if "optimize" not in copied_ops_dir:
                        command = [
                            "otx",
                            "optimize",
                            "--checkpoint",
                            str(exported_model_path),
                            "--work_dir",
                            str(sub_work_dir),
                        ]
                        for key, value in dataset.extra_overrides.get("optimize", {}).items():
                            command.append(f"--{key}")
                            command.append(str(value))
                        self._run_command(command)

                    optimized_model_path = sub_work_dir / ".latest" / "optimize" / "optimized_model.xml"
                    if not optimized_model_path.exists():
                        optimized_model_path = sub_work_dir / ".latest" / "optimize" / "optimized_model_decoder.xml"

                    self._run_test(
                        sub_work_dir,
                        dataset,
                        tags,
                        criteria,
                        checkpoint=optimized_model_path,
                        what2test="optimize",
                    )

                # Force memory clean up
                gc.collect()
            except Exception as e:  # noqa: PERF203
                exceptions.append((seed, str(e)))

        if exceptions:
            # Raise the custom exception with all collected errors
            raise AggregateError(exceptions)

        result = self.load_result(work_dir)
        if result is None:
            return None
        result = summary.average(result, keys=["task", "model", "data_group", "data"])  # Average out seeds
        return result.set_index(["task", "model", "data_group", "data"])

    def _prepare_resume(self, tags: dict[str, str], work_dir: Path) -> list[str]:
        copied_ops_dir = []
        if self.resume_from is None:
            return copied_ops_dir
        prev_work_dir = self._find_corresponding_dir(tags)
        if prev_work_dir is None:
            return copied_ops_dir

        latest_dir = work_dir / ".latest"

        if self.test_only is None:
            copy_until = "train"
        elif self.test_only == "all":
            copy_until = "optimize"
        else:
            copy_until = self.test_only

        for otx_cmd in ["train", "export", "optimize"]:
            prev_symlink = prev_work_dir / ".latest" / otx_cmd
            try:  # check symlink exists
                prev_symlink.readlink()
            except FileNotFoundError:
                break
            prev_cmd_dir_name = prev_symlink.resolve().name
            prev_cmd_dir = prev_work_dir / prev_cmd_dir_name
            if not prev_cmd_dir.exists():
                break

            if not latest_dir.exists():
                latest_dir.mkdir(parents=True)

            shutil.copytree(prev_cmd_dir, work_dir / prev_cmd_dir_name, ignore_dangling_symlinks=True)
            (latest_dir / otx_cmd).symlink_to(Path("..") / (work_dir / prev_cmd_dir_name).relative_to(work_dir))

            copied_ops_dir.append(otx_cmd)
            if otx_cmd == copy_until:
                break

        if copy_until != otx_cmd:
            log.warning(
                f"There is no {otx_cmd} directory for {work_dir} in resume directory. "
                f"{work_dir} starts from {otx_cmd}.",
            )

        return copied_ops_dir

    def _find_corresponding_dir(self, tags: dict[str, str]) -> Path | None:
        if self.resume_from is None:
            return None
        for csv_file in self.resume_from.rglob("benchmark.raw.csv"):
            if csv_file.parent.name == ".latest":
                continue
            raw_data = pd.read_csv(csv_file)
            if all(  # check meta info is same
                str(raw_data.iloc[0].get(key, "NOT_IN_CSV")) == tags.get(key, "NOT_IN_TAG")
                for key in ["data_group", "data", "model", "task", "seed"]
            ):
                return csv_file.parent.parent
        return None

    def _run_test(
        self,
        work_dir: Path | str,
        dataset: Dataset,
        tags: dict[str, str],
        criteria: list[Criterion],
        checkpoint: Path | str | None = None,
        what2test: Literal["train", "export", "optimize"] = "train",
    ) -> None:
        """Run otx test and update result csv file to align it's indices to the current task."""
        if self.test_only not in ["all", what2test, None]:
            return

        replace_map = {
            "train": {"test_": "test/", "{pre}": "test/"},
            "export": {"test": what2test, "{pre}": f"{what2test}/"},
            "optimize": {"test": what2test, "{pre}": f"{what2test}/"},
        }

        command = [
            "otx",
            "test",
            "--work_dir",
            str(work_dir),
        ]
        if checkpoint is not None:
            command.extend(["--checkpoint", str(checkpoint)])
        for key, value in dataset.extra_overrides.get("test", {}).items():
            command.append(f"--{key}")
            command.append(str(value))

        start_time = time()
        self._run_command(command)
        extra_metrics = {f"test({what2test})/e2e_time": time() - start_time}

        self._rename_raw_data(
            work_dir=work_dir / ".latest" / "test",
            replaces=replace_map[what2test],
        )
        self._log_metrics(
            work_dir=work_dir / ".latest" / "test",
            tags=tags,
            criteria=criteria,
            extra_metrics=extra_metrics,
        )

    def _run_command(self, command: list[str], **kwargs) -> None:
        """Run command using 'subprocess.run'.

        Args:
            command (list[str]): command to execute.
            kwags: arguments to 'subprocess.run'.
        """
        print(" ".join(command))
        kwargs["check"] = True
        if not self.dry_run:
            subprocess.run(command, **kwargs)  # noqa: S603, PLW1510

    def _log_metrics(
        self,
        work_dir: Path,
        tags: dict[str, str],
        criteria: list[Criterion],
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        if not work_dir.exists():
            return

        # Load raw metrics
        csv_files = work_dir.glob("**/metrics.csv")
        raw_data = [pd.read_csv(csv_file) for csv_file in csv_files]
        raw_data = pd.concat(raw_data, ignore_index=True)
        if extra_metrics:
            for k, v in extra_metrics.items():
                raw_data[k] = v

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
        replaces = {**self.NAME_MAPPING, **replaces}

        def _rename_col(col_name: str) -> str:
            for src_str, dst_str in replaces.items():
                if src_str == "{pre}":
                    if not col_name.startswith(dst_str):
                        col_name = dst_str + col_name
                elif src_str == "{post}":
                    if not col_name.endswith(dst_str):
                        col_name = col_name + dst_str
                else:
                    col_name = col_name.replace(src_str, dst_str)
            return col_name

        csv_files = work_dir.glob("**/metrics.csv")
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            data = data.rename(columns=_rename_col)  # Column names
            data = data.replace(replaces)  # Values
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

        return pd.concat(results, ignore_index=True)

    def check(self, result: pd.DataFrame, criteria: list[Criterion]):
        """Check result w.r.t. reference data.

        Args:
            result (pd.DataFrame): Result data frame
            criteria (list[Criterion]): Criteria to check results
        """
        if result is None:
            print("[Check] No results loaded. Skipping result checking.")
            return

        if self.reference_results is None:
            print("[Check] No benchmark references loaded. Skipping result checking.")
            return

        for key, result_entry in result.iterrows():
            if key not in self.reference_results.index:
                print(f"[Check] No benchmark reference for {key} loaded. Skipping result checking.")
                continue
            target_entry = self.reference_results.loc[key]
            if isinstance(target_entry, pd.DataFrame):
                # Match num_repeat of result and target
                result_seed_average = result_entry["seed"]
                result_num_repeat = 2 * result_seed_average + 1  # (0+1+2+3+4)/5 = 2.0 -> 2*2.0+1 = 5
                target_entry = target_entry.query(f"seed < {result_num_repeat}")
                target_entry = target_entry.mean(numeric_only=True)  # N-row pd.DataFrame to pd.Series

            for criterion in criteria:
                criterion(result_entry, target_entry)

    NAME_MAPPING: dict[str, str] = {}  # noqa: RUF012
