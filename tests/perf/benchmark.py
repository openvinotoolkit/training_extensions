"""OTX Benchmark based on tools/experiment.py."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import glob
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional

from tests.test_suite.run_test_command import check_run


class OTXBenchmark:
    """Benchmark runner based on tools/experiment.py in OTX1.x.

    Example:
        >>> bm = OTXBenchmark(['random_sample1', 'random_sample2'], data_root='./data/coco')
        >>> atss_result = bm.run('MobileNetV2-ATSS')
        >>> yolox_result = bm.run('YOLOX-TINY')

    Args:
        datasets (List[str]): Paths to datasets relative to the data_root.
            Intended for, but not restricted to different sampling based on same dataset.
        data_root (str): Path to the root of dataset directories. Defaults to './data'.
        num_epoch (int): Overrides the per-model default number of epoch settings.
            Defaults to 0, which means no overriding.
        num_repeat (int): Number for trials with different random seed, which would be set
            as range(0, num_repeat). Defaults to 1.
        train_params (dict): Additional training parameters.
            e.x) {'learning_parameters.num_iters': 2}. Defaults to {}.
        track_resources (bool): Whether to track CPU & GPU usage metrics. Defaults to False.
        eval_upto (str): The last serial operation to evaluate. Choose one of ('train', 'export', 'optimize').
            Operations include the preceeding ones.
            e.x) Eval up to 'optimize': train -> eval -> export -> eval -> optimize -> eval
            Default to 'train'.
        output_root (str): Output root dirctory for logs and results. Defaults to './otx-benchmark'.
        dry_run (bool): Whether to just print the OTX command without execution. Defaults to False.
        tags (dict): Key-values pair metadata for the experiment. Defaults to {}.
    """

    def __init__(
        self,
        datasets: List[str],
        data_root: str = "data",
        num_epoch: int = 0,
        num_repeat: int = 1,
        train_params: dict | None = None,
        track_resources: bool = False,
        eval_upto: str = "train",
        output_root: str = "otx-benchmark",
        dry_run: bool = False,
        tags: dict | None = None,
        subset_dir_names: dict | None = None,
    ):
        self.datasets = datasets
        self.data_root = data_root
        self.num_epoch = num_epoch
        self.num_repeat = num_repeat
        self.train_params = train_params or {}
        self.track_resources = track_resources
        self.eval_upto = eval_upto
        self.output_root = output_root
        self.dry_run = dry_run
        self.tags = tags or {}
        self.subset_dir_names = subset_dir_names or {"train": "", "val": "", "test": ""}

    def run(
        self,
        model_id: str,
        train_params: dict = {},
        tags: dict = {},
    ) -> pd.DataFrame | None:
        """Run configured benchmark with given model and return the result.

        Args:
            model_id (str): Target model identifier
            train_params (dict): Overrides global benchmark train params
            tags (dict): Overrides global benchmark tags

        Retruns:
            pd.DataFrame | None: Table with benchmark metrics
        """

        # Build config file
        cfg = self._build_config(model_id, train_params, tags)
        cfg_dir = Path(cfg["output_path"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "cfg.yaml"
        with open(cfg_path, "w") as cfg_file:
            yaml.dump(cfg, cfg_file, indent=2)
        cmd = [
            "python",
            "tools/experiment.py",
            "-f",
            cfg_path,
        ]
        if self.dry_run:
            cmd.append("-d")
        # Run benchmark
        check_run(cmd)
        # Load result
        result = self.load_result(cfg_dir)
        return result

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
            return pd.concat(results, ignore_index=True)
        else:
            return None

    def _build_config(
        self,
        model_id: str,
        train_params: dict = {},
        tags: dict = {},
    ) -> dict:
        """Build config for tools/expeirment.py."""
        all_train_params = self.train_params.copy()
        all_train_params.update(train_params)
        all_tags = self.tags.copy()
        all_tags.update(tags)

        cfg = {}
        cfg["tags"] = all_tags  # metadata
        cfg["output_path"] = os.path.abspath(Path(self.output_root) / "-".join(list(all_tags.values()) + [model_id]))
        cfg["constants"] = {
            "dataroot": os.path.abspath(self.data_root),
        }
        cfg["variables"] = {
            "model": [model_id],
            "data": self.datasets,
        }
        cfg["repeat"] = self.num_repeat
        cfg["command"] = []
        resource_param = ""
        if self.track_resources:
            resource_param = "--track-resource-usage all"
        if self.num_epoch > 0:
            self._set_num_epoch(model_id, all_train_params, self.num_epoch)
        params_str = " ".join([f"--{k} {v}" for k, v in all_train_params.items()])
        cfg["command"].append(
            "otx train ${model}"
            " --train-data-roots ${dataroot}/${data}" + f"/{self.subset_dir_names['train']}"
            " --val-data-roots ${dataroot}/${data}" + f"/{self.subset_dir_names['val']}"
            " --deterministic"
            f" {resource_param}"
            f" params {params_str}"
        )
        cfg["command"].append("otx eval --test-data-roots ${dataroot}/${data}" + f"/{self.subset_dir_names['test']}")
        if self.eval_upto == "train":
            return cfg

        cfg["command"].append("otx export")
        cfg["command"].append("otx eval --test-data-roots ${dataroot}/${data}" + f"/{self.subset_dir_names['test']}")
        if self.eval_upto == "export":
            return cfg

        cfg["command"].append("otx optimize")
        cfg["command"].append("otx eval --test-data-roots ${dataroot}/${data}" + f"/{self.subset_dir_names['test']}")
        return cfg

    @staticmethod
    def _set_num_epoch(model_id: str, train_params: dict, num_epoch: int):
        """Set model specific num_epoch parameter."""
        if "padim" in model_id:
            return  # No configurable parameter for num_epoch
        elif "stfpm" in model_id:
            train_params["learning_parameters.max_epochs"] = num_epoch
        else:
            train_params["learning_parameters.num_iters"] = num_epoch
