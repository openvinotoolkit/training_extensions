# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Util functions to run HPO."""

from __future__ import annotations

import json
from pathlib import Path


def find_file_recursively(directory: Path, file_name: str) -> Path | None:
    """Find the file from the direcotry recursively."""
    if found_file := list(directory.rglob(file_name)):
        return found_file[0]
    return None


def find_trial_file(hpo_workdir: Path, trial_id: str) -> Path | None:
    """Find a trial file which store trial record.

    Args:
        hpo_workdir (Path): HPO work directory.
        trial_id (str): trial id.

    Returns:
        Path | None: trial file. If it doesn't exist, return None.
    """
    return find_file_recursively(hpo_workdir, f"{trial_id}.json")


def get_best_hpo_weight(weight_dir: Path, trial_file: Path) -> Path | None:
    """Get best model weight path of the HPO trial.

    Args:
        weight_dir (Path): directory where model weights are saved.
        trial_file (Path): json format trial file which stores trial record.

    Returns:
        Path | None: best HPO model weight. If it doesn't exist, return None.
    """
    if not trial_file.exists():
        return None

    with trial_file.open("r") as f:
        trial_output = json.load(f)

    best_epochs = []
    best_score = None
    for eph, score in trial_output["score"].items():
        eph = str(int(eph) - 1)  # lightning uses index starting from 0
        if best_score is None:
            best_score = score
            best_epochs.append(eph)
        elif best_score < score:
            best_score = score
            best_epochs = [eph]
        elif best_score == score:
            best_epochs.append(eph)

    best_epochs.sort(key=int, reverse=True)
    for best_epoch in best_epochs:
        if (best_weight_path := find_file_recursively(weight_dir, f"epoch_*{best_epoch}.ckpt")) is not None:
            return best_weight_path

    return None


def get_hpo_weight_dir(hpo_workdir: Path, trial_id: str) -> Path:
    """Get HPO weight directory. If it doesn't exist, directory is made.

    Args:
        hpo_workdir (Path): HPO work directory.
        trial_id (str): trial id.

    Returns:
        Path: HPO weight directory path.
    """
    hpo_weight_dir: Path = hpo_workdir / "weight" / trial_id
    if not hpo_weight_dir.exists():
        hpo_weight_dir.mkdir(parents=True)
    return hpo_weight_dir


def remove_unused_files(directory: Path, pattern: str, file_to_leave: Path | None = None) -> None:
    """Remove all files macthing to pattern except file_to_leave.

    Args:
        directory (Path): direcetory to find files to remove.
        pattern (str): pattern to match a file name.
        file_not_to_remove (Path | None, optional): files to leave. Defaults to None.
    """
    for weight in directory.rglob(pattern):
        if weight != file_to_leave:
            weight.unlink()
