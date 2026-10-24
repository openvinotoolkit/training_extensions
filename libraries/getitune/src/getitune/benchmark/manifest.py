# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark manifest - parsing, filtering, and experiment enumeration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from getitune.utils import RECIPE_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Threshold:
    """A single metric regression threshold."""

    compare: str  # ">=" or "<="
    margin: float


@dataclass(frozen=True)
class Scenario:
    """A benchmark scenario (default or a parameter override)."""

    name: str
    description: str = ""
    tag: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    datasets: list[str] | None = None  # restrict to these dataset names
    models: list[str] | None = None  # restrict to these model names
    num_seeds: int | None = None  # override global num_seeds

    @staticmethod
    def default() -> Scenario:
        """Return the implicit 'default' scenario."""
        return Scenario(name="default", description="Default recipe configuration")


@dataclass(frozen=True)
class ModelEntry:
    """A model declared in the manifest.

    The recipe path is always derived from ``task`` and ``name`` as
    ``<task>/<name>.yaml`` under ``RECIPE_PATH`` -- there is no manual override.
    """

    name: str
    task: str
    priority: str = "core"

    @property
    def recipe_path(self) -> Path:
        """Absolute path to the recipe YAML."""
        return RECIPE_PATH / self.task / f"{self.name}.yaml"


@dataclass(frozen=True)
class CriteriaConfig:
    """Criteria section from the manifest for a single task."""

    accuracy_metric: str
    thresholds: dict[str, Threshold]


@dataclass(frozen=True)
class TaskSection:
    """All benchmark configuration for one task."""

    task: str
    models: list[ModelEntry]
    datasets: list[str]  # references into the catalog by name
    scenarios: list[Scenario]
    criteria: CriteriaConfig


@dataclass(frozen=True)
class ManifestDefaults:
    """Global defaults from the manifest."""

    num_seeds: int = 3
    eval_upto: str = "optimize"
    deterministic: bool = True
    rotation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkManifest:
    """Parsed representation of ``benchmark_manifest.yaml``."""

    version: int
    defaults: ManifestDefaults
    experiments: dict[str, TaskSection]  # task key -> section

    # -- querying ----------------------------------------------------------

    def all_tasks(self) -> list[str]:
        """Return all task keys declared in the manifest."""
        return list(self.experiments.keys())

    def get_task(self, task: str) -> TaskSection:
        """Look up a task section by key."""
        if task not in self.experiments:
            msg = f"Task '{task}' not found in manifest."
            raise KeyError(msg)
        return self.experiments[task]


# ---------------------------------------------------------------------------
# Experiment - an immutable unit of work
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """A single ``(task, model, dataset, scenario)`` combination.

    Seeds are iterated over externally by the runner so that each seed
    can be individually resumed.
    """

    task: str
    model: ModelEntry
    dataset_name: str
    scenario: Scenario
    eval_upto: str
    num_seeds: int
    criteria: CriteriaConfig

    @property
    def run_id(self) -> str:
        """Unique directory-safe identifier."""
        parts = [self.task, self.model.name, self.dataset_name]
        if self.scenario.name != "default":
            parts.append(self.scenario.name)
        return "/".join(parts)

    @property
    def recipe_path(self) -> Path:
        """Absolute recipe path for this experiment."""
        return self.model.recipe_path


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@dataclass
class ManifestFilters:
    """Runtime filters applied when enumerating experiments."""

    tasks: list[str] | None = None
    models: list[str] | None = None
    datasets: list[str] | None = None
    size_tiers: list[str] | None = None
    data_groups: list[str] | None = None
    priorities: list[str] | None = None
    scenarios: list[str] | None = None
    scenario_tags: list[str] | None = None
    dry_run: bool = False


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _resolve_metric_placeholder(raw: str, metric: str) -> str:
    """Replace ``{metric}`` in threshold keys with the task's accuracy metric."""
    return raw.replace("{metric}", metric)


def _parse_thresholds(
    raw: dict[str, Any],
    accuracy_metric: str,
) -> dict[str, Threshold]:
    resolved: dict[str, Threshold] = {}
    for key_template, spec in raw.items():
        key = _resolve_metric_placeholder(key_template, accuracy_metric)
        resolved[key] = Threshold(compare=spec["compare"], margin=spec["margin"])
    return resolved


def _parse_scenario(raw: dict[str, Any]) -> Scenario:
    return Scenario(
        name=raw["name"],
        description=raw.get("description", ""),
        tag=raw.get("tag", ""),
        overrides=raw.get("overrides", {}),
        train_kwargs=raw.get("train_kwargs", {}),
        datasets=raw.get("datasets"),
        models=raw.get("models"),
        num_seeds=raw.get("num_seeds"),
    )


def load_manifest(path: Path) -> BenchmarkManifest:
    """Parse ``benchmark_manifest.yaml``."""
    with Path(path).open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    version = raw.get("version", 1)

    # Defaults
    dfl_raw = raw.get("defaults", {})
    defaults = ManifestDefaults(
        num_seeds=dfl_raw.get("num_seeds", 3),
        eval_upto=dfl_raw.get("eval_upto", "optimize"),
        deterministic=dfl_raw.get("deterministic", True),
        rotation=dfl_raw.get("rotation", {}),
    )

    experiments: dict[str, TaskSection] = {}
    for task_key, section_raw in raw.get("experiments", {}).items():
        # Models
        models = [
            ModelEntry(
                name=m["name"],
                task=task_key,
                priority=m.get("priority", "core"),
            )
            for m in section_raw.get("models", [])
        ]

        # Datasets (list of names referencing catalog)
        dataset_names: list[str] = section_raw.get("datasets", [])

        # Scenarios - the "default" scenario is always implicit
        scenarios = [Scenario.default()]
        scenarios.extend(_parse_scenario(s_raw) for s_raw in section_raw.get("scenarios", []))

        # Criteria
        crit_raw = section_raw.get("criteria", {})
        accuracy_metric = crit_raw.get("accuracy_metric", "mAP")
        thresholds = _parse_thresholds(crit_raw.get("thresholds", {}), accuracy_metric)
        criteria = CriteriaConfig(accuracy_metric=accuracy_metric, thresholds=thresholds)

        experiments[task_key] = TaskSection(
            task=task_key,
            models=models,
            datasets=dataset_names,
            scenarios=scenarios,
            criteria=criteria,
        )

    return BenchmarkManifest(version=version, defaults=defaults, experiments=experiments)


# ---------------------------------------------------------------------------
# Experiment enumeration
# ---------------------------------------------------------------------------


def iter_experiments(
    manifest: BenchmarkManifest,
    filters: ManifestFilters | None = None,
    catalog_names: set[str] | None = None,
    *,
    size_tier_map: dict[str, str] | None = None,
    data_group_map: dict[str, str] | None = None,
) -> Iterator[Experiment]:
    """Yield :class:`Experiment` instances after applying runtime filters.

    Args:
        manifest: Parsed benchmark manifest.
        filters: Runtime filters (tasks, models, priorities, etc.).
        catalog_names: Set of dataset names present in the catalog. If
            provided, experiments referencing unknown datasets are skipped
            with a warning.
        size_tier_map: Mapping of ``{dataset_name: size_tier}`` from the
            catalog.  When provided together with
            ``filters.size_tiers``, datasets whose tier is not in the
            filter are excluded.
        data_group_map: Mapping of ``{dataset_name: data_group}`` from the
            catalog.  When provided together with ``filters.data_groups``,
            datasets whose data_group is not ``"all"`` and not in the filter
            are excluded.
    """
    f = filters or ManifestFilters()

    for task_key, section in manifest.experiments.items():
        if f.tasks and task_key not in f.tasks:
            continue

        for model in section.models:
            if f.models and model.name not in f.models:
                continue
            if f.priorities and model.priority not in f.priorities:
                continue
            # Deferred models are excluded from unfiltered/automated runs -- they only
            # run when explicitly requested via --priority deferred or --model <name>.
            if model.priority == "deferred" and not f.models and not (f.priorities and "deferred" in f.priorities):
                continue

            for scenario in section.scenarios:
                # Filter scenarios
                if f.scenarios and scenario.name not in f.scenarios:
                    continue
                if f.scenario_tags and scenario.tag not in f.scenario_tags and scenario.name != "default":
                    continue
                # Scenario may restrict which models it applies to
                if scenario.models and model.name not in scenario.models:
                    continue

                for ds_name in section.datasets:
                    if f.datasets and ds_name not in f.datasets:
                        continue
                    if catalog_names and ds_name not in catalog_names:
                        logger.warning(
                            "Dataset '%s' referenced in manifest but not in catalog; skipping.",
                            ds_name,
                        )
                        continue
                    # Size-tier filtering: cross-reference the catalog tier
                    if f.size_tiers and size_tier_map:
                        tier = size_tier_map.get(ds_name)
                        if tier is None or tier not in f.size_tiers:
                            continue
                    # Data-group filtering: cross-reference the catalog data_group;
                    # entries with data_group "all" (the default) always pass.
                    if f.data_groups and data_group_map:
                        group = data_group_map.get(ds_name, "all")
                        if group != "all" and group not in f.data_groups:
                            continue
                    # Scenario may restrict datasets
                    if scenario.datasets and ds_name not in scenario.datasets:
                        continue

                    num_seeds = scenario.num_seeds if scenario.num_seeds is not None else manifest.defaults.num_seeds

                    yield Experiment(
                        task=task_key,
                        model=model,
                        dataset_name=ds_name,
                        scenario=scenario,
                        eval_upto=manifest.defaults.eval_upto,
                        num_seeds=num_seeds,
                        criteria=section.criteria,
                    )


def count_experiments(
    manifest: BenchmarkManifest,
    filters: ManifestFilters | None = None,
) -> int:
    """Return the number of experiments that would be generated."""
    return sum(1 for _ in iter_experiments(manifest, filters))
