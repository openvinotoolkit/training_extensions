# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.manifest."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from getitune.benchmark.manifest import (
    BenchmarkManifest,
    Experiment,
    ManifestDefaults,
    ManifestFilters,
    ModelEntry,
    Scenario,
    count_experiments,
    iter_experiments,
    load_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


MANIFEST_YAML = textwrap.dedent("""\
    version: 1
    defaults:
      num_seeds: 2
      eval_upto: export
      deterministic: true
      rotation:
        extended_groups: 2

    experiments:
      detection:
        models:
          - name: yolox_s
            priority: core
          - name: ssd_mobilenetv2
            priority: extended
        datasets:
          - pothole_tiny
          - wgisd_small
        scenarios:
          - name: tiling
            description: "Tiling variant"
            recipe_suffix: "_tile"
            datasets: [wgisd_small]
          - name: lr_high
            description: "High LR"
            tag: configurable
            overrides:
              model.init_args.optimizer.init_args.lr: 0.002
            models: [yolox_s]
        criteria:
          accuracy_metric: mAP
          thresholds:
            "training:val/{metric}": { compare: ">=", margin: 0.10 }
            "training:e2e_time":     { compare: "<=", margin: 0.10 }

      classification:
        models:
          - name: efficientnet_b0
            priority: core
          - name: mobilenet_v3_large
            priority: extended
        datasets:
          - pneumonia_tiny
        criteria:
          accuracy_metric: f1-score
          thresholds:
            "training:val/{metric}": { compare: ">=", margin: 0.10 }

      instance_segmentation:
        models:
          - name: rfdetr_seg_nano
            priority: core
          - name: rfdetr_seg_small
            priority: extended
          - name: rfdetr_seg_medium
            priority: extended
          - name: rfdetr_seg_xlarge
            priority: extended
          - name: rfdetr_seg_2xl
            priority: extended
        datasets:
          - grapes
        criteria:
          accuracy_metric: mAP
          thresholds:
            "training:val/{metric}": { compare: ">=", margin: 0.10 }
""")


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST_YAML)
    return p


@pytest.fixture
def manifest(manifest_path: Path) -> BenchmarkManifest:
    return load_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_version(self, manifest: BenchmarkManifest) -> None:
        assert manifest.version == 1

    def test_defaults(self, manifest: BenchmarkManifest) -> None:
        assert manifest.defaults.num_seeds == 2
        assert manifest.defaults.eval_upto == "export"
        assert manifest.defaults.deterministic is True

    def test_tasks(self, manifest: BenchmarkManifest) -> None:
        assert set(manifest.all_tasks()) == {"detection", "classification", "instance_segmentation"}

    def test_models_parsed(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        names = [m.name for m in det.models]
        assert "yolox_s" in names
        assert "ssd_mobilenetv2" in names

    def test_scenarios_include_default(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        scenario_names = [s.name for s in det.scenarios]
        assert "default" in scenario_names
        assert "tiling" in scenario_names

    def test_metric_placeholder_resolved(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        assert "training:val/mAP" in det.criteria.thresholds
        assert "training:e2e_time" in det.criteria.thresholds

    def test_classification_metric(self, manifest: BenchmarkManifest) -> None:
        cls_ = manifest.get_task("classification")
        assert cls_.criteria.accuracy_metric == "f1-score"
        assert "training:val/f1-score" in cls_.criteria.thresholds
        assert "training:val/accuracy" not in cls_.criteria.thresholds

    def test_instance_segmentation_models_parsed(self, manifest: BenchmarkManifest) -> None:
        inst = manifest.get_task("instance_segmentation")
        names = [m.name for m in inst.models]
        assert names[0] == "rfdetr_seg_nano"
        assert "rfdetr_seg_small" in names
        assert "rfdetr_seg_2xl" in names

    def test_unknown_task_raises(self, manifest: BenchmarkManifest) -> None:
        with pytest.raises(KeyError, match="segmentation"):
            manifest.get_task("segmentation")


# ---------------------------------------------------------------------------
# Experiment enumeration
# ---------------------------------------------------------------------------


class TestIterExperiments:
    def test_total_count_no_filters(self, manifest: BenchmarkManifest) -> None:
        """detection: 2 models x 2 ds x default + tiling(1 ds x 2 models) + lr_high(1 model x 2 ds)
        + classification: 2 x 1 x default + instance seg: 5 x 1 x default
        """
        exps = list(iter_experiments(manifest))
        # detection:
        #   default: 2 models x 2 datasets = 4
        #   tiling:  2 models x 1 dataset (wgisd_small) = 2
        #   lr_high: 1 model (yolox_s) x 2 datasets = 2
        # classification:
        #   default: 2 x 1 = 2
        # instance_segmentation:
        #   default: 5 x 1 = 5
        assert len(exps) == 15

    def test_filter_by_task(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(tasks=["classification"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.task == "classification" for e in exps)

    def test_filter_by_model(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(models=["yolox_s"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.model.name == "yolox_s" for e in exps)

    def test_filter_by_priority(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(priorities=["core"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.model.priority == "core" for e in exps)
        # ssd_mobilenetv2 is extended → excluded
        assert not any(e.model.name == "ssd_mobilenetv2" for e in exps)

    def test_filter_by_scenario(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(scenarios=["default"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.scenario.name == "default" for e in exps)

    def test_filter_by_scenario_tag(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(scenario_tags=["configurable"])
        exps = list(iter_experiments(manifest, f))
        # Should include "default" (always) + "configurable"-tagged scenarios
        scenario_names = {e.scenario.name for e in exps}
        assert "default" in scenario_names
        assert "lr_high" in scenario_names
        # Tiling has no tag → excluded
        assert "tiling" not in scenario_names

    def test_scenario_dataset_restriction(self, manifest: BenchmarkManifest) -> None:
        """Tiling scenario only applies to wgisd_small."""
        f = ManifestFilters(scenarios=["tiling"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.dataset_name == "wgisd_small" for e in exps)

    def test_scenario_model_restriction(self, manifest: BenchmarkManifest) -> None:
        """lr_high scenario only applies to yolox_s."""
        f = ManifestFilters(scenarios=["lr_high"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.model.name == "yolox_s" for e in exps)

    def test_unknown_catalog_dataset_skipped(self, manifest: BenchmarkManifest) -> None:
        """Datasets not in the catalog are skipped with a warning."""
        catalog_names = {"pothole_tiny"}  # wgisd_small and pneumonia_tiny missing
        exps = list(iter_experiments(manifest, catalog_names=catalog_names))
        ds_names = {e.dataset_name for e in exps}
        assert "wgisd_small" not in ds_names
        assert "pneumonia_tiny" not in ds_names

    def test_experiment_run_id(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(models=["yolox_s"], scenarios=["tiling"])
        exps = list(iter_experiments(manifest, f))
        assert len(exps) == 1
        assert exps[0].run_id == "detection/yolox_s/wgisd_small/tiling"

    def test_experiment_run_id_default_scenario(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(tasks=["classification"], scenarios=["default"])
        exps = list(iter_experiments(manifest, f))
        assert exps[0].run_id == "classification/efficientnet_b0/pneumonia_tiny"

    def test_count_experiments(self, manifest: BenchmarkManifest) -> None:
        assert count_experiments(manifest) == 15


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class TestScenario:
    def test_default_factory(self) -> None:
        s = Scenario.default()
        assert s.name == "default"
        assert s.overrides == {}

    def test_override_passthrough(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        lr_high = next(s for s in det.scenarios if s.name == "lr_high")
        assert lr_high.overrides["model.init_args.optimizer.init_args.lr"] == 0.002


# ---------------------------------------------------------------------------
# ModelEntry
# ---------------------------------------------------------------------------


class TestModelEntry:
    def test_recipe_path_is_absolute(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        model = next(m for m in det.models if m.name == "yolox_s")
        assert model.recipe_path.is_absolute()
        assert str(model.recipe_path).endswith("detection/yolox_s.yaml")

    def test_default_values(self) -> None:
        m = ModelEntry(name="test_model", task="detection")
        assert m.priority == "core"
        assert str(m.recipe_path).endswith("detection/test_model.yaml")


# ---------------------------------------------------------------------------
# Experiment properties
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_recipe_path(self) -> None:
        """Experiment.recipe_path should return the model's recipe path."""
        model = ModelEntry(name="yolox_s", task="detection")
        scenario = Scenario(name="default")
        exp = Experiment(
            task="detection",
            model=model,
            dataset_name="ds",
            scenario=scenario,
            eval_upto="train",
            num_seeds=1,
            criteria=None,  # type: ignore[arg-type]
        )
        assert str(exp.recipe_path).endswith("detection/yolox_s.yaml")

    def test_recipe_path_no_suffix(self) -> None:
        model = ModelEntry(name="yolox_s", task="detection")
        scenario = Scenario.default()
        exp = Experiment(
            task="detection",
            model=model,
            dataset_name="ds",
            scenario=scenario,
            eval_upto="train",
            num_seeds=1,
            criteria=None,  # type: ignore[arg-type]
        )
        assert str(exp.recipe_path).endswith("detection/yolox_s.yaml")

    def test_run_id_with_scenario(self) -> None:
        model = ModelEntry(name="m", task="det")
        scenario = Scenario(name="tiling")
        exp = Experiment(
            task="det",
            model=model,
            dataset_name="ds",
            scenario=scenario,
            eval_upto="train",
            num_seeds=1,
            criteria=None,  # type: ignore[arg-type]
        )
        assert exp.run_id == "det/m/ds/tiling"

    def test_run_id_default_scenario(self) -> None:
        model = ModelEntry(name="m", task="det")
        exp = Experiment(
            task="det",
            model=model,
            dataset_name="ds",
            scenario=Scenario.default(),
            eval_upto="train",
            num_seeds=1,
            criteria=None,  # type: ignore[arg-type]
        )
        assert exp.run_id == "det/m/ds"


# ---------------------------------------------------------------------------
# ManifestDefaults
# ---------------------------------------------------------------------------


class TestManifestDefaults:
    def test_defaults(self) -> None:
        d = ManifestDefaults()
        assert d.num_seeds == 3
        assert d.eval_upto == "optimize"
        assert d.deterministic is True
        assert d.rotation == {}

    def test_rotation_parsed(self, manifest: BenchmarkManifest) -> None:
        assert manifest.defaults.rotation == {"extended_groups": 2}


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_threshold_fields(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        th = det.criteria.thresholds["training:val/mAP"]
        assert th.compare == ">="
        assert th.margin == 0.10


# ---------------------------------------------------------------------------
# Additional filtering
# ---------------------------------------------------------------------------


class TestIterExperimentsAdvanced:
    def test_filter_by_dataset_name(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(datasets=["pothole_tiny"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.dataset_name == "pothole_tiny" for e in exps)

    def test_scenario_num_seeds_override(self, tmp_path: Path) -> None:
        """A scenario with num_seeds should override the global default."""
        yaml_content = textwrap.dedent("""\
            version: 1
            defaults:
              num_seeds: 5

            experiments:
              detection:
                models:
                  - name: m
                datasets:
                  - ds
                scenarios:
                  - name: fast
                    description: "Fewer seeds"
                    num_seeds: 1
                criteria:
                  accuracy_metric: mAP
                  thresholds: {}
        """)
        p = tmp_path / "manifest.yaml"
        p.write_text(yaml_content)
        manifest = load_manifest(p)

        exps = list(iter_experiments(manifest))
        fast_exp = next(e for e in exps if e.scenario.name == "fast")
        default_exp = next(e for e in exps if e.scenario.name == "default")
        assert fast_exp.num_seeds == 1
        assert default_exp.num_seeds == 5

    def test_scenario_tag_filter_includes_default(self, manifest: BenchmarkManifest) -> None:
        """Filtering by tag always includes the 'default' scenario."""
        f = ManifestFilters(scenario_tags=["configurable"])
        exps = list(iter_experiments(manifest, f))
        assert any(e.scenario.name == "default" for e in exps)

    def test_multiple_tasks_filter(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(tasks=["detection", "classification"])
        exps = list(iter_experiments(manifest, f))
        task_set = {e.task for e in exps}
        assert task_set == {"detection", "classification"}

    def test_combined_model_and_scenario_filter(self, manifest: BenchmarkManifest) -> None:
        f = ManifestFilters(models=["ssd_mobilenetv2"], scenarios=["default"])
        exps = list(iter_experiments(manifest, f))
        assert all(e.model.name == "ssd_mobilenetv2" for e in exps)
        assert all(e.scenario.name == "default" for e in exps)

    def test_size_tier_filter(self, manifest: BenchmarkManifest) -> None:
        """Size tier filter should exclude datasets whose tier doesn't match."""
        size_tier_map = {"pothole_tiny": "tiny", "wgisd_small": "small", "pneumonia_tiny": "tiny"}
        f = ManifestFilters(size_tiers=["tiny"])
        exps = list(iter_experiments(manifest, f, size_tier_map=size_tier_map))
        # Only tiny datasets should remain
        ds_names = {e.dataset_name for e in exps}
        assert "wgisd_small" not in ds_names
        assert "pothole_tiny" in ds_names

    def test_size_tier_filter_no_map(self, manifest: BenchmarkManifest) -> None:
        """Size tier filter without a map should not filter anything."""
        f = ManifestFilters(size_tiers=["tiny"])
        exps_filtered = list(iter_experiments(manifest, f))
        exps_all = list(iter_experiments(manifest))
        # Without size_tier_map, size_tiers filter is a no-op
        assert len(exps_filtered) == len(exps_all)

    def test_data_group_filter(self, manifest: BenchmarkManifest) -> None:
        """Data group filter should exclude datasets restricted to another lane."""
        data_group_map = {"pothole_tiny": "weekly", "wgisd_small": "extended", "pneumonia_tiny": "all"}
        f = ManifestFilters(data_groups=["weekly"])
        exps = list(iter_experiments(manifest, f, data_group_map=data_group_map))
        ds_names = {e.dataset_name for e in exps}
        # "weekly" explicitly matches; "all" (default) always matches; "extended" is excluded.
        assert "wgisd_small" not in ds_names
        assert "pothole_tiny" in ds_names
        assert "pneumonia_tiny" in ds_names

    def test_data_group_filter_no_map(self, manifest: BenchmarkManifest) -> None:
        """Data group filter without a map should not filter anything."""
        f = ManifestFilters(data_groups=["weekly"])
        exps_filtered = list(iter_experiments(manifest, f))
        exps_all = list(iter_experiments(manifest))
        assert len(exps_filtered) == len(exps_all)


# ---------------------------------------------------------------------------
# Scenario details
# ---------------------------------------------------------------------------


class TestScenarioAdvanced:
    def test_scenario_tag(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        lr_high = next(s for s in det.scenarios if s.name == "lr_high")
        assert lr_high.tag == "configurable"

    def test_scenario_recipe_suffix_removed(self, manifest: BenchmarkManifest) -> None:
        """recipe_suffix was removed; Scenario should no longer expose it."""
        det = manifest.get_task("detection")
        tiling = next(s for s in det.scenarios if s.name == "tiling")
        assert not hasattr(tiling, "recipe_suffix")

    def test_scenario_dataset_restriction(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        tiling = next(s for s in det.scenarios if s.name == "tiling")
        assert tiling.datasets == ["wgisd_small"]

    def test_scenario_model_restriction(self, manifest: BenchmarkManifest) -> None:
        det = manifest.get_task("detection")
        lr_high = next(s for s in det.scenarios if s.name == "lr_high")
        assert lr_high.models == ["yolox_s"]
