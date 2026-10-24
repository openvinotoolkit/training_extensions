# Design Document: Automated Model Benchmarking Workflow

| Field       | Value      |
| ----------- | ---------- |
| **Status**  | Draft      |
| **Date**    | 2026-04-08 |
| **Authors** | Albert     |

---

## Table of Contents

1. [Objective](#1-objective)
2. [Motivation & Background](#2-motivation--background)
3. [Current State Analysis](#3-current-state-analysis)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Dataset Catalog](#5-dataset-catalog)
6. [Benchmark Manifest](#6-benchmark-manifest)
7. [Benchmark Runner](#7-benchmark-runner)
8. [Experiment Tracking with MLflow](#8-experiment-tracking-with-mlflow)
9. [Reporting & Regression Detection](#9-reporting--regression-detection)
10. [PR Change Detection & Targeted Benchmarks](#10-pr-change-detection--targeted-benchmarks)
11. [CI/CD Integration (GitHub Actions)](#11-cicd-integration-github-actions)
12. [Hardware & Accelerator Support](#12-hardware--accelerator-support)
13. [Failure Handling](#13-failure-handling)
14. [Directory & File Layout](#14-directory--file-layout)
15. [Migration from Existing Code](#15-migration-from-existing-code)
16. [Rollout Plan](#16-rollout-plan)
17. [Open Questions](#17-open-questions)

---

## 1. Objective

Set up a fully automated, CI-integrated workflow that evaluates the performance
of every supported OTX model on a curated set of testing datasets. The workflow
must produce comparable, versioned reports so that engineers can detect
regressions in accuracy, training time, and inference speed across releases,
branches, and hardware targets.

---

## 2. Motivation & Background

OTX is a framework for training vision models. Engineers continuously add new
architectures, loss functions, data augmentations, and training tricks. Every
change risks introducing regressions that are hard to detect during code
review alone — models must actually be trained and evaluated.

Today this evaluation is **manual**: an engineer picks a model, picks a
dataset, trains locally, and eyeballs the numbers. This approach is:

- **Time-consuming** — a full matrix of models × datasets × seeds takes days of
  human effort.
- **Error-prone** — inconsistent hyperparameters, forgotten datasets, typos in
  metric comparisons.
- **Non-reproducible** — no single source of truth for "what was the accuracy of
  `yolox_s` on `pothole_tiny` last week?"
- **Opaque** — results live on individual engineers' machines and are not
  accessible to the wider team.

An automated benchmark workflow eliminates all four problems and lets
engineers focus on _building_ while the CI system takes care of _validating_.

---

## 3. Current State Analysis

The repository already contains two partially overlapping systems for model
evaluation. Both have significant limitations that motivate a clean redesign.

### 3.1 `tests/perf_v2/` — Performance Benchmark Suite

| Aspect               | Description                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| Entry point          | `python -m tests.perf_v2.run --task DETECTION ...` or `python -m tests.perf_v2.benchmark ...`  |
| Runner               | Spawns one subprocess per `(model, dataset, seed)` triple via `subprocess.run`                 |
| Engine               | Uses `OTXEngine` / `OVEngine` Python API directly                                              |
| Reporting            | Writes `benchmark.raw.csv` per experiment; `summary.py` aggregates into Excel/CSV pivot tables |
| Tracking             | None — no MLflow, no DB, no dashboard                                                          |
| CI integration       | None — no workflow file exists                                                                 |
| Dataset provisioning | Assumes datasets are pre-downloaded into a `data/` root; no automated download                 |
| Failure handling     | Retries up to 2 times; collects failures into `failed_jobs.json`                               |
| Accelerator support  | `--device gpu\|xpu\|cpu` flag; single device assumption                                        |

**Key limitations:**

- No CI workflow — the suite can only be run manually.
- Datasets must already exist on the machine; there is no download/provisioning step.
- Results are flat CSV files with no experiment-tracking UI, making cross-run comparison tedious.
- The `Criterion` checking is all-or-nothing assertions that raise on first violation; not suitable for a continuous monitoring system that should _report_ regressions, not just crash.
- The model/dataset/criteria registrations are scattered across `tasks/*.py` files with no single declarative manifest.
- No support for scenario variants (e.g. tiling, different hyperparameter profiles).

### 3.2 `tests/regression/` — MLflow Regression Tests

| Aspect               | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| Entry point          | `pytest tests/regression/test_regression.py`                               |
| Runner               | pytest parametrize over model × dataset × seed                             |
| Engine               | Shells out to `otx train` / `otx test` CLI via subprocess                  |
| Tracking             | MLflow — creates experiment, logs metrics step-by-step, logs CSV artifacts |
| Reporting            | Relies entirely on MLflow UI for visualization                             |
| Dataset provisioning | Same assumption — pre-existing `--data-root`                               |

**Key limitations:**

- Tightly coupled to pytest — parametrization, fixtures, and assertions are deeply intertwined. This makes it hard to run a single model ad-hoc, resume a partial run, or integrate with non-pytest CI systems.
- Test class hierarchy is copy-pasted per task; adding a new task means duplicating ~60 lines of boilerplate.
- MLflow integration is promising but incomplete (see `TODO` comments about `metrics.csv` not being produced correctly for the `test` subcommand).
- No reference comparison logic — metrics are logged but never compared against a baseline.
- No support for export/optimize evaluation or tiling variants.

### 3.3 Assessment

Neither system is complete enough to serve as the automated benchmark. Rather
than patching the gaps, we propose a unified redesign that takes the best ideas
from both (MLflow tracking from `regression/`, the `Benchmark` orchestration
from `perf_v2/`) while fixing the structural issues.

---

## 4. Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          GitHub Actions Workflow                         │
│                                                                          │
│   trigger: schedule (weekly) | workflow_dispatch | PR label              │
│                                                                          │
│   ┌────────────────┐    ┌────────────────┐    ┌───────────────────────┐  │
│   │ Provision      │───▶│ Run            │───▶│ Report &              │  │
│   │ Datasets       │    │ Benchmarks     │    │ Regression Check      │  │
│   └───────┬────────┘    └───────┬────────┘    └───────────┬───────────┘  │
│           │                     │                         │              │
│           ▼                     ▼                         ▼              │
│   ┌────────────────┐    ┌────────────────┐    ┌───────────────────────┐  │
│   │ Preparation    │    │ Benchmark      │    │ MLflow                │  │
│   │ Scripts        │    │ Runner         │    │ Tracking Server       │  │
│   │ (scripts/      │    │ (Python)       │    │ (local or remote)     │  │
│   │  benchmark_    │    │                │    │                       │  │
│   │  datasets/)    │    │                │    │                       │  │
│   └────────────────┘    └───────┬────────┘    └───────────┬───────────┘  │
│                                 │                         │              │
│                                 ▼                         ▼              │
│                         ┌────────────────┐    ┌───────────────────────┐  │
│                         │ OTXEngine      │    │ Markdown / HTML       │  │
│                         │ (train/test/   │    │ Summary Report        │  │
│                         │  export/       │    │ (PR comment /         │  │
│                         │  optimize)     │    │  artifact / GH Pages) │  │
│                         └────────────────┘    └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

The system is composed of four independent layers:

| Layer                | Responsibility                                              | Artifact                                                 |
| -------------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| **Dataset Catalog**  | Declare datasets, their preparation scripts, and size class | `benchmark_catalog.yaml` + `scripts/benchmark_datasets/` |
| **Benchmark Runner** | Provision data via scripts, run experiments, log to MLflow  | Python package `getitune.benchmark`                      |
| **Report Generator** | Compare against baselines, produce human-readable report    | Markdown + CSV                                           |
| **CI Workflow**      | Orchestrate the above on GitHub Actions runners             | `.github/workflows/benchmark.yml`                        |

Each layer is independently testable and usable outside CI (e.g., an engineer
can run the benchmark runner locally with `python -m getitune.benchmark run ...`).

---

## 5. Dataset Catalog

### 5.1 Principles

- Every dataset used for benchmarking is **declared in a single YAML manifest**
  (`benchmark_catalog.yaml`) rather than scattered across Python files.
- Datasets are classified by **size tier**: `tiny`, `small`, `medium`, `large`.
- Each entry references a **preparation script** — a Python script that lives in
  the repository under `scripts/benchmark_datasets/`. The script is responsible
  for downloading the raw dataset from its source (e.g. a public URL, a
  cloud bucket, or any other origin), applying any necessary transformations
  (format conversion, annotation adjustments, subset sampling), and placing
  the final dataset in the designated data directory.
- Datasets must have **verified, meaningful annotations**. Because scripts are
  version-controlled, the exact transformation logic is reviewable and
  reproducible.

#### Size Tier Definitions

| Tier       | Item Count (images / samples) | Typical Use                                          |
| ---------- | ----------------------------- | ---------------------------------------------------- |
| **tiny**   | **< 50**                      | Unit-level smoke tests, PR checks, fast iteration    |
| **small**  | **50 – 200**                  | Nightly quick benchmarks, scenario sweeps            |
| **medium** | **200 – 1 000**               | Weekly full benchmarks, release validation           |
| **large**  | **> 1 000**                   | On-demand deep evaluation, publication-grade results |

> **Guideline:** Every task should have at least one `tiny` and one `small`
> dataset so that PR-level and nightly benchmarks can run quickly. `medium` and
> `large` datasets are reserved for weekly and on-demand runs where statistical
> significance and real-world representativeness matter more than speed.

### 5.2 Schema

Datasets are declared as a **flat list** — each dataset is listed once and
referenced by name from the benchmark manifest. Since the same dataset can be
used by multiple tasks (e.g. a multi-annotation dataset used for detection,
segmentation, and classification), the catalog is **task-independent**. The
manifest (§6) declares which datasets each task uses.

Each entry specifies a `script` field — the path (relative to the repository
root) to a Python preparation script. The script must accept two arguments:

1. `--output-dir` — the directory where the final dataset should be placed.
2. `--name` — the dataset name (used by the script to determine its output
   sub-directory within the output dir).

```yaml
# benchmark_catalog.yaml
version: 1

datasets:
  - name: pothole_tiny
    script: "scripts/benchmark_datasets/prepare_pothole.py"
    size_tier: tiny

  - name: wgisd_small
    script: "scripts/benchmark_datasets/prepare_wgisd.py"
    size_tier: small

  - name: diopsis_medium
    script: "scripts/benchmark_datasets/prepare_diopsis.py"
    size_tier: medium

  - name: visdrone_large
    script: "scripts/benchmark_datasets/prepare_visdrone.py"
    size_tier: large

  - name: pneumonia_tiny
    script: "scripts/benchmark_datasets/prepare_pneumonia.py"
    size_tier: tiny
  # ...
```

### 5.3 Script-Based Provisioning & Cache

A utility (`getitune.benchmark.catalog`) will:

1. Read the catalog.
2. For each required dataset, check if the directory `<data_root>/<name>`
   already exists. If it does, the dataset is considered ready and the
   script is skipped.
3. If the directory does not exist, run the preparation script via
   `python <script> --output-dir <data_root> --name <name>`.
4. On CI runners, use the GitHub Actions cache (`actions/cache`) keyed on the
   hash of all preparation scripts to avoid re-provisioning on every run.
   To force re-provisioning (e.g. after a script change), delete the dataset
   directory or bust the CI cache.

Because datasets are stored in a flat `<data_root>/<name>/` layout (no task
prefix), datasets shared across multiple tasks are only prepared and stored
once.

#### 5.3.1 Preparation Script Contract

Each preparation script in `scripts/benchmark_datasets/` must:

- Accept `--output-dir <path>` and `--name <dataset_name>` CLI arguments.
- Download raw data from its source (URL, cloud storage, etc.).
- Apply any transformations (format conversion, annotation fixes, sampling).
- Place the final dataset in `<output-dir>/<name>/`.
- Be **idempotent** — running the script twice produces the same result.
- Exit with code 0 on success, non-zero on failure.

Scripts should use the shared helpers in `getitune.benchmark.dataset_helpers` to
avoid duplicating boilerplate. The module provides:

| Helper              | Purpose                                                             |
| ------------------- | ------------------------------------------------------------------- |
| `parse_args()`      | Parse `--output-dir` / `--name`; returns a `DatasetArgs` dataclass  |
| `download()`        | Download a URL to a local directory; returns the local file path    |
| `extract_archive()` | Extract `.tar.gz`, `.tar`, or `.zip` into a destination directory   |
| `DatasetArgs`       | Dataclass with `.dest` (final path), `.archive_dir` (temp archives) |

A simple "download + extract" dataset needs only ~10 lines:

```python
#!/usr/bin/env python3
# scripts/benchmark_datasets/prepare_pothole.py
"""Download and prepare the pothole_tiny benchmark dataset."""

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

_URL = "https://storage.geti.intel.com/test-data/pothole_tiny.tar.gz"


def main() -> None:
    args = parse_args(description="Prepare the pothole_tiny dataset.")

    archive = download(_URL, dest_dir=args.archive_dir, filename=f"{args.name}.tar.gz")
    extract_archive(archive, args.dest)
    archive.unlink(missing_ok=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
```

For datasets that need transformation after extraction, add custom logic
between `extract_archive()` and the final print:

```python
def main() -> None:
    args = parse_args(description="Prepare the wgisd dataset (subsampled to 200 images).")

    archive = download(_URL, dest_dir=args.archive_dir, filename=f"{args.name}.zip")
    extract_archive(archive, args.dest)
    archive.unlink(missing_ok=True)

    # Dataset-specific: subsample to 200 images for the "small" tier
    _subsample_images(args.dest / "images", max_count=200)
    _rewrite_annotations(args.dest / "annotations.json")

    print(f"Dataset '{args.name}' ready at {args.dest}")
```

```python
# getitune/benchmark/catalog.py  (simplified)
def provision_datasets(
    catalog: DatasetCatalog,
    data_root: Path,
    *,
    entries: list[DatasetEntry] | None = None,
) -> dict[str, Path]:
    """Run preparation scripts for all datasets (or a filtered subset).

    Returns a mapping {dataset_name: prepared_path}.
    """
    ...
```

Provisioning is resilient per-dataset: if one entry fails (missing credentials, a
transient network error, an unresolvable `local_path`/`raw_dir`), the error is logged and
that entry is simply omitted from the returned mapping rather than aborting the whole
batch. The runner already treats a missing dataset name as "skip the experiments that
need it" (§8), so a single problematic dataset never blocks unrelated experiments.

#### 5.3.2 Credentialed, Manually-Placed, and Local Datasets

Not every dataset can be freely auto-downloaded: some require an account/license
(e.g. Kaggle), and some are large enough that re-fetching them on every run is wasteful.
Rather than build a generic pluggable "dataset source" abstraction, the catalog schema
gets two additional, optional fields that compose with the existing script contract:

- **`local_path`** (alternative to `script`) — a directory that is already fully
  prepared (placed manually, mounted from a shared/network location, or copied from a
  prior script run). No script runs; the path is used as-is, after `${VAR}`/`~`
  expansion so the same entry resolves differently per machine/CI runner. Mutually
  exclusive with `script`.
- **`raw_dir`** (optional, alongside `script`) — a pre-fetched raw archive/directory,
  forwarded to the script as `--raw-dir`. This lets a script skip only its network fetch
  step (e.g. because the raw data was obtained once, manually, with a person's own
  Kaggle credentials) while still running its transform/export logic, keeping that logic
  shared, version-controlled, and testable. It's a best-effort accelerant: if it doesn't
  resolve, the script falls back to its normal download behavior instead of failing.

```yaml
datasets:
  - name: brain_tumor
    script: "scripts/benchmark_datasets/prepare_brain_tumor.py"
    raw_dir: "${GETITUNE_BENCHMARK_EXTERNAL_DATA}/brain_tumor_raw"
    size_tier: small
```

Scripts sourcing from Kaggle use `getitune.benchmark.dataset_helpers.download_kaggle_dataset()`,
which checks for `kagglehub` and API credentials (`KAGGLE_API_TOKEN` or
`~/.kaggle/access_token`) up front and raises a clear, actionable error rather than a
bare dependency failure. `kagglehub` itself is an opt-in optional dependency extra
(`benchmark`; e.g. `uv sync --extra benchmark`), not installed by default. In CI, a `KAGGLE_API_TOKEN`
repository secret is wired into the `benchmark-dataset-scripts` job; the corresponding
real-download test is additionally `skipif`-gated on credentials being present, so
forked-repo PRs (which never receive secrets) skip it instead of failing.

This directly answers §17 Open Question #1 for the credentialed/oversized subset of
datasets: rather than depending on a single hosting service, `local_path` lets any
externally-managed location (shared drive, network mount, manually-populated directory)
serve as a dataset's source of truth, while `raw_dir` keeps the credentialed/huge case
inside the same reviewable script pipeline as every other dataset.

---

## 6. Benchmark Manifest

The **benchmark manifest** defines _what_ to run. It is separate from the
dataset catalog (which defines _what data exists_) and from the runner (which
defines _how_ to run).

### 6.1 Schema

```yaml
# benchmark_manifest.yaml
version: 1

# Global defaults (can be overridden per-experiment)
defaults:
  num_seeds: 3
  eval_upto:
    optimize # The *last* phase to execute (all preceding phases are included).
    # train → train only
    # export → train + export
    # optimize → train + export + optimize
  deterministic: true # see note below

experiments:
  detection:
    models:
      - name: atss_mobilenetv2
        priority: core
        recipe: detection/atss_mobilenetv2.yaml
      - name: yolox_s
        priority: core
        recipe: detection/yolox_s.yaml
      - name: dfine_x
        priority: core
        recipe: detection/dfine_x.yaml
      - name: yolox_tiny
        priority: extended
        recipe: detection/yolox_tiny.yaml
      - name: ssd_mobilenetv2
        priority: extended
        recipe: detection/ssd_mobilenetv2.yaml
      - name: deformable_detr
        priority: exploratory
        recipe: detection/deformable_detr.yaml
      # ...full list auto-discoverable from recipe directory

    datasets:
      # References entries in benchmark_catalog.yaml by name
      - pothole_tiny
      - wgisd_small
      - diopsis_medium
      - visdrone_large

    scenarios:
      # The "default" scenario (train with unmodified recipe) is always
      # included implicitly, it does not need to be declared here.

      # Tiling is a built-in variant that uses a separate recipe file.
      - name: tiling
        description: "Train with tiling enabled"
        recipe_suffix: "_tile" # appended to model recipe name
        datasets: [wgisd_small] # only run tiling on specific datasets

      # ──────────────────────────────────────────────────────────────
      # The scenarios below are OPTIONAL parameter overrides.
      # They are NOT run by default, a user must explicitly request
      # them via --scenario <name> or --scenario-tag configurable.
      # ──────────────────────────────────────────────────────────────
      - name: lr_high
        description: "Train with a high learning rate"
        tag: configurable
        overrides:
          model.init_args.optimizer.init_args.lr: 0.002
      - name: lr_low
        description: "Train with a low learning rate"
        tag: configurable
        overrides:
          model.init_args.optimizer.init_args.lr: 0.0005
      - name: batch_size_small
        description: "Train with a small batch size"
        tag: configurable
        overrides:
          overrides.data.train_subset.batch_size: 4
      - name: batch_size_large
        description: "Train with a large batch size"
        tag: configurable
        overrides:
          overrides.data.train_subset.batch_size: 32
      - name: input_size_small
        description: "Train with reduced input resolution (e.g. 320×320 for 640-default models)"
        tag: configurable
        overrides:
          overrides.data.input_size: [320, 320]
      - name: optimizer_adamw
        description: "Replace default optimizer with AdamW"
        tag: configurable
        overrides:
          model.init_args.optimizer:
            class_path: torch.optim.AdamW
            init_args:
              lr: 0.001
              weight_decay: 0.05
      - name: precision_32
        description: "Train with full FP32 precision (baseline for numerical comparison)"
        tag: configurable
        train_kwargs:
          precision: "32"
      - name: no_augmentation
        description: "Train without GPU augmentations to isolate augmentation impact"
        tag: configurable
        overrides:
          overrides.data.train_subset.augmentations_gpu: []

    criteria:
      # The primary accuracy metric for this task. The `{metric}` placeholder
      # in threshold keys below is replaced with this value at manifest load
      # time. For detection this produces "training:val/mAP", etc.
      # Each task section declares its own accuracy_metric (e.g. "accuracy"
      # for classification, "Dice" for segmentation, "PCK" for keypoints).
      # This mirrors the existing TASK_METRIC_MAP in tests/perf_v2/summary.py.
      accuracy_metric: mAP
      thresholds:
        "training:val/{metric}": { compare: ">=", margin: 0.10 }
        "torch:test/{metric}": { compare: ">=", margin: 0.10 }
        "export:test/{metric}": { compare: ">=", margin: 0.10 }
        "training:e2e_time": { compare: "<=", margin: 0.10 }
        "training:gpu_mem": { compare: "<=", margin: 0.10 }
        "torch:test/latency": { compare: "<=", margin: 0.15 }

  classification/multi_class_cls:
    models:
      - name: efficientnet_b0
        # ...
    datasets:
      - pneumonia_tiny
      # ...
    criteria:
      accuracy_metric: accuracy
      # ...

  # ...one section per task
```

> **`deterministic: true` trade-off:** Enabling deterministic mode disables
> cuDNN benchmark/auto-tuner and forces CUDA to use deterministic (but
> potentially slower) algorithm variants. This can make training 10–30% slower
> than real-world non-deterministic training. This is acceptable for benchmark
> purposes because (a) regression detection relies on _relative_ comparisons
> where the overhead cancels out, and (b) reproducibility across seeds and runs
> is essential for meaningful statistical comparison. Benchmark reports should
> note that absolute timing numbers are measured under deterministic mode and
> may not reflect production training speed.

### 6.2 Model Selection at Runtime

The manifest is the _full_ matrix. At invocation time, the user can filter:

```bash
# Run everything
python -m getitune.benchmark run

# Single model
python -m getitune.benchmark run --model yolox_s

# All models for a task
python -m getitune.benchmark run --task detection

# Specific list
python -m getitune.benchmark run --model yolox_s --model atss_mobilenetv2

# Only tiny/small datasets (quick smoke test)
python -m getitune.benchmark run --size-tier tiny --size-tier small

# Specific scenario (optional, see §6.3; omit for default recipe config)
python -m getitune.benchmark run --scenario tiling --task detection

# Preview what would run without executing (see also §6.4.4)
python -m getitune.benchmark run --task detection --size-tier tiny --dry-run
```

All commands above run each model with its **default recipe configuration**
unless `--scenario` is explicitly provided.

This directly satisfies the requirement:

> _"Allow the developers to select a subset of models to test (single model, all models, or a list)."_

### 6.3 Configurable Parameter Scenarios (Optional)

By default, every benchmark run uses the **model's default recipe
configuration**, no overrides, no parameter tweaks. This is the standard mode
and covers the vast majority of benchmark runs (nightly, weekly, PR).

Optionally, a user running a benchmark can request **parameter scenario
overrides** to measure the impact of different hyperparameters, optimizers,
input resolutions, augmentation pipelines, precision modes, or batch sizes.
These scenarios are **never run automatically** unless explicitly requested via
the `--scenario` CLI flag or a PR label. Rather than creating separate recipe
files for every variation, the manifest supports **scenario-level overrides**
that are applied on top of the model's default recipe at runtime.

#### 6.3.1 Override Schema

Each scenario can declare two optional fields:

| Field          | Purpose                                                                      | Applied to                                |
| -------------- | ---------------------------------------------------------------------------- | ----------------------------------------- |
| `overrides`    | Key-value pairs that override recipe YAML fields before engine instantiation | `OTXEngine.from_config(..., **overrides)` |
| `train_kwargs` | Keyword arguments passed directly to `engine.train()`                        | `engine.train(**train_kwargs)`            |

**`overrides`** uses dot-notation paths matching the recipe YAML structure. For
example, `model.init_args.optimizer.init_args.lr` maps to the nested field in
the recipe file. Values are **literal replacements** (e.g. `0.002` for a learning rate, `[320, 320]`
for `input_size`, or a full optimizer dict for swapping `SGD` → `AdamW`).

**`train_kwargs`** are passed as-is to `engine.train()`. This covers
engine-level parameters that live outside the recipe (e.g., `precision`,
`max_epochs`, `deterministic`).

#### 6.3.2 Supported Parameter Categories

The following categories of configurable parameters are first-class targets
for scenario-based benchmarking:

| Category              | Example Scenarios                               | Recipe Path / Kwarg                                |
| --------------------- | ----------------------------------------------- | -------------------------------------------------- |
| **Learning rate**     | `lr_high` (0.002), `lr_low` (0.0005)            | `model.init_args.optimizer.init_args.lr`           |
| **Batch size**        | `batch_size_small` (4), `batch_size_large` (32) | `overrides.data.train_subset.batch_size`           |
| **Input resolution**  | `input_size_small` (320×320)                    | `overrides.data.input_size`                        |
| **Optimizer**         | `optimizer_adamw`, `optimizer_adam`             | `model.init_args.optimizer` (full replacement)     |
| **Scheduler**         | `scheduler_cosine`                              | `model.init_args.scheduler` (full replacement)     |
| **Precision**         | `precision_32` (FP32)                           | `train_kwargs.precision`                           |
| **Augmentation**      | `no_augmentation`, `heavy_augmentation`         | `overrides.data.train_subset.augmentations_gpu`    |
| **Early stopping**    | `patience_20`                                   | callback override                                  |
| **Weight decay**      | `wd_high` (0.01), `wd_low` (0.00001)            | `model.init_args.optimizer.init_args.weight_decay` |
| **Gradient clipping** | `grad_clip_none`                                | `overrides.gradient_clip_val`                      |

Not every scenario needs to be run for every model — the `models` and
`datasets` fields on a scenario can restrict which combinations it applies to:

```yaml
scenarios:
  - name: lr_high
    description: "High learning rate"
    overrides:
      model.init_args.optimizer.init_args.lr: 0.002
    models: [yolox_s, atss_mobilenetv2] # only these models
    datasets: [pothole_tiny, wgisd_small] # only smaller datasets (quick)
    num_seeds: 1 # override global seed count
```

#### 6.3.3 CLI Usage

```bash
# Run only the lr_high scenario for detection
python -m getitune.benchmark run --scenario lr_high --task detection

# Run all configurable-parameter scenarios (exclude default and tiling)
python -m getitune.benchmark run --scenario-tag configurable

# Run a one-off parameter variation without a manifest entry
python -m getitune.benchmark run \
    --model yolox_s \
    --dataset pothole_tiny \
    --override model.init_args.optimizer.init_args.lr=0.01 \
    --override overrides.data.train_subset.batch_size=16 \
    --train-kwarg precision=32
```

The `--override` and `--train-kwarg` flags enable ad-hoc parameter sweeps from
the command line without editing the manifest. This is useful for quick local
experimentation.

#### 6.3.4 How Overrides Are Applied

The override resolution happens inside `ExperimentExecutor` before engine
construction. `OTXEngine.from_config()` delegates to `get_instantiated_classes()`
which converts `**kwargs` into CLI-style arguments (`--{key} {value}`) and
passes them through the jsonargparse/omegaconf parser. This means:

- **Scalar dot-path overrides work natively.** A kwarg like
  `model.init_args.optimizer.init_args.lr=0.002` becomes the CLI argument
  `--model.init_args.optimizer.init_args.lr 0.002`, which jsonargparse resolves
  against the recipe YAML before instantiation.
- **Complex (dict/list) overrides require JSON serialization.** Because
  `get_instantiated_classes()` converts values via `str()`, a Python dict would
  produce invalid CLI input. Complex values must be serialized to a JSON string
  so that jsonargparse can parse them correctly.

The resolver handles both scalar and complex overrides:

```python
import json

def _resolve_overrides(
    scenario_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Resolve scenario overrides into kwargs for OTXEngine.from_config().

    Args:
        scenario_overrides: Override spec from the scenario definition.

    Returns:
        Flat dict of resolved key-value pairs suitable for passing as
        ``**kwargs`` to ``OTXEngine.from_config()``. Complex values are
        JSON-serialized so that the underlying jsonargparse CLI parser
        can consume them.
    """
    resolved = {}
    for dotpath, value in scenario_overrides.items():
        if isinstance(value, (dict, list)):
            # Complex values: serialize to JSON string for jsonargparse
            resolved[dotpath] = json.dumps(value)
        else:
            # Scalar literal replacement
            resolved[dotpath] = value
    return resolved
```

#### 6.3.5 Tracking & Comparison

Configurable-parameter runs are tracked in MLflow with additional tags:

| Tag                   | Example   |
| --------------------- | --------- |
| `scenario`            | `lr_high` |
| `override.lr`         | `0.002`   |
| `override.batch_size` | `16`      |
| `override.precision`  | `32`      |

This allows MLflow queries like "compare `yolox_s` on `pothole_tiny` across all
LR scenarios" or "show all `precision_32` runs grouped by model."

Regression comparison for configurable-parameter scenarios uses **separate
baselines** per scenario. A `lr_high` run is compared against the previous
`lr_high` baseline — not against the `default` baseline — because the expected
accuracy differs by design. This is handled automatically by the MLflow
baseline query (§9.1.1), which includes `tags.scenario` in its filter. No
separate storage is needed — the scenario tag on each MLflow run is sufficient
to scope the comparison.

> **Note:** Only the `default` scenario is benchmarked by automated CI runs.
> Scenario-specific baselines are created on-demand when a user runs parameter
> scenarios; those results are logged to MLflow and become the baseline for
> future runs of the same scenario.

#### 6.3.6 When to Run Parameter Scenarios

Parameter scenarios are **opt-in only**. No automated CI trigger runs them
unless a user explicitly requests it:

| Trigger                         | Models                                     | Scenarios Run                                                                |
| ------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------- |
| **Weekly full benchmark**       | `core` + `extended`                        | `default` only                                                               |
| **Nightly quick run**           | `core` only                                | `default` only (tiny+small datasets)                                         |
| **PR benchmark**                | Affected `core` models                     | `default` only (tiny datasets, single seed)                                  |
| **Manual `workflow_dispatch`**  | User-selected via `--model` / `--priority` | User-selected via `--scenario` (defaults to `default` if omitted)            |
| **PR label `benchmark:params`** | Affected `core` models                     | `default` + all `configurable`-tagged scenarios (tiny datasets, single seed) |

The `benchmark:params` PR label is the only way to trigger parameter scenarios
from CI without using `workflow_dispatch`. This is intended for PRs that
intentionally change hyperparameter defaults or optimizer configurations, where
validating the impact across parameter variations is useful.

For ad-hoc local use, engineers can run any scenario directly:

```bash
# Run a specific parameter scenario
python -m getitune.benchmark run --scenario lr_high --task detection

# Run all configurable-parameter scenarios
python -m getitune.benchmark run --scenario-tag configurable --task detection
```

### 6.4 Scaling to Hundreds of Models

As the model catalog grows from tens to hundreds of models, running every model
every week becomes infeasible. The design addresses this through **model
priority tiers**, **rotating schedules**, and **smart filtering** so that CI
budgets remain bounded while still guaranteeing coverage over time.

#### 6.4.1 Model Priority Tiers

Every model in the manifest declares a `priority` field:

| Priority          | Description                                                                                                                 | Expected Count                | Benchmark Cadence                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------ |
| **`core`**        | Flagship models, the most-used or strategically important models per task. These are the models that _must not_ regress.    | ~1–3 per task (~15–25 total)  | Every nightly + Weekly               |
| **`extended`**    | Fully supported models that are not flagged as core. Important to validate but can tolerate slightly less frequent testing. | ~3–10 per task (~30–80 total) | Weekly (rotated — see §6.4.2)        |
| **`exploratory`** | Experimental, deprecated, or niche models. Included in the manifest for completeness but never benchmarked automatically.   | Unbounded                     | On-demand only (`workflow_dispatch`) |

Assigning priorities is a **human decision** made when adding a model to the
manifest. The guideline is:

- If a regression in this model would block a release → `core`.
- If a regression matters but can be fixed in the next sprint → `extended`.
- If this model is experimental or rarely used in production → `exploratory`.

#### 6.4.2 Rotating Schedule for Extended Models

With 50+ `extended` models, running them all every week may still exceed the CI
budget. The runner supports a **rotation policy** that splits extended models
into N equally-sized groups and runs one group per week, cycling through all
groups over N weeks.

```yaml
# benchmark_manifest.yaml — global scheduling config
defaults:
  num_seeds: 3
  eval_upto: optimize
  deterministic: true
  rotation:
    extended_groups: 4 # split extended models into 4 groups
```

The active rotation group is determined by the current ISO week number:
`active_group = ISO_week_number % extended_groups`. All runs within the same
calendar week benchmark the same group. For example, with `extended_groups: 4`:

- Week 14 → group `14 % 4 = 2`
- Week 15 → group `15 % 4 = 3`
- Week 16 → group `16 % 4 = 0`
- Week 17 → group `17 % 4 = 1` full cycle complete, all groups covered.

Each _model_ is permanently assigned to a group via `hash(model_name) % N`.
The runner then filters: only models whose assigned group matches the current
active group are included in the run.

The rotation assignment is deterministic `group = hash(model_name) % N` so
the same model always lands in the same group, and engineers can predict when
their model will next be benchmarked.

```bash
# Rotation happens automatically based on the current week
python -m getitune.benchmark run --priority extended   # runs only this week's group

# Override to force a specific rotation group
python -m getitune.benchmark run --priority extended --rotation-group 2

# Run ALL extended models regardless of rotation (e.g. before a release)
python -m getitune.benchmark run --priority extended --no-rotation
```

Over a 4-week cycle every `extended` model is benchmarked at least once, while
any single week's load is only ~25% of the full extended set.

#### 6.4.3 Coverage Guarantees

The tiering + rotation system provides the following guarantee:

| Priority      | Nightly          | Weekly             | Before Release              | On-demand (`workflow_dispatch`)              |
| ------------- | ---------------- | ------------------ | --------------------------- | -------------------------------------------- |
| `core`        | Yes (tiny+small) | Yes (all tiers)    | Yes                         | Yes                                          |
| `extended`    | —                | Yes (1/N rotation) | Yes (full, `--no-rotation`) | Yes                                          |
| `exploratory` | —                | —                  | —                           | Yes (`--priority core,extended,exploratory`) |

**Before a release**, the CI runs with `--no-rotation --priority core,extended`
to ensure full coverage of all supported models. This is triggered manually or
by a `benchmark:release` label on the release PR.

#### 6.4.4 CLI Support

```bash
# Filter by priority
python -m getitune.benchmark run --priority core
python -m getitune.benchmark run --priority core,extended

# Combine with other filters
python -m getitune.benchmark run --priority core --task detection --size-tier tiny

# See what would run (dry-run)
python -m getitune.benchmark run --priority extended --dry-run
# Output: "Would run 15 of 60 extended models (rotation group 2 of 4)"
```

#### 6.4.5 Staleness Detection

Models that have not been benchmarked within their expected cadence are flagged
as **stale** in the report. The reporter checks MLflow for the last successful
run of each model and warns if it exceeds the expected interval:

| Priority   | Expected Cadence         | Stale After |
| ---------- | ------------------------ | ----------- |
| `core`     | Weekly                   | 2 weeks     |
| `extended` | Every N weeks (rotation) | 2N weeks    |

`exploratory` models are not tracked for staleness since they have no
automated cadence, they are run purely on-demand.

Stale models are highlighted in the Markdown report and can optionally be
force-included in the next run via `--include-stale`.

---

## 7. Benchmark Runner

The runner is the core execution engine. It replaces both `tests/perf_v2/benchmark.py`
and `tests/regression/test_regression.py`.

### 7.1 Package Structure

```
src/getitune/benchmark/
├── __init__.py
├── __main__.py          # CLI entry: `python -m getitune.benchmark`
├── cli.py               # Argument parsing
├── catalog.py           # Dataset catalog loading & provisioning
├── dataset_helpers.py   # Shared helpers for preparation scripts (download, extract, parse_args)
├── manifest.py          # Benchmark manifest loading & filtering
├── runner.py            # Core orchestration loop
├── experiment.py        # Single experiment (train/test/export/optimize)
├── tracking.py          # MLflow integration
└── report.py            # Report generation, baseline resolution & regression comparison
```

### 7.2 Core Loop (`runner.py`)

> **Sequential execution by design.** The core loop runs one experiment at a
> time. This is intentional: benchmark experiments are GPU-bound, and running
> multiple training jobs concurrently on a single GPU causes memory contention,
> non-deterministic scheduling, and unreliable timing measurements. Sequential
> execution guarantees that each experiment has exclusive access to the
> accelerator, producing stable and reproducible metrics.
>
> For future multi-GPU runners, the runner can be extended with a `--jobs N`
> flag that assigns experiments to different GPUs via `CUDA_VISIBLE_DEVICES`
> round-robin. Each GPU still runs one experiment at a time. This is not
> planned for v1, the current single-GPU sequential model matches the
> self-hosted runner setup described in §11.

```python
class BenchmarkRunner:
    """Orchestrates the full benchmark run."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.tracker = MLflowTracker(config.tracking)
        self.reporter = Reporter(config.reporting)
        self.results: list[ExperimentResult] = []
        self.failures: list[ExperimentFailure] = []

    def run(self) -> BenchmarkReport:
        manifest = load_manifest(self.config.manifest_path, filters=self.config.filters)
        catalog = load_catalog(self.config.catalog_path)
        required_names = {exp.dataset_name for exp in manifest.iter_experiments()}
        required_entries = catalog.filter(names=required_names)
        dataset_paths = provision_datasets(catalog, self.config.data_root,
                                           entries=required_entries)

        for experiment in manifest.iter_experiments():
            # experiment = (task, model, dataset, scenario, seed)
            for seed in range(experiment.num_seeds):
                result = self._run_single(experiment, seed)
                if result.success:
                    self.results.append(result)
                else:
                    self.failures.append(result.failure)

        report = self.reporter.generate(
            results=self.results,
            failures=self.failures,
            baseline=self.config.baseline,
        )
        return report

    def _run_single(self, experiment: Experiment, seed: int) -> ExperimentResult:
        """Run a single (model, dataset, scenario, seed) combination.

        Catches all exceptions so that one failure does not stop the entire suite.
        Retries exactly once on failure before giving up.
        """
        max_attempts = 2
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                with self.tracker.start_run(experiment, seed) as run:
                    executor = ExperimentExecutor(
                        experiment=experiment,
                        seed=seed,
                        config=self.config,
                    )

                    # Phase 1: Train
                    train_metrics = executor.train()
                    run.log_metrics(train_metrics, phase="train")

                    # Phase 2: Test (PyTorch checkpoint)
                    test_metrics = executor.test(checkpoint_type="torch")
                    run.log_metrics(test_metrics, phase="test/torch")

                    # Phase 3: Export + Test (OpenVINO IR)
                    if experiment.eval_upto in ("export", "optimize"):
                        executor.export()
                        export_metrics = executor.test(checkpoint_type="export")
                        run.log_metrics(export_metrics, phase="test/export")

                    # Phase 4: Optimize (NNCF/POT) + Test
                    if experiment.eval_upto == "optimize":
                        executor.optimize()
                        opt_metrics = executor.test(checkpoint_type="optimize")
                        run.log_metrics(opt_metrics, phase="test/optimize")

                    return ExperimentResult(experiment, seed, run.all_metrics())

            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Experiment %s seed=%d failed (attempt %d/%d), retrying: %s",
                        experiment, seed, attempt, max_attempts, exc,
                    )
                else:
                    logger.error(
                        "Experiment %s seed=%d failed after %d attempts: %s",
                        experiment, seed, max_attempts, exc,
                    )

        return ExperimentResult.failure(experiment, seed, last_exc)
```

### 7.3 Experiment Executor (`experiment.py`)

Wraps `OTXEngine` / `OVEngine` with timing instrumentation, GPU memory
tracking, and structured metric collection. This is a thin wrapper, the
actual training logic stays in the engine.

```python
class ExperimentExecutor:
    def __init__(self, experiment: Experiment, seed: int, config: RunConfig):
        self.experiment = experiment
        self.seed = seed
        self.config = config
        self.work_dir = config.output_root / experiment.run_id / str(seed)

    def train(self) -> dict[str, float]:
        # 1. Resolve scenario overrides (JSON-serialize complex values)
        recipe_overrides = _resolve_overrides(
            scenario_overrides=self.experiment.scenario.overrides,
        )

        # 2. Build engine with resolved overrides applied on top of recipe
        engine = OTXEngine.from_config(
            config_path=self.experiment.recipe_path,
            data_root=self.data_path,
            work_dir=self.work_dir / "train",
            device=self.config.accelerator,
            **recipe_overrides,        # <-- scenario parameter overrides
        )

        # 3. Merge train_kwargs from scenario (e.g. precision, max_epochs)
        kwargs = {"seed": self.seed, "deterministic": self.config.deterministic}
        if self.config.max_epochs > 0:
            kwargs["max_epochs"] = self.config.max_epochs
        kwargs.update(self.experiment.scenario.train_kwargs)

        start = time.time()
        engine.train(**kwargs)
        wall_time = time.time() - start

        return {
            "training:e2e_time": wall_time,
            "training:gpu_mem": _get_peak_gpu_memory_mb(),
            "training:scenario": self.experiment.scenario.name,
            # ... parse metrics.csv for val metrics, iter time
        }
    # ... test(), export(), optimize() similarly
```

### 7.4 Key Design Decisions

| Decision                                                       | Rationale                                                                                                                                                                                                                                                                                                                         |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sequential execution on a single GPU**                       | Benchmark experiments are GPU-bound. Running multiple training jobs concurrently on one GPU causes memory contention and unreliable timing measurements. Sequential execution guarantees exclusive accelerator access and reproducible metrics. Multi-GPU parallelism (one experiment per GPU) can be added later via `--jobs N`. |
| **Not pytest-based**                                           | The benchmark is an _application_, not a test suite. Pytest's fixture/assertion model adds coupling without benefit. A standalone CLI is simpler to invoke from CI, run locally, and resume.                                                                                                                                      |
| **YAML manifests over Python registrations**                   | A single YAML file is readable, diffable, and editable without understanding Python dataclasses. Adding a model means adding 2 lines of YAML, not touching 3 Python files.                                                                                                                                                        |
| **Catch-and-continue error handling**                          | A failure in one experiment must never abort the others. Failures are collected and reported at the end.                                                                                                                                                                                                                          |
| **Experiment = immutable record**                              | Each `(task, model, dataset, scenario, seed)` tuple is a self-contained unit. It can be resumed, retried, or skipped independently.                                                                                                                                                                                               |
| **MLflow for tracking, CSV/Markdown for reporting**            | MLflow provides the interactive dashboard and historical comparison. CSV and Markdown reports are generated for CI artifacts and PR comments — they work without a running MLflow server.                                                                                                                                         |
| **Configurable parameters as scenarios, not separate recipes** | Override-based scenarios allow testing any parameter variation without duplicating recipe YAML files. All override values are explicit literals, making scenario definitions self-contained and readable without consulting recipe defaults.                                                                                      |

---

## 8. Experiment Tracking with MLflow

### 8.1 Why MLflow

- Provides experiment comparison UI out-of-the-box.
- Supports remote tracking servers for team-wide visibility.
- Tracks parameters, metrics, artifacts, and system tags.

### 8.2 Tracking Schema

Each benchmark invocation creates an **MLflow Experiment** named with a
structured convention:

```
otx-benchmark/{branch}/{trigger}
```

For example: `otx-benchmark/develop/weekly`, `otx-benchmark/develop/nightly`,
`otx-benchmark/feature/new-aug/pr`. All runs for a given branch and trigger
type land in the same experiment, making historical comparison straightforward
in the MLflow UI. Individual benchmark invocations are distinguished by their
run timestamps and `git_sha` tags, not by experiment name. Multiple runs on
the same day (e.g., a manual `workflow_dispatch` followed by a nightly) simply
add more runs to the same experiment, which is the intended MLflow workflow.

Each `(model, dataset, scenario, seed)` run maps to one **MLflow Run**, with:

| Field                | MLflow Concept               | Example                 |
| -------------------- | ---------------------------- | ----------------------- |
| Task                 | Tag `task`                   | `DETECTION`             |
| Model                | Tag `model`                  | `yolox_s`               |
| Dataset              | Tag `dataset`                | `pothole_tiny_1`        |
| Scenario             | Tag `scenario`               | `default`               |
| Seed                 | Tag `seed`                   | `0`                     |
| Size tier            | Tag `size_tier`              | `tiny`                  |
| OTX version          | Tag `otx_version`            | `2.5.0`                 |
| Git ref              | Tag `git_sha`                | `a1b2c3d`               |
| Branch               | Tag `branch`                 | `develop`               |
| Accelerator          | Tag `accelerator`            | `NVIDIA A100`           |
| Val accuracy         | Metric `training:val/mAP`    | `0.87`                  |
| Train wall time      | Metric `training:e2e_time`   | `342.5`                 |
| Peak GPU memory (MB) | Metric `training:gpu_mem`    | `4821`                  |
| Export latency       | Metric `export:test/latency` | `0.012`                 |
| Recipe YAML          | Artifact                     | `atss_mobilenetv2.yaml` |
| Raw metrics CSV      | Artifact                     | `metrics.csv`           |

### 8.3 Deployment Options

| Option                           | When to use                                                                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Local file store** (`mlruns/`) | Local developer runs, quick experiments. Artifacts stored as CI job output.                                                                    |
| **Remote tracking server**       | Team-wide dashboard. A long-running MLflow server (e.g. on an internal VM or managed service). CI jobs post results via `MLFLOW_TRACKING_URI`. |

The runner will default to a local file store and accept `--mlflow-tracking-uri`
to point to a remote server. The CI workflow will set this via a GitHub
Actions secret.

---

## 9. Reporting & Regression Detection

### 9.1 Baseline System

Every benchmark run logs all metrics to MLflow with structured tags (`branch`,
`git_sha`, `accelerator`, `scenario`, etc.). To determine the baseline for
comparison, the reporter simply queries MLflow for the most recent successful
`develop` run matching the same (model, dataset, scenario, accelerator)
combination. MLflow is the single source of truth for both historical results
and baselines.

#### 9.1.1 How Baselines Are Resolved

When the reporter needs a baseline for comparison, it queries MLflow:

```python
def get_baseline(
    tracking_uri: str,
    experiment_name: str,
    model: str,
    dataset: str,
    scenario: str = "default",
    accelerator: str = "gpu",
    branch: str = "develop",
) -> dict[str, float] | None:
    """Fetch the most recent successful baseline from MLflow.

    Queries for the latest run on the given branch that matches the
    (model, dataset, scenario, accelerator) combination. Returns
    the run's metrics dict, or None if no baseline exists yet.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.branch = '{branch}' "
            f"AND tags.model = '{model}' "
            f"AND tags.dataset = '{dataset}' "
            f"AND tags.scenario = '{scenario}' "
            f"AND tags.accelerator = '{accelerator}' "
            f"AND tags.status = 'success'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0].data.metrics
```

The query filters ensure that:

- **Accelerator-scoped**: A GPU run is only compared against a previous GPU
  baseline, never against XPU or CPU.
- **Scenario-scoped**: A `lr_high` run is compared against the previous
  `lr_high` baseline, not against the `default` baseline.
- **Branch-scoped**: Only `develop` runs serve as baselines (not feature
  branches), ensuring a stable reference point.
- **Seed-aggregated**: The reporter averages across seeds for the current run
  and queries the most recent run (which was similarly averaged when it was
  reported). Alternatively, individual seed-level runs can be compared by
  adding a `tags.seed` filter.

#### 9.1.2 First-Run Behavior

When no baseline exists yet (e.g., a new model or the very first benchmark
run), the reporter marks all metrics as `status: "no_baseline"` in the report.
No regression can be detected, but the metrics are still logged to MLflow and
will serve as the baseline for subsequent runs.

#### 9.1.3 Local / Offline Fallback

When running locally without a remote MLflow server (`mlruns/` file store),
baselines are resolved against the local MLflow store. This means:

- **Local developer runs**: Compare against whatever runs exist locally. If the
  developer has previously run a benchmark, those results serve as the baseline.
  If not, no baseline exists and the report simply shows current metrics.
- **CI runs**: The self-hosted runner's MLflow file store persists across runs
  (it lives on the runner's disk, not in the ephemeral job workspace). Each
  successful `develop` benchmark automatically becomes the baseline for the
  next run.

> **When a remote MLflow server is deployed** (Phase 5), the tracking URI is
> set via `MLFLOW_TRACKING_URI` and all CI runs share a single server. This
> gives the richest baseline history and enables cross-run comparison from
> the MLflow UI.

### 9.2 Comparison Logic

After a benchmark run completes, the reporter loads the appropriate baseline(s)
and compares each metric:

```python
@dataclass
class RegressionResult:
    experiment: str
    metric: str
    baseline_value: float
    current_value: float
    threshold: float
    direction: str          # "higher_is_better" | "lower_is_better"
    status: str             # "pass" | "regression" | "improvement" | "missing"
```

A metric is flagged as **regression** when:

- For accuracy-like metrics (`higher_is_better`): `current < baseline * (1 - margin)`
- For cost-like metrics (`lower_is_better`): `current > baseline * (1 + margin)`

Margins are defined per-metric in the benchmark manifest (Section 6).

### 9.3 Report Outputs

| Output                | Format                  | Destination                                                               |
| --------------------- | ----------------------- | ------------------------------------------------------------------------- |
| **Summary table**     | Markdown                | PR comment (via `peter-evans/create-or-update-comment`) or CI job summary |
| **Detailed CSV**      | CSV                     | CI artifact (downloadable)                                                |
| **Regression alerts** | Markdown callout blocks | PR comment                                                                |
| **Failure list**      | Markdown                | PR comment                                                                |
| **MLflow dashboard**  | Web UI                  | Link in PR comment                                                        |

Example summary (Markdown):

```markdown
## OTX Benchmark Report — 2026-03-26

**Branch:** `feature/new-augmentation` | **Commit:** `a1b2c3d` | **Accelerator:** NVIDIA A100

### Detection

| Model            | Dataset        | Val mAP ↑    | Test mAP ↑ | Export mAP ↑ | Train Time ↓ | GPU Mem ↓ | Test Latency ↓ | Status                      |
| ---------------- | -------------- | ------------ | ---------- | ------------ | ------------ | --------- | -------------- | --------------------------- |
| yolox_s          | pothole_tiny   | 0.87 (±0.01) | 0.85       | 0.84         | 5m 12s       | 4.8 GB    | 12ms           | ✅                          |
| atss_mobilenetv2 | diopsis_medium | 0.72 (±0.02) | 0.70       | 0.68         | 12m 05s      | 3.2 GB    | 8ms            | ⚠️ val mAP -13% vs baseline |

### Failures (1)

| Model   | Dataset        | Seed | Error                    |
| ------- | -------------- | ---- | ------------------------ |
| dfine_x | visdrone_large | 2    | OOM — CUDA out of memory |

📊 [Full results on MLflow](https://mlflow.example.com/experiments/42)
```

---

## 10. PR Change Detection & Targeted Benchmarks

> **Note:** This section describes a future enhancement. The detailed design
> (dependency map format, resolution algorithm, CI workflow) will be worked out
> in a follow-up document once the core benchmark runner (Sections 5–9) is
> operational.

The weekly full-matrix benchmark is thorough but expensive. Most PRs only touch
a handful of models or a single shared component. Running the _entire_ benchmark
on every PR would waste hours of GPU time and slow down the development loop.

The idea is to introduce a **change-detection layer** that analyzes the files
modified in a PR, determines which models are affected, and triggers only the
relevant subset of benchmarks on a reduced dataset scope (e.g. `tiny` + `small`
only, single seed, train-only). This keeps PR feedback in the 10–30 minute
range.

The codebase lends itself naturally to this because of its layered structure:

- **Global** — shared infrastructure (`src/otx/engine/`, `src/otx/data/`,
  `models/modules/`, `models/common/`) → changes here affect all models.
- **Task-level** — task-specific modules (`models/detection/heads/`,
  `metrics/fmeasure.py`, base data configs) → changes affect all models in
  that task.
- **Model-specific** — individual model files and recipe YAMLs
  (`models/detection/yolox.py`, `recipe/detection/yolox_s.yaml`) → changes
  affect only that model family.

A mapping from file-glob patterns to models can be declared in the benchmark
manifest itself (via `global_triggers`, `task_triggers`, and per-model
`triggers` fields). A lightweight `detect-changes` CI job would diff the PR,
resolve affected models, and pass them as a filter to the benchmark runner.

Manual overrides (e.g. PR labels `benchmark:full` / `benchmark:skip`) should
also be supported for edge cases.

---

## 11. CI/CD Integration (GitHub Actions)

### 11.1 Workflow File

```yaml
# .github/workflows/benchmark.yml
name: Model Benchmark

on:
  schedule:
    - cron: "0 2 * * 0" # Weekly full: Sunday 02:00 UTC
    - cron: "0 2 * * 1-6" # Nightly quick: Mon–Sat 02:00 UTC
  pull_request:
    types: [labeled] # Trigger on benchmark labels
  workflow_dispatch:
    inputs:
      task:
        description: "Task filter (e.g. detection, all)"
        default: "all"
      model:
        description: "Model filter (comma-separated, or 'all')"
        default: "all"
      priority:
        description: "Model priority tiers to include (see §6.4)"
        default: "core,extended"
        type: choice
        options: [core, "core,extended", "core,extended,exploratory"]
      size_tier:
        description: "Dataset size tiers to include"
        default: "tiny,small,medium"
      scenario:
        description: "Scenario filter (default, tiling, lr_high, ... or 'all')"
        default: "default"
      accelerator:
        description: "Accelerator target"
        default: "gpu"
        type: choice
        options: [gpu, xpu, cpu]
      eval_upto:
        description: "Last evaluation phase to execute (all preceding phases are included: train → export → optimize)"
        default: "optimize"
        type: choice
        options: [train, export, optimize]

concurrency:
  group: benchmark-${{ github.event_name }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ── Step 0: Determine run parameters based on trigger ──────────────
  configure:
    runs-on: ubuntu-latest
    outputs:
      priority: ${{ steps.cfg.outputs.priority }}
      size_tier: ${{ steps.cfg.outputs.size_tier }}
      num_seeds: ${{ steps.cfg.outputs.num_seeds }}
      eval_upto: ${{ steps.cfg.outputs.eval_upto }}
      scenario: ${{ steps.cfg.outputs.scenario }}
      should_run: ${{ steps.cfg.outputs.should_run }}
    steps:
      - id: cfg
        run: |
          # Defaults for workflow_dispatch, use inputs directly
          if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
            echo "priority=${{ inputs.priority }}" >> "$GITHUB_OUTPUT"
            echo "size_tier=${{ inputs.size_tier }}" >> "$GITHUB_OUTPUT"
            echo "num_seeds=3" >> "$GITHUB_OUTPUT"
            echo "eval_upto=${{ inputs.eval_upto }}" >> "$GITHUB_OUTPUT"
            echo "scenario=${{ inputs.scenario }}" >> "$GITHUB_OUTPUT"
            echo "should_run=true" >> "$GITHUB_OUTPUT"

          # Weekly full: Sunday schedule
          elif [[ "${{ github.event.schedule }}" == "0 2 * * 0" ]]; then
            echo "priority=core,extended" >> "$GITHUB_OUTPUT"
            echo "size_tier=tiny,small,medium" >> "$GITHUB_OUTPUT"
            echo "num_seeds=3" >> "$GITHUB_OUTPUT"
            echo "eval_upto=optimize" >> "$GITHUB_OUTPUT"
            echo "scenario=default" >> "$GITHUB_OUTPUT"
            echo "should_run=true" >> "$GITHUB_OUTPUT"

          # Nightly quick: Mon–Sat schedule
          elif [[ "${{ github.event.schedule }}" == "0 2 * * 1-6" ]]; then
            echo "priority=core" >> "$GITHUB_OUTPUT"
            echo "size_tier=tiny,small" >> "$GITHUB_OUTPUT"
            echo "num_seeds=1" >> "$GITHUB_OUTPUT"
            echo "eval_upto=export" >> "$GITHUB_OUTPUT"
            echo "scenario=default" >> "$GITHUB_OUTPUT"
            echo "should_run=true" >> "$GITHUB_OUTPUT"

          # PR label: benchmark:run / benchmark:params / benchmark:full
          elif [[ "${{ github.event_name }}" == "pull_request" ]]; then
            LABEL="${{ github.event.label.name }}"
            if [[ "$LABEL" == "benchmark:run" ]]; then
              echo "priority=core" >> "$GITHUB_OUTPUT"
              echo "size_tier=tiny" >> "$GITHUB_OUTPUT"
              echo "num_seeds=1" >> "$GITHUB_OUTPUT"
              echo "eval_upto=train" >> "$GITHUB_OUTPUT"
              echo "scenario=default" >> "$GITHUB_OUTPUT"
              echo "should_run=true" >> "$GITHUB_OUTPUT"
            elif [[ "$LABEL" == "benchmark:params" ]]; then
              echo "priority=core" >> "$GITHUB_OUTPUT"
              echo "size_tier=tiny" >> "$GITHUB_OUTPUT"
              echo "num_seeds=1" >> "$GITHUB_OUTPUT"
              echo "eval_upto=train" >> "$GITHUB_OUTPUT"
              echo "scenario=default,configurable" >> "$GITHUB_OUTPUT"
              echo "should_run=true" >> "$GITHUB_OUTPUT"
            elif [[ "$LABEL" == "benchmark:full" ]]; then
              echo "priority=core,extended" >> "$GITHUB_OUTPUT"
              echo "size_tier=tiny,small,medium" >> "$GITHUB_OUTPUT"
              echo "num_seeds=3" >> "$GITHUB_OUTPUT"
              echo "eval_upto=optimize" >> "$GITHUB_OUTPUT"
              echo "scenario=default" >> "$GITHUB_OUTPUT"
              echo "should_run=true" >> "$GITHUB_OUTPUT"
            else
              echo "should_run=false" >> "$GITHUB_OUTPUT"
            fi
          fi

  # ── Step 1: Run all benchmarks on a single self-hosted GPU runner ──
  benchmark:
    needs: configure
    if: needs.configure.outputs.should_run == 'true'
    runs-on: [self-hosted, "${{ inputs.accelerator || 'gpu' }}"]
    timeout-minutes: 2880 # 48h safety net for large runs
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
    steps:
      - uses: actions/checkout@v4

      - name: Install OTX
        working-directory: library
        run: uv sync --frozen --extra ${{ inputs.accelerator || 'cuda' }}

      - name: Restore dataset cache
        uses: actions/cache@v4
        with:
          path: data/
          key: benchmark-datasets-${{ hashFiles('scripts/benchmark_datasets/**') }}

      - name: Provision datasets
        run: python -m getitune.benchmark provision --catalog benchmark_catalog.yaml --data-root data/

      - name: Run benchmarks
        run: |
          python -m getitune.benchmark run \
            --manifest benchmark_manifest.yaml \
            --catalog benchmark_catalog.yaml \
            --data-root data/ \
            --output-root results/ \
            --task "${{ inputs.task || 'all' }}" \
            --model "${{ inputs.model || 'all' }}" \
            --priority "${{ needs.configure.outputs.priority }}" \
            --size-tier "${{ needs.configure.outputs.size_tier }}" \
            --eval-upto "${{ needs.configure.outputs.eval_upto }}" \
            --accelerator "${{ inputs.accelerator || 'gpu' }}" \
            --num-seeds "${{ needs.configure.outputs.num_seeds }}" \
            --scenario "${{ needs.configure.outputs.scenario }}"

      - name: Generate report
        if: always()
        run: |
          python -m getitune.benchmark report \
            --results-dir results/ \
            --output report.md

      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.sha }}
          path: |
            results/**/*.csv
            report.md

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: peter-evans/create-or-update-comment@v4
        with:
          issue-number: ${{ github.event.pull_request.number }}
          body-path: report.md
```

### 11.2 Workflow Variants by Hardware

The same workflow file is parameterized by `accelerator`. Self-hosted runners
are labeled by hardware class:

| Label | Hardware                    | Use case                         |
| ----- | --------------------------- | -------------------------------- |
| `gpu` | NVIDIA A100 / A10           | Primary GPU benchmark            |
| `xpu` | Intel Arc / Data Center GPU | XPU support validation           |
| `cpu` | Any x86_64                  | CPU-only export/optimize testing |

The weekly schedule runs on `gpu` and `xpu`. The `workflow_dispatch` trigger lets
engineers run on any target on-demand.

### 11.3 Runner Environment

No dedicated Docker image is required. The benchmark runner is a plain Python
CLI (`python -m getitune.benchmark`) with no dependencies beyond what OTX and its
dev group already provide (`mlflow`, `py-cpuinfo`, etc. are already in
`pyproject.toml` `[dependency-groups] dev`).

The CI job installs OTX directly on the self-hosted runner:

```yaml
- name: Install OTX
  working-directory: library
  run: |
    uv sync --frozen --extra ${{ inputs.accelerator || 'cuda' }}
```

---

## 12. Hardware & Accelerator Support

### 12.1 Parameterization

The `accelerator` parameter flows through every layer:

1. **CI workflow** → selects the runner label and container image.
2. **Benchmark runner** → passes `--device gpu|xpu|cpu` to `OTXEngine`.
3. **MLflow tags** → records the exact accelerator model for comparison.
4. **Baselines** → MLflow queries are scoped by `tags.accelerator` (see §9.1.1).

### 12.2 Multi-Accelerator Baselines

Baselines are scoped by accelerator via MLflow tags (see §9.1.1). The
`get_baseline()` query includes `tags.accelerator` in its filter, so a GPU
run is only compared against a previous GPU baseline, never against XPU or
CPU results.

This means each accelerator class builds up its own independent baseline
history within MLflow. When a new accelerator (e.g., XPU) runs benchmarks for
the first time, the reporter will show `no_baseline` for all metrics. After
that first successful `develop` run, subsequent XPU runs will compare against
the XPU-specific baseline.

Comparisons are always made against the same accelerator class. Comparing GPU
vs. XPU timings is meaningless; accuracy should be consistent across all.

---

## 13. Failure Handling

### 13.1 Principles

1. **Isolate**: Each `(model, dataset, scenario, seed)` runs in its own
   try/except block. A CUDA OOM in one experiment must not affect others.
2. **Record**: Failed experiments are logged with the full exception traceback,
   the experiment identity, and the last-known state.
3. **Report**: The final report includes a dedicated "Failures" section.
4. **Retry once**: If an experiment fails for any reason, the runner retries it
   exactly once. If the retry also fails, the experiment is recorded as failed
   and the runner moves on.
5. **Resume**: If a run is interrupted (e.g. CI timeout), it can be re-invoked
   with `--resume-from <output-root>`. The runner detects completion status
   at phase granularity (see §13.3).

### 13.2 Exit Codes

| Code | Meaning                                                                 |
| ---- | ----------------------------------------------------------------------- |
| `0`  | All experiments passed                                                  |
| `1`  | At least one experiment failed (but others completed)                   |
| `2`  | Infrastructure error (e.g. dataset download failed, MLflow unreachable) |

The CI workflow uses `if: always()` for the report step so that partial results
are always surfaced.

### 13.3 Resume Semantics

When `--resume-from <output-root>` is specified, the runner scans the output
directory to determine which experiments (and which phases within each
experiment) have already completed. Completion is detected per-phase by
checking for specific marker files:

| Phase           | Completion Marker                                               | Reusable Artifact  |
| --------------- | --------------------------------------------------------------- | ------------------ |
| Train           | `<seed>/train/metrics.csv` exists AND contains at least one row | Trained checkpoint |
| Test (torch)    | `<seed>/test/torch/result.json` exists                          | —                  |
| Export          | `<seed>/export/exported_model.xml` exists                       | Exported IR model  |
| Test (export)   | `<seed>/test/export/result.json` exists                         | —                  |
| Optimize        | `<seed>/optimize/optimized_model.xml` exists                    | Optimized model    |
| Test (optimize) | `<seed>/test/optimize/result.json` exists                       | —                  |

The resume logic follows these rules:

1. **All phases complete** → skip the entire experiment.
2. **Training complete, later phases missing** → reuse the trained checkpoint,
   resume from the first incomplete phase (e.g. export).
3. **Training incomplete** → delete the partial training directory and restart
   from scratch (partial checkpoints are unreliable).
4. **Marker file exists but is empty/corrupt** → treat as incomplete, re-run
   that phase.

```python
def _should_skip(self, experiment: Experiment, seed: int) -> tuple[bool, str | None]:
    """Check if an experiment can be skipped or partially resumed.

    Returns:
        (skip_entirely, resume_from_phase)
        - (True, None): all phases done, skip completely
        - (False, None): start from scratch
        - (False, "export"): training done, resume from export phase
    """
    seed_dir = self.config.output_root / experiment.run_id / str(seed)

    # Check training completion
    metrics_csv = seed_dir / "train" / "metrics.csv"
    if not metrics_csv.exists() or metrics_csv.stat().st_size == 0:
        # Training not done or corrupt, start over
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        return False, None

    # Training is done, check subsequent phases
    checkpoint = seed_dir / "train" / "best_checkpoint.ckpt"
    if not checkpoint.exists():
        # Training metrics exist but checkpoint doesn't, corrupt state
        shutil.rmtree(seed_dir)
        return False, None

    # Determine the first incomplete phase
    phase_markers = [
        ("test/torch", seed_dir / "test" / "torch" / "result.json"),
        ("export", seed_dir / "export" / "exported_model.xml"),
        ("test/export", seed_dir / "test" / "export" / "result.json"),
        ("optimize", seed_dir / "optimize" / "optimized_model.xml"),
        ("test/optimize", seed_dir / "test" / "optimize" / "result.json"),
    ]

    for phase_name, marker in phase_markers:
        if not marker.exists():
            return False, phase_name

    # All markers present, fully complete
    return True, None
```

> **Note on existing `perf_v2` resume:** The existing `benchmark.py` has a
> `resume_from` parameter that checks for the presence of checkpoint files to
> skip training. The new design improves on this by (a) checking at phase
> granularity rather than only training, (b) validating marker file integrity,
> and (c) cleaning up corrupt partial state rather than silently reusing it.

---

## 14. Directory & File Layout

### 14.1 Source Code

```
libraries/getitune/
├── src/getitune/benchmark/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── catalog.py
│   ├── manifest.py
│   ├── runner.py
│   ├── experiment.py
│   ├── tracking.py
│   └── report.py
├── scripts/
│   └── benchmark_datasets/
│       ├── prepare_pothole.py
│       ├── prepare_wgisd.py
│       ├── prepare_diopsis.py
│       └── ...
├── benchmark_catalog.yaml
└── benchmark_manifest.yaml
```

> **Note:** Baselines are resolved by querying MLflow for the most recent
> successful `develop` run (see §9.1.1).

### 14.2 Runtime Output (`results/`)

```
results/
├── detection/
│   ├── yolox_s/
│   │   ├── pothole_tiny_1/
│   │   │   ├── 0/           # seed 0
│   │   │   │   ├── train/
│   │   │   │   │   ├── metrics.csv
│   │   │   │   │   └── best_checkpoint.ckpt
│   │   │   │   ├── test/
│   │   │   │   │   ├── torch/
│   │   │   │   │   ├── export/
│   │   │   │   │   └── optimize/
│   │   │   │   ├── export/
│   │   │   │   │   └── exported_model.xml
│   │   │   │   └── optimize/
│   │   │   │       └── optimized_model.xml
│   │   │   ├── 1/           # seed 1
│   │   │   └── 2/           # seed 2
│   │   └── ...
│   └── ...
├── aggregated.csv           # all results, averaged across seeds
├── failed_experiments.json  # structured failure log
└── report.md                # human-readable summary
```

---

## 15. Migration from Existing Code

### 15.1 What We Keep

| Component                         | From                                                                     | Reuse                              |
| --------------------------------- | ------------------------------------------------------------------------ | ---------------------------------- |
| `OTXEngine` / `OVEngine` API      | `src/otx/backend/native/engine.py`, `src/otx/backend/openvino/engine.py` | Used as-is by `ExperimentExecutor` |
| MLflow integration pattern        | `tests/regression/conftest.py`                                           | Adapted into `tracking.py`         |
| Metric parsing from `metrics.csv` | `tests/perf_v2/benchmark.py` `_log_metrics()`                            | Ported into `experiment.py`        |
| Version/HW tag collection         | `tests/perf_v2/utils.py` `build_tags()`                                  | Ported into `tracking.py`          |
| Summary pivot tables              | `tests/perf_v2/summary.py`                                               | Adapted into `report.py`           |

### 15.2 What We Replace

| Old                                                         | New                                                  | Reason                                                |
| ----------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| `tests/perf_v2/tasks/*.py` (model/dataset registration)     | `benchmark_manifest.yaml` + `benchmark_catalog.yaml` | Declarative over imperative; single source of truth   |
| `tests/perf_v2/run.py` (subprocess-per-experiment)          | `runner.py` (in-process with isolation)              | Simpler, faster, better error handling                |
| `tests/perf_v2/utils.py::Criterion` (assert-based checking) | `report.py` (MLflow-based baseline comparison)       | Regressions should be _reported_, not crash the suite |
| `tests/regression/test_regression.py` (pytest parametrize)  | `runner.py` CLI                                      | Not a test suite; it's an application                 |
| Copy-pasted test classes per task                           | Manifest-driven iteration                            | Zero boilerplate per task                             |

### 15.3 Deprecation

- `tests/perf_v2/` and `tests/regression/` are kept in the repository for one
  release cycle with a deprecation notice, then removed.
- Existing CSV history in `csv/version_*` remains as read-only archival data.
  The new system builds baselines from MLflow history starting fresh.

---

## 16. Rollout Plan

### Phase 1 — Foundation (2 weeks)

- [ ] Create `src/getitune/benchmark/` package skeleton.
- [ ] Implement `catalog.py`: YAML parsing, script execution, sentinel-based caching (script hash re-verification).
- [ ] Implement `manifest.py`: YAML parsing, filtering, experiment enumeration, `{metric}` placeholder resolution.
- [ ] Write `benchmark_catalog.yaml` for detection (one task, end-to-end proof of concept).
- [ ] Write preparation scripts in `scripts/benchmark_datasets/` for detection datasets.
- [ ] Write `benchmark_manifest.yaml` for detection.
- [ ] Implement `experiment.py`: thin wrapper around `OTXEngine` with timing, JSON serialization for complex overrides.
- [ ] Implement `runner.py`: core loop with catch-and-continue, phase-level resume support (§13.3).
- [ ] Unit tests for catalog, manifest, runner (mocked engine).

### Phase 2 — Tracking & Reporting (1 week)

- [ ] Implement `tracking.py`: MLflow integration (local file store first).
- [ ] Implement `report.py`: Markdown report generation with MLflow-based baseline resolution (§9.1.1).
- [ ] End-to-end local test: run detection benchmark on a single GPU, verify report.

### Phase 3 — CI Integration (1 week)

- [ ] Create `.github/workflows/benchmark.yml` with all triggers (weekly full, nightly quick, PR label, `workflow_dispatch`).
- [ ] Implement the `configure` job for trigger-dependent parameter selection.
- [ ] Set up dataset caching in GitHub Actions.
- [ ] Wire up PR comment reporting.
- [ ] Test `workflow_dispatch` with different parameter combinations.
- [ ] Implement automatic disk space cleanup: delete training checkpoints and
      exported models after metrics are extracted, keeping only `metrics.csv` and
      `result.json` files. Add `--keep-checkpoints` flag for when artifacts are
      needed. This must be in place before CI runs accumulate data on the self-hosted runner.
- [ ] Implement MLflow run retention policy (auto-delete runs older than N weeks
      on local file store; configure TTL on remote server).

### Phase 4 — Full Coverage (2 weeks)

- [ ] Extend `benchmark_catalog.yaml` and `benchmark_manifest.yaml` to all tasks:
      classification (multi-class, multi-label, h-label), semantic segmentation,
      instance segmentation, keypoint detection, rotated detection.
- [ ] **Rotated detection note:** This task has no existing `perf_v2` coverage
      (no `tasks/rotated_detection.py` exists). Models, datasets, and criteria must
      be defined from scratch. Coordinate with rotated detection model owners to
      identify appropriate datasets and accuracy metrics.
- [ ] Curate and write preparation scripts for all datasets.
- [ ] Run full matrix on GPU runner; generate initial baselines (first `develop` run logs to MLflow, establishing the baseline for future comparisons).
- [ ] Add tiling scenario configurations.

### Phase 5 — Polish & Multi-HW (1 week)

- [ ] Set up XPU runner and run initial benchmarks (first run establishes XPU baselines in MLflow).
- [ ] Connect remote MLflow tracking server (if available).
- [ ] Deprecation notices on `tests/perf_v2/` and `tests/regression/`.
- [ ] Documentation for contributors: "How to add a new model to the benchmark."

---

## 17. Open Questions

| #   | Question                                            | Proposed Default                                                        | Notes                                                                                                                                                                                                                                                                                                                                                                                             |
| --- | --------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Where to host benchmark datasets?                   | Produced by in-repo preparation scripts (`scripts/benchmark_datasets/`) | Scripts download from any source, transform, and place data locally. No dependency on a single hosting service. Credentialed (e.g. Kaggle) or oversized datasets additionally support `local_path` (fully externally-managed, no script) and `raw_dir` (skip only the script's network fetch) — see §5.3.2.                                                                                       |
| 2   | Remote MLflow server or local-only?                 | Start local (`mlruns/`), move to remote server in Phase 5               | Avoids infra dependency in early phases.                                                                                                                                                                                                                                                                                                                                                          |
| 3   | How many seeds per experiment?                      | 3                                                                       | Balances statistical confidence vs. CI cost. 5 seeds would be better but almost doubles runtime.                                                                                                                                                                                                                                                                                                  |
| 4   | Should the benchmark block PR merges on regression? | No (advisory only)                                                      | Flaky GPU tests + margin of error make hard-gating risky initially. Revisit after baseline stability is established.                                                                                                                                                                                                                                                                              |
| 5   | Should we support multi-GPU training benchmarks?    | Not in v1                                                               | The existing `benchmark.py` has `num_devices` support. Add as a scenario later if needed (`num_devices > 1`).                                                                                                                                                                                                                                                                                     |
| 6   | Maximum acceptable CI runtime for weekly run?       | ~20 hours on a single A100                                              | All benchmarks run sequentially on one self-hosted GPU runner. The runner is given a 48h timeout. Use `--priority`, `--size-tier`, and model rotation (§6.4.2) to keep runtime within budget. Nightly quick runs (~5 hours) use `core` models + `tiny`+`small` datasets only.                                                                                                                     |
| 7   | How to handle dataset version changes?              | Treat as a baseline-breaking change                                     | When a preparation script changes, the sentinel hash mismatches and the dataset is re-provisioned automatically. All baselines for experiments using that dataset become stale. The reporter should flag these as `no_baseline` rather than `regression`. Document the process: update preparation script → re-run benchmark on `develop` → new MLflow results become the baseline automatically. |
| 8   | `deterministic: true` performance impact            | Document as known trade-off                                             | Setting `deterministic: true` disables cuDNN benchmark mode and uses slower deterministic CUDA algorithms. Benchmark timings under this setting may be 10–30% slower than real-world training. This is acceptable for regression detection (relative comparisons) but should be noted in published reports.                                                                                       |
