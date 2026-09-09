# GetiTune Benchmarking

This module runs repeatable model benchmarks for GetiTune with a CLI:

- dataset provisioning from `benchmark_catalog.yaml`
- experiment selection from `benchmark_manifest.yaml`
- phased execution (`train -> test/torch -> export -> test/export -> optimize -> test/optimize`)
- MLflow tracking and baseline comparison
- report generation (`report.md`, `aggregated.csv`, optional `failed_experiments.json`)

## Where things live

- CLI entry point: `library/src/getitune/benchmark/__main__.py`
- CLI implementation: `library/src/getitune/benchmark/cli.py`
- Runner/orchestration: `library/src/getitune/benchmark/runner.py`
- Dataset catalog: `library/benchmark_catalog.yaml`
- Benchmark manifest: `library/benchmark_manifest.yaml`
- Optional MLflow server compose file: `library/src/getitune/benchmark/docker-compose.yaml`

Run commands from `library/`.

## Quick start

1. Sync dependencies (if needed):

   ```bash
   just venv --device <cpu|cuda|xpu>
   ```

2. See all benchmark commands:

   ```bash
   python -m getitune.benchmark --help
   ```

3. Provision datasets referenced by the catalog/filters:

   ```bash
   python -m getitune.benchmark provision
   ```

4. Run a small benchmark slice:

   ```bash
   python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --accelerator <cpu|cuda|xpu> --num-seeds 1 --no-tracking
   ```

## CI benchmark schedule

The `Model Benchmark` GitHub workflow runs in two modes: a weekly scheduled run and a manual run.

| Trigger             | Timeline                  | Source ref                   | Model groups / categories                                         | Dataset size tiers          | Dataset data group           | Scenario                   | Eval phase                                    | Seeds                       | Accelerators                   |
| ------------------- | ------------------------- | ---------------------------- | ----------------------------------------------------------------- | --------------------------- | ---------------------------- | -------------------------- | --------------------------------------------- | --------------------------- | ------------------------------ |
| `schedule`          | Every Friday, `19:00 UTC` | `develop`                    | `core,extended`                                                   | `tiny,small,medium,large`   | `weekly`                     | `default`                  | `optimize`                                    | `2`                         | `gpu` and `xpu`                |
| `workflow_dispatch` | On demand                 | User-selected (`source_ref`) | User-selected (`core`, `core,extended`, `core,extended,deferred`) | User-selected (`size_tier`) | User-selected (`data_group`) | User-selected (`scenario`) | User-selected (`train`, `export`, `optimize`) | User-selected (`num_seeds`) | User-selected (`gpu` or `xpu`) |

### Rotation and timeline details

- Weekly runs use model rotation for `extended` priority models.
- Weekly runs use `--data-group weekly`, which selects only datasets explicitly marked `data_group: weekly` in the catalog rather than every dataset.
- Rotation group is computed as `ISO week number % extended_groups`.
- `extended_groups` is read from `benchmark_manifest.yaml` (`defaults.rotation.extended_groups`, currently `2`).
- `core` models are not rotated and are always included when `priority=core,extended`.
- `deferred` models are never included in an unfiltered/automated run (including rotation-group runs) — they only run when explicitly requested via `--priority deferred` or `--model <name>`.

Practical effect with `extended_groups: 2`:

- Week A: all `core` models + one half of `extended` models.
- Week B: all `core` models + the other half of `extended` models.
- After two weekly runs, the full `extended` set has been covered once (per accelerator).

## Useful run examples

Dry run only (prints what would execute):

```bash
python -m getitune.benchmark run --dry-run --task detection --size-tier small --num-seeds 1 --no-tracking
```

Run only train+torch-test phases:

```bash
python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --eval-upto train --num-seeds 1 --no-tracking
```

Run with custom output location:

```bash
python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --output-root results/local-smoke --num-seeds 1 --no-tracking
```

Run with ad-hoc config overrides:

```bash
python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --override model.init_args.optimizer.init_args.lr=0.01 --max-epochs 10 --train-kwarg precision=32 --no-tracking
```

## Report and cleanup commands

Regenerate a report from tracked MLflow runs:

```bash
python -m getitune.benchmark report --mlflow-uri http://localhost:5000 --output-root results --accelerator <cpu|cuda|xpu>
```

Clean old MLflow runs (dry run first):

```bash
python -m getitune.benchmark clean --dry-run --max-age-days 90
```

## Output artifacts

By default (`--output-root results`), the benchmark writes:

- `results/report.md` - markdown summary with pass/regression/failure sections
- `results/aggregated.csv` - flattened metrics table
- `results/failed_experiments.json` - structured failure details (only when failures exist)

Per-seed work directories are created under:

- `results/<task>/<model>/<dataset>/<seed>/` (default scenario)
- `results/<task>/<model>/<dataset>/<scenario>/<seed>/` (non-default scenario)

Dataset provisioning writes readiness markers at:

- `data/<dataset>/.ready` (script-provisioned datasets only — `local_path` datasets are
  externally managed and never get a `.ready` marker; see "Datasets requiring
  credentials or manual placement" below)

## Memory profiling

The `train` and `test/torch` phases record peak accelerator and host memory
alongside their accuracy/timing metrics:

- `training:gpu_mem`, `torch:test/gpu_mem` - peak CUDA/XPU memory allocated by
  PyTorch during that phase (MB).
- `training:ram_mem`, `torch:test/ram_mem` - peak host RSS during that phase (MB),
  summed across the process and any DataLoader worker subprocesses.

These metrics flow through the same MLflow logging, CSV export, and
baseline-comparison paths as every other metric, so they support the usual
regression thresholds in `benchmark_manifest.yaml` (10% margin by default) and
appear in `aggregated.csv`; the Markdown report additionally surfaces the
train-phase values as "GPU Mem" and "Peak RAM" columns.

RAM sampling requires `psutil`, which is only installed via the `benchmark`
extra (`just venv-benchmark`, or `uv sync --extra benchmark`); without it,
`*_ram_mem` values are best-effort and read as `0.0`. `export`/`optimize`
(OpenVINO) phases are not currently instrumented.

## Add a new dataset or model

Use this flow when extending benchmark coverage.

### Add a new dataset

1. Add a dataset provisioning script under `scripts/benchmark_datasets/` (for example, `prepare_my_dataset.py`).
2. Add an entry in `benchmark_catalog.yaml` that points `script` to that provisioning script.
3. Reference the dataset name from the relevant task in `benchmark_manifest.yaml` under `experiments.<task>.datasets`.
4. Provision the dataset.
5. Verify the new task+dataset combination appears in a dry run.

Catalog example (`benchmark_catalog.yaml`):

```yaml
datasets:
  - name: my_dataset
    script: "scripts/benchmark_datasets/prepare_my_dataset.py"
    size_tier: medium
    description: "Short description of dataset size/content."
    compatible_tasks:
      - detection
```

`size_tier` normally reflects rough dataset size (`tiny`, `small`, `medium`, `large`).
`data_group` (`weekly`, `extended`, or `all` — default `all` when omitted) controls which
benchmark lane(s) include the dataset: `weekly` datasets only run in weekly-scheduled
benchmarks, `extended` datasets only run in extended/full runs, and `all` runs in both.

Manifest reference example (`benchmark_manifest.yaml`):

```yaml
experiments:
  detection:
    datasets:
      - wgisd
      - my_dataset
```

Commands (run from `library/`):

```bash
python -m getitune.benchmark provision --dataset my_dataset
python -m getitune.benchmark run --dry-run --task detection --dataset my_dataset --num-seeds 1 --no-tracking
```

Provisioning script contract:

- The script path is read from `benchmark_catalog.yaml` (`script: ...`).
- `provision` runs the script as `python <script> --output-dir <data_root> --name <dataset_name>`.
- The script should create `data/<dataset_name>/`; the runner writes `data/<dataset_name>/.ready` after success.

### Datasets requiring credentials or manual placement

Some datasets can't be auto-downloaded by CI: they're gated behind an account/license
(e.g. Kaggle), or they're large enough that re-downloading on every run is impractical.
Two catalog fields cover this, and can be combined or used independently.

#### Option A — `local_path`: reference an already-prepared directory directly

Use this when the dataset was (or will be) fully prepared out-of-band — manually, on a
different machine, or copied once from a prior script run — and should simply be reused
as-is, with **no script execution at all**.

```yaml
datasets:
  - name: my_private_dataset
    local_path: "${GETITUNE_BENCHMARK_EXTERNAL_DATA}/my_private_dataset"
    size_tier: medium
```

- `local_path` supports `${VAR}`/`~` expansion, so the same catalog entry resolves to a
  different location per machine/CI runner — set `GETITUNE_BENCHMARK_EXTERNAL_DATA` (or
  whatever variable name you choose) to wherever that machine keeps such datasets.
- The directory must already contain a dataset in a format the engine can load (native
  Datumaro `metadata.json`+`data.parquet`, or a recognizable COCO/YOLO/VOC layout) —
  nothing converts it.
- No `.ready` sentinel is written and no existence check is cached: the directory is
  assumed to be externally managed.
- Mutually exclusive with `script` — a catalog entry must declare exactly one of the two.
- If the path doesn't resolve (unset environment variable) or doesn't exist, provisioning
  logs a clear error and skips just that dataset's experiments (see "Provisioning
  resilience" below) — it does not abort the whole run.

#### Option B — `raw_dir`: let a script skip its own network download

Use this when you still want the script's transform/export logic to run (kept shared,
version-controlled, and testable) but need to skip a credentialed or slow network fetch.

```yaml
datasets:
  - name: brain_tumor
    script: "scripts/benchmark_datasets/prepare_brain_tumor.py"
    raw_dir: "${GETITUNE_BENCHMARK_EXTERNAL_DATA}/brain_tumor_raw"
    size_tier: small
```

- `raw_dir` is only valid together with `script`, and is forwarded to it as `--raw-dir
<resolved-path>`.
- It's a **best-effort accelerant, not a requirement**: if the variable is unset or the
  path doesn't exist, provisioning logs a warning and falls back to the script's normal
  (e.g. network/credentialed) download path instead of failing.
- In the script, use `getitune.benchmark.dataset_helpers.resolve_raw_source(args,
download_fn)` — it returns `args.raw_dir` directly (extracting it first if it's a
  single archive file) when set, or calls `download_fn` otherwise. See
  `scripts/benchmark_datasets/prepare_brain_tumor.py` for a full example.

#### Kaggle-sourced datasets

For datasets hosted on Kaggle (like `brain_tumor`), use
`getitune.benchmark.dataset_helpers.download_kaggle_dataset()` instead of calling the
`kaggle` CLI directly — it gives a clear, actionable error (instead of a subprocess
traceback) when the CLI isn't installed or credentials aren't configured, and points at
`--raw-dir` as an alternative.

Setup:

- Install the downloader: `just venv-benchmark` or `uv sync --extra benchmark`
  (from `library/`; not installed by default).
- Configure credentials, either:
  - Environment variable: `KAGGLE_API_TOKEN`, or
  - A credentials file: `~/.kaggle/access_token` — see https://www.kaggle.com/docs/api.
- Alternatively, skip credentials entirely and use `--raw-dir` / `local_path` with a
  manually-downloaded copy.

**CI:** the `benchmark-dataset-scripts` job in `.github/workflows/lib-lint-and-test.yaml`
reads `KAGGLE_API_TOKEN` from a repository secret of the same name. Set it with
(requires a Kaggle account and repo admin access):

```bash
gh secret set KAGGLE_API_TOKEN
```

GitHub never exposes secrets to `pull_request` workflows triggered from forks, so the
real-download test is additionally skipped (not failed) whenever credentials aren't
present — see `tests/unit/scripts/test_prepare_brain_tumor.py`. A future
scheduled benchmark workflow (see "CI benchmark schedule" above) should reuse the same
token secret.

#### Provisioning resilience

A single dataset failing to provision (missing credentials, transient network error, a
`local_path`/`raw_dir` that doesn't resolve) is logged and that dataset is skipped —
`python -m getitune.benchmark run`/`provision` continues with everything else rather than
aborting the whole invocation.

### Add a new model

1. Ensure the model recipe exists under `src/getitune/recipe/<task>/<name>.yaml` — the recipe path is
   always derived from the model's `name` and its task, there's no separate path to configure.
2. Add a model entry under `experiments.<task>.models` in `benchmark_manifest.yaml`.
3. Run a focused benchmark slice for the new model.

Manifest model example (`benchmark_manifest.yaml`):

```yaml
experiments:
  detection:
    models:
      - name: my_detector
        priority: extended
```

`priority` is one of `core`, `extended`, or `deferred`:

- `core` / `extended` models are included in unfiltered/automated runs (see "Rotation and timeline details" above).
- `deferred` models are declared for completeness but are never picked up by an unfiltered/automated run — only
  when explicitly requested via `--priority deferred` or `--model my_detector`.

Focused run command (from `library/`):

```bash
python -m getitune.benchmark run --task detection --model my_detector --dataset wgisd --num-seeds 1 --no-tracking
```

Notes:

- `datasets` in the manifest must reference names declared in `benchmark_catalog.yaml`.
- A `default` scenario is always present implicitly; optional `scenarios` can further restrict `datasets` and `models`.

## Optional: centralized MLflow server

You can run the included MLflow+PostgreSQL stack:

```bash
docker compose -f src/getitune/benchmark/docker-compose.yaml up -d
```

Then point benchmark commands at it:

```bash
python -m getitune.benchmark run --mlflow-uri http://localhost:5000 --task detection --model yolox_s --dataset wgisd --num-seeds 1
```

If you prefer local file-based tracking (the CLI default `./mlruns`), set:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
```

without that variable, newer MLflow versions can reject file-store tracking backends.
