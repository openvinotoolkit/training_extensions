---
name: geti-library-dev
description: Develop and validate changes in `libraries/getitune/` for the `getitune` Python package. Use when changing library source or tests as well as packaging, recipes and model manifests. This includes Python APIs and CLI behavior across training and export. Covers environment setup; accelerator extras; multi-backend architecture; model and recipe additions; focused checks.
---

# Geti Library Development

> For the full architecture reference (package layout, multi-backend design, how
> to add models, recipes, and manifests) read `libraries/getitune/AGENTS.md`.

## Quick Start

- Work from `libraries/getitune/`.
- Create or refresh the environment with `just venv --device cpu` for routine work.
- Switch to `just venv --device cuda` or `just venv --device xpu` only when the task needs accelerator-specific behavior.
- Run `just lint` before wider test runs.

## Architecture Essentials

- Source lives under `src/getitune/`. Public API entry points must stay stable
  because `application/backend/` consumes them.
- **Multi-backend design**: compute backends live under
  `src/getitune/backend/` — `lightning/` (PyTorch Lightning training),
  `openvino/` (inference-only), and optional `ultralytics/`. `src/getitune/models/`
  re-exports model classes from each backend into one namespace.
- **Device abstraction**: `DeviceType` (`src/getitune/types/device.py`) and
  `DeviceConfig` (`src/getitune/config/device.py`) abstract accelerator choice.
  Guard device-specific (CUDA vs XPU) code with capability checks in
  `src/getitune/utils/device.py` — never at import time.
- **Recipes**: YAML configs under `src/getitune/recipe/<task>/<model>.yaml` bind
  a task + model `class_path` + training config. Recipes are self-discovering
  via `src/getitune/utils/recipes.py` (`list_models`) — no registry to update.
- **Adding a model**: implement the model class under
  `src/getitune/backend/lightning/models/<task>/`, inheriting the task base
  class (ultimately `LightningModel`); export it from the task `__init__.py`;
  add a recipe YAML. See `libraries/getitune/AGENTS.md` for the full walkthrough.

## Workflow

1. Confirm the change belongs in `libraries/getitune/`. If the task is mainly FastAPI or React work, switch to the matching backend or UI skill.
2. Inspect the nearest module and tests before editing. Keep changes inside the existing package boundaries under `src/getitune/`.
3. Make the smallest change that resolves the task. Avoid lockfile churn unless dependencies changed intentionally.
4. Run the smallest relevant checks first and widen only if the changed behavior crosses package or task boundaries.

## Verification

- Use `just lint` for formatting, lint, and type issues.
- Use `just test-unit -- tests/unit/...` or `just test-unit -- -k <expr>` for normal Python behavior changes.
- Use the backend-scoped recipes for model code: `just test-unit-lightning -- <pytest args>`,
  `just test-unit-ultralytics -- <pytest args>`, or `just test-unit-openvino -- <pytest args>`.
- Use `just test-integration -- <pytest args>` only when the change affects end-to-end training, export, or integration behavior.

## Coordination Notes

- `application/backend` consumes `../../libraries/getitune` as a local editable dependency. If the change affects shared runtime behavior, validate the backend too.
- Update docs or examples when public library behavior changes.
- Prefer project `just` targets over ad hoc dependency-install commands so the pinned `uv` workflow stays consistent with CI.
