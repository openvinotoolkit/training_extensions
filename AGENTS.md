# Geti Agent Guide

## Role of This File

- `AGENTS.md` is the canonical repo-wide instruction file for agentic tools.
- `CLAUDE.md` imports this file for Claude Code compatibility.
- Per-component guides live in `library/AGENTS.md`, `application/backend/AGENTS.md`,
  and `application/ui/AGENTS.md`; read the matching one before working in that area.
- Canonical task skills live in `skills/`, grouped into `skills/library/` and
  `skills/application/` buckets.
- `.agents/skills/` and `.claude/skills/` are committed symlink adapters pointing
  back at `skills/<bucket>/<name>`; do not edit them directly. Run
  `python3 .github/scripts/skills/agent_skills.py sync` after adding or renaming a
  skill (each skill directory must contain a `SKILL.md`). Adapters and layout are
  CI-enforced via `.github/workflows/skills.yaml`; verify locally with
  `python3 .github/scripts/skills/agent_skills.py validate`.
- Keep always-on repository rules here; keep task-specific workflows in skills.

## Repository Map

- `library/`: `getitune` Python package (the Geti training library; source in `src/getitune/`), recipes, and tests. See `library/AGENTS.md`.
- `application/backend/`: FastAPI backend named `geti`; consumes `../../library` as an editable `uv` source. See `application/backend/AGENTS.md`.
- `application/ui/`: React 19 + TypeScript + RSBuild frontend. See `application/ui/AGENTS.md`.
- `integrations/geti-mcp/`: standalone `geti-mcp` Python package, a Model Context Protocol (stdio) server over the Geti REST API. Depends on Geti only over HTTP — not on the backend package or `getitune`. See `integrations/geti-mcp/README.md`.
- `application/README.md`: overview and entry point for the application; links to the installation guide.
- `application/docs/`: Markdown docs for the application (installation, upgrade, API, pipeline, jobs, dataset import/export, models, quantization).
- `library/README.md`: overview and quick-start for the `getitune` library.
- `library/docs/design/`: design notes for the library; user-facing library docs live on the external documentation website.
- `integrations/geti-mcp/README.md`: installation, configuration, tool reference, and API compatibility for the MCP server.
- `README.md`: root-level project overview.
- `.github/workflows/`: CI source of truth for path-based checks and required jobs.

## Data & State

- The backend stores persistent data under `application/backend/data/` by default (i.e., `data/` when running from `application/backend/`); override via the `DATA_DIR` setting. The directory is git-ignored and may not exist until the server runs.
- This includes:
  - the SQLite database (`geti.db`)
  - datasets with media artifacts (images, videos) + trained models with logs under `projects/`
  - job outputs (e.g. dataset exports) under `output/` and in-progress imports under `staged_datasets/`
  - SSL certificates under `certs/`
  - pretrained weights downloaded as per manifest urls under `pretrained_weights/`

## Documentation

Keep documentation close to the code and update it in the same PR as the behavior
it describes. Use the `geti-docs-update` skill for documentation changes.

Where each kind of documentation lives:

- `README.md` (root): project overview and the shortest possible quick-start. It
  lists the available install options and documents only the most basic case,
  then links to `application/docs/install.md` for anything more advanced. Do not
  duplicate detailed steps here.
- `application/README.md`: application overview and entry point. It summarises
  the install options and links to the installation guide; it does not repeat
  the detailed steps.
- `application/docs/install.md`: the **single source of truth for installing and
  running the application**. It covers every scenario (Windows MSIX app, Docker
  with pre-built or self-built images, install script, and run-from-source for
  development), plus prerequisites, accelerator support, TLS/TURN, and
  air-gapped setup. Add or change installation steps here first.
- `application/docs/upgrade.md`: upgrading an existing installation (Docker or
  Windows MSIX), data migration, and rollback.
- `library/README.md`: overview and quick-start for the `getitune` training
  library. Detailed library guides live on the external documentation website
  (`https://docs.geti.intel.com/docs/user-guide/library/`), not in this repo.
- `library/docs/design/`: design notes for the library.
- `application/docs/`: Markdown docs for the application (API, pipeline, jobs,
  dataset import/export, models, quantization).

Installation docs follow an **install-guide-first** model:
`application/docs/install.md` is authoritative and detailed, while the root
`README.md`, `application/README.md`, and the public docs website
(`https://docs.geti.intel.com`) show only the basic case and link to it for
advanced scenarios. When you change how Geti is installed or run, update
`application/docs/install.md` first, then reconcile the two READMEs so they do
not drift; the docs website lives outside this repo and may also need updating.

## Choose the Right Workflow

Contributor/development skills (changing the codebase):

- Use the `library` workflow for changes under `library/src`, `library/tests`, or model, training, export, and CLI logic.
- Use the `backend` workflow for changes under `application/backend/app`, backend tests, backend packaging, or backend API schemas.
- Use the `ui` workflow for changes under `application/ui/src`, frontend tests, build config, or generated API client types.
- Use the OpenAPI sync workflow whenever backend API contracts change and the UI consumes those changes.
- Use the documentation update workflow to keep `README.md`, `application/README.md`, `application/docs/`, or `library/README.md` in sync with code changes.

User-facing skills (using Geti, not changing it):

- Use `getitune-discovering-models` to list available models, recipes, and tasks before training.
- Use `getitune-preparing-datasets` to lay out and point a dataset at the library (COCO/YOLO/VOC/native).
- Use `getitune-training-a-model` to train or evaluate a model with `create_engine` / the `getitune` CLI.
- Use `getitune-exporting-a-model` to export a trained model to OpenVINO IR or ONNX.
- Use `getitune-optimizing-a-model` to quantize an exported model to INT8 with NNCF.
- Use `getitune-running-inference` to run predictions/evaluation with PyTorch, OpenVINO, or ONNX models.
- Use `geti-using-the-pipeline` to drive the Geti application end to end (project → dataset → train → deploy) via its REST API.

## Commands: Library

- Work from `library/`.
- Create or refresh the environment with `just venv --device cpu`, `just venv --device cuda`, or `just venv --device xpu`.
- Run lint and type checks with `just lint`.
- Run unit tests with `just test-unit -- <pytest args>`.
- Run backend-focused unit tests with `just test-unit-lightning -- <pytest args>` (Lightning models),
  `just test-unit-ultralytics -- <pytest args>` (Ultralytics/YOLO models), or
  `just test-unit-openvino -- <pytest args>` (OpenVINO).
- Run integration tests with `just test-integration -- <pytest args>`.

## Commands: Backend

- Work from `application/backend/`.
- Create or refresh the environment with `just venv --accelerator cpu`, `just venv --accelerator cuda`, or `just venv --accelerator xpu`.
- Run lint and type checks with `just lint`.
- Run unit tests with `just test-unit -- <pytest args>`.
- Run integration tests with `just test-integration -- <pytest args>`.
- Run BDD tests with `just test-bdd -- <behave args>`.
- Generate an OpenAPI spec with `just gen-api-spec --output-path openapi-spec.json`.
- Start the local server with `just run-server`.
- Seed demo projects/sources/sinks on startup with `just run-server --setup-demo` (add `--force-import` to bypass schema-version checks).
- Treat `_clean_data` and `run-server --clean` as destructive operations; use them only when the task explicitly calls for data reset.

## Commands: UI

- Work from `application/ui/`.
- Use Node `>=24.2.0` and npm `>=11.14.0`.
- Install dependencies with `npm ci`.
- Build with `npm run build`.
- Run formatting checks with `npm run format:check`.
- Run lint with `npm run lint`.
- Check cyclic imports with `npm run cyclic-deps-check`.
- Run the TypeScript checker with `npm run type-check`.
- Run unit tests with `npm run test:unit` or `npm run test:unit:coverage`.
- Use `npm run test:component` and `npm run test:e2e` only when the change actually affects those layers.
- Build API typings from an existing spec with `npm run build:api`.
- Pull a spec from a running backend with `npm run update-spec` when `https://localhost:7860` is available (the backend serves self-signed TLS).

## Commands: Geti MCP

- Work from `integrations/geti-mcp/`.
- Create or refresh the environment with `just venv`.
- Run lint and type checks with `just lint`.
- Run unit tests with `just test-unit -- <pytest args>` and integration tests with `just test-integration -- <pytest args>`; neither needs a running Geti.
- Refresh the OpenAPI contract snapshot with `just update-contract` after a backend API change.

## Cross-Area Rules

- Do not assume commands from one area apply to another; `library`, `application/backend`, and `application/ui` use different runtimes and toolchains.
- Backend changes can require validating `library/` because `application/backend` depends on the local editable package.
- Backend REST contract changes can break `integrations/geti-mcp`; refresh its snapshot with `just update-contract` and run its integration tests.
- Do not hand-edit generated UI OpenAPI typings when regeneration is possible.
- When backend request or response schemas change, regenerate the UI OpenAPI spec and TypeScript definitions in the same change set.
- Use `.github/workflows/lib-lint-and-test.yaml`, `.github/workflows/backend-lint-and-test.yaml`, and `.github/workflows/ui-lint-and-test.yaml` as the source of truth for CI expectations if local commands are ambiguous.

## Change Discipline

- Prefer minimal, area-scoped edits.
- Follow existing Ruff, import-linter, ESLint, Prettier, and TypeScript configuration instead of introducing parallel style rules.
- Update tests or docs when behavior changes.
- When changing how Geti is installed or run, edit `application/docs/install.md` (the source of truth) and reconcile `README.md` and `application/README.md` so they do not drift.
- Run the narrowest relevant verification before finishing and state exactly what was not run.
