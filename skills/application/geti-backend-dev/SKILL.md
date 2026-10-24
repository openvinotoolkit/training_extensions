---
name: geti-backend-dev
description: Develop and validate changes in `application/backend/` for the FastAPI `geti` service. Use when changing backend source or tests as well as packaging and configuration. This includes API routers and schemas; services and repositories; database code; UI-facing backend contracts. Covers environment setup; targeted tests; OpenAPI generation; local server workflows.
---

# Geti Backend Development

> For the full architecture reference (layered design, app/ layout, adding an
> endpoint end-to-end, library integration and job execution) read
> `application/backend/AGENTS.md`.

## Quick Start

- Work from `application/backend/`.
- Create or refresh the environment with `just venv --accelerator cpu` for normal local work.
- Switch to `cuda` or `xpu` only when the task depends on accelerator-specific behavior.
- Start with `just lint`, then run the narrowest relevant test target.

## Workflow

1. Keep the change inside the existing backend boundaries unless the task explicitly crosses into `libraries/getitune/` or `application/ui/`.
2. Keep routers thin and move business logic into services or repositories that match the existing package structure.
3. Generate a fresh OpenAPI spec when router or schema changes affect the API contract.
4. Hand off to the $geti-openapi-sync skill after backend contract changes so the UI types stay aligned.

## Architecture Reminders

- The request flow is layered: `app.api` → `app.services` → `app.repositories`
  → `app.db`. Respect the import-linter contracts in `pyproject.toml`.
- Keep `app.api.routers` above `app.api.schemas`, and `app.api.schemas` above
  `app.api.dependencies`.
- Pydantic API schemas (`app/api/schemas/`), domain models (`app/models/`), and
  SQLAlchemy ORM models (`app/db/schema.py`) are separate layers — never return
  raw ORM models from a router.
- Schema changes require an Alembic migration under `app/alembic/versions/`.
- Long-running work (training, quantization, dataset import/export) runs
  out-of-process via `app/core/jobs/` and lazily-imported builders in
  `app/execution/`; training calls into the `getitune` library.
- Treat `run-server --clean` and `_clean_data` as destructive helpers.

## Verification

- Use `just lint` for Ruff, import-linter, and pyrefly checks.
- Use `just test-unit -- tests/unit/...` or `just test-unit -- -k <expr>` for routine backend changes.
- Use `just test-integration -- <pytest args>` when the change crosses service, persistence, or API boundaries.
- Use `just test-bdd -- <behave args>` when behavior is covered by BDD specs.
- Use `just gen-api-spec --output-path openapi-spec.json` after intentional API contract changes.

## Coordination Notes

- `application/backend` depends on the local editable `../../libraries/getitune`. Validate `libraries/getitune/` too when shared model or training behavior changes.
- Prefer project `just` targets over custom shell commands so local work matches CI.
