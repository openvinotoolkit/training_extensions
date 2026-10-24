---
applyTo: "libraries/getitune/**"
---

## Conventions

- Source under `libraries/getitune/src/getitune/`.
- Recipes (YAML) define task + model + training config. When adding a model,
  add or update its recipe under the appropriate task directory.
- Public API entry points must stay stable — consumed by `application/backend/`.

## Code style

- Use `pyrefly` baseline (`libraries/getitune/pyrefly-baseline.json`) — do not regress.
- Use `logging` (`logging.getLogger(__name__)`).
- Avoid hard CUDA dependencies. XPU is first-class — guard device-specific
  code with capability checks, not import-time failures.
- Prefer `lightning.pytorch` over `pytorch_lightning` imports.

## Commands

| Task              | Recipe                              |
| ----------------- | ----------------------------------- |
| Create venv       | `just venv --device cpu\|xpu\|cuda` |
| Refresh lockfile  | `just venv-lock`                    |
| Lint + type-check | `just lint`                         |
| Auto-fix lint     | `just ruff-fix`                     |
| Type-check only   | `just pyrefly`                      |

## Testing

- Tests in `libraries/getitune/tests/` — organized as `unit/`, `integration/`, `regression/`.
- Default device for local tests: `cpu`.
- Do not commit datasets — use existing fixtures under `libraries/getitune/tests/assets/`.

## Do not

- Do not import from `application/`.
