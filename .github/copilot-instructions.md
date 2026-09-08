## Repository layout

Monorepo with three components — different languages, toolchains, and conventions.

| Path                   | What it is                                                 | Primary stack                                                    |
| ---------------------- |------------------------------------------------------------| ---------------------------------------------------------------- |
| `library/`             | `getitune` — low-code transfer-learning CV library (PyPI). | Python 3.11+, PyTorch 2.10, OpenVINO, Lightning, Datumaro        |
| `application/backend/` | Geti™ app server (`geti` package).                         | Python 3.14, FastAPI, SQLAlchemy 2 (async), Pydantic v2, Alembic |
| `application/ui/`      | Geti™ web/desktop UI.                                      | Node 24.2+, React, TypeScript, rsbuild, Tauri                    |

The application can be built and deployed in three ways:

1. As a single Docker container.
2. As a desktop app for Windows (MSIX).
3. With server and UI as standalone processes, for development only.

The library is consumed by the backend (`getitune[cpu|xpu|cuda]` extras).

## General rules

- Assume the virtual environment is already activated — do not prepend `uv run`
  or activate a venv in every command.
- Assume the working directory is already the component root you are working on
  (e.g. `application/backend/`). Do not `cd` into it at the start of every
  command.
- When the user is on Windows with WSL2, generate plain Linux commands — do not
  wrap them in `wsl -d … -- bash -c "…"` or use Windows paths like
  `/mnt/wsl.localhost/`. The terminal is already inside WSL.
- Do not mix code or conventions between the three sub-projects.
- Prefer **absolute imports** within each Python package. Relative imports are
  acceptable when they help avoid circular dependencies.
- Prefer editing existing files over creating new ones; match surrounding style.
- Code must pass pre-commit checks (see `.pre-commit-config.yaml`). Run them
  locally with `prek`.
- Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `ci:`.
- **New source files require a copyright + SPDX header** (current year, `#` for Python/shell, `//` for TS/JS/Rust):

  ```
  # Copyright (C) 2026 Intel Corporation
  # SPDX-License-Identifier: Apache-2.0
  ```

- **Use `just`** for developer workflows (`application/ui/` uses `npm` scripts instead).
  Never invent ad-hoc `uv` / `docker` commands when a recipe exists.

## Python conventions (`library/` and `application/backend/`)

- Type-hint every public function, method, and module-level variable.
- Modern typing: `list[int]`, `X | None` — not `List`, `Optional`.
- Prefer `pathlib.Path` over `os.path`.
- Write code that is portable across the main platforms (Linux, Windows, macOS).
- No bare `print`.
- Google-style docstrings (`Args`, `Returns`, `Raises`).
- Logging: `library/` uses stdlib `logging`; `application/backend/` uses
  `loguru`. Do not mix them.
- Tests use `pytest`; new features require unit tests.
- **Test placement**: unit tests live next to the code they test in
  `tests/unit/` (mirroring the source tree); integration tests go in
  `tests/integration/`. Do not duplicate fixtures across directories — reuse
  existing `conftest.py` fixtures at the appropriate scope.
- **Running tests**: after writing or modifying tests, run the specific new test
  file (e.g. `pytest tests/unit/path/to/test_foo.py`). Running the full suite
  is optional and not required for validation.

## Things to avoid

- Do not change Python version pins (they ensure CI reproducibility).
- Do not add new third-party dependencies when stdlib or already-declared
  dependencies can solve the problem.
