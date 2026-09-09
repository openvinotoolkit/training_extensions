# Contributing to Geti™

We welcome your input! 👐

We want to make it as simple and straightforward as possible to contribute to this project, whether it is a:

- Bug Report
- Discussion
- Feature Request
- Creating a Pull Request (PR)
- Becoming a maintainer

## Repository layout

This is a monorepo with three independent components, each with its own language and toolchain:

| Path                   | What it is                                                 | Primary stack                                                    |
| ---------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| `library/`             | `getitune` — low-code transfer-learning CV library (PyPI). | Python 3.11+, PyTorch, OpenVINO, Lightning, Datumaro             |
| `application/backend/` | Geti™ app server (`geti` package).                        | Python 3.13, FastAPI, SQLAlchemy 2 (async), Pydantic v2, Alembic |
| `application/ui/`      | Geti™ web/desktop UI.                                     | Node 24.2+, React, TypeScript, rsbuild, Tauri                    |

See [`AGENTS.md`](AGENTS.md) for a more detailed map of the repository, and the
per-component guides (`library/AGENTS.md`, `application/backend/AGENTS.md`,
`application/ui/AGENTS.md`) for conventions specific to each area.

## Bug Report

We use GitHub issues to track the bugs. Report a bug by using our Bug Report Template in [Issues](https://github.com/open-edge-platform/geti/issues/new?template=bug_report.md).

## Discussion

We enabled [GitHub Discussions](https://github.com/open-edge-platform/geti/discussions) to welcome the community to ask questions and/or propose ideas/solutions. This will not only provide a medium for the community to discuss Geti™ but also help us de-clutter [Issues](https://github.com/open-edge-platform/geti/issues/).

## Feature Request

We utilize GitHub issues to track the feature requests as well. If you are certain regarding the feature you are interested and have a solid proposal, you could then create the feature request by using our [Feature Request Template](https://github.com/open-edge-platform/geti/issues/new?template=feature_request.md) in Issues. If it's still in the idea phase, you could discuss it with the community in our [Discussion](https://github.com/open-edge-platform/geti/discussions/categories/ideas).

## Development & PRs

We actively welcome your pull requests:

### Getting Started

#### 1. Fork and Clone the Repository

First, fork the repository by following the GitHub documentation on [forking a repo](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). Then, clone your forked repository to your local machine and create a new branch from `develop`.

#### 2. Set Up Your Development Environment

Each component manages its own environment, but the tooling is consistent across the repo:

- Python components (`library/`, `application/backend/`) use [`uv`](https://docs.astral.sh/uv/) for dependency and virtual environment management, and expose their workflows through [`just`](https://github.com/casey/just) recipes.
- The UI (`application/ui/`) uses `npm` with Node `>=24.2.0`.
- Code quality hooks are managed with [`prek`](https://github.com/j178/prek) (a drop-in `pre-commit` replacement), configured in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

<details>
<summary>Library (<code>getitune</code>) setup</summary>

```bash
cd library
just venv --device cpu   # or --device cuda / --device xpu
```

Run checks with:

```bash
just lint
just test-unit -- <pytest args>
just test-integration -- <pytest args>
```

</details>

<details>
<summary>Backend (<code>geti</code>) setup</summary>

```bash
cd application/backend
just venv --accelerator cpu   # or --accelerator cuda / --accelerator xpu
```

Run checks with:

```bash
just lint
just test-unit -- <pytest args>
just test-integration -- <pytest args>
```

</details>

<details>
<summary>UI setup</summary>

```bash
cd application/ui
npm ci
```

Run checks with:

```bash
npm run format:check
npm run lint
npm run type-check
npm run test:unit
```

</details>

<details>
<summary>Pre-commit hooks (all components)</summary>

`prek` is used instead of `pre-commit` to run the shared hooks (Ruff, Prettier, Hadolint, etc.):

```bash
prek install
prek run --all-files
```

</details>

Never invent ad-hoc `uv`/`docker` commands when a `just` recipe already exists; run `just --list` from a component root to see what's available.

### Making Changes

1. **Write Code:** Follow the conventions of the component you're editing (see `AGENTS.md` and the per-component guides). Keep changes minimal and scoped to what's needed.

2. **Add Tests:** If your code includes new functionality, add corresponding tests (`pytest` for `library`/`application/backend`, Vitest/Playwright for `application/ui`) to maintain coverage and reliability.

3. **Update Documentation:** If you've changed APIs or added new features, update the relevant documentation (`README.md`, `application/docs/`, `library/README.md`, or docstrings) in the same change set.

4. **Pass Tests and Quality Checks:** Ensure the test suite and lint/type checks pass for the component(s) you touched, using the `just`/`npm` commands above.

5. **Check Licensing:** Ensure you own the code or have rights to use it, adhering to appropriate licensing. New source files require a copyright + SPDX header (see [`AGENTS.md`](AGENTS.md)).

### Submitting Pull Requests

Once you've followed the above steps and are satisfied with your changes:

1. Push your changes to your forked repository.
2. Go to the original repository you forked and click "New pull request".
3. Choose your fork and the branch with your changes to open a pull request.
4. Fill in the pull request template with the necessary details about your changes. Make the title and the description are accurate and clear, this will help reviewers to understand your code.

We look forward to your contributions!

## License

You accept that your contributions will be licensed under the [Apache-2.0 License](https://choosealicense.com/licenses/apache-2.0/) if you contribute to this repository. If this is a concern, please notify the maintainers.
