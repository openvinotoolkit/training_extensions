---
name: geti-docs-update
description: Update documentation (READMEs, application docs, or inline docstrings) to reflect changes, fixes, or new features. Use when a PR modifies behavior that should be documented for users or developers.
---

# Geti Documentation Update

## Quick Start

- Identify affected documentation: `README.md`, `application/README.md`,
  `application/docs/*.md`, `libraries/getitune/README.md`, `libraries/getitune/docs/design/*.md`, or
  docstrings.
- Use clear, concrete wording. Avoid vague or marketing-heavy language.

## Where Documentation Lives

- `README.md` (root): project overview and the shortest possible quick-start.
  Lists install options and the most basic case, then links to
  `application/docs/install.md` for advanced instructions. Do not duplicate
  detailed steps here.
- `application/README.md`: application overview and entry point. Summarises the
  install options and links to the installation guide; it does not repeat the
  detailed steps.
- `application/docs/install.md`: **single source of truth for installing and
  running the application** (Windows MSIX app, Docker with pre-built or
  self-built images, install script, and run-from-source for development), plus
  prerequisites, accelerator support, TLS/TURN, and air-gapped setup. Add or
  change installation steps here first.
- `application/docs/upgrade.md`: upgrading an existing installation, data
  migration, and rollback.
- `application/docs/`: Markdown docs for the application (API, pipeline, jobs,
  dataset import/export, models, quantization).
- `libraries/getitune/README.md`: overview and quick-start for the `getitune` library.
- `libraries/getitune/docs/design/`: design notes for the library.

> Detailed user-facing library guides are **not** in this repo — they live on the
> documentation website (`https://docs.geti.intel.com/docs/user-guide/library/`).
> If a change needs them updated, flag it in the PR instead of adding new docs
> under `libraries/getitune/docs/`.

## Installation Docs (install-guide-first)

Installation docs follow an **install-guide-first** model:

- `application/docs/install.md` is authoritative and detailed — it documents
  every install scenario.
- The root `README.md`, `application/README.md`, and the public docs website
  (`https://docs.geti.intel.com`) show only the basic case and link to
  `application/docs/install.md` for advanced scenarios.
- When you change how Geti is installed or run, update
  `application/docs/install.md` first, then reconcile the two READMEs so they do
  not drift. The docs website lives outside this repo and may also need
  updating — flag it in the PR.

## Workflow

1. **Analyze Changes**: Review the code changes or fixes to determine what needs to be documented.
2. **Locate Docs**: Find the relevant documentation files:
   - Installation/runtime steps: `application/docs/install.md` (then reconcile `README.md` and `application/README.md`).
   - Upgrade/migration steps: `application/docs/upgrade.md`.
   - Application features/API/pipeline: `application/docs/`.
   - Library user-facing features: `libraries/getitune/README.md` (deeper guides live on the docs website).
   - Library design notes: `libraries/getitune/docs/design/` or inline docstrings.
3. **Draft Updates**: Apply the documentation changes, matching the existing style and tone.
4. **Verify**:
   - For Markdown: Ensure formatting is consistent and internal links resolve to files that exist.

## Style Guidelines

- Use active voice and concrete examples.
- Match the existing frequency of comments and documentation.
- Ensure all technical terms match the actual implementation.

## Coordination Notes

- When installation or runtime behavior changes, edit
  `application/docs/install.md` first and reconcile `README.md` and
  `application/README.md` in the same PR.
- If a change affects multiple areas (e.g., Backend and UI), ensure documentation reflects the end-to-end impact.
