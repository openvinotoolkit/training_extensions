# Skill Evaluation Scenarios — Application

Use these prompts to test whether an agent correctly invokes and follows each
application skill. Each scenario checks one realistic failure mode. Run the agent
from the repo root with no extra hints beyond the prompt.

Expected rubric per scenario:

- **Activates the right skill** — the loaded `SKILL.md` matches the topic.
- **Uses real paths and commands** — references `application/backend/app/...`,
  `application/ui/src/...`, `just ...`, `npm run ...` as documented.
- **Follows the workflow checklist** — does not skip required checks or verify
  steps.
- **Produces a checkable artifact** — a command run, a file written, or a result.

## `geti-backend-dev`

### Scenario 1: Add an endpoint and test it

> "Add a small read-only endpoint to the backend and verify it with a targeted
> test."

Expected behavior:

- Works from `application/backend/`, edits under `app/api/` and related
  services/schemas.
- Runs `just test-unit -- <path>` scoped to the change and `just lint`.
- Does not run destructive data-reset commands.

### Scenario 2: Run the local server

> "Start the backend locally so I can hit it from the UI."

Expected behavior:

- Uses `just run-server` from `application/backend/`.
- Does not pass `--clean` or `_clean_data` unless explicitly asked.

## `geti-ui-dev`

### Scenario 3: Set up and test the UI

> "Install the UI dependencies and run the unit tests."

Expected behavior:

- Uses Node `>=24.2.0`, runs `npm ci` from `application/ui/`.
- Runs `npm run test:unit`, and prefers unit tests over Playwright/e2e.

### Scenario 4: Respect generated API types

> "Update the UI to use a new API field."

Expected behavior:

- Does not hand-edit generated API types (`src/api/openapi-spec.d.ts`); regenerates them instead.
- Runs `npm run type-check` and `npm run lint` before finishing.

## `geti-openapi-sync`

### Scenario 5: Regenerate the contract after a schema change

> "I changed a backend response schema. Update the UI so the types match."

Expected behavior:

- Generates the backend spec with `just gen-api-spec`, places it under
  `application/ui/src/api/`.
- Regenerates TypeScript typings with `npm run build:api`.
- Runs the minimal backend and UI checks to confirm the contract matches.

### Scenario 6: Pull a spec from a running backend

> "The backend is running on localhost:7860 — sync the UI spec from it."

Expected behavior:

- Uses `npm run update-spec` when the backend is available.
- Regenerates typings and does not hand-edit the generated `.d.ts`.

## `geti-using-the-pipeline`

### Scenario 7: Create a project and train via REST

> "Using the REST API, create a project, upload an image, annotate it, and start
> training."

Expected behavior:

- Follows the `/api/...` project → media → annotation → job workflow.
- Uses the async job model to track training status rather than assuming it is
  synchronous.

### Scenario 8: Configure and enable inference

> "Set up a source → model → sink pipeline and turn on live inference."

Expected behavior:

- Configures the pipeline via the documented endpoints and enables inference.
- Does not describe backend code changes; stays at the REST-usage layer.

## `geti-docs-update`

### Scenario 9: Document a behavior change

> "I changed how a feature works — update the docs to match."

Expected behavior:

- Updates the correct target (`README.md`, `application/docs/`, `libraries/getitune/README.md`,
  or docstrings).
- Keeps the change scoped and consistent with the code, without inventing
  behavior.

## `geti-annotating-and-managing-labels`

### Scenario 10: Create a project and annotate media

> "Using the REST API, create a detection project with two labels, upload an
> image, and add a bounding-box annotation to it."

Expected behavior:

- Uses `POST /api/projects` with a task type and initial labels, then
  `POST /api/projects/<id>/dataset/media` and
  `POST /api/projects/<id>/dataset/media/<media_id>/annotations`.
- Matches the annotation shape (bounding box) to the detection task type.
- Stays at the REST-usage layer; does not propose backend code changes.

### Scenario 11: Edit labels on an existing project

> "Add a new label and rename an existing one on my project."

Expected behavior:

- Uses `PATCH /api/projects/<id>/labels` with `labels_to_add` / `labels_to_edit`.
- Does not attempt to change the project task type or reparent labels.

## `geti-import-export-datasets`

### Scenario 12: Import an existing dataset as a new project

> "I have a COCO dataset archive — import it into Geti as a new project."

Expected behavior:

- Uploads the archive to staging (`POST /api/staged_datasets`), runs a
  `prepare_dataset_for_import` job, reviews detected task/labels, then submits
  `import_dataset_as_new_project` via `POST /api/jobs`.
- Tracks each operation with the async job model (`GET /api/jobs/<id>` or the
  `/status` stream).
- Does not claim external trained models (BYOM) can be imported.

### Scenario 13: Export a project's dataset with filtering

> "Export my project's dataset in COCO format, only the training subset."

Expected behavior:

- Submits an `export_dataset` job with the format and subset filter, then
  downloads the archive via `GET /api/staged_datasets/<id>/zip`.
- Uses the staging area rather than assuming a synchronous download.

## Running the evaluations

1. Reset the agent context between scenarios.
2. Give only the prompt; do not hint at which skill to use.
3. Score against the expected-behavior rubric above.
4. If an agent fails a scenario, update the corresponding `SKILL.md` and rerun.
