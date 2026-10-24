# Skill Evaluation Scenarios — Library

Use these prompts to test whether an agent correctly invokes and follows each
library skill. Each scenario checks one realistic failure mode. Run the agent
from the repo root with no extra hints beyond the prompt.

Expected rubric per scenario:

- **Activates the right skill** — the loaded `SKILL.md` matches the topic.
- **Uses real paths and commands** — references `libraries/getitune/src/getitune/...`,
  `getitune ...`, `just ...` as documented.
- **Follows the workflow checklist** — does not skip required checks or verify
  steps.
- **Produces a checkable artifact** — a command run, a file written, or a result.

## `geti-library-dev`

### Scenario 1: Set up and run targeted tests

> "I changed a utility under `libraries/getitune/src/getitune`. Set up the environment for
> CPU and run only the relevant unit tests."

Expected behavior:

- Creates the env with `just venv --device cpu` from `libraries/getitune/`.
- Runs `just test-unit -- <path or -k filter>` scoped to the change, not the full
  suite.
- Runs `just lint` before declaring success.

### Scenario 2: Add a unit test next to the code

> "Add a unit test for a new helper in the library and run just that test file."

Expected behavior:

- Places the test under `libraries/getitune/tests/unit/` mirroring the source tree.
- Reuses existing `conftest.py` fixtures instead of duplicating them.
- Runs the single new test file with `just test-unit -- tests/unit/...`.

## `getitune-discovering-models`

### Scenario 3: List models for a task

> "Which detection models can I train with getitune?"

Expected behavior:

- Uses `list_models(...)` or `getitune find` filtered by task.
- Explains name/task filtering rather than guessing a model list.

### Scenario 4: Resolve an ambiguous model name

> "`getitune train` fails saying the model name matches multiple tasks. How do I
> fix it?"

Expected behavior:

- Explains that the same recipe name exists under multiple tasks.
- Shows disambiguation by passing the task explicitly.

## `getitune-preparing-datasets`

### Scenario 5: Point a COCO dataset at the engine

> "I have a COCO detection dataset. How do I pass it to `create_engine`?"

Expected behavior:

- Describes the expected COCO layout and Datumaro auto-detection.
- Sets `data=` (API) or `--data_root` (CLI) to the dataset root.

### Scenario 6: Diagnose format auto-detection failure

> "Auto-detection can't identify my dataset format. What's wrong?"

Expected behavior:

- Checks the on-disk layout against the supported formats.
- Suggests a corrected layout or an explicit format, not a random guess.

## `getitune-training-a-model`

### Scenario 7: Train from the Python API

> "Write a short script that trains a classification model on my dataset for a
> quick smoke run."

Expected behavior:

- Uses `create_engine(...)` then `engine.train()`.
- Selects a device explicitly and keeps the run short for a smoke test.

### Scenario 8: Warm-start from a checkpoint

> "Resume or fine-tune from an existing checkpoint. Show the API and CLI."

Expected behavior:

- Passes the checkpoint to `engine.train()` and the equivalent `getitune train`
  flag.
- Does not invent flags that are not documented in the skill.

## `getitune-exporting-a-model`

### Scenario 9: Export to OpenVINO FP16

> "Export my trained model to OpenVINO IR at FP16."

Expected behavior:

- Uses `engine.export(...)` with `ExportFormat` and
  `export_precision=Precision.FP16`.
- Reports where the exported artifacts are written.

### Scenario 10: Choose between IR and ONNX

> "Should I export to OpenVINO IR or ONNX for my deployment?"

Expected behavior:

- Explains the trade-offs and the `ExportFormat`/`Precision` options.
- Does not claim ONNX support for steps that require OpenVINO IR.

## `getitune-optimizing-a-model`

### Scenario 11: Quantize an exported model

> "Quantize my exported model to INT8."

Expected behavior:

- Uses `OVEngine.optimize()` / `engine.optimize()` on an OpenVINO IR (`.xml`)
  model.
- Notes the calibration-set requirement and re-validates after quantization.

### Scenario 12: Reject an unsupported input

> "Can I quantize my ONNX export directly?"

Expected behavior:

- States that optimization operates on OpenVINO IR, not ONNX.
- Points to exporting IR first.

## `getitune-running-inference`

### Scenario 13: Predict with a PyTorch checkpoint

> "Run inference on a folder of images with my trained checkpoint."

Expected behavior:

- Uses `engine.predict()` / `getitune predict` with the checkpoint.
- Produces a checkable output (predictions written or summarized).

### Scenario 14: Run inference with an exported model

> "Run the same inference with my exported OpenVINO model instead."

Expected behavior:

- Loads the `.xml` IR (or `.onnx`) through the OpenVINO/ONNX backend.
- Does not mix up the PyTorch and OpenVINO load paths.

## Running the evaluations

1. Reset the agent context between scenarios.
2. Give only the prompt; do not hint at which skill to use.
3. Score against the expected-behavior rubric above.
4. If an agent fails a scenario, update the corresponding `SKILL.md` and rerun.
