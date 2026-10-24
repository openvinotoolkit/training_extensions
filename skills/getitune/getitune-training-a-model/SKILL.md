---
name: getitune-training-a-model
description: Train a computer-vision model with the getitune library (the Geti training library) using its Python API or CLI. Use when a user wants to train, fine-tune, or evaluate a model with `create_engine(...)` and `engine.train()/engine.test()`, run `getitune train`/`getitune test`, pick or override a recipe under `getitune.recipe.<task>`, choose a device (cpu/gpu/xpu/cuda), warm-start from a checkpoint, or debug a training run. Covers classification, detection, instance/semantic segmentation, and keypoint detection.
---

# Training a model with getitune

`getitune` is a low-code transfer-learning library. Training is driven by an
`Engine` created with `create_engine(...)`, which pairs a **model/recipe** with a
**dataset** and returns a runnable engine. Recipes (YAML under
`libraries/getitune/src/getitune/recipe/<task>/`) bundle model + data pipeline + training
config, so a model name alone gives a strong baseline.

There are two equal entry points that share the same objects and recipes:

- **Python API** — `from getitune.engine import create_engine`, then
  `engine.train()` / `engine.test()`. Preferred for notebooks, scripts, tests,
  and library integration. See `libraries/getitune/README.md` ("Quick Start") and the
  [getitune documentation](https://docs.geti.intel.com/docs/user-guide/library/get-started/quick-start).
- **CLI** — `getitune train --data_root <path> --model <name|recipe.yaml>`.
  Preferred for reproducible experiments and shell workflows. See
  `getitune train --help` and the
  [getitune documentation](https://docs.geti.intel.com/docs/user-guide/library/get-started/intro).

Run everything from `libraries/getitune/`. Install with the extra that matches your
hardware: `uv sync` (cpu), `uv sync --extra xpu`, or `uv sync --extra cuda`.

## Python API workflow

```python
from getitune.engine import create_engine

engine = create_engine(
    model="efficientnet_b0",          # model name, recipe .yaml path, or model class
    data="/path/to/dataset_root",     # dataset root (COCO/YOLO/VOC/native), auto-detected
    work_dir="./my_workspace",        # checkpoints + logs; defaults to ./getitune-workspace
    device="auto",                    # "auto", "cpu", "gpu", "xpu", "cuda", "0", ...
)
engine.train(max_epochs=50)
engine.test()
```

1. **Pick the model/recipe.** Pass a model name (`"efficientnet_b0"`), a recipe
   path (`"src/getitune/recipe/detection/yolox_s.yaml"`), or a model class. If a
   name matches recipes under several tasks, pass `task=` (e.g.
   `task="DETECTION"`) to disambiguate. Use the `getitune-discovering-models`
   skill to list options.
   - Done when: `create_engine(...)` returns without a `ValueError`/`FileNotFoundError`.
2. **Point `data=` at the dataset root.** Format is auto-detected by Datumaro; see
   the `getitune-preparing-datasets` skill.
   - Done when: the engine builds a datamodule without a format/feature error.
3. **Smoke-test the wiring first** with a tiny run (`engine.train(max_epochs=1)`
   or a small subset) before a long run.
   - Done when: one train + one validation pass complete without shape errors.
4. **Train**, overriding hyperparameters as needed
   (`engine.train(max_epochs=50)`).
   - Done when: checkpoints appear under `work_dir`.
5. **Evaluate** with `engine.test()` and confirm the task metric moves, not just
   loss. Record the model + `work_dir` that produced it.

Warm-start from existing weights with
`create_engine(..., checkpoint="/path/to/weights.pt")`.

## CLI workflow

```bash
# from libraries/getitune/
# 1. Simplest: data only — getitune picks a default model for the task
getitune train --data_root /path/to/dataset

# 2. Choose a model or recipe
getitune train --data_root /path/to/dataset --model yolox_s

# 3. Override hyperparameters
getitune train --data_root /path/to/dataset --model yolox_s \
  --max_epochs 200 --checkpoint /path/to/weights.pt

# 4. Run a full, resolved config file
getitune train --data_root /path/to/dataset --config src/getitune/recipe/detection/yolox_s.yaml
```

`getitune test` and `getitune predict` share the same `--model` / `--data_root`
shape. Use `getitune <cmd> --help -v` (and `-vv`) for the full overridable
argument list.

## Choosing a device

- `device="auto"` selects an available accelerator; force with `"cpu"`, `"gpu"`,
  `"xpu"`, `"cuda"`, or an index like `"0"`.
- The device must match the installed extra — `--extra xpu` for Intel GPUs,
  `--extra cuda` for NVIDIA. Guard nothing yourself; the library handles
  capability checks.

## Debugging a run

- Run one epoch on a small dataset first to isolate construction vs. dataloading
  vs. training failures.
- Shape/feature mismatches usually mean the dataset's labels or task disagree
  with the model — recheck `task=` and the dataset format
  (`getitune-preparing-datasets`).
- Dataset auto-detection failures: confirm the folder matches one supported
  layout (COCO/YOLO/VOC/native).

## Verify

```bash
# from libraries/getitune/
just lint
just test-unit -- -k engine        # when you changed engine/training code
```

For API-facing work, add or run a short Python smoke test that calls
`create_engine(...)` + `engine.train(max_epochs=1)` on a tiny fixture rather than
a long real run.

## Related skills

- `getitune-discovering-models` — list models/recipes and disambiguate by task.
- `getitune-preparing-datasets` — the `data=` half of the engine.
- `getitune-exporting-a-model` — export a trained checkpoint to OpenVINO/ONNX.
- `getitune-running-inference` — run predictions with a trained or exported model.
- `geti-library-dev` — when the libraries/getitune/model code itself needs changes.
