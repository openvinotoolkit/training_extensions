# Library agent skills

Skills for `libraries/getitune/` (`getitune`, the Geti training library): models, recipes,
datasets, training, export, optimization, and inference. Library skills cover
both the Python API and the `getitune` CLI whenever both surfaces exist.

Run commands from `libraries/getitune/` unless noted otherwise (`just venv`, `just lint`,
`just test-unit`, `getitune ...`).

## Skills

| Skill                         | Covers                                                                                                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geti-library-dev`            | Develop and validate changes under `libraries/getitune/src`, `libraries/getitune/tests`, recipes, and model manifests; `uv`/`just` setup, cpu/cuda/xpu extras, targeted lint and tests. |
| `getitune-discovering-models` | `list_models(...)`, `getitune find`, filtering by task/name, resolving the "matches multiple tasks" error.                                                                              |
| `getitune-preparing-datasets` | Supported formats and Datumaro auto-detection, the `data=` / `--data_root` argument, COCO/YOLO/VOC/native layouts, zip archives, YOLO `data.yaml`.                                      |
| `getitune-training-a-model`   | `create_engine(...)`, `engine.train()/test()`, `getitune train/test`, device selection, recipe overrides, warm-starting, debugging runs.                                                |
| `getitune-exporting-a-model`  | `engine.export(...)`, `getitune export`, `ExportFormat`/`Precision`, OpenVINO IR vs ONNX, FP32 vs FP16, artifact locations.                                                             |
| `getitune-optimizing-a-model` | `OVEngine.optimize()` INT8 NNCF quantization, calibration-set requirements, re-validating a quantized model.                                                                            |
| `getitune-running-inference`  | `engine.predict()/test()`, `getitune predict/test`, PyTorch checkpoint vs OpenVINO IR vs ONNX backends.                                                                                 |

New library skills must pass at least three scenarios in
[`EVALUATION.md`](EVALUATION.md).

## Add a library skill

```bash
NAME=getitune-my-workflow
mkdir -p "skills/getitune/$NAME"
$EDITOR "skills/getitune/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Global authoring rules: [`../README.md`](../README.md).
