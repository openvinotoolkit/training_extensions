---
name: getitune-exporting-a-model
description: Export a trained getitune model (the Geti training library) to a deployable format. Use when a user wants to run `engine.export(...)` or `getitune export`, choose between OpenVINO IR and ONNX, set FP32 vs FP16 precision with `ExportFormat` / `Precision`, or understand where exported artifacts are written and how they load back for inference. Covers the export/load contract between training and OpenVINO/ONNX inference.
---

# Exporting a model with getitune

After training, export a model to a deployable format with `engine.export(...)`
(Python API) or `getitune export` (CLI). getitune exports to **OpenVINO IR**
(default) or **ONNX**, each at **FP32** (default) or **FP16** precision. Exported
artifacts load back for inference via the OpenVINO/ONNX path (see the
`getitune-running-inference` skill).

Run everything from `libraries/getitune/`.

## Python API workflow

```python
from getitune.engine import create_engine
from getitune.types import ExportFormat, Precision

engine = create_engine(
    model="efficientnet_b0",
    data="/path/to/dataset",
    work_dir="./my_workspace",
)
engine.train(max_epochs=50)

# FP32 OpenVINO IR (default) -> returns the .xml path
ov_ir_path = engine.export()

# FP32 ONNX
onnx_path = engine.export(export_format=ExportFormat.ONNX)

# FP16 ONNX (same pattern works for OpenVINO IR)
onnx_fp16 = engine.export(export_format=ExportFormat.ONNX, export_precision=Precision.FP16)
```

1. **Train or load a model** into the engine first (export operates on the
   engine's current model).
   - Done when: `engine.test()` produces sensible metrics before you export.
2. **Choose format and precision.** Default is FP32 OpenVINO IR. Use
   `export_format=ExportFormat.ONNX` for ONNX; `export_precision=Precision.FP16`
   to halve size for supported hardware. Both enums live in `getitune.types`.
   - Done when: `engine.export(...)` returns a path to the written artifact.
3. **Confirm the artifact exists** under `work_dir` (`.xml` + `.bin` for
   OpenVINO IR, `.onnx` for ONNX).
   - Done when: the returned path exists on disk.
4. **Validate parity** by loading the exported model back and running
   `engine.test()` — accuracy should closely match the trained model (small
   FP16 drift is expected). See `getitune-running-inference`.
   - Done when: exported-model metrics are within tolerance of the trained model.

## CLI workflow

```bash
# from libraries/getitune/
getitune export --data_root /path/to/dataset --model efficientnet_b0
# use --help -v for export-format / precision flags
```

## Export/load contract

- Each model implements `forward_for_tracing(...)` under
  `libraries/getitune/src/getitune/backend/lightning/models/<task>/`; that is what defines
  the exported graph. If you change model I/O, keep this method in sync or export
  parity breaks.
- Exported OpenVINO IR / ONNX models are loaded for inference through the
  OpenVINO backend (`OVEngine`) using
  [ModelAPI](https://github.com/open-edge-platform/model_api).
- Each task also ships an `openvino_model.yaml` recipe for loading a
  pre-exported IR model directly.

## Verify

```bash
# from libraries/getitune/
just lint
just test-unit -- -k export        # when you touched export/tracing code
```

## Related skills

- `getitune-training-a-model` — produce the checkpoint to export.
- `getitune-running-inference` — load and validate the exported model.
- `getitune-optimizing-a-model` — quantize an exported OpenVINO model to INT8.
