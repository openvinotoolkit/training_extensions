---
name: getitune-optimizing-a-model
description: Optimize an exported getitune model (the Geti training library) with post-training quantization. Use when a user wants to run `OVEngine.optimize()` / `engine.optimize()` to produce an INT8 model via NNCF, understands calibration-set requirements, or needs to re-validate and run inference with a quantized model versus the original FP32/FP16 model. Covers OpenVINO NNCF post-training quantization and the accuracy/size trade-off.
---

# Optimizing (quantizing) a model with getitune

`getitune` applies **post-training quantization (PTQ)** via
[NNCF](https://github.com/openvinotoolkit/nncf) to shrink an exported OpenVINO
model and speed up inference. Quantization runs on an **OpenVINO** model (an
exported `.xml`), producing an INT8 version.

Run everything from `libraries/getitune/`.

## Workflow

```python
from getitune.engine import create_engine

# Load an exported OpenVINO model, then quantize it
ov_engine = create_engine(
    model="/path/to/exported_model.xml",
    data="/path/to/dataset",
)
ov_engine.optimize()                 # INT8 post-training quantization via NNCF
int8_metrics = ov_engine.test()      # validate the quantized model
predictions = ov_engine.predict()    # run inference with the quantized model
```

1. **Start from an exported OpenVINO model** (`.xml`). If you only have a
   checkpoint, export it first with the `getitune-exporting-a-model` skill.
   - Done when: `create_engine(model="....xml", data=...)` builds an `OVEngine`.
2. **Provide a calibration dataset.** Calibration images are taken automatically
   from the training subset; 200-500 images is the recommended calibration size.
   - Done when: `optimize()` runs without a "not enough calibration data" issue.
3. **Run `optimize()`.** This replaces the engine's model in place with the INT8
   version.
   - Done when: the call completes and subsequent `test()`/`predict()` use INT8.
4. **Re-validate accuracy** with `test()` and compare against the FP32/FP16
   baseline; a small accuracy drop is expected in exchange for size/latency.
   - Done when: the INT8 metric is within your acceptable tolerance of baseline.

## Comparing against the original model

After `optimize()` the engine holds the INT8 model. To re-check the original
FP32/FP16 model, either pass the original `.xml` path directly to `.test()` /
`.predict()`, or create the engine again from the original `.xml`.

## Notes

- Quantization is OpenVINO/NNCF-based and applies to exported IR models — it is
  not a training-time step.
- **Only OpenVINO IR (`.xml`) is supported.** An ONNX model must be converted to
  OpenVINO IR first before it can be optimized.
- In the Geti application this is exposed as the `quantize` job
  (`application/backend/app/execution/quantization/`); library `optimize()` is
  the same capability without the job/queue wrapper.

## Verify

```bash
# from libraries/getitune/
just lint
just test-unit -- -k optimize      # when you touched optimization code
```

## Related skills

- `getitune-exporting-a-model` — produce the OpenVINO `.xml` to quantize.
- `getitune-running-inference` — run inference with the quantized model.
