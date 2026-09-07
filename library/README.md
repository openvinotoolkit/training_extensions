<div align="center">

<img src="../assets/geti-tune-header.png" alt="Geti Library">

**A low-code transfer learning framework for training, evaluating, optimizing, and deploying computer vision models**

---

[Key Features](#key-features) •
[Supported Tasks & Models](#supported-tasks--models) •
[Installation](#installation) •
[Usage](#usage) •
[Command-Line Scripts](#command-line-scripts) •
[Docs](https://docs.geti.intel.com/docs/user-guide/library/get-started/intro) •
[License](#license)

[![PyPI](https://img.shields.io/pypi/v/getitune)](https://pypi.org/project/getitune)

<!-- markdownlint-disable MD042 -->

[![python](https://img.shields.io/badge/python-3.11%E2%80%933.14-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.12-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2026.3-purple)]()
[![numpy](https://img.shields.io/badge/numpy-%E2%89%A52.0-blue)]()

<!-- markdownlint-enable  MD042 -->

[![Codecov](https://codecov.io/gh/open-edge-platform/training_extensions/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/open-edge-platform/training_extensions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/open-edge-platform/training_extensions/badge)](https://securityscorecards.dev/viewer/?uri=github.com/open-edge-platform/training_extensions)
[![Pre-Merge Test](https://github.com/open-edge-platform/training_extensions/actions/workflows/pre_merge.yaml/badge.svg)](https://github.com/open-edge-platform/training_extensions/actions/workflows/pre_merge.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/getitune?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/getitune)

---

</div>

# Geti Library - getitune

The Geti™ library (`getitune`) is a low-code transfer learning framework for Computer Vision.
Its API and CLI let you train, evaluate, optimize, and deploy models quickly, even without deep expertise in deep learning.
It supports diverse combinations of model architectures, learning methods, and task types built on [PyTorch](https://pytorch.org) and the [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

Each supported task ships with curated "recipes": YAML files that bundle the model, data pipeline, and training configuration into a single one-stop entry point. Recipes are validated on standard datasets so you get a strong baseline out of the box.

## Key Features

- **Multi-task support**: classification, object detection, instance segmentation, semantic segmentation, and keypoint detection, see the [full model list below](#supported-tasks--models).
- **Tiling** for large images across detection and segmentation tasks.
- **Multiple backends**: train with PyTorch Lightning, Ultralytics YOLO, export and run inference with ONNX and OpenVINO™.
- **Hardware acceleration**: Intel GPU (XPU) and NVIDIA CUDA support.
- [Datumaro](https://github.com/open-edge-platform/datumaro/tree/develop/src/datumaro/experimental) **data frontend** with automatic format detection (COCO, YOLO, VOC, native).
- **Distributed training** across multiple GPUs.
- **Mixed-precision training** to reduce memory and increase batch size.
- **Class-incremental learning** to extend an existing model with new classes.
- **Deployment** to OpenVINO™ IR and ONNX formats, with inference via [OpenVINO™ ModelAPI](https://github.com/open-edge-platform/model_api).

---

## Supported Tasks & Models

All recipes live under `src/getitune/recipe/<task>/`. Pass any of these YAMLs directly to the API as `model=...`. Recipes whose name ends in `_tile` enable the tiling pipeline for large images.

| Task                                       | Recipe directory                                                                                                                               | Example recipes                                                                                                                                                                                                                                                                   |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification (multi-class / multi-label) | [multi_class_cls](src/getitune/recipe/classification/multi_class_cls/), [multi_label_cls](src/getitune/recipe/classification/multi_label_cls/) | `dino_v2`, `vit_tiny`, `efficientnet_b0`, `efficientnet_b3`, `efficientnet_v2`, `mobilenet_v3_large`, `yolo26_{n,s,m,l,x}_cls`, `timm_generic` (any [timm](https://github.com/huggingface/pytorch-image-models) backbone by name, e.g. `resnet50.a1_in1k`, `convnext_tiny.in12k`) |
| Object detection                           | [detection](src/getitune/recipe/detection/)                                                                                                    | `atss_mobilenetv2`, `ssd_mobilenetv2`, `yolox_{tiny,s,l,x}`, `rtdetr_50`, `dfine_x`, `deim_dfine_{l,m,x}`, `deimv2_{s,m,l}`, `edgecrafter_{s,m,l,x}`, `rfdetr_{nano,small,medium,large}`, `yolo11_{n,s,m,l,x}`, `yolo12_{n,s,m,l,x}`, `yolo26_{n,s,m,l,x}`                        |
| Instance segmentation                      | [instance_segmentation](src/getitune/recipe/instance_segmentation/)                                                                            | `maskrcnn_{r50,swint,efficientnetb2b}`, `rtmdet_inst_tiny`, `rfdetr_seg_{nano,small,medium,large,xlarge,2xlarge}`, `yolo11_{n,s,m,l,x}_seg`, `yolo26_{n,s,m,l,x}_seg`                                                                                                             |
| Semantic segmentation                      | [semantic_segmentation](src/getitune/recipe/semantic_segmentation/)                                                                            | `dino_v2`, `litehrnet_{s,18,x}`, `segnext_{t,s,b}`, `yolo26_{n,s,m,l,x}_sem` (with `_tile` variants)                                                                                                                                                                              |
| Keypoint detection                         | [keypoint_detection](src/getitune/recipe/keypoint_detection/)                                                                                  | `rtmpose_tiny`                                                                                                                                                                                                                                                                    |

Each task directory also ships an `openvino_model.yaml` recipe for running and optimizing pre-exported OpenVINO IR models via `OVEngine`.

> [!NOTE]
> Ultralytics YOLO models are distributed under the AGPL-3.0 license, an OSI approved license ideal for open-source research, academic, and personal projects. For commercial use, enhanced support, and tailored licensing terms, please explore flexible Ultralytics licensing options at https://www.ultralytics.com/license.

---

## Installation

```bash
# With uv (recommended)
# CPU-only by default
uv pip install "getitune"

# Or with pip
pip install "getitune"

# For hardware-specific PyTorch wheels, see "Advanced Installation: Specify Hardware Backend" below.
```

> [!IMPORTANT]
> Due to licensing constraints, the PyPI package doesn't include Ultralytics YOLO models.
>
> To use Ultralytics YOLO models, you must [install from source](#advanced-install-from-source).

<details>
<summary><strong> Advanced Installation: Specify Hardware Backend</strong></summary>

`getitune` ships three mutually exclusive extras that select the right PyTorch wheel for your hardware:

| Extra    | PyTorch wheel                                                          | Use when                             | Setup Guide                                                                      |
| -------- | ---------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------- |
| `[cpu]`  | `torch==2.12.1+cpu` (Linux/Windows) or default `torch==2.12.1` (macOS) | No GPU, or running on Apple silicon. | —                                                                                |
| `[xpu]`  | `torch==2.12.1+xpu` + `triton-xpu`                                     | Intel discrete or integrated GPUs.   | [Intel GPU drivers](https://github.com/intel/compute-runtime/releases)           |
| `[cuda]` | `torch==2.12.1+cu130`                                                  | NVIDIA GPUs with CUDA 13.0 drivers.  | [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-13-0-0-download-archive) |

```bash
# Intel GPU (XPU)
uv pip install "getitune[xpu]" --extra-index-url https://download.pytorch.org/whl/xpu

# NVIDIA GPU (CUDA 13.0)
uv pip install "getitune[cuda]" --extra-index-url https://download.pytorch.org/whl/cu128

# CPU-only (no extra index needed)
uv pip install "getitune[cpu]"
```

> [!NOTE]
> For **macOS** users: PyTorch's `+cpu` wheel is only published for Linux and Windows. The `[cpu]` extra resolves this automatically and installs the default `torch==2.12.1` wheel on macOS.

</details>

<details>
<summary><a id="advanced-install-from-source"></a><strong> Advanced Installation: Install from Source with Ultralytics YOLO Support</strong></summary>

```bash
git clone https://github.com/open-edge-platform/geti.git
cd geti/library

# Recommended: use uv to honor the lockfile
uv sync                      # CPU-only
uv sync --extra xpu          # Intel GPU (XPU) — setup: https://github.com/intel/compute-runtime/releases
uv sync --extra cuda         # NVIDIA GPU (CUDA 13.0) — setup: https://developer.nvidia.com/cuda-13-0-0-download-archive

# Or with pip in a virtual environment
python -m venv .venv && source .venv/bin/activate

# CPU-only
pip install -e ".[cpu]"

# Intel GPU (XPU)
pip install -e ".[xpu]" \
  --extra-index-url https://download.pytorch.org/whl/xpu

# NVIDIA GPU (CUDA 13.0)
pip install -e ".[cuda]" \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

> [!NOTE]
> For **Ultralytics YOLO models**, add `--extra ultralytics` for `uv sync` or `[ultralytics]` for `pip install`:
>
> ```bash
> uv sync --extra xpu --extra ultralytics  # Intel GPU + YOLO
>
> # or with pip
> pip install -e ".[xpu,ultralytics]" --extra-index-url https://download.pytorch.org/whl/xpu  #Intel GPU + YOLO
> ```

</details>

<details>
<summary><a id="legacy-cuda-support"></a><strong> Advanced Installation: Install from Source with Legacy CUDA Support (12.6)</strong></summary>

By default, the `[cuda]` extra builds `torch`/`torchvision` against CUDA 13.0, which doesn't support older NVIDIA GPU
architectures (Volta, Maxwell, Pascal). If you have one of these GPUs, patch the project to use CUDA 12.6 instead:

```bash
git clone https://github.com/open-edge-platform/geti.git
cd geti/library

# Replace CUDA 13.0 with CUDA 12.6 in pyproject.toml and refresh the lockfile
just patch-for-legacy-gpu-support

# Then install as usual, e.g. with uv
uv sync --extra cuda         # NVIDIA GPU (CUDA 12.6) — setup: https://developer.nvidia.com/cuda-12-6-0-download-archive

# Or with pip in a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cuda]" \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

> [!TIP]
> Refer to [this thread](https://github.com/pytorch/pytorch/issues/178665) in the PyTorch project for more details
> about what GPUs are supported for each CUDA version.

</details>

---

## Usage

### Discovering Recipes and Models

To explore available models and recipes:

```python
from getitune.utils import list_models

# List all available models names
all_models = list_models()

# List all available recipes / configuration files (full YAML paths)
all_recipes = list_models(return_recipes=True)

# Filter by task
detection_models = list_models(task="DETECTION")

# Filter by pattern
efficient_models = list_models(pattern="*efficient*")
```

Then pass any model name to `create_engine(model="...", data="...")`.

> [!NOTE]
> Model resolution depends on the provided input type:
>
> - If you pass a **recipe YAML path** (with `.yaml` or `.yml` suffix) that doesn't exist on disk, a `FileNotFoundError` is raised.
> - If you pass a **model name** that matches recipes under multiple tasks, a `ValueError` is raised listing the matches. Pass `task=` to disambiguate.
> - Use `list_models(task="...", return_recipes=True)` to get full recipe paths instead of just model names.

---

### Training

Getitune supports an API-based training approach:

```python
from getitune.engine import create_engine

# Initialize and train using a recipe and dataset
engine = create_engine(
    model="src/getitune/recipe/classification/multi_class_cls/efficientnet_b0.yaml", # any supported model name from the model catalog (see below example), recipe.yaml path or instantiated model class directly
    data="tests/assets/classification_cifar10", # path to dataset root (any supported format, e.g., COCO, VOC, YOLO, Datumaro) or DataModule instance
    work_dir="./my_workspace",  # Defaults to "./getitune-workspace"
    device="auto",              # "auto", "cpu", "gpu", "0", "xpu", etc.
)
engine.train(max_epochs=50)
engine.test()
```

> [!TIP]
> Pass `model=` as a recipe YAML path or a model name (e.g., `"efficientnet_b0"`), or a weights path (`.xml`, `.onnx`).
> If a model name matches recipes under multiple tasks, pass `task=` to disambiguate (e.g., `task="DETECTION"`).
> If you want to use an Ultralytics YOLO model, you can pass a YAML file in [Ultralytics format](https://docs.ultralytics.com/datasets/) as `data=`.

---

### Export

Export a trained model to OpenVINO IR or ONNX format:

```python
from getitune.engine import create_engine
from getitune.types import ExportFormat, ExportPrecision

engine = create_engine(
    model="efficientnet_b0",
    data="/path/to/dataset",
    work_dir="./my_workspace",
)
engine.train(max_epochs=50)

# Export to FP32 OpenVINO IR (default)
ov_ir_path = engine.export()

# Export to FP32 ONNX
onnx_path = engine.export(export_format=ExportFormat.ONNX)

# Export to FP16 ONNX. Same for OpenVINO IR.
onnx_path = engine.export(export_format=ExportFormat.ONNX, precision=ExportPrecision.FP16)
```

NMS-based detection and instance-segmentation models are exported without NMS in the graph by default; ModelAPI applies NMS using the embedded metadata. `MaskRCNN`, `MaskRCNNTV`, and `RTMDetInst` always embed NMS because their export paths do not support raw outputs, regardless of `export_nms`. To embed NMS in the exported graph for models that support the option, pass `export_nms=True` explicitly:

```python
onnx_path = engine.export(export_format=ExportFormat.ONNX, export_nms=True)
```

---

### Validation and Inference

Getitune provides inference via PyTorch and OpenVINO backends (utilizing [ModelAPI](https://github.com/open-edge-platform/model_api)):

```python
from getitune.engine import create_engine

# PyTorch inference with model name
engine = create_engine(
    model="efficientnet_b0",
    data="/path/to/dataset",
)
test_metrics = engine.test() # test on test subset
predictions = engine.predict() # predict on test subset
```

```python

# OpenVINO inference (from exported model)
ov_engine = create_engine(
    model="/path/to/exported_model.xml",
    data="/path/to/dataset",
)
ov_engine.test() # test on test subset
ov_engine.predict() # predict on test subset
```

```python

# ONNX inference (from exported model)
ov_engine = create_engine(
    model="/path/to/exported_model.onnx",
    data="/path/to/dataset",
)
ov_engine.test() # test on test subset
ov_engine.predict() # predict on test subset
```

---

### Optimization

Apply post-training quantization to reduce model size and accelerate inference:

```python
from getitune.engine import create_engine

# Load an exported OpenVINO model and optimize it
ov_engine = create_engine(
    model="/path/to/exported_model.xml",
    data="/path/to/dataset",
)
ov_engine.optimize()  # post-training quantization via NNCF
test_metrics = ov_engine.test() # test on test subset with optimized model
predictions = ov_engine.predict() # predict on test subset with optimized model
```

> [!NOTE]
> The recommended calibration set size for optimization is around 300 images. Calibration images are automatically taken from the training subset of your dataset.
>
> After `.optimize()` the model in the Engine is replaced with an INT8 quantized version. To re-validate or run inference with the original FP32/FP16 model, pass the model XML path directly to `.test()` / `.predict()` or create the Engine again from the original `.xml`.

---

### Dataset Support

When you pass a path to `data=`, getitune uses [Datumaro](https://github.com/open-edge-platform/datumaro/tree/develop/src/datumaro/experimental) to auto-detect the dataset format. Supported formats:

| Format                | Detection method                                        |
| --------------------- | ------------------------------------------------------- |
| **COCO**              | `annotations/` directory with COCO JSON files           |
| **YOLO**              | `data.yaml` file (Ultralytics layout)                   |
| **Pascal VOC**        | `JPEGImages/`, `Annotations/`, `ImageSets/` directories |
| **Datumaro (native)** | `metadata.json` + `data.parquet` at root                |

Zip archives are also accepted, Datumaro extracts them on import.

```python
# Works the same regardless of format, just point to the dataset root
engine = create_engine(
    data="/path/to/dataset_root",
    model="src/getitune/recipe/detection/yolox_s.yaml",
)
engine.train()
```

> [!NOTE]
> If you are working with Ultralytics YOLO models, you can pass a [YOLO Ultralytics](https://docs.ultralytics.com/datasets/) `data.yaml` file directly to the `data=` argument of `create_engine`.

---

### Advanced Usage

<details>
<summary><strong>Direct Backend Engine Usage</strong></summary>

The backends engines can also be instantiated directly:

```python
# -- Lightning Backend --
from getitune.backend.lightning.engine import LightningEngine
from getitune.models import EfficientNet

engine = LightningEngine(
    model=EfficientNet(label_info=10, model_name="efficientnet_b0"),
    data="/path/to/dataset",
    work_dir="./my_workspace",
    device="auto",
)
engine.train(max_epochs=50)
engine.test()
engine.export()


# -- Ultralytics Backend --
from getitune.backend.ultralytics.engine import UltralyticsEngine
from getitune.backend.ultralytics.models import UltralyticsDetectionModel

engine = UltralyticsEngine(
    model=UltralyticsDetectionModel(model_name="yolo26s", label_info=10),  # or datamodule.label_info
    data="/path/to/yolo_dataset/data.yaml",
    work_dir="./yolo_workspace",
    device="auto",
)
engine.train(epochs=50)
engine.test()
engine.export()

> [!NOTE]
> Ultralytics YOLO models and the `UltralyticsEngine` backend require [installing from source](#advanced-installation-install-from-source) with the `[ultralytics]` extra.
> The PyPI package does **not** include Ultralytics support.


# -- OpenVINO Backend (inference) --
from getitune.backend.openvino.engine import OVEngine

engine = OVEngine(
    model="/path/to/exported_model.xml",
    data="/path/to/dataset",
)
engine.test()
engine.optimize()
```

</details>

<details>
<summary><strong>Common Engine Parameters</strong></summary>

Customize engine creation with the following parameters:

```python
engine = create_engine(
    model="efficientnet_b0",
    data="/path/to/data",
    work_dir="./my_workspace",         # Working directory for checkpoints/logs; defaults to "./getitune-workspace"
    device="gpu",                       # "auto", "cpu", "gpu", "0", "1", "xpu", etc.
    checkpoint="/path/to/weights.pt",  # Optional pretrained checkpoint for warm-start training
    task="MULTI_CLASS_CLS",             # Required only if model name matches multiple tasks
)
engine.train(max_epochs=50)
```

</details>

<details>
<summary><strong>Override training hyperparameters</strong></summary>

You can override engine-level training parameters directly:

```python
engine.train(
    max_epochs=50,
    seed=42,
    deterministic=True,
    precision="16-mixed",          # mixed-precision training
    gradient_clip_val=1.0,
    check_val_every_n_epoch=5,
)
```

For model-level hyperparameters like learning rate and optimizer, you can pass them when instantiating a model class directly:

```python
from torch.optim import AdamW
from getitune.models import EfficientNet

model = EfficientNet(
    label_info=datamodule.label_info, # or simply num_classes, e.g., int value
    model_name="efficientnet_b0",
    optimizer=lambda params: AdamW(params, lr=0.001, weight_decay=0.01),
)

engine = create_engine(data=datamodule, model=model)
engine.train(max_epochs=100)
```

Alternatively, you can set these in a custom recipe YAML (copy and modify an existing one):

```yaml
# my_recipe.yaml — custom learning rate and optimizer
task: MULTI_CLASS_CLS
model:
  class_path: getitune.backend.lightning.models.classification.multiclass_models.efficientnet.EfficientNetMulticlassCls
  init_args:
    label_info: 1000
    model_name: efficientnet_b0

    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 0.001
        weight_decay: 0.01

    scheduler:
      class_path: getitune.backend.lightning.schedulers.LinearWarmupSchedulerCallable
      init_args:
        num_warmup_steps: 5
        main_scheduler_callable:
          class_path: lightning.pytorch.cli.ReduceLROnPlateau
          init_args:
            mode: max
            factor: 0.5
            patience: 3
            monitor: val/accuracy

data: src/getitune/recipe/_base_/data/classification.yaml

overrides:
  max_epochs: 100
```

</details>

<details>
<summary><strong>Override augmentations and datamodule</strong></summary>

For augmentations, override the data config. Augmentations run on CPU (`augmentations_cpu`) and GPU (`augmentations_gpu`) separately:

```yaml
# my_data.yaml — stronger augmentations
task: MULTI_CLASS_CLS
input_size: [224, 224]
train_subset:
  subset_name: train
  batch_size: 32
  num_workers: 8
  augmentations_cpu:
    - class_path: torchvision.transforms.v2.RandomResizedCrop
      init_args:
        size: [224, 224]
        scale: [0.08, 1.0]
  augmentations_gpu:
    - class_path: kornia.augmentation.RandomHorizontalFlip
      init_args:
        p: 0.5
    - class_path: kornia.augmentation.ColorJiggle
      init_args:
        brightness: 0.4
        contrast: 0.4
        saturation: 0.4
        hue: 0.1
        p: 0.8
    - class_path: kornia.augmentation.RandomGaussianBlur
      init_args:
        kernel_size: [3, 3]
        sigma: [0.1, 2.0]
        p: 0.3
    - class_path: kornia.augmentation.Normalize
      init_args:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
```

Then reference your custom data config from your recipe with `data: my_data.yaml`.

Augmentations can also be overridden directly via the API by creating a `DataModule` instance and passing it to the engine:

```python
from getitune.data.module import DataModule
from getitune.config.data import SubsetConfig
from getitune.models import EfficientNet
import kornia.augmentation as K
from torchvision.transforms import v2

datamodule = DataModule(
    task="MULTI_CLASS_CLS",
    data_root="tests/assets/classification_cifar10",
    train_subset=SubsetConfig(
        augmentations_cpu=[
            v2.Resize((256, 256)),
        ],
        augmentations_gpu=[
            K.RandomErasing(p=0.5),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ],
        batch_size=32,
        num_workers=8,
    ),
    val_subset=SubsetConfig(
        subset_name="val",
        augmentations_cpu=[
            v2.Resize((256, 256)),
        ],
        batch_size=32,
        num_workers=8,
    ),
    test_subset=SubsetConfig(
        subset_name="test",
        augmentations_cpu=[
            v2.Resize((256, 256)),
        ],
        batch_size=32,
        num_workers=8,
    ),
)

model = EfficientNet(label_info=datamodule.label_info, model_name="efficientnet_b0")
engine = create_engine(data=datamodule, model=model)
engine.train()
```

> [!NOTE]
> GPU augmentations (`augmentations_gpu`) are supported only for the Lightning backend and will be ignored for Ultralytics. For YOLO models, all augmentations should be placed on CPU via `torchvision.transforms.v2`.

Available model classes:

- **Detection:** `ATSS`, `SSD`, `YOLOX`, `RTDETR`, `DFine`, `DEIMDFine`, `DEIMV2`, `RFDETR`, `EdgeCrafter`, `UltralyticsDetectionModel`
- **Instance segmentation:** `MaskRCNN`, `MaskRCNNTV`, `RTMDetInst`, `RFDETRSeg`, `UltralyticsInstSegModel`
- **Semantic segmentation:** `DinoV2Seg`, `LiteHRNet`, `SegNext`, `UltralyticsSemanticSegModel`
- **Classification:** `EfficientNet`, `MobileNetV3`, `VisionTransformer`, `TimmModel`, `TVModel`, `UltralyticsMultiClassClsModel`
- **Keypoint:** `RTMPose`

</details>

---

## Command-Line Scripts

The `scripts/` directory ships ready-to-use command-line entry points that wrap the `getitune` API for common workflows (training, evaluation, export, benchmarking, and prediction). They are plain Python scripts — run them directly with your `getitune` environment active.

### `train.py`

Train, test, and optionally export/quantize a `getitune` model from the CLI. Accepts a model name or recipe, dataset root, device, epochs, batch size, and precision, with flags to disable early stopping and to export (`--export`) to OpenVINO/ONNX and quantize (`--quantize`) to INT8.

```bash
# Train, test, and export to OpenVINO FP16
python scripts/train.py \
    --model src/getitune/recipe/detection/yolox_s.yaml \
    --data-root /path/to/dataset \
    --epochs 50 \
    --export --export-format openvino --export-precision fp16

# Train and quantize the exported model to INT8
python scripts/train.py \
    --model yolox_s \
    --data-root /path/to/dataset \
    --export --quantize
```

### `test.py`

Evaluate a trained model (torch `.ckpt`/`.pt`/`.pth`, or exported OV/ONNX) on a dataset and print the resulting metrics. Supports a task-aware `--metric` override (e.g. `f1-score`, `map`, `dice`, `miou`, `pck`) mapped per task, so invalid combinations like `map` for multiclass classification are rejected.

```bash
# Evaluate an exported OpenVINO model with the default recipe metrics
python scripts/test.py \
    --model /path/to/exported_model.xml \
    --data-root /path/to/dataset

# Evaluate a torch checkpoint with explicit metrics (DETECTION task)
python scripts/test.py \
    --model ckpt.ckpt --recipe yolox_s --task DETECTION \
    --data-root /path/to/dataset \
    --metric f1-score map

# Supported metrics per task:
#   MULTI_CLASS_CLS:        accuracy, f1-score
#   MULTI_LABEL_CLS:        accuracy, f1-score, map
#   DETECTION:              f1-score, map
#   INSTANCE_SEGMENTATION:  f1-score, map
#   SEMANTIC_SEGMENTATION:  dice, miou
#   KEYPOINT_DETECTION:     pck, pck-score
```

### `export.py`

Export a `getitune` model to OpenVINO or ONNX at FP32/FP16, or to INT8 via post-training quantization. Defaults to OpenVINO FP16; INT8 requires a dataset root for the calibration step.

```bash
# Export to OpenVINO FP16 (default)
python scripts/export.py \
    --model yolox_s --checkpoint /path/to/ckpt.ckpt \
    --data-root /path/to/dataset

# Export to ONNX FP32 from a recipe/model and checkpoint
python scripts/export.py \
    --model yolox_s --checkpoint /path/to/ckpt.ckpt \
    --data-root /path/to/dataset \
    --format onnx --precision fp32

# Export to OpenVINO INT8 (quantized)
python scripts/export.py \
    --model yolox_s --checkpoint /path/to/ckpt.ckpt \
    --data-root /path/to/dataset \
    --precision int8
```

### `predict.py`

Run inference with an OV/ONNX/torch model on a dataset or images folder and write the predictions to a COCO-format JSON file (bounding boxes, RLE masks for segmentation, and keypoints). Uses `coco_utils.py` for the conversion.

```bash
# Predict with an exported OpenVINO model on an images folder
python scripts/predict.py \
    --model /path/to/exported_model.xml \
    --input /path/to/images \
    --output predictions.json

# Predict with a torch checkpoint
python scripts/predict.py \
    --model ckpt.ckpt --recipe yolox_s --task DETECTION \
    --input /path/to/dataset \
    --output predictions.json
```

### `benchmark.py`

Prepare a model (export a recipe or quantize to INT8 as needed) and run OpenVINO `benchmark_app` against it, forwarding device, batch, input shape, hint, and iteration/time options for latency/throughput measurement.

```bash
# Benchmark an exported OpenVINO model on CPU
python scripts/benchmark.py \
    --model /path/to/exported_model.xml \
    --data-root /path/to/dataset \
    --device CPU

# Export + quantize a recipe to INT8, then benchmark
python scripts/benchmark.py \
    --model yolox_s --checkpoint /path/to/ckpt.ckpt \
    --data-root /path/to/dataset \
    --precision int8 --device CPU
```

---

## License

The core Geti™ Library (`getitune`) is licensed under [Apache License Version 2.0](https://github.com/open-edge-platform/training_extensions/blob/develop/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

Ultralytics YOLO models are distributed under the AGPL-3.0 license, an OSI approved license ideal for open-source research, academic, and personal projects. For commercial use, enhanced support, and tailored licensing terms, please explore flexible Ultralytics licensing options at https://www.ultralytics.com/license.

---
