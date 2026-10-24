# Proposal: Integrate the Full `timm` Classification Catalog via Adaptive Model Manifests

- Issue: [#5230 — Integrate pytorch-image-models (timm) for classification](https://github.com/open-edge-platform/geti/issues/5230)
- Milestone: Geti 3.1
- Scope: Classification only — `MULTI_CLASS_CLS` and `MULTI_LABEL_CLS`. Detection, segmentation, and hierarchical classification are explicitly out of scope.
- Affected areas: `libraries/getitune/` (getitune), `application/backend/` (geti), `application/ui/`

---

## 1. Goal and `timm` API Usage

`timm` (PyTorch Image Models) exposes 1400+ pretrained classification backbones. This proposal describes how to make all of them selectable in Geti with minimal, maintainable code, without regressing the existing curated experience.

This catalog is intended for experienced users who want a specific architecture from the full timm ecosystem. We would not recommend models from this list by default.

The UI should include a short user-facing note on the timm card explaining what it is for: selecting and training a specific architecture from the timm Hub catalog.

For users who do not want to reason about architecture details, we can additionally expose a smaller set of separately curated classic model manifests. Those models should be tested against the existing catalog and shipped with strong default hyperparameters before we recommend them.

Decisions driving this design (agreed with stakeholders):

1. Keep all existing models, manifests, and top-picks untouched — zero regression for curated architectures.
2. Expose the entire timm pretrained catalog as user-selectable, with no accuracy guarantee, but fully configurable and shipped with sensible default hyperparameters so a default fine-tune never collapses to ~0 accuracy.
3. Generate manifests adaptively from a pinned timm catalog snapshot, rather than committing ~1400 static YAML files.
4. Image mean/std, optimizer, and model instantiation happen inside the library, on the fly, from the timm architecture's own configuration. They are not modeled in the manifest, API, or UI. Learning rate remains exposed and editable, along with the existing epochs / batch size / input size / etc.
5. The UI shows a single "timm backbone" card with a searchable architecture selector, not 1400 individual cards.

### Goal

- Massive model coverage with minimal additional code.
- One parametrized recipe + one generated catalog instead of thousands of files.
- A single, searchable entry point in the Train dialog that scales to 1400+ architectures.
- Per-architecture default hyperparameters resolved dynamically on selection.
- Clear, architecture-specific defaults while retaining the existing curated-model experience.

---

### `timm` API Usage

The integration consumes the pinned `timm` package from the library environment. The backend does not import `timm` or `torch`; it consumes a JSON snapshot generated from these APIs when the pinned `timm` version changes.

```python
import timm
import torch

from getitune.utils.utils import measure_flops

# Names that have pretrained weights. This is the catalog exposed by Geti.
model_names = timm.list_models(pretrained=True)

# Metadata for one named pretrained configuration. It includes the preprocessing
# configuration Geti must apply for this particular model.
cfg = timm.get_pretrained_cfg("vit_base_patch16_224.augreg2_in21k_ft_in1k")
print(cfg.input_size)      # (3, 224, 224)
print(cfg.mean)            # (0.5, 0.5, 0.5), model dependent
print(cfg.std)             # (0.5, 0.5, 0.5), model dependent
print(cfg.interpolation)   # "bicubic", model dependent

# The library creates a headless feature extractor and obtains its feature width.
# global_pool is intentionally left at the architecture's own default (see 3.3):
# forcing "avg" or "" is not universal across all 1400+ architectures.
model = timm.create_model(
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    pretrained=True,
    num_classes=0,
)
print(model.num_features)

# The integration resolves optimizer options from the selected architecture.
# These options, including default_lr, are Geti policy; timm creates the
# optimizer but does not prescribe fine-tuning learning-rate defaults.
family = "vit"
optimizer_config = (
    {"name": "adamw", "default_lr": 1e-4, "weight_decay": 0.05}
    if family in {"vit", "deit", "beit", "swin"}
    else {"name": "sgd", "default_lr": 7e-3, "momentum": 0.9, "weight_decay": 1e-4}
)

# The existing model base calls this optimizer callable; timm constructs the
# concrete optimizer from the model parameters and the resolved options.
optimizer = timm.optim.create_optimizer_v2(
    model,
    opt=optimizer_config["name"],
    lr=optimizer_config["default_lr"],
    weight_decay=optimizer_config["weight_decay"],
)

# Catalog statistics are calculated once while generating the pinned snapshot.
# FLOPs are for one backbone forward pass at cfg.input_size.
parameters = sum(parameter.numel() for parameter in model.parameters())
inputs = torch.zeros((1, *cfg.input_size))
flops = measure_flops(lambda: model(inputs))
print(f"{parameters / 1e6:.1f} M parameters")
print(f"{flops / 1e9:.1f} GFLOPs")
```

The snapshot generator records `model_name`, the fields from `get_pretrained_cfg`, calculated family/version metadata, resolved optimizer name/options/default learning rate, and model statistics. The worker calls `timm.create_model` again at train time with the selected name; the snapshot is catalog metadata, not a serialized model.

---

## 2. Design Constraints and Current Architecture

### Library baseline

- A generic timm backbone already exists: `TimmBackbone` (`src/getitune/backend/lightning/models/classification/backbones/timm.py`) wraps `timm.create_model(...)` and accepts any `model_name: str`.
- Task models already exist and accept `model_name`:
  - `TimmModelMulticlassCls` and `TimmModelMultilabelCls` (`.../classification/{multiclass,multilabel}_models/timm_model.py`).
  - A `TimmModel` factory (`.../classification/factory.py`) dispatches by task and exposes `TimmModel.list_models()` → `timm.list_models(pretrained=True)`

Blockers at scale:

- One recipe YAML per model does not scale to 1400.
- `TimmBackbone` currently hardcodes `num_classes=1000`, sets `self.model.classifier = None`, and relies on `forward_features` + `GlobalAveragePooling(dim=2)`. This assumes a CNN-style 2D feature map and a `.classifier`-named head. Many timm families (ViT/DeiT/BEiT/Swin/hybrids) use different head names and non-spatial outputs, so they will fail or silently mispreprocess → poor/zero accuracy. This is the single biggest correctness risk.
- The recipe hardcodes a single optimizer (`torch.optim.SGD`). One optimizer for all 1400 architectures is wrong — transformer families need AdamW. Optimizer construction must be internal to the timm model wrapper (see §3.2).

### Backend baseline

- Manifests are static YAML with 3-level inheritance: `manifests/base.yaml` → `manifests/<task>/base.yaml` → `manifests/<task>/<model>.yaml`, deep-merged in `app/services/model_manifest_service.py`.
- `ModelManifestService.get_model_manifests()` is `@cache`d and eager: it parses every manifest once into `ModelManifest` (`app/models/model_manifest.py`). At 1400 models that is ~4200 file reads + 1400 Pydantic constructions on first call.
- `ModelManifest` uses `extra="forbid"` on the top-level model and on `ModelStats`, `BenchmarkMetrics`, `Capabilities`, `PretrainedWeights`. Any unexpected auto-generated field hard-fails loading.
- `pretrained_weights` is required with `{url, mirror_url, sha_sum}` — all required strings. timm downloads weights itself from HF Hub, so this model does not fit timm.
- `learning_rate` has no default in `AlgoLevelTrainingParameters` — effectively required per model.
- No optimizer field exists anywhere — not in `AlgoLevelTrainingParameters`, the manifest schema, or the API view. `HyperparametersUpdater._update_learning_rate` only writes `optimizer["init_args"]["lr"]`; it never sets the optimizer `class_path`. This is convenient for us: it confirms the optimizer is already an internal recipe/library concern, so keeping it library-side is consistent with the current design.
- `GetiConfigConverter.TEMPLATE_ID_MAPPING` (`app/execution/common/geti_config_converter.py`) is a hardcoded dict mapping manifest `id` → recipe path (~71 entries today). It also routes classification via `sub_task_type`. 1400 entries here is untenable.
- API (`app/api/routers/model_architectures.py`): `GET /api/model_architectures?task=X` returns the entire flat list (`ModelArchitectureView`), unpaged; `top_picks` come from a hardcoded `RECOMMENDED_MODEL_ARCHITECTURES` map (`app/supported_models/model_recommendations.py`).

Deployment note: most timm backbones should still go through the standard Geti export/inference path, so the proposal should treat exportability as a requirement to validate rather than assume. The first release should cover representative models with training, OpenVINO export, ModelAPI load, and optional NNCF optimization.
For that I propose to create a simple script that runs through a set of timm models (2 per family (smallest and largest)) and validates that they can be exported and loaded in ModelAPI.

### UI baseline

- Architecture selection happens only in the Train Model dialog (`src/features/models/train-model/`).
- `use-get-model-architectures.hook.ts` runs a single `useSuspenseQuery('get', '/api/model_architectures')` and merges `top_picks` into each item as a `performanceCategory`.
- Rendering: a flat CSS `Grid` of radio-selectable cards inside one `RadioGroup` (`model-architectures-list-layout.component.tsx`).
- Hyperparameters ("Training configuration") are configured in `train-model/advanced-settings/` and fetched per selected architecture via `GET /api/projects/{project_id}/training_configuration?model_architecture_id=…`. The panel renders generically: it walks the returned `parameters` tree and chooses a widget from each param's `value_type`/`allowed_values` (`ParameterField`). Nothing is hardcoded by name. Implication: whatever fields the backend serves render automatically — no new UI widgets are needed for LR/epochs/etc.

Key insight: the `id → recipe` / `id → manifest` relationships are 1:1 static files today. Making timm a single parametrized recipe + a dynamically generated, per-id manifest removes the scaling problem. The UI treats the concrete architecture id (`image-classification-timm-<arch>`) as the selection key, so the existing per-id config/training pipeline is reused unchanged.

---

## 3. Required Library Changes

### 3.1 Generic timm recipes

Add one recipe per supported classification sub-task; `model_name`, preprocessing parameters, and `learning_rate` are injected by the backend. Optimizer construction remains inside the timm model wrapper (§3.2). Example:

```yaml
# libraries/getitune/src/getitune/recipe/classification/multi_class_cls/timm_generic.yaml
task: MULTI_CLASS_CLS
model:
  class_path: getitune.backend.lightning.models.classification.multiclass_models.timm_model.TimmModelMulticlassCls
  init_args:
    label_info: 1000
    model_name: DYNAMIC # <-- overwritten per user selection
    learning_rate: DYNAMIC # <-- overwritten from manifest/user (see §3.1)
    # NOTE: no `optimizer:` block. The model supplies a timm optimizer callable.
engine: { device: auto }
callback_monitor: val/f1-score
data: ../../_base_/data/classification.yaml
callbacks:
  - class_path: getitune.backend.lightning.callbacks.adaptive_early_stopping.EarlyStoppingWithWarmup
    init_args: { patience: 5 }
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      dirpath: ""
      monitor: val/f1-score
      mode: max
      save_top_k: 1
      save_last: true
      auto_insert_metric_name: false
      filename: "checkpoints/epoch_{epoch:03d}"
overrides:
  max_epochs: 90
```

Two files total: `multi_class_cls/timm_generic.yaml` and `multi_label_cls/timm_generic.yaml`.

### 3.2 Architecture-specific optimizer construction

No separate `optim_policy.py` is needed. The existing `LightningModel.configure_optimizers()` constructs `self.optimizer_callable(self.parameters())`; the timm wrappers should provide that callable in their constructor. The wrapper chooses the `opt` and static optimizer options from a small, local mapping, then delegates construction to `timm.optim.create_optimizer_v2`.

```python
# .../classification/multiclass_models/timm_model.py (conceptual change)
from functools import partial

import timm

_ADAMW_PREFIXES = ("vit", "deit", "beit", "swin", "cait", "xcit", "maxvit", "coat", "twins", "pvt")

def _timm_optimizer(model_name: str, learning_rate: float):
    family = model_name.split("_")[0].split(".")[0].lower()
    kwargs = (
        {"opt": "adamw", "lr": learning_rate, "weight_decay": 0.05}
        if family.startswith(_ADAMW_PREFIXES)
        else {"opt": "sgd", "lr": learning_rate, "momentum": 0.9, "weight_decay": 1e-4}
    )
    return partial(timm.optim.create_optimizer_v2, **kwargs)

class TimmModelMulticlassCls(LightningMulticlassClsModel):
    def __init__(self, label_info, *, model_name, learning_rate, **kwargs):
        super().__init__(
            label_info=label_info,
            model_name=model_name,
            optimizer=_timm_optimizer(model_name, learning_rate),
            **kwargs,
        )
```

`configure_optimizers()` is deliberately not overridden: the base implementation already invokes the supplied callable and configures the existing scheduler. The optimizer stays absent from the manifest, API, and UI. `learning_rate` remains editable and is passed to the callable.

### 3.3 Harden `TimmBackbone` and apply preprocessing dynamically

Required so defaults don't produce ~0 accuracy across families:

- Replace `num_classes=1000` + `self.model.classifier = None` with timm's canonical headless feature-extractor mode, without forcing a specific pooling mode:
  ```python
  self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
  ```
  `num_classes=0` is intentional and does not need the datamodule's real label count: Geti always attaches its own task head externally (`LinearClsHead` / `MultiLabelLinearClsHead`) on top of `backbone.num_features`, the same as every other Geti backbone. timm's own `fc` is pretrained for ImageNet-1000 and would have to be replaced anyway for any other label count, so setting `num_classes=0` simply turns `fc` into `Identity` and skips building a classifier head we would immediately discard. This keeps the backbone/head separation consistent with the rest of the codebase and is head-name agnostic (no reliance on `.classifier`).
- `global_pool` is deliberately left at each model's own default rather than forced to `"avg"` (or `""`). Empirically, `global_pool="avg"` is not universal: `pit_b_224` asserts `global_pool in ("token",)` and rejects `"avg"` outright at construction time. Explicitly requesting `forward_head(x, pre_logits=True)` is not universal either — several families override `forward_head()` with a signature that does not accept `pre_logits` (e.g. `inception_v3`, `hgnet*`, `tiny_vit_*`), and others return a `list` of multi-branch features from `forward_features()` (e.g. `coat_*`, `crossvit_*`) that `SelectAdaptivePool2d` cannot pool directly.

  The combination that works across the catalog is to keep `num_classes=0` (so `fc` becomes `Identity`) and simply call the model's own `forward(x)`, letting each architecture apply its own default pooling and head logic internally:

  ```python
  def extract_features(self, x: torch.Tensor) -> torch.Tensor:
      return self.model(x)  # fc is Identity; each model pools with its own default global_pool
  ```

  This was verified against 180 randomly sampled architectures from the full `timm.list_models(pretrained=True)` catalog (two independent 80- and 100-model samples, batch size 2 to avoid batch-norm's batch=1 restriction): all 180 returned a flat `[B, num_features]` tensor with no changes beyond `num_classes=0`. No custom neck, output-shape detection, or per-family pooling logic is needed in `TimmBackbone`.

- Read `model.pretrained_cfg` after creating the selected model and turn it into `DataInputParams`. This happens during model construction, so the selected architecture determines its own defaults:
  ```python
  cfg = self.model.pretrained_cfg
  return DataInputParams(
      input_size=tuple(cfg["input_size"][-2:]),
      mean=tuple(cfg["mean"]),
      std=tuple(cfg["std"]),
  )
  ```
  The backend also puts these values into the dynamic manifest before training, allowing the UI to show the correct input size. The converter injects `data_input_params` into the recipe. Resize interpolation must be mapped from `cfg["interpolation"]` into Geti's augmentation configuration so training and inference use the architecture's expected interpolation.
- Expose `num_features` reliably for the downstream head.

### 3.4 Catalog snapshot generation

The backend must never import `timm`/`torch` in the API process. A committed snapshot, generated offline from the pinned timm version, is the single data source for names, families, stats, and per-arch defaults:

```jsonc
// application/backend/app/supported_models/timm_catalog_snapshot.json
{
  "timm_version": "1.0.3",
  "backbones": [
    {
      "model_name": "tf_efficientnetv2_s.in21k",
      "family": "efficientnet",
      "version": "v2_s.in21k",
      "input_size": [3, 300, 300], // from pretrained_cfg
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225],
      "interpolation": "bicubic",
      "default_lr": 0.007, // aligned with the optimizer the library will pick (SGD)
      "gigaflops": 8.4,
      "trainable_parameters": 21.5,
      "imagenet_top1_accuracy": 83.9
    },
    {
      "model_name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
      "family": "vit",
      "version": "base_patch16_224",
      "input_size": [3, 224, 224],
      "mean": [0.5, 0.5, 0.5],
      "std": [0.5, 0.5, 0.5],
      "interpolation": "bicubic",
      "default_lr": 0.0001, // AdamW default
      "gigaflops": 17.6,
      "trainable_parameters": 86.6,
      "imagenet_top1_accuracy": 85.1
    }
    // ... ~1400 entries
  ]
}
```

Generation script (run only on timm upgrades) reads `timm.list_models(pretrained=True)`, `results-imagenet.csv`, and each model's `pretrained_cfg`. It assigns the catalog's default LR using the same family-to-`timm.optim.create_optimizer_v2` options as the timm wrapper. A CI check fails if `timm_version` drifts from the pinned dependency.

#### Dynamic model statistics

FLOPs and parameter count are model metadata, not training hyperparameters. `timm` supplies model construction and preprocessing configuration, but it does not provide a complete, normalized FLOPs/parameter catalog for every architecture. The snapshot generator therefore calculates statistics from the pinned `timm` build:

```python
import timm
import torch

from getitune.utils.utils import measure_flops

def collect_stats(model_name: str) -> dict[str, float]:
    cfg = timm.get_pretrained_cfg(model_name)
    model = timm.create_model(model_name, pretrained=False, num_classes=0)  # default global_pool (see 3.3)
    params = sum(parameter.numel() for parameter in model.parameters())
    inputs = torch.zeros((1, *cfg.input_size))
    flops = measure_flops(lambda: model(inputs))
    return {
        "trainable_parameters": params / 1_000_000,
        "gigaflops": flops / 1_000_000_000,
    }
```

The generator stores these values in the snapshot beside the selected name. Consequently, `TimmManifestProvider.build_manifest(name)` returns the correct precomputed FLOPs and parameter count whenever a user selects that name, without creating a `timm` model in the backend API process.

The documented meaning must be unambiguous:

- `trainable_parameters` is the headless timm backbone parameter count at snapshot generation time.
- `gigaflops` is one forward-pass estimate at the architecture's `pretrained_cfg.input_size`; it is not a training-step cost and changes if the user changes input size.
- The final Geti model additionally contains a task head whose size depends on the project label count. It is excluded from catalog statistics so the value stays stable while browsing. Optionally, the training-configuration endpoint can return a separate `effective_trainable_parameters` after it knows the project's labels and selected input size.

## 4. Required Backend Changes

### 4.1 `TimmManifestProvider`: build manifests dynamically

Concrete generator that turns one snapshot entry into a `ModelManifest`, on demand, by id:

```python
# application/backend/app/supported_models/timm_catalog.py
import json
from functools import lru_cache
from importlib import resources

from app.models.model_manifest import (
    ModelManifest, ModelStats, BenchmarkMetrics, Capabilities,
    ModelManifestDeprecationStatus,
)
from app.models.training_configuration import AlgoLevelParameters, AlgoLevelTrainingParameters
from app.models import TaskType

_ID_PREFIX = "image-classification-timm-"

def model_name_to_id(model_name: str) -> str:
    # Dots and underscores are valid in URL paths and database identifiers.
    return _ID_PREFIX + model_name

def id_to_model_name(manifest_id: str) -> str:
    return manifest_id.removeprefix(_ID_PREFIX)

@lru_cache(maxsize=1)
def _snapshot() -> dict[str, dict]:
    raw = json.loads(resources.files("app.supported_models")
                      .joinpath("timm_catalog_snapshot.json").read_text())
    return {e["model_name"]: e for e in raw["backbones"]}

class TimmManifestProvider:
    """Adaptively builds ModelManifest objects for timm backbones from the snapshot."""

    @staticmethod
    def is_timm_id(manifest_id: str) -> bool:
        return manifest_id.startswith(_ID_PREFIX)

    @classmethod
    def build_manifest(cls, model_name: str) -> ModelManifest:
        e = _snapshot()[model_name]
        _, h, w = e["input_size"]
        return ModelManifest(
            id=model_name_to_id(model_name),
            name=f"timm · {model_name}",
            family=e["family"],                 # new schema field (§5)
            version=e.get("version"),           # new schema field (§5)
            task=TaskType.CLASSIFICATION,
            description=f"timm backbone '{model_name}'.",
            pretrained_weights=None,             # weights are timm-managed (§6)
            weights_source="timm_managed",      # new schema field (§6)
            support_status=ModelManifestDeprecationStatus.ACTIVE,
            capabilities=Capabilities(xai=True, tiling=False),
            stats=ModelStats(
                gigaflops=e.get("gigaflops", 0.0),
                trainable_parameters=e.get("trainable_parameters", 0.0),
                benchmark_metrics=BenchmarkMetrics(
                    imagenet_top1_accuracy=e.get("imagenet_top1_accuracy"),
                ),
            ),
            # ---- Dynamic hyperparameters keyed on the chosen architecture ----
            hyperparameters=AlgoLevelParameters(
                training=AlgoLevelTrainingParameters(
                    learning_rate=e["default_lr"],   # arch-appropriate default, editable
                    input_size_width=w,              # from timm pretrained_cfg
                    input_size_height=h,
                    # everything else (epochs, batch, scheduler, early stopping,
                    # augmentation) inherits classification/base.yaml defaults.
                ),
                # dataset_preparation augmentation inherited from task base.
            ),
        )

    @classmethod
    def get_preprocessing(cls, model_name: str) -> dict:
        e = _snapshot()[model_name]
        _, height, width = e["input_size"]
        return {
            "input_size": (height, width),
            "mean": tuple(e["mean"]),
            "std": tuple(e["std"]),
            "interpolation": e["interpolation"],
        }
```

### 4.2 Wire the provider into `ModelManifestService`

Keep static manifests untouched; resolve timm ids lazily by id (the critical path for the config fetch), and enumerate via the snapshot (cheap, no manifest construction):

```python
# application/backend/app/services/model_manifest_service.py  (additions)
from app.supported_models.timm_catalog import TimmManifestProvider, id_to_model_name

class ModelManifestService:
    @classmethod
    def get_model_manifest_by_id(cls, model_manifest_id: str) -> ModelManifest:
        # 1) existing static/curated manifests (unchanged)
        static = cls.get_model_manifests()
        if model_manifest_id in static:
            return static[model_manifest_id]
        # 2) lazily materialize a single timm manifest on demand
        if TimmManifestProvider.is_timm_id(model_manifest_id):
            return TimmManifestProvider.build_manifest(id_to_model_name(model_manifest_id))
        raise ManifestNotFoundException(model_manifest_id=model_manifest_id)
```

No eager parse of 1400 manifests: the snapshot is loaded once, while a concrete `ModelManifest` is built for the architecture selected by the user. Selecting another name triggers the existing training-configuration request with that name's id and builds a fresh manifest for that new selection. The preceding selection is not mutated or reused.

### 4.3 Resolve recipes and inject the selected model configuration

Two focused changes to the converter:

```python
# application/backend/app/execution/common/geti_config_converter.py  (resolver)
from app.supported_models.timm_catalog import TimmManifestProvider, id_to_model_name

@staticmethod
def _resolve_recipe(config: dict) -> tuple[Path, str | None]:
    manifest_id = config["model_manifest_id"]
    if TimmManifestProvider.is_timm_id(manifest_id):
        sub = config["sub_task_type"].lower()  # multi_class_cls | multi_label_cls
        recipe = RECIPE_PATH / "classification" / sub / "timm_generic.yaml"
        return recipe, id_to_model_name(manifest_id)   # also return model_name to inject
    return TEMPLATE_ID_MAPPING[manifest_id]["recipe_path"], None   # unchanged path
```

After loading the recipe, inject the chosen backbone and route LR to the model (since there is no `optimizer` block for timm recipes):

```python
recipe_path, timm_model_name = GetiConfigConverter._resolve_recipe(config)
default_config = load_recipe_config(recipe_path)
if timm_model_name is not None:
    timm_config = TimmManifestProvider.get_preprocessing(timm_model_name)
    default_config["model"]["init_args"]["model_name"] = timm_model_name
    default_config["model"]["init_args"]["data_input_params"] = {
        "input_size": timm_config["input_size"],
        "mean": timm_config["mean"],
        "std": timm_config["std"],
    }
```

The manifest supplies the selected architecture's `input_size` to the UI. The provider reads that same selected snapshot entry for mean, std, and interpolation; the converter injects mean/std/input size into `data_input_params` and maps interpolation into the resize transform. Thus selecting a new name rebuilds its manifest and produces a recipe configured with that name's own preprocessing values, rather than relying on a generic ImageNet default.

`HyperparametersUpdater._update_learning_rate` gains a fallback: if there is no `optimizer` block (timm recipes), write the model's `learning_rate` init-arg instead of `optimizer.init_args.lr`:

```python
@staticmethod
def _update_learning_rate(param_value: float | None, config: dict) -> None:
    if param_value is None:
        return
    init_args = config["model"]["init_args"]
    optimizer = init_args.get("optimizer")
    if isinstance(optimizer, dict) and "init_args" in optimizer:
        optimizer["init_args"]["lr"] = param_value        # curated recipes (unchanged)
    else:
        init_args["learning_rate"] = param_value          # timm recipes (library builds optimizer)
```

---

### 4.4 Family metadata and manifest schema

Because the UI is a single card + selector (§5.1), families are used to filter/group the selector list, not to render cards.

- Derive `family` from timm's architecture prefix/module (`resnet`, `efficientnet`, `convnext`, `vit`, `swin`, `deit`, `regnet`, `mobilenet`, `beit`, `maxvit`, …); `version` holds the size/tag (`b0`, `tiny`, `v2_s.in21k`, …). Both are precomputed in the snapshot.
- Schema additions to `ModelManifest` (domain) and `ModelArchitectureView` (API DTO): `family: str`, `version: str | None`.

---

### 4.5 Weights handling

timm fetches pretrained weights from HF Hub; Geti's `PretrainedWeights{url, mirror_url, sha_sum}` (all required, `extra="forbid"`) does not fit.

- Make `pretrained_weights` optional on `ModelManifest`.
- Add `weights_source: Literal["geti_hosted", "timm_managed"]` (default `geti_hosted`).
- For `timm_managed`, the training worker lets `timm.create_model(pretrained=True)` download/cache weights into `data/pretrained_weights/` (respecting `TIMM_HOME`/`HF_HOME`).
- Curated models keep `geti_hosted` + existing URL/sha behavior — no change.
- Offline / air-gapped: document a cache pre-seeding step; selecting a model whose weights are not cached surfaces a clear error.

---

### 4.6 Dynamic configuration defaulting

The whole point: defaults are resolved from the architecture the user picks, so training never starts from nonsense values.

Layered resolution, in order of precedence (highest last):

1. Global classification defaults — `manifests/classification/base.yaml` (epochs, batch size, scheduler shape, augmentation, early stopping). Inherited unchanged.
2. Per-architecture defaults from the snapshot — `learning_rate` (aligned with the timm optimizer options) and `input_size`, mean, std, and interpolation (from `pretrained_cfg`), injected by `TimmManifestProvider.build_manifest` (§4.1).
3. User overrides — anything the user edits in Advanced Settings (PATCH to `training_configuration`) wins over the defaults.

Optimizer is not in this list — it is constructed inside the library from `model_name` (§3.2), tracking the same architecture the LR default was computed for. `learning_rate` is the exposed, editable representation of that decision.

---

### 4.7 API contract changes

Two endpoints:

4.7.1 Main list - one synthetic timm entry (the card). `GET /api/model_architectures?task=classification` returns the curated models plus a single synthetic entry so the UI can render the timm card:

```jsonc
{
  "id": "image-classification-timm", // synthetic entry point (not trainable directly)
  "name": "Custom backbone (timm)",
  "family": "timm",
  "support_status": "active",
  "task": "classification",
  "description": "Choose any of 1400+ timm backbones.",
  "capabilities": { "xai": true, "tiling": false }
}
```

4.7.2 New backbone-search endpoint - feeds the selector. Backed entirely by the snapshot - cheap, paginated, filterable:

```
GET /api/model_architectures/timm/backbones?search=&family=&limit=&offset=
```

```jsonc
{
  "total": 1412,
  "limit": 50,
  "offset": 0,
  "families": [
    { "name": "efficientnet", "count": 38 },
    { "name": "vit", "count": 121 }
  ],
  "backbones": [
    {
      "id": "image-classification-timm-tf_efficientnetv2_s.in21k",
      "model_name": "tf_efficientnetv2_s.in21k",
      "family": "efficientnet",
      "version": "v2_s.in21k",
      "stats": {
        "gigaflops": 8.4,
        "trainable_parameters": 21.5,
        "imagenet_top1_accuracy": 83.9
      }
    }
    // ...
  ]
}
```

When the user selects a backbone, the UI uses its concrete `id` (`image-classification-timm-<arch>`) for the existing per-id flow: `GET .../training_configuration?model_architecture_id=<id>` and the training request. No other endpoint changes.

Process: implement in API/service, regenerate the OpenAPI spec (`just gen-api-spec --output-path openapi-spec.json`), then UI types (`npm run build:api`). Use the `geti-openapi-sync` workflow.

### 4.8 Export and optimization compatibility

Architecture discovery and training support do not imply deployment support. A
timm backbone is eligible for the first release only after a representative
smoke test has completed the full path:

1. instantiate the headless backbone and attach the Geti classification head;
2. export the trained model to OpenVINO;
3. load the exported model through ModelAPI and run one inference; and
4. apply the supported NNCF optimization flow where the architecture is
   compatible with it.

The validation matrix should include at least the smallest and largest
representative model from every discovered family. The catalog may still list
architectures that fail this validation, but those entries must be marked
unsupported for deployment and must not be presented as production-ready.
Export or optimization failures should identify the selected `model_name` and
the failed stage, rather than silently falling back to another architecture.

---

## 5. Required UI Changes

### 5.1 Single card and searchable selector

All under `application/ui/src/features/models/train-model/`. Curated cards are unchanged.

1. New timm card. When the architectures list contains the `family: "timm"` entry, render a dedicated `TimmBackboneCard` instead of a plain radio card.
2. Searchable selector inside the card. A `SearchField` + results list driven by the new `/timm/backbones` endpoint (debounced query, `family` facet filter, infinite scroll / pagination). Each result shows `model_name`, family, parameters, and top-1 when available. No support-status or verification labels are shown.
3. Selection wiring. Picking a backbone sets the provider's `selectedModelArchitectureId = <concrete timm id>`:
   ```tsx
   // train-model-provider.component.tsx (conceptual)
   const onSelectBackbone = (backbone: TimmBackbone) =>
     setSelectedModelArchitectureId(backbone.id); // image-classification-timm-<arch>
   ```
   Everything downstream (the `advanced-settings` config fetch and the train mutation) already keys off `selectedModelArchitectureId` and needs no change.
4. Dynamic hyperparameters (no new widgets). On selection, `useGetModelArchitectureTrainingConfiguration({ modelArchitectureId })` refetches; the backend returns the arch-specific `learning_rate` and `input_size`, which the generic `Parameters`/`ParameterField` renderer displays automatically. Learning rate appears as an editable `FloatParameterView`; optimizer never appears.
5. Pre-selection state. Until a backbone is chosen, disable the "Start training" action and either gray out the Advanced Settings LR/input-size fields or show explicit placeholder values. The UI team can choose the final presentation; both are acceptable as long as fields are clearly architecture-dependent.
6. Default backbone. Pre-select a documented baseline architecture (for example, EfficientNetV2-S) so the card is trainable out of the box.

### 5.2 UI configuration behavior

The selector stores the concrete architecture id. Changing the selection invalidates the previous training-configuration query and requests the newly selected id, so the backend builds a new dynamic manifest and returns new defaults.

### 5.3 Testing

- Library unit/model tests: for representative models from CNN and transformer families, verify the `timm.optim.create_optimizer_v2` options, then instantiate → 1-step train → export (OpenVINO/ONNX).
- Backend unit tests: `model_name ↔ id` round-trip; `TimmManifestProvider.build_manifest` produces a schema-valid `ModelManifest` (respecting `extra="forbid"`); lazy `get_model_manifest_by_id` for timm ids; `GetiConfigConverter._resolve_recipe` + LR-injection fallback; `/timm/backbones` search/pagination/facets against a fixture snapshot.
- E2E testing: each run should execute a small rotating sample of random timm
  architectures, while the complete validation matrix guarantees at least one
  representative model from every discovered family. The smallest and largest
  representative models from each family should be included in the export and
  ModelAPI smoke tests described in §4.8.
- UI: component tests for the selector (search, family filter, selection → id), and that the generic parameter panel renders the returned LR field; MSW handlers for both new endpoints.
- Acceptance criteria (from the issue): model discovery, manifest correctness, basic instantiation flows covered; no regression in existing classification workflows.

---

### 5.4 Complete Flow Example

```text
User opens the Train dialog
  -> GET /api/model_architectures?task=classification
  -> curated model cards + one "Custom backbone (timm)" card

User searches for "vit"
  -> GET /api/model_architectures/timm/backbones?search=vit&limit=50&offset=0
  -> backend returns snapshot metadata:
       model_name = vit_base_patch16_224.augreg2_in21k_ft_in1k
       family = vit
       params = 86.6 M
       GFLOPs = 17.6 at 224 x 224

User selects the architecture
  -> UI stores selectedModelArchitectureId =
  image-classification-timm-vit_base_patch16_224.augreg2_in21k_ft_in1k
  -> GET /api/projects/{project_id}/training_configuration
       ?model_architecture_id=<selected id>
  -> backend decodes the id to the timm model name
  -> TimmManifestProvider builds a fresh manifest for this selection:
       model_name = vit_base_patch16_224.augreg2_in21k_ft_in1k
       input_size = 224 x 224
       mean/std/interpolation = from pretrained_cfg
       parameters = 86.6 M
       GFLOPs = 17.6
       optimizer = AdamW (internal library setting)
       learning_rate = 1e-4 (editable UI default)

UI renders the selected model's configuration
  -> model statistics are displayed as read-only metadata
  -> input size and learning rate are populated dynamically
  -> learning rate remains editable
  -> optimizer name is not an API or UI setting

User changes the architecture to tf_efficientnetv2_s.in21k
  -> UI replaces selectedModelArchitectureId
  -> the previous configuration query is invalidated
  -> a new configuration request builds a new manifest:
       input_size = 300 x 300
       parameters = 21.5 M
       GFLOPs = 8.4
       optimizer = SGD (internal library setting)
       learning_rate = 7e-3 (editable UI default)

User starts training
  -> training request contains the concrete timm model id
  -> backend resolves timm_generic.yaml
  -> backend injects model_name, data_input_params, and the user LR
  -> library creates the selected timm model with pretrained=True
  -> timm constructs the internal optimizer with the architecture-specific options
  -> Geti trains and exports the selected architecture
```

---

## Appendix — Key Source References

- Library timm: `libraries/getitune/src/getitune/backend/lightning/models/classification/backbones/timm.py`, `.../{multiclass,multilabel}_models/timm_model.py`, `.../classification/factory.py`
- Backend manifests: `application/backend/app/supported_models/manifests/`, `app/services/model_manifest_service.py`, `app/models/model_manifest.py`, `app/supported_models/model_recommendations.py`
- Backend training config: `app/models/training_configuration/` (`AlgoLevelTrainingParameters`), `app/services/training_configuration_service.py`, `app/api/schemas/training_configuration.py`, `app/api/routers/training_configurations.py`
- Backend converter: `application/backend/app/execution/common/geti_config_converter.py` (`TEMPLATE_ID_MAPPING`, `HyperparametersUpdater`, `TransformsUpdater`)
- Backend API: `application/backend/app/api/routers/model_architectures.py`, `app/api/schemas/model_architecture.py`
- UI: `application/ui/src/features/models/train-model/` (`model-architectures-list/`, `advanced-settings/` generic `Parameters`/`ParameterField`, `train-model-provider.component.tsx`), `src/features/models/hooks/api/use-get-model-architectures.hook.ts`, `src/api/shared-types.ts`
