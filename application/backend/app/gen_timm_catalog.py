#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate the offline timm backbone catalog snapshot.

Builds ``timm_catalog_snapshot.json`` - the single data source the backend
(`application/backend/app/supported_models/`) uses to expose the timm
backbone catalog without importing `timm`/`torch` in the API process.

For every model returned by ``timm.list_pretrained()`` in the pinned `timm`
build, this script records:

- ``family`` / ``version`` / ``pretrained``: derived from timm's own module
  grouping.
- preprocessing defaults (``input_size``, ``mean``, ``std``) from ``pretrained_cfg``.
- ``default_lr`` / ``default_weight_decay``: aligned with the optimizer family
  the timm model wrapper will pick at train time (see ``TimmModelMulticlassCls``/§3.2
  of the design doc), so the exposed learning rate and weight decay always matches
  the optimizer actually constructed.
- ``trainable_parameters`` / ``gigaflops``: computed once per model via a
  headless (``num_classes=0``) forward pass and ``measure_flops``.

Incremental behaviour: this script is expensive (~1400 model instantiations
+ FLOP counts), so if an existing snapshot already has an entry for a given
model name, its ``trainable_parameters``/``gigaflops`` are reused as-is and
*not* recomputed. Cheap, purely metadata-derived fields (family, version,
preprocessing, default_lr) are always recomputed, since they cost nothing
and must never drift from the pinned timm build.

Run only when the pinned `timm` version changes (see `application/backend/Justfile`
`gen-timm-catalog` recipe, wired to `just gen-timm-catalog` from `application/backend`).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_CODE_LICENSES = frozenset({"Apache-2.0", "BSD-3-Clause", "MIT"})
_REFUSED_WEIGHTS_LICENSES = frozenset({"not specified", "unknown"})
_UNKNOWN_LICENSE = "unknown"
# Architecture prefixes that should use AdamW at train time (matches the
# optimizer selection in TimmModelMulticlassCls / TimmModelMultilabelCls).
# TODO(https://github.com/open-edge-platform/geti/issues/7097): import constants from library (single source of truth)
_ADAMW_PREFIXES = ("vit", "deit", "beit", "swin", "cait", "xcit", "maxvit", "coat", "twins", "pvt")
_ADAMW_DEFAULTS = {
    "learning_rate": 1e-4,
    "weight_decay": 5e-2,
}
_SGD_DEFAULTS = {
    "learning_rate": 7e-3,
    "weight_decay": 1e-3,
}

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "timm_catalog_snapshot.json"


def _family_of(model_name: str) -> str:
    """Return the family (optimizer-selection) prefix for *model_name*.

    Mirrors the heuristic in the timm model wrapper so the default LR
    recorded here always matches the optimizer chosen at train time.
    """
    return model_name.removeprefix("tf_").split("_")[0].lower()


def _defaults_for(model_name: str) -> dict[str, float]:
    family = _family_of(model_name)
    if family.startswith(_ADAMW_PREFIXES):
        return _ADAMW_DEFAULTS
    return _SGD_DEFAULTS


def _module_family_map() -> dict[str, str]:
    """Map every pretrained model name to its timm module (architecture family)."""
    import timm

    mapping: dict[str, str] = {}
    for module in timm.list_modules():
        for name in timm.list_models(module=module, pretrained=True):
            mapping[name] = module
    return mapping


def _tags(model_name: str, family: str) -> tuple[str, str]:
    """Best-effort variant & pretrained tags: model name with the leading family token stripped."""
    variant, _, pretrained = model_name.partition(".")
    variant = variant.removeprefix(f"tf_{family}_" if "tf_" in variant else f"{family}_")
    return variant, pretrained


def _load_imagenet_top1(csv_path: Path | None) -> dict[str, float]:
    """Optionally load top-1 accuracy from a locally provided results-imagenet.csv.

    This file is not shipped with the `timm` PyPI package (it lives in the
    timm GitHub repo's `results/` folder), so this is best-effort: pass
    ``--imagenet-results-csv`` with a manually downloaded copy to populate
    ``imagenet_top1_accuracy``. When omitted, the field is left as ``None``.
    """
    if csv_path is None:
        return {}
    if not csv_path.exists():
        logger.warning("ImageNet results CSV not found at %s; skipping top1 accuracy.", csv_path)
        return {}
    top1: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model = row.get("model")
            top1_str = row.get("top1")
            if model and top1_str:
                try:
                    top1[model] = float(top1_str)
                except ValueError:
                    continue
    return top1


def _load_model_licenses(csv_path: Path | None) -> dict[str, dict[str, str]]:
    """Optionally load per-model weight/code license info from a local CSV copy.

    Expected source: timm_license_extractor's `timm_model_licenses.csv`
    (https://github.com/itallix/timm_license_extractor). Pass
    ``--model-licenses-csv`` with a locally downloaded copy to populate
    ``weight_license`` and to filter out models whose license isn't allowed.
    When omitted, no license info is added and no models are filtered on this basis.
    """
    if csv_path is None:
        return {}
    if not csv_path.exists():
        logger.warning("Model licenses CSV not found at %s; skipping license info.", csv_path)
        return {}
    licenses: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model_id = row.get("model_id")
            if not model_id:
                continue
            licenses[model_id] = {
                "weights_license": row.get("weights_license", "") or "",
                "code_license": row.get("code_license", "") or "",
            }
    return licenses


def _is_license_allowed(model_name: str, licenses: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_name*'s code license is allowed and its weight license isn't refused."""
    info = licenses.get(model_name)
    if info is None:
        return False

    code_license = info.get("code_license", "")
    if code_license not in _ALLOWED_CODE_LICENSES:
        return False

    weights_license = (info.get("weights_license", "")).strip()
    return bool(weights_license) and weights_license.lower() not in _REFUSED_WEIGHTS_LICENSES


def _compute_stats(model_name: str) -> dict[str, float]:
    """Compute headless parameter count and single-forward-pass GFLOPs for *model_name*."""
    import timm
    import torch
    from getitune.utils.utils import measure_flops

    cfg = timm.get_pretrained_cfg(model_name)
    if cfg is None:
        raise ValueError(f"Pretrained config not found for {model_name}; cannot compute stats.")
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    model.eval()

    params = sum(parameter.numel() for parameter in model.parameters())
    inputs = torch.zeros((1, *cfg.input_size))

    flops = measure_flops(lambda: model(inputs))

    return {
        "trainable_parameters": round(params / 1_000_000, 1),
        "gigaflops": round(flops / 1_000_000_000, 2),
    }


def _build_entry(
    model_name: str,
    family_map: dict[str, str],
    imagenet_top1: dict[str, float],
    model_licenses: dict[str, dict[str, str]],
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    import timm

    cfg = timm.get_pretrained_cfg(model_name)
    if cfg is None:
        raise ValueError(f"Pretrained config not found for {model_name}; cannot build entry.")
    family = family_map.get(model_name, _family_of(model_name))
    version_tag, pretrained_tag = _tags(model_name, family)
    default_params = _defaults_for(model_name)

    entry: dict[str, Any] = {
        "model_name": model_name,
        "family": family,
        "version": version_tag,
        "pretrained": pretrained_tag,
        "input_size": list(cfg.input_size),
        "mean": list(cfg.mean),
        "std": list(cfg.std),
        "default_lr": default_params["learning_rate"],
        "default_weight_decay": default_params["weight_decay"],
        "imagenet_top1_accuracy": imagenet_top1.get(model_name),
        "license": model_licenses.get(model_name, {}).get("weights_license", _UNKNOWN_LICENSE),
    }

    cached_keys = {"trainable_parameters", "gigaflops"}
    if existing is not None and cached_keys.issubset(existing.keys()):
        logger.debug("Reusing cached stats for %s", model_name)
        entry.update({k: existing[k] for k in cached_keys})
    else:
        logger.info("Computing stats for %s", model_name)
        entry.update(_compute_stats(model_name))

    return entry


def _load_existing(output: Path) -> dict[str, dict[str, Any]]:
    if not output.exists():
        return {}
    try:
        data = json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Existing snapshot at %s is not valid JSON; regenerating from scratch.", output)
        return {}
    return {entry["model_name"]: entry for entry in data.get("backbones", [])}


def generate_snapshot(
    output: Path,
    imagenet_results_csv: Path | None,
    model_licenses_csv: Path | None,
    limit: int | None,
) -> dict[str, Any]:
    """Build (or incrementally update) the timm catalog snapshot."""
    import timm

    timm_version = pkg_version("timm")
    existing_by_name = _load_existing(output)
    family_map = _module_family_map()
    imagenet_top1 = _load_imagenet_top1(imagenet_results_csv)
    model_licenses = _load_model_licenses(model_licenses_csv)

    model_names = sorted(timm.list_models(pretrained=True))
    if model_licenses:
        model_names = [name for name in model_names if _is_license_allowed(name, model_licenses)]
    if limit is not None:
        model_names = model_names[:limit]

    backbones = [
        _build_entry(name, family_map, imagenet_top1, model_licenses, existing_by_name.get(name))
        for name in model_names
    ]

    return {"timm_version": timm_version, "backbones": backbones}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path to write the snapshot JSON to.",
    )
    parser.add_argument(
        "--imagenet-results-csv",
        type=Path,
        default=None,
        help="Optional local copy of timm's results/results-imagenet.csv to populate top1 accuracy.",
    )
    parser.add_argument(
        "--model-licenses-csv",
        type=Path,
        default=None,
        help="Optional local copy of timm_license_extractor's timm_model_licenses.csv "
        "to populate weight_license and filter models by allowed code license.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N models (for quick local testing).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    snapshot = generate_snapshot(
        output=args.output,
        imagenet_results_csv=args.imagenet_results_csv,
        model_licenses_csv=args.model_licenses_csv,
        limit=args.limit,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d backbone entries to %s", len(snapshot["backbones"]), args.output)


if __name__ == "__main__":
    main()
