# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from types import MappingProxyType


@lru_cache(maxsize=1)
def _snapshot() -> Mapping[str, dict]:
    raw = json.loads(resources.files("app.supported_models").joinpath("timm_catalog_snapshot.json").read_text())
    return MappingProxyType({e["model_name"]: e for e in raw["backbones"]})


class TimmCatalog:
    """The single entry point for querying available timm backbones."""

    @staticmethod
    def count_backbones() -> int:
        """Return the number of timm backbones in the catalog."""
        return len(_snapshot())

    @staticmethod
    def list_families() -> list[str]:
        """Return every distinct architecture family present in the catalog."""
        return sorted({e["family"] for e in _snapshot().values()})

    @staticmethod
    def list_variants(family: str) -> list[str]:
        """Return every distinct architecture variant within *family*."""
        return sorted({e["version"] for e in _snapshot().values() if e["family"] == family})

    @staticmethod
    def list_pretrained_tags(family: str, version: str) -> list[str]:
        """Return every pretrained tag available for *family*/*version*."""
        return sorted(
            {e["pretrained"] for e in _snapshot().values() if e["family"] == family and e["version"] == version}
        )

    @staticmethod
    def get_model_name(family: str, version: str, pretrained_tag: str) -> str | None:
        """Return the model ID for a given family, version, and pretrained tag."""
        for model_name, entry in _snapshot().items():
            if entry["family"] == family and entry["version"] == version and entry["pretrained"] == pretrained_tag:
                return model_name
        return None
