# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build the trimmed Geti OpenAPI snapshot used by the REST contract tests.

The full specification is large and mostly irrelevant to this adapter, so the snapshot keeps
only the operations ``geti_mcp`` calls plus every schema they transitively reference.

Regenerate after a backend API change:

    cd application/backend && just gen-api-spec --output-path /tmp/geti-openapi.json
    python tests/assets/build_snapshot.py /tmp/geti-openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: Operations the adapter calls. Keep in sync with REQUIRED_OPERATIONS in the contract tests.
KEEP: dict[str, tuple[str, ...]] = {
    "/api/system/info": ("get",),
    "/api/system/devices/training": ("get",),
    "/api/projects": ("get",),
    "/api/projects/{project_id}": ("get",),
    "/api/projects/{project_id}/dataset/statistics": ("get",),
    "/api/projects/{project_id}/dataset/media": ("get",),
    "/api/projects/{project_id}/dataset/media/{media_id}/binary": ("get",),
    "/api/model_architectures": ("get",),
    "/api/projects/{project_id}/training_configuration": ("get",),
    "/api/projects/{project_id}/models": ("get",),
    "/api/projects/{project_id}/models/{model_id}": ("get",),
    "/api/jobs": ("get", "post"),
    "/api/jobs/{job_id}": ("get",),
    "/api/jobs/{job_id}:cancel": ("post",),
}

#: Keys dropped to keep the snapshot small. Only removed from specification objects, never
#: from a ``properties`` map, where the same words are legitimate property names.
NOISE = frozenset({"example", "examples", "description", "title", "summary", "tags", "operationId"})


def _referenced(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            found.add(ref.rsplit("/", 1)[1])
        for value in node.values():
            _referenced(value, found)
    elif isinstance(node, list):
        for item in node:
            _referenced(item, found)


def _strip(node: Any, *, in_properties: bool = False) -> Any:
    """Drop documentation noise, leaving ``properties`` keys untouched."""
    if isinstance(node, list):
        return [_strip(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: _strip(value, in_properties=key == "properties")
        for key, value in node.items()
        if in_properties or key not in NOISE
    }


def main(source: Path, destination: Path) -> None:
    spec = json.loads(source.read_text())
    missing = [path for path in KEEP if path not in spec["paths"]]
    if missing:
        raise SystemExit(f"Specification is missing expected paths: {missing}")

    paths = {
        path: {method: spec["paths"][path][method] for method in methods if method in spec["paths"][path]}
        for path, methods in KEEP.items()
    }

    schemas = spec["components"]["schemas"]
    needed: set[str] = set()
    _referenced(paths, needed)
    while True:
        discovered: set[str] = set()
        for name in needed:
            _referenced(schemas[name], discovered)
        if discovered <= needed:
            break
        needed |= discovered

    snapshot = {
        "openapi": spec["openapi"],
        "info": {"title": spec["info"]["title"], "version": spec["info"]["version"]},
        "paths": _strip(paths),
        "components": {"schemas": {name: _strip(schemas[name]) for name in sorted(needed)}},
    }
    destination.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {destination} ({len(paths)} paths, {len(needed)} schemas).")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).with_name("geti-openapi-snapshot.json"))
