#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Select one lightweight representative model per timm architecture family.

For each distinct ``family`` in ``timm_catalog_snapshot.json``, picks the
backbone with the lowest ``gigaflops``, so the resulting set is cheap to
train while still covering every architectural family at least once. Used to
seed the smoke-test suite in ``tests/bdd/features/timm_training.feature``.

Usage (from ``application/backend/``)::

    uv run python app/select_timm_smoke_models.py
    uv run python app/select_timm_smoke_models.py --format gherkin
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

_DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "supported_models" / "timm_catalog_snapshot.json"


def select_lightest_per_family(snapshot_path: Path) -> list[tuple[str, str]]:
    """Return a sorted list of (family, model_name) tuples, one per family, minimizing gigaflops."""
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    lightest_by_family: dict[str, dict] = {}

    for entry in data["backbones"]:
        family = entry["family"]
        current = lightest_by_family.get(family)
        if current is None or entry["gigaflops"] < current["gigaflops"]:
            lightest_by_family[family] = entry

    return sorted((family, entry["model_name"]) for family, entry in lightest_by_family.items())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT, help="Path to the timm catalog snapshot.")
    parser.add_argument(
        "--format",
        choices=["python", "gherkin"],
        default="python",
        help="Output format: 'python' for a repr'd list of tuples, 'gherkin' for an Examples table.",
    )
    return parser.parse_args()


def _format_gherkin_table(headers: list[str], rows: Sequence[tuple[str, ...]], indent: str = "      ") -> str:
    """Render *rows* as a column-aligned Gherkin table with *headers*, matching standard Gherkin style.

    Each column is padded to the width of its longest cell (header included), and cells are
    left-aligned with a single space of padding on each side of the `|` separators, e.g.::

        | family  | model_architecture_id   |
        | byobnet | bat_resnext26ts.ch_in1k |
        | resnet  | resnet10t.c3_in1k       |
    """
    columns = list(zip(headers, *rows, strict=True)) if rows else [(h,) for h in headers]
    widths = [max(len(str(cell)) for cell in column) for column in columns]

    def _format_row(cells: tuple[str, ...]) -> str:
        padded = (str(cell).ljust(width) for cell, width in zip(cells, widths, strict=True))
        return f"{indent}| " + " | ".join(padded) + " |"

    lines = [_format_row(tuple(headers)), *(_format_row(row) for row in rows)]
    return "\n".join(lines)


def main() -> None:
    """Entry point for the script."""
    args = _parse_args()
    selection = select_lightest_per_family(args.snapshot)

    if args.format == "gherkin":
        print(_format_gherkin_table(["family", "model_architecture_id"], selection))
    else:
        print(f"{len(selection)} families:")
        for pair in selection:
            print(pair)


if __name__ == "__main__":
    main()
