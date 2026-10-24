#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Maintain Geti agent skills (adapters, layout, frontmatter).

Canonical skills live under ``skills/<bucket>/<name>/`` with ``bucket`` one of
``getitune`` or ``application``. Client adapters under ``.claude/skills/`` and
``.agents/skills/`` are symlinks (directory junctions on Windows without symlink
support) pointing back at the canonical directories, so a skill is authored once
and consumed by every agent client.
"""

from __future__ import annotations

import argparse
import os
import platform
import re

# subprocess is only used for a fixed `cmd /c mklink /J` fallback with no shell.
import subprocess  # nosec B404
import sys
from pathlib import Path

_SCRIPT = "python3 .github/scripts/skills/agent_skills.py"

BUCKETS = ("getitune", "application")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def skill_dirs(root: Path) -> list[tuple[str, str]]:
    """Return (bucket, skill_name) for each canonical skill directory."""
    found: list[tuple[str, str]] = []
    for bucket in BUCKETS:
        bucket_path = root / "skills" / bucket
        if not bucket_path.is_dir():
            continue
        for child in sorted(bucket_path.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                found.append((bucket, child.name))
    return found


def adapter_target(bucket: str, name: str) -> str:
    return f"../../skills/{bucket}/{name}"


def read_link(path: Path) -> str | None:
    if path.is_symlink():
        return os.readlink(path)
    return None


def _norm(value: str | None) -> str | None:
    """Normalize separators so Windows backslash links compare equal to POSIX.

    Git stores symlink blobs with forward slashes on every platform, but the
    materialized working-tree link on Windows uses backslashes. Comparisons
    must be separator-insensitive.
    """
    return value.replace("\\", "/") if value is not None else None


def resolves_to(link: Path, expected_target: str) -> bool:
    """Check if a symlink or junction points at the expected target.

    On non-Windows, adapters are symlinks and ``read_link`` suffices.
    On Windows, ``sync`` may fall back to a directory junction when
    symlinks are unavailable; ``read_link`` returns ``None`` for those,
    so resolve the link and compare against the resolved expected target.
    """
    if link.is_symlink():
        return _norm(read_link(link)) == expected_target
    if link.is_dir():
        return link.resolve() == (link.parent / expected_target).resolve()
    return False


def remove_adapter(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"{path} is a real directory; remove it manually")


def create_adapter(link: Path, target: str) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    remove_adapter(link)
    abs_target = (link.parent / target).resolve()
    if not abs_target.is_dir():
        raise RuntimeError(f"canonical skill missing: {abs_target}")

    if platform.system() == "Windows":
        _create_windows_link(link, target, abs_target)
    else:
        link.symlink_to(target, target_is_directory=True)


def _create_windows_link(link: Path, target: str, abs_target: Path) -> None:
    # Prefer a relative symlink. Git normalizes the on-disk backslashes back to
    # forward slashes on ``add``, so the committed blob stays portable (mode
    # 120000, content ``../../skills/...``) while the working-tree link uses
    # native separators and is followable. Works for normal users when Windows
    # Developer Mode is enabled. Fall back to a directory junction (which git
    # records as a real directory) only when symlinks are unavailable.
    try:
        os.symlink(target.replace("/", os.sep), link, target_is_directory=True)
        return
    except OSError:
        pass

    # Fixed argument list, no shell, and paths are repo-internal.
    subprocess.run(  # nosec B603
        ["cmd", "/c", "mklink", "/J", str(link), str(abs_target)],
        check=True,
        capture_output=True,
        text=True,
    )


def cmd_sync(root: Path) -> int:
    adapters = (root / ".claude" / "skills", root / ".agents" / "skills")

    for bucket, name in skill_dirs(root):
        target = adapter_target(bucket, name)
        for adapter_root in adapters:
            link = adapter_root / name
            current = _norm(read_link(link))
            if current == target and link.exists():
                continue
            create_adapter(link, target)

    names = sorted({name for _, name in skill_dirs(root)})
    for adapter_root in adapters:
        print(f"{adapter_root.relative_to(root)}:")
        for name in names:
            if (adapter_root / name).exists():
                print(f"  {name}")
    return 0


def cmd_check_adapters(root: Path) -> int:
    errors: list[str] = []
    adapters = (root / ".claude" / "skills", root / ".agents" / "skills")

    for bucket, name in skill_dirs(root):
        target = adapter_target(bucket, name)
        for adapter_root in adapters:
            link = adapter_root / name
            if resolves_to(link, target):
                continue
            current = read_link(link)
            errors.append(f"{link}: expected adapter to {target!r}, got {current!r}")

    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        print(f"Run: {_SCRIPT} sync", file=sys.stderr)
        return 1
    return 0


def frontmatter_name(skill_md: Path) -> str | None:
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    for line in match.group(1).splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def cmd_validate(root: Path) -> int:
    errors: list[str] = []

    for bucket in BUCKETS:
        bucket_dir = root / "skills" / bucket
        if not bucket_dir.is_dir():
            errors.append(f"Missing {bucket_dir}")
            continue

        for child in sorted(bucket_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue

            name = child.name
            fm_name = frontmatter_name(skill_md)
            if not fm_name:
                errors.append(f"{skill_md}: missing frontmatter name")
            elif fm_name != name:
                errors.append(
                    f"{skill_md}: frontmatter name {fm_name!r} "
                    f"must match directory {name!r}"
                )

            for entry in child.iterdir():
                if entry.is_symlink():
                    errors.append(
                        f"{child}: must not contain symlinks "
                        "(canonical skill content only)"
                    )
                    break

            target = adapter_target(bucket, name)
            for adapter_root in (
                root / ".claude" / "skills",
                root / ".agents" / "skills",
            ):
                link = adapter_root / name
                if not link.exists():
                    errors.append(f"Missing adapter {link} (run: {_SCRIPT} sync)")
                    continue
                if not resolves_to(link, target):
                    errors.append(
                        f"{link}: expected adapter to {target!r}, "
                        f"got {read_link(link)!r}"
                    )
                    continue
                if not (link / "SKILL.md").is_file():
                    errors.append(f"Broken adapter {link}")

    if errors:
        for msg in errors:
            print(f"::error::{msg}", file=sys.stderr)
        print(f"Skills validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Skills validation passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Maintain agent skills: sync .claude/.agents adapters and validate layout.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="create or refresh client adapter symlinks")
    sub.add_parser(
        "validate",
        help="check SKILL.md frontmatter, layout, and adapters (no writes)",
    )
    sub.add_parser(
        "check-adapters",
        help="verify adapter symlinks only (no writes)",
    )
    args = parser.parse_args()
    root = repo_root()

    if args.command == "sync":
        return cmd_sync(root)
    if args.command == "validate":
        return cmd_validate(root)
    if args.command == "check-adapters":
        return cmd_check_adapters(root)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
