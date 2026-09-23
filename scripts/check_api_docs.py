#!/usr/bin/env python3
"""Check public API docs against the curated and authoritative contracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sync_api_openapi import INFERENCE_OPERATIONS, build_spec


HTTP_METHODS = {"delete", "get", "patch", "post", "put"}
LOCALES = {"cn", "de", "es", "fr"}
ACTIVE_LEGACY_INFERENCE = (
    re.compile(r"https://api\.fastino\.ai/inference(?:\b|[/?#])"),
    re.compile(r"\|\s*`POST`\s*\|\s*`/inference`\s*\|"),
)


def _operations(spec: dict[str, object]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return result
    for path, path_item in paths.items():
        if not isinstance(path, str) or not isinstance(path_item, dict):
            continue
        result.update(
            (method.upper(), path)
            for method in path_item
            if method.lower() in HTTP_METHODS
        )
    return result


def _english_docs(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.mdx")
        if path.relative_to(root).parts[0] not in LOCALES
    ]


def _line_findings(root: Path, path: Path, text: str) -> list[str]:
    findings: list[str] = []
    relative = path.relative_to(root)
    for number, line in enumerate(text.splitlines(), start=1):
        if "api.pioneer.ai" in line and not (
            relative == Path("api-reference/overview.mdx")
            and "update its base URL" in line
        ):
            findings.append(f"{relative}:{number}: stale api.pioneer.ai origin")
        if "PIONEER_API_KEY" in line and not (
            (relative == Path("api-reference/overview.mdx") and "Fastino CLI" in line)
            or relative == Path("CLI-Installation.mdx")
        ):
            findings.append(f"{relative}:{number}: stale PIONEER_API_KEY name")
        if "/v1/v1/" in line and not (
            relative == Path("api-reference/overview.mdx") and "not" in line
        ):
            findings.append(f"{relative}:{number}: doubled /v1 prefix")
        if "/v1/completions" in line:
            findings.append(f"{relative}:{number}: removed /v1/completions route")
        if "/v1/felix/evaluations" in line:
            findings.append(f"{relative}:{number}: removed legacy evaluation route")
        if any(pattern.search(line) for pattern in ACTIVE_LEGACY_INFERENCE):
            findings.append(f"{relative}:{number}: active removed POST /inference route")
        if "POST /inference" in line and not any(
            word in line.lower() for word in ("legacy", "migrat", "removed")
        ):
            findings.append(f"{relative}:{number}: active removed POST /inference text")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pioneer-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = json.loads((root / "openapi.json").read_text(encoding="utf-8"))
    generated = build_spec(root, args.pioneer_root.resolve())
    findings: list[str] = []

    actual_operations = _operations(destination)
    expected_operations = _operations(generated)
    for operation in sorted(expected_operations - actual_operations):
        findings.append(f"openapi.json missing {operation[0]} {operation[1]}")
    for operation in sorted(actual_operations - expected_operations):
        findings.append(f"openapi.json has undocumented {operation[0]} {operation[1]}")
    if destination != generated:
        findings.append("openapi.json content is stale; run scripts/sync_api_openapi.py")

    docs = _english_docs(root)
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    for method, path in INFERENCE_OPERATIONS:
        visible_path = path.replace("{inference_id}", ":id")
        if path not in corpus and visible_path not in corpus:
            findings.append(f"English docs do not mention {method} {path}")
    for path in docs:
        findings.extend(_line_findings(root, path, path.read_text(encoding="utf-8")))

    if findings:
        print("\n".join(findings))
        return 1
    print(f"API docs parity passed: {len(actual_operations)} curated operations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
