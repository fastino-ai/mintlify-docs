#!/usr/bin/env python3
"""Build the public training/inference OpenAPI from Pioneer's contract."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


HTTP_METHODS = {"delete", "get", "patch", "post", "put"}
INFERENCE_OPERATIONS = (
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/messages"),
    ("GET", "/v1/models"),
    ("POST", "/v1/responses"),
    ("GET", "/inferences"),
    ("GET", "/inferences/{inference_id}"),
    ("GET", "/inferences/{inference_id}/feedback"),
    ("POST", "/inferences/{inference_id}/feedback"),
)
API_FRONTMATTER = re.compile(r'^api:\s+"(GET|POST|PATCH|PUT|DELETE) ([^"]+)"$', re.MULTILINE)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_operations(docs_root: Path) -> list[tuple[str, str]]:
    operations: list[tuple[str, str]] = []
    for path in sorted((docs_root / "api-reference" / "training-jobs").glob("*.mdx")):
        match = API_FRONTMATTER.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"{path}: missing API frontmatter")
        operations.append((match.group(1), match.group(2)))
    return operations


def _route_index(
    manifest: dict[str, object],
) -> dict[tuple[str, str], tuple[str, str]]:
    index: dict[tuple[str, str], tuple[str, str]] = {}
    routes = manifest.get("routes")
    if not isinstance(routes, list):
        raise ValueError("route manifest has no routes list")
    for route in routes:
        if not isinstance(route, dict):
            continue
        source = route.get("source_path")
        target = route.get("target_path")
        methods = route.get("methods")
        if not isinstance(source, str) or not isinstance(target, str) or not isinstance(methods, list):
            continue
        for method in methods:
            if not isinstance(method, str):
                continue
            index[(method, source)] = (source, target)
            index[(method, target)] = (source, target)
    return index


def _operation(
    source_spec: dict[str, object],
    method: str,
    source_path: str,
    target_path: str,
) -> dict[str, object]:
    paths = source_spec.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Pioneer OpenAPI has no paths object")
    for candidate in (target_path, source_path):
        path_item = paths.get(candidate)
        if isinstance(path_item, dict):
            value = path_item.get(method.lower())
            if isinstance(value, dict):
                return copy.deepcopy(value)
    raise ValueError(f"Pioneer OpenAPI is missing {method} {source_path} (target {target_path})")


def _refs(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str):
            yield ref
        for nested in value.values():
            yield from _refs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _refs(nested)


def _referenced_components(
    source_spec: dict[str, object],
    paths: dict[str, object],
) -> dict[str, object]:
    source_components = source_spec.get("components")
    if not isinstance(source_components, dict):
        return {}
    selected: dict[str, dict[str, object]] = {}
    pending = list(_refs(paths))
    seen: set[str] = set()
    while pending:
        ref = pending.pop()
        if ref in seen or not ref.startswith("#/components/"):
            continue
        seen.add(ref)
        parts = ref.split("/")
        if len(parts) != 4:
            raise ValueError(f"unsupported component reference: {ref}")
        _, _, category, name = parts
        category_values = source_components.get(category)
        if not isinstance(category_values, dict) or name not in category_values:
            raise ValueError(f"missing OpenAPI component: {ref}")
        component = copy.deepcopy(category_values[name])
        selected.setdefault(category, {})[name] = component
        pending.extend(_refs(component))
    security_schemes = source_components.get("securitySchemes")
    if isinstance(security_schemes, dict):
        selected["securitySchemes"] = copy.deepcopy(security_schemes)
    return selected


def build_spec(docs_root: Path, pioneer_root: Path) -> dict[str, object]:
    """Return the curated public spec using Pioneer's generated contracts."""
    source_spec = _load_json(pioneer_root / "api" / "openapi.json")
    manifest = _load_json(pioneer_root / "docs" / "route_manifest.json")
    route_index = _route_index(manifest)
    operations = [*_training_operations(docs_root), *INFERENCE_OPERATIONS]
    paths: dict[str, dict[str, object]] = {}
    for method, documented_path in operations:
        source_path, target_path = route_index.get(
            (method, documented_path),
            (documented_path, documented_path),
        )
        paths.setdefault(target_path, {})[method.lower()] = _operation(
            source_spec,
            method,
            source_path,
            target_path,
        )

    source_info = source_spec.get("info")
    version = source_info.get("version", "1.0.0") if isinstance(source_info, dict) else "1.0.0"
    result: dict[str, object] = {
        "openapi": source_spec.get("openapi", "3.1.0"),
        "info": {
            "title": "Fastino Training and Inference API",
            "version": version,
            "description": (
                "Public Fastino training and inference operations documented at "
                "https://docs.fastino.ai."
            ),
        },
        "servers": [{"url": "https://api.fastino.ai", "description": "Production"}],
        "paths": dict(sorted(paths.items())),
        "components": _referenced_components(source_spec, paths),
    }
    security = source_spec.get("security")
    if isinstance(security, list):
        result["security"] = copy.deepcopy(security)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pioneer-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    docs_root = Path(__file__).resolve().parents[1]
    destination = docs_root / "openapi.json"
    rendered = json.dumps(build_spec(docs_root, args.pioneer_root.resolve()), indent=2) + "\n"
    if args.check:
        if destination.read_text(encoding="utf-8") != rendered:
            print("openapi.json is stale; run scripts/sync_api_openapi.py", file=sys.stderr)
            return 1
        return 0
    destination.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
