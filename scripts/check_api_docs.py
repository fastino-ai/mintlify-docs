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
FASTINO_URL = re.compile(r"https://api\.fastino\.ai([^\s\"'`<\\]+)")
METHOD_PATH = re.compile(r"\b(GET|POST|PATCH|PUT|DELETE)\s+(/[^`\s\"'|,)]+)")
RoutePatterns = dict[str, list[re.Pattern[str]]]


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


def _documentation_files(root: Path, *, include_locales: bool) -> list[Path]:
    return [
        path
        for path in root.rglob("*.mdx")
        if include_locales or path.relative_to(root).parts[0] not in LOCALES
    ]


def _route_patterns(manifest: dict[str, object]) -> RoutePatterns:
    routes = manifest.get("routes")
    if not isinstance(routes, list):
        raise TypeError("route manifest has no routes list")
    patterns: RoutePatterns = {}
    for route in routes:
        if not isinstance(route, dict):
            continue
        target = route.get("target_path")
        methods = route.get("methods")
        environments = route.get("environments")
        if (
            not isinstance(target, str)
            or not isinstance(methods, list)
            or route.get("classification") == "tombstone"
            or not isinstance(environments, list)
            or "prod" not in environments
        ):
            continue
        expression = re.escape(target)
        expression = re.sub(r"\\\{[^}]+\\\}", r"[^/?#]+", expression)
        pattern = re.compile(f"^{expression}/?$")
        for method in methods:
            if isinstance(method, str):
                patterns.setdefault(method, []).append(pattern)
    return patterns


def _path_matches_route(
    url_path: str,
    patterns: RoutePatterns,
    method: str | None = None,
) -> bool:
    path = url_path.split("?", maxsplit=1)[0].rstrip(".,)")
    if "$" in path or path in {"", "/v1"}:
        return True
    concrete_path = re.sub(r"(\{[^}]+\}|YOUR_[A-Z_]+|[A-Z_]{2,}|:[a-z_]+)", "VALUE", path)
    candidates = patterns.get(method, []) if method is not None else [
        pattern for method_patterns in patterns.values() for pattern in method_patterns
    ]
    if path.endswith("/*"):
        escaped_prefix = f"^{re.escape(path[:-1])}"
        return any(pattern.pattern.startswith(escaped_prefix) for pattern in candidates)
    return any(
        pattern.fullmatch(concrete_path) or pattern.fullmatch(path)
        for pattern in candidates
    )


def _is_migration_line(line: str) -> bool:
    return any(
        word in line.lower()
        for word in (
            "legacy",
            "migrat",
            "removed",
            "supprim",
            "elimin",
            "entfernt",
            "旧版",
            "移除",
        )
    )


def _line_findings(
    root: Path,
    path: Path,
    text: str,
    route_patterns: RoutePatterns,
) -> list[str]:
    findings: list[str] = []
    relative = path.relative_to(root)
    is_overview = relative.as_posix().endswith("api-reference/overview.mdx")
    is_cli = relative.name == "CLI-Installation.mdx"
    for number, line in enumerate(text.splitlines(), start=1):
        if "api.pioneer.ai" in line and not (
            is_overview and "api.fastino.ai" in line
        ):
            findings.append(f"{relative}:{number}: stale api.pioneer.ai origin")
        if "PIONEER_API_KEY" in line and not (
            is_overview or is_cli
        ):
            findings.append(f"{relative}:{number}: stale PIONEER_API_KEY name")
        if "/v1/v1/" in line and not is_overview:
            findings.append(f"{relative}:{number}: doubled /v1 prefix")
        if "/v1/completions" in line and not _is_migration_line(line):
            findings.append(f"{relative}:{number}: removed /v1/completions route")
        if "/v1/embeddings" in line and not _is_migration_line(line):
            findings.append(f"{relative}:{number}: removed /v1/embeddings route")
        if "/v1/felix" in line and not _is_migration_line(line):
            findings.append(f"{relative}:{number}: retired /v1/felix route prefix")
        if "pio_sk_" in line and not _is_migration_line(line):
            findings.append(f"{relative}:{number}: retired pio_sk_ API key prefix")
        if "/v1/felix/evaluations" in line:
            findings.append(f"{relative}:{number}: removed legacy evaluation route")
        if (
            relative == Path("api-reference/inference/anthropic-compatible.mdx")
            and "base_url" in line
            and "https://api.fastino.ai/v1" in line
        ):
            findings.append(
                f"{relative}:{number}: Anthropic SDK base URL would double the /v1 prefix"
            )
        if any(pattern.search(line) for pattern in ACTIVE_LEGACY_INFERENCE):
            findings.append(f"{relative}:{number}: active removed POST /inference route")
        if "POST /inference" in line and not _is_migration_line(line):
            findings.append(f"{relative}:{number}: active removed POST /inference text")
        for match in FASTINO_URL.finditer(line):
            if not _path_matches_route(match.group(1), route_patterns):
                findings.append(
                    f"{relative}:{number}: Fastino URL is absent from route manifest: "
                    f"{match.group(0)}"
                )
        if not _is_migration_line(line):
            for method, documented_path in METHOD_PATH.findall(line):
                if not _path_matches_route(documented_path, route_patterns, method):
                    findings.append(
                        f"{relative}:{number}: documented operation is absent from the "
                        f"production route manifest: {method} {documented_path}"
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pioneer-root", type=Path, required=True)
    parser.add_argument("--include-locales", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = json.loads((root / "openapi.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (args.pioneer_root.resolve() / "docs" / "route_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    generated = build_spec(root, args.pioneer_root.resolve())
    route_patterns = _route_patterns(manifest)
    findings: list[str] = []

    actual_operations = _operations(destination)
    expected_operations = _operations(generated)
    for operation in sorted(expected_operations - actual_operations):
        findings.append(f"openapi.json missing {operation[0]} {operation[1]}")
    for operation in sorted(actual_operations - expected_operations):
        findings.append(f"openapi.json has undocumented {operation[0]} {operation[1]}")
    if destination != generated:
        findings.append("openapi.json content is stale; run scripts/sync_api_openapi.py")
    serialized_destination = json.dumps(destination)
    for retired_contract in ('"felix"', "/felix/training-jobs", "pio_sk_"):
        if retired_contract in serialized_destination:
            findings.append(
                f"openapi.json contains retired public contract text: {retired_contract}"
            )

    docs = _documentation_files(root, include_locales=args.include_locales)
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    for method, path in INFERENCE_OPERATIONS:
        visible_path = path.replace("{inference_id}", ":id")
        if path not in corpus and visible_path not in corpus:
            findings.append(f"English docs do not mention {method} {path}")
    for path in docs:
        findings.extend(
            _line_findings(
                root,
                path,
                path.read_text(encoding="utf-8"),
                route_patterns,
            )
        )

    if findings:
        print("\n".join(findings))
        return 1
    print(f"API docs parity passed: {len(actual_operations)} curated operations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
