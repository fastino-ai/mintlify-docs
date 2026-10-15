"""Check public API docs against Pioneer's route contract."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from generate_localized_llms import localized_index_findings

HTTP_METHODS = {"delete", "get", "patch", "post", "put"}
INFERENCE_OPERATIONS = {
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/systemone"),
    ("POST", "/v1/gliner-2"),
    ("POST", "/v1/gliner-2/async"),
    ("GET", "/v1/gliner-2/jobs/{job_id}"),
    ("POST", "/v1/messages"),
    ("GET", "/v1/models"),
    ("POST", "/v1/responses"),
    ("GET", "/v1/inferences"),
    ("GET", "/v1/inferences/{inference_id}"),
    ("GET", "/v1/inferences/{inference_id}/feedback"),
    ("POST", "/v1/inferences/{inference_id}/feedback"),
}
TRAINING_OPERATIONS = {
    ("GET", "/v1/base-models"),
    ("GET", "/v1/datasets"),
    ("POST", "/v1/datasets/upload/process"),
    ("POST", "/v1/datasets/upload/url"),
    ("DELETE", "/v1/datasets/{name}"),
    ("GET", "/v1/datasets/{name}"),
    ("GET", "/v1/training/jobs"),
    ("POST", "/v1/training/jobs"),
    ("POST", "/v1/training/jobs/create-model"),
    ("DELETE", "/v1/training/jobs/{job_id}"),
    ("GET", "/v1/training/jobs/{job_id}"),
    ("PATCH", "/v1/training/jobs/{job_id}"),
    ("GET", "/v1/training/jobs/{job_id}/billing"),
    ("GET", "/v1/training/jobs/{job_id}/checkpoints/{kind}/{name}"),
    ("DELETE", "/v1/training/jobs/{job_id}/checkpoints/{kind}/{name}"),
    ("POST", "/v1/training/jobs/{job_id}/checkpoints/sampler_weights/{name}/deploy"),
    ("POST", "/v1/training/jobs/{job_id}/checkpoints/{kind}/{name}/publish"),
    ("DELETE", "/v1/training/jobs/{job_id}/checkpoints/{kind}/{name}/publish"),
    ("PUT", "/v1/training/jobs/{job_id}/checkpoints/{kind}/{name}/ttl"),
    ("GET", "/v1/training/jobs/{job_id}/checkpoints"),
    ("POST", "/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/deploy"),
    ("GET", "/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/download"),
    ("GET", "/v1/training/jobs/{job_id}/deployments"),
    ("GET", "/v1/training/jobs/{job_id}/download"),
    ("POST", "/v1/training/jobs/{job_id}/forward"),
    ("POST", "/v1/training/jobs/{job_id}/forward-backward"),
    ("GET", "/v1/training/jobs/{job_id}/futures/{future_id}"),
    ("POST", "/v1/training/jobs/{job_id}/futures/{future_id}/cancel"),
    ("POST", "/v1/training/jobs/{job_id}/load-weights"),
    ("GET", "/v1/training/jobs/{job_id}/logs"),
    ("POST", "/v1/training/jobs/{job_id}/optim-step"),
    ("PATCH", "/v1/training/jobs/{job_id}/model-name"),
    ("POST", "/v1/training/jobs/{job_id}/push-to-hub"),
    ("POST", "/v1/training/jobs/{job_id}/sample"),
    ("POST", "/v1/training/jobs/{job_id}/save-weights"),
    ("POST", "/v1/training/jobs/{job_id}/save-weights-for-sampler"),
    ("POST", "/v1/training/jobs/{job_id}/stop"),
    ("POST", "/v1/training/jobs/{job_id}/sync"),
    ("POST", "/v1/training/jobs/{job_id}/terminate"),
    ("GET", "/v1/training/usage"),
}
EXPECTED_OPERATIONS = INFERENCE_OPERATIONS | TRAINING_OPERATIONS
LOCALES = {"cn", "de", "es", "fr"}
EXPECTED_SKILLS = {
    "fastino-datasets",
    "fastino-fine-tune",
    "fastino-glide",
    "fastino-gliner",
    "fastino-inference",
}
ACTIVE_LEGACY_INFERENCE = (
    re.compile(r"https://api\.fastino\.ai/inference(?:\b|[/?#])"),
    re.compile(r"\|\s*`POST`\s*\|\s*`/inference`\s*\|"),
)
FASTINO_URL = re.compile(r"https://api\.fastino\.ai([^\s\"'`<\\]+)")
METHOD_PATH = re.compile(r"\b(GET|POST|PATCH|PUT|DELETE)\s+(/[A-Za-z0-9_{}\-./:*?=&%]+)")
LLMS_PAGE_URL = re.compile(r"https://docs\.fastino\.ai/([^\s)]+)\.md")
API_BINDING = re.compile(
    r'^api:\s*"(GET|POST|PATCH|PUT|DELETE) ([^"]+)"\s*$',
    re.MULTILINE,
)
FENCED_BLOCK = re.compile(
    r"^[ \t]*```(?P<language>[\w-]*)[^\n]*\n(?P<body>.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
PARAM_FIELD = re.compile(r'<ParamField\s+(header|path|query|body)="([^"]+)"')
CODE_SPAN = re.compile(r"`([^`\n]+)`")
REQUIRED_AGENT_RESOURCES = {
    "https://docs.fastino.ai/.well-known/agent-skills/index.json",
    "https://docs.fastino.ai/llms-full.txt",
    "https://docs.fastino.ai/openapi.json",
}
RoutePatterns = dict[str, list[re.Pattern[str]]]
LOCALE_PARITY_SCOPES = (
    "concepts/datasets.mdx",
    "training.mdx",
    "training/",
    "api-reference/training/",
    "api-reference/training-jobs/",
)


@dataclass(frozen=True)
class TinkerSdkContract:
    pin: str
    base_path: str
    private_child_marker: str
    unsupported_calls: frozenset[str]


TINKER_SDK = TinkerSdkContract(
    pin="tinker==0.32.0",
    base_path="/v1/training/compat/tinker",
    private_child_marker="/api/v1/",
    unsupported_calls=frozenset(
        {"retrieve_futures", "save_weights_external", "external_weights_urls"}
    ),
)
SDK_BASE_PATHS = {TINKER_SDK.base_path}


@dataclass(frozen=True)
class LocaleSkeleton:
    api_binding: str | None
    fences: tuple[tuple[str, str], ...]
    code_spans: frozenset[str]
    param_fields: tuple[tuple[str, str], ...]


def _visible_navigation_pages(node: object) -> set[str]:
    if isinstance(node, str):
        return {node}
    if not isinstance(node, dict) or node.get("hidden") is True:
        return set()
    pages: set[str] = set()
    for key in ("groups", "pages"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                pages.update(_visible_navigation_pages(child))
    return pages


def _llms_findings(root: Path) -> list[str]:
    config = json.loads((root / "docs.json").read_text(encoding="utf-8"))
    navigation = config.get("navigation")
    languages = navigation.get("languages") if isinstance(navigation, dict) else None
    english = next(
        (
            language
            for language in languages or []
            if isinstance(language, dict) and language.get("language") == "en"
        ),
        None,
    )
    if not isinstance(english, dict):
        return ["docs.json has no English navigation"]

    visible_pages: set[str] = set()
    divisions = english.get("tabs")
    if not isinstance(divisions, list):
        divisions = english.get("groups")
    if isinstance(divisions, list):
        for division in divisions:
            visible_pages.update(_visible_navigation_pages(division))

    llms_text = (root / "llms.txt").read_text(encoding="utf-8")
    indexed_pages = set(LLMS_PAGE_URL.findall(llms_text))
    findings = [
        f"llms.txt is missing visible English page: {page}"
        for page in sorted(visible_pages - indexed_pages)
    ]
    findings.extend(
        f"llms.txt is missing agent resource: {url}"
        for url in sorted(REQUIRED_AGENT_RESOURCES)
        if f"]({url})" not in llms_text
    )
    findings.extend(localized_index_findings(root))
    return findings


def _api_reference_findings(
    root: Path,
    destination: dict[str, object],
) -> list[str]:
    """Return OpenAPI binding and API-reference navigation findings."""
    findings: list[str] = []
    actual_operations = _operations(destination)

    bindings: dict[tuple[str, str], list[str]] = {}
    for path in _documentation_files(root, include_locales=False):
        match = API_BINDING.search(path.read_text(encoding="utf-8"))
        if match is not None:
            bindings.setdefault((match.group(1), match.group(2)), []).append(
                path.relative_to(root).with_suffix("").as_posix()
            )
    for operation in sorted(actual_operations):
        pages = bindings.get(operation, [])
        if len(pages) != 1:
            findings.append(
                f"{operation[0]} {operation[1]} must have exactly one API binding, "
                f"found {pages}"
            )
    findings.extend(_stale_binding_findings(bindings, actual_operations))

    config = json.loads((root / "docs.json").read_text(encoding="utf-8"))
    languages = config.get("navigation", {}).get("languages", [])
    english = next(
        (
            language
            for language in languages
            if isinstance(language, dict) and language.get("language") == "en"
        ),
        None,
    )
    tabs = english.get("tabs") if isinstance(english, dict) else None
    if not isinstance(tabs, list):
        return [*findings, "English navigation has no tabs"]
    reference = next(
        (
            tab
            for tab in tabs
            if isinstance(tab, dict) and tab.get("tab") == "API Reference"
        ),
        None,
    )
    groups = reference.get("pages") if isinstance(reference, dict) else None
    if not isinstance(groups, list):
        return [*findings, "English navigation has no API Reference tab"]
    names = [
        group.get("group")
        for group in groups
        if isinstance(group, dict) and group.get("hidden") is not True
    ]
    if names != ["Inference", "Training"]:
        findings.append(
            "API Reference must contain exactly Inference and Training groups, "
            f"found {names}"
        )
    visible_reference_pages = _visible_navigation_pages(reference)
    for operation, pages in sorted(bindings.items()):
        if operation not in actual_operations:
            continue
        if pages[0] not in visible_reference_pages:
            findings.append(
                f"API binding is outside the API Reference tab: {pages[0]}"
            )
    return findings


def _journey_order_findings(root: Path) -> list[str]:
    """Require task guides to precede their operation contracts for agents."""
    text = (root / "llms.txt").read_text(encoding="utf-8")
    pairs = (
        ("inference/systemone.md", "api-reference/inference/systemone.md"),
        (
            "inference/chat-completions.md",
            "api-reference/inference/chat-completions.md",
        ),
        ("training.md", "api-reference/training-jobs/create.md"),
        ("concepts/datasets.md", "api-reference/datasets/upload-url.md"),
    )
    findings: list[str] = []
    for guide, contract in pairs:
        guide_index = text.find(guide)
        contract_index = text.find(contract)
        if guide_index < 0 or contract_index < 0:
            findings.append(
                f"llms.txt must index both guide and contract: {guide}, {contract}"
            )
        elif guide_index > contract_index:
            findings.append(
                f"llms.txt must place the guide before its contract: {guide}"
            )
    return findings


def _skill_findings(root: Path) -> list[str]:
    findings: list[str] = []
    skills_root = root / ".mintlify" / "skills"
    actual_skills = (
        {path.name for path in skills_root.iterdir() if path.is_dir()}
        if skills_root.is_dir()
        else set()
    )
    if actual_skills != EXPECTED_SKILLS:
        findings.append(
            "Mintlify skills differ from the expected public set: "
            f"expected {sorted(EXPECTED_SKILLS)}, found {sorted(actual_skills)}"
        )
    for skill_name in sorted(actual_skills):
        skill_path = skills_root / skill_name / "SKILL.md"
        if not skill_path.is_file():
            findings.append(f"{skill_path.relative_to(root)} is missing")
            continue
        match = re.search(
            r"\A---\s*\n.*?^name:\s*([^\s]+)\s*$.*?^---\s*$",
            skill_path.read_text(encoding="utf-8"),
            flags=re.DOTALL | re.MULTILINE,
        )
        if match is None:
            findings.append(f"{skill_path.relative_to(root)} has invalid frontmatter")
        elif match.group(1) != skill_name:
            findings.append(
                f"{skill_path.relative_to(root)} name is {match.group(1)!r}, "
                f"expected {skill_name!r}"
            )
    for path in root.rglob("*.mdx"):
        relative = path.relative_to(root)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "npx skills add https://api.fastino.ai" in line:
                findings.append(
                    f"{relative}:{number}: stale API-hosted skill install"
                )
            if "https://api.fastino.ai/.well-known/agent-skills/" in line:
                findings.append(
                    f"{relative}:{number}: stale API-hosted skill source"
                )
    return findings


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


def _stale_binding_findings(
    bindings: dict[tuple[str, str], list[str]],
    actual_operations: set[tuple[str, str]],
) -> list[str]:
    return [
        f"{method} {path} has API bindings but is absent from openapi.json: {pages}"
        for (method, path), pages in sorted(bindings.items())
        if (method, path) not in actual_operations
    ]


def _openapi_patterns(spec: dict[str, object]) -> RoutePatterns:
    patterns: RoutePatterns = {}
    for method, path in _operations(spec):
        expression = re.escape(path)
        expression = re.sub(r"\\\{[^}]+\\\}", r"[^/?#]+", expression)
        patterns.setdefault(method, []).append(re.compile(f"^{expression}/?$"))
    return patterns


def _claim_files(root: Path) -> list[Path]:
    paths = list(root.rglob("*.mdx"))
    agents = root / "AGENTS.md"
    if agents.is_file():
        paths.append(agents)
    skills = root / ".mintlify" / "skills"
    if skills.is_dir():
        paths.extend(skills.glob("*/SKILL.md"))
    return sorted(set(paths))


def _sdk_claim_findings(root: Path, files: list[Path]) -> list[str]:
    findings: list[str] = []
    pin_pattern = re.compile(r"\btinker==[0-9A-Za-z.-]+")
    child_path_pattern = re.compile(re.escape(TINKER_SDK.base_path) + r"/")
    for path in files:
        relative = path.relative_to(root)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in pin_pattern.finditer(line):
                if match.group(0) != TINKER_SDK.pin:
                    findings.append(
                        f"{relative}:{number}: SDK pin is {match.group(0)}, "
                        f"expected {TINKER_SDK.pin}"
                    )
            if (
                TINKER_SDK.private_child_marker in line
                and (
                    "tinker" in relative.as_posix().lower()
                    or "tinker" in line.lower()
                )
            ):
                findings.append(
                    f"{relative}:{number}: compatibility child route is not public"
                )
            elif child_path_pattern.search(line):
                findings.append(
                    f"{relative}:{number}: compatibility child route is not public"
                )
    return findings


def _skill_route_findings(
    root: Path,
    destination: dict[str, object],
) -> list[str]:
    findings: list[str] = []
    patterns = _openapi_patterns(destination)
    skills = root / ".mintlify" / "skills"
    if not skills.is_dir():
        return findings
    for path in sorted(skills.glob("*/SKILL.md")):
        relative = path.relative_to(root)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for method, documented_path in METHOD_PATH.findall(line):
                normalized_path = documented_path.split("?", maxsplit=1)[0].rstrip(".,)")
                if (
                    normalized_path in SDK_BASE_PATHS
                    or not _path_matches_route(documented_path, patterns, method)
                ):
                    findings.append(
                        f"{relative}:{number}: {method} {normalized_path} "
                        "is not a public OpenAPI operation"
                    )
    return findings


def _locale_skeleton(text: str) -> LocaleSkeleton:
    binding = API_BINDING.search(text)
    without_fences = FENCED_BLOCK.sub("", text)
    return LocaleSkeleton(
        api_binding=(
            f"{binding.group(1)} {binding.group(2)}" if binding is not None else None
        ),
        fences=tuple(
            (match.group("language"), match.group("body"))
            for match in FENCED_BLOCK.finditer(text)
        ),
        code_spans=frozenset(CODE_SPAN.findall(without_fences)),
        param_fields=tuple(PARAM_FIELD.findall(text)),
    )


def _is_locale_parity_page(relative: Path) -> bool:
    value = relative.as_posix()
    return any(
        value == scope or (scope.endswith("/") and value.startswith(scope))
        for scope in LOCALE_PARITY_SCOPES
    )


def _locale_skeleton_findings(root: Path) -> list[str]:
    findings: list[str] = []
    english_pages = [
        path
        for path in _documentation_files(root, include_locales=False)
        if _is_locale_parity_page(path.relative_to(root))
    ]
    for english_path in sorted(english_pages):
        relative = english_path.relative_to(root)
        expected = _locale_skeleton(english_path.read_text(encoding="utf-8"))
        for locale in sorted(LOCALES):
            locale_path = root / locale / relative
            if not locale_path.is_file():
                findings.append(f"{locale_path.relative_to(root)}: locale page is missing")
                continue
            actual = _locale_skeleton(locale_path.read_text(encoding="utf-8"))
            differences: list[str] = []
            if actual.api_binding != expected.api_binding:
                differences.append("API binding")
            if actual.fences != expected.fences:
                differences.append("code fences")
            if actual.code_spans != expected.code_spans:
                differences.append("code spans")
            if actual.param_fields != expected.param_fields:
                differences.append("ParamFields")
            if differences:
                findings.append(
                    f"{locale_path.relative_to(root)}: locale skeleton differs from "
                    f"English in {', '.join(differences)}"
                )
    return findings


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
    if "$" in path or path in {"", "/v1"} | SDK_BASE_PATHS:
        return True
    concrete_path = re.sub(r"(\{[^}]+\}|YOUR_[A-Z_]+|[A-Z_]{2,}|:[a-z_]+)", "VALUE", path)
    candidates = patterns.get(method, []) if method is not None else [
        pattern for method_patterns in patterns.values() for pattern in method_patterns
    ]
    if path.endswith("/*"):
        escaped_prefix = f"^{re.escape(path[:-1])}"
        return any(pattern.pattern.startswith(escaped_prefix) for pattern in candidates)
    return any(
        pattern.fullmatch(candidate)
        for pattern in candidates
        for candidate in {path, concrete_path}
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
            "eingestellt",
            "außer betrieb",
            "retir",
            "heredad",
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


def local_docs_findings(
    root: Path,
    destination: dict[str, object],
) -> list[str]:
    """Return checks that need only this documentation repository."""
    findings = _llms_findings(root)
    findings.extend(_skill_findings(root))
    findings.extend(_api_reference_findings(root, destination))
    findings.extend(_journey_order_findings(root))
    claim_files = _claim_files(root)
    findings.extend(_sdk_claim_findings(root, claim_files))
    findings.extend(_skill_route_findings(root, destination))
    findings.extend(_locale_skeleton_findings(root))

    serialized_destination = json.dumps(destination)
    for retired_contract in ('"felix"', "/felix/training-jobs", "pio_sk_"):
        if retired_contract in serialized_destination:
            findings.append(
                f"openapi.json contains retired public contract text: {retired_contract}"
            )

    docs = _documentation_files(root, include_locales=False)
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    for method, path in INFERENCE_OPERATIONS:
        visible_path = path.replace("{inference_id}", ":id")
        if path not in corpus and visible_path not in corpus:
            findings.append(f"English docs do not mention {method} {path}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pioneer-root", type=Path, required=True)
    parser.add_argument("--include-locales", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination_path = root / "openapi.json"
    destination = json.loads(destination_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (args.pioneer_root.resolve() / "docs" / "route_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    route_patterns = _route_patterns(manifest)
    actual_operations = _operations(destination)
    findings = local_docs_findings(root, destination)
    docs = _documentation_files(root, include_locales=args.include_locales)
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
    print(f"API docs parity passed: {len(actual_operations)} published operations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
