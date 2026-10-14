from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn

import agent_harness as HARNESS

CENSUS_PATH = HARNESS.ROOT / "scripts" / "training_snippets_census.json"
SECRET_TOKEN = re.compile(r"fast_sk_[A-Za-z0-9_-]+")
JOB_ID_IN_PATH = re.compile(
    r"/v1/training/jobs/(?:JOB_ID|\$TRAINING_JOB_ID|\$env:TRAINING_JOB_ID)"
)
DATASET_IN_PATH = re.compile(r"/v1/datasets/[^/?]+")
SAFE_GET_TEMPLATES = frozenset(
    {
        "/v1/base-models",
        "/v1/training/jobs/{job_id}",
        "/v1/training/jobs/{job_id}/logs",
        "/v1/training/jobs/{job_id}/checkpoints",
        "/v1/datasets/{dataset_name}",
    }
)
JOB_GET_TEMPLATES = frozenset(
    {
        "/v1/training/jobs/{job_id}",
        "/v1/training/jobs/{job_id}/logs",
        "/v1/training/jobs/{job_id}/checkpoints",
    }
)
INTERACTIVE_JOB_PATH = re.compile(
    r"/v1/training/(?:usage|jobs/(?:create-model|[^/]+/(?:forward|forward-backward|optim-step"
    r"|save-weights|save-weights-for-sampler|load-weights|sample|stop|futures/.+"
    r"|checkpoints(?:/.+)?)))"
)
REST_PYTHON_MARKERS = ("import requests", "call(", "wait(", "loss_fn_inputs")
INCOMPLETE_PYTHON_NAMES = frozenset(
    {
        "advantages",
        "model_input",
        "parse_answer",
        "resume",
        "sample_result",
        "service_client",
        "target_tokens",
        "training_client",
    }
)


@dataclass(frozen=True)
class FencedSnippet:
    page: str
    ordinal: int
    language: str
    body: str
    body_sha256: str


@dataclass(frozen=True)
class Classification:
    kind: str
    rule: str
    coverage: str
    risk: str
    method: str | None = None
    path: str | None = None
    mapped_command: str | None = None


@dataclass(frozen=True)
class CensusEntry:
    page: str
    ordinal: int
    language: str
    kind: str
    rule: str
    coverage: str
    risk: str
    body_sha256: str


@dataclass(frozen=True)
class LivePlan:
    action: str
    method: str | None
    path: str | None
    payload: object | None
    detail: str
    snippet: FencedSnippet
    classification: Classification


def redact_secrets(text: str) -> str:
    return SECRET_TOKEN.sub("fast_sk_[REDACTED]", text)


def fail_redacted(message: str) -> NoReturn:
    HARNESS.fail(redact_secrets(message))


def collect_pages(root: Path | None = None) -> tuple[str, ...]:
    del root
    return tuple(
        f"{page}.mdx" for page in HARNESS._ordered_navigation_pages(HARNESS.TRAINING_NAVIGATION)
    )


def collect_snippets(root: Path | None = None) -> tuple[FencedSnippet, ...]:
    base = root or HARNESS.ROOT
    snippets: list[FencedSnippet] = []
    for page in collect_pages(base):
        text = (base / page).read_text(encoding="utf-8")
        for ordinal, match in enumerate(HARNESS.FENCED_BLOCK.finditer(text)):
            body = match.group("body")
            snippets.append(
                FencedSnippet(
                    page=page,
                    ordinal=ordinal,
                    language=match.group("language") or "",
                    body=body,
                    body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                )
            )
    return tuple(snippets)


def match_safe_template(path: str) -> str | None:
    concrete = path.split("?", maxsplit=1)[0].rstrip(".,)")
    for template in SAFE_GET_TEMPLATES:
        expression = re.sub(r"\\\{[^}]+\\\}", r"[^/]+", re.escape(template))
        if re.fullmatch(expression, concrete):
            return template
    return None


def parse_http(language: str, body: str) -> tuple[str, str] | None:
    if language.startswith("powershell"):
        uri = re.search(r'-Uri\s+"([^"]+)"', body)
        if uri is None:
            return None
        url_match = HARNESS.API_URL.search(uri.group(1))
        if url_match is None:
            return None
        method_match = re.search(r"-Method\s+([A-Za-z]+)", body)
        method = method_match.group(1).upper() if method_match else "GET"
        return method, url_match.group("path")
    if "curl" not in body:
        return None
    urls = list(HARNESS.API_URL.finditer(body))
    if not urls:
        return None
    path = urls[-1].group("path")
    method_match = re.search(r"-X\s+(GET|POST|PATCH|PUT|DELETE)", body)
    if method_match:
        method = method_match.group(1)
    elif "--upload-file" in body:
        method = "PUT"
    elif re.search(r"(?:^|\s)(-d|--data|--data-raw|--data-binary)\s", body):
        method = "POST"
    else:
        method = "GET"
    return method, path


def _has_preview_model_type(body: str) -> bool:
    return '"type": "config"' in body and '"model":' in body


def _has_config_create_body(body: str) -> bool:
    return all(
        field in body
        for field in ('"model_name":', '"datasets":', '"base_model":')
    )


def _shell_family(language: str) -> str:
    if language.startswith("powershell"):
        return "powershell"
    return "bash"


def _rule_jsonl(snippet: FencedSnippet) -> Classification | None:
    if snippet.language != "jsonl":
        return None
    return Classification(kind="jsonl-dataset", rule="jsonl-parse", coverage="static", risk="none")


def _rule_json(snippet: FencedSnippet) -> Classification | None:
    if snippet.language != "json":
        return None
    if _has_preview_model_type(snippet.body):
        return Classification(
            kind="json-request",
            rule="json-preview-model-type",
            coverage="static",
            risk="preview-mismatch",
            mapped_command="training-replay",
        )
    if _has_config_create_body(snippet.body):
        return Classification(
            kind="json-request",
            rule="json-config-create",
            coverage="static",
            risk="mutating",
            mapped_command="training-lifecycle",
        )
    return Classification(kind="json-request", rule="json-parse", coverage="static", risk="none")


def _rule_text(snippet: FencedSnippet) -> Classification | None:
    if snippet.language != "text":
        return None
    return Classification(kind="text-diagram", rule="text-accept", coverage="static", risk="none")


def _is_known_incomplete_python(body: str) -> bool:
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return "continue" in body or "break" in body
    defined: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            defined.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            defined.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return bool((used - defined) & INCOMPLETE_PYTHON_NAMES)


def _rule_python(snippet: FencedSnippet) -> Classification | None:
    if snippet.language != "python":
        return None
    if "def get_last_checkpoint" in snippet.body:
        return Classification(
            kind="python-checkpoint-parser",
            rule="python-checkpoint-exec",
            coverage="static",
            risk="none",
        )
    if "import tinker" in snippet.body or re.search(r"\btinker\.", snippet.body):
        return Classification(
            kind="python-tinker",
            rule="python-tinker-compile",
            coverage="static",
            risk="custom-loop",
            mapped_command="blocker",
        )
    if any(marker in snippet.body for marker in REST_PYTHON_MARKERS):
        return Classification(
            kind="python-rest",
            rule="python-rest-compile",
            coverage="static",
            risk="custom-loop",
            mapped_command="blocker",
        )
    if _is_known_incomplete_python(snippet.body):
        return Classification(
            kind="python-incomplete",
            rule="python-incomplete-fragment",
            coverage="static",
            risk="incomplete",
        )
    return None


def _classify_http(language: str, method: str, path: str, body: str) -> Classification | None:
    family = _shell_family(language)
    if method == "GET" and match_safe_template(path) is not None:
        return Classification(
            kind=f"{family}-safe-get",
            rule="http-get-safe",
            coverage="live",
            risk="none",
            method="GET",
            path=path,
        )
    concrete = path.split("?", maxsplit=1)[0].rstrip("/")
    if method == "POST" and concrete == "/v1/training/jobs":
        return Classification(
            kind=f"{family}-mutating",
            rule="http-post-create",
            coverage="static",
            risk="preview-mismatch" if _has_preview_model_type(body) else "mutating",
            method="POST",
            path=path,
            mapped_command="training-replay",
        )
    if method == "POST" and concrete.endswith("/datasets/upload/url"):
        return Classification(
            kind=f"{family}-mutating",
            rule="http-post-upload",
            coverage="static",
            risk="mutating",
            method="POST",
            path=path,
            mapped_command="blocker",
        )
    if method == "POST" and concrete.endswith("/datasets/upload/process"):
        return Classification(
            kind=f"{family}-mutating",
            rule="http-post-process",
            coverage="static",
            risk="mutating",
            method="POST",
            path=path,
            mapped_command="blocker",
        )
    if method == "POST" and concrete.endswith("/deploy"):
        return Classification(
            kind=f"{family}-mutating",
            rule="http-post-deploy",
            coverage="static",
            risk="mutating",
            method="POST",
            path=path,
            mapped_command="training-lifecycle",
        )
    if method == "POST" and concrete == "/v1/chat/completions":
        return Classification(
            kind=f"{family}-mutating",
            rule="http-post-inference",
            coverage="static",
            risk="mutating",
            method="POST",
            path=path,
            mapped_command="blocker",
        )
    if INTERACTIVE_JOB_PATH.fullmatch(concrete):
        return Classification(
            kind=f"{family}-interactive",
            rule="http-interactive-job",
            coverage="static",
            risk="none" if method == "GET" else "mutating",
            method=method,
            path=path,
            mapped_command="blocker",
        )
    if method == "PUT":
        return Classification(
            kind=f"{family}-mutating",
            rule="http-put-upload",
            coverage="static",
            risk="mutating",
            method="PUT",
            path=path,
            mapped_command="blocker",
        )
    return None


def _rule_http(snippet: FencedSnippet) -> Classification | None:
    if not (
        snippet.language.startswith("bash") or snippet.language.startswith("powershell")
    ):
        return None
    parsed = parse_http(snippet.language, snippet.body)
    if parsed is None:
        return None
    classification = _classify_http(snippet.language, parsed[0], parsed[1], snippet.body)
    if classification is None:
        HARNESS.fail(f"UNCLASSIFIED: {snippet.page}#{snippet.ordinal}")
    return classification


def _rule_put_upload(snippet: FencedSnippet) -> Classification | None:
    if not snippet.language.startswith("bash"):
        return None
    if "--upload-file" not in snippet.body and not re.search(r"-X\s+PUT", snippet.body):
        return None
    return Classification(
        kind="bash-mutating",
        rule="http-put-upload",
        coverage="static",
        risk="mutating",
        method="PUT",
        mapped_command="blocker",
    )


def _rule_shell_install(snippet: FencedSnippet) -> Classification | None:
    body = snippet.body
    if snippet.language.startswith("bash") and (
        "uv add" in body or "uv init" in body or "pip install" in body
    ):
        return Classification(
            kind="bash-install",
            rule="bash-n-install",
            coverage="static",
            risk="custom-loop",
            mapped_command="blocker",
        )
    if snippet.language.startswith("powershell") and (
        "pip install" in body or "py -m venv" in body
    ):
        return Classification(
            kind="powershell-install",
            rule="powershell-shape-install",
            coverage="static",
            risk="custom-loop",
            mapped_command="blocker",
        )
    return None


def _rule_shell_run(snippet: FencedSnippet) -> Classification | None:
    if snippet.language.startswith("bash") and re.search(
        r"^python(?:3)?\s+\S+", snippet.body, re.MULTILINE
    ):
        return Classification(
            kind="bash-run-script",
            rule="bash-n-run",
            coverage="static",
            risk="custom-loop",
            mapped_command="blocker",
        )
    return None


def _rule_shell_env(snippet: FencedSnippet) -> Classification | None:
    if snippet.language.startswith("bash") and (
        "export " in snippet.body or snippet.body.lstrip().startswith("test ")
    ):
        return Classification(
            kind="bash-env",
            rule="bash-n-env",
            coverage="static",
            risk="none",
        )
    if snippet.language.startswith("powershell") and "$env:" in snippet.body:
        return Classification(
            kind="powershell-env",
            rule="powershell-shape-env",
            coverage="static",
            risk="none",
        )
    return None


_RULES: tuple[Callable[[FencedSnippet], Classification | None], ...] = (
    _rule_jsonl,
    _rule_json,
    _rule_text,
    _rule_python,
    _rule_http,
    _rule_put_upload,
    _rule_shell_install,
    _rule_shell_run,
    _rule_shell_env,
)


def classify(snippet: FencedSnippet) -> Classification:
    for rule in _RULES:
        classification = rule(snippet)
        if classification is not None:
            return classification
    HARNESS.fail(f"UNCLASSIFIED: {snippet.page}#{snippet.ordinal}")


def census_entries(
    pages: tuple[str, ...],
    snippets: tuple[FencedSnippet, ...],
) -> tuple[CensusEntry, ...]:
    grouped: dict[str, list[FencedSnippet]] = {page: [] for page in pages}
    for snippet in snippets:
        grouped.setdefault(snippet.page, []).append(snippet)
    entries: list[CensusEntry] = []
    for page in pages:
        page_snippets = grouped.get(page, [])
        if not page_snippets:
            entries.append(
                CensusEntry(
                    page=page,
                    ordinal=0,
                    language="",
                    kind="empty-page",
                    rule="no-fences",
                    coverage="none",
                    risk="none",
                    body_sha256=hashlib.sha256(b"").hexdigest(),
                )
            )
            continue
        for snippet in page_snippets:
            classification = classify(snippet)
            entries.append(
                CensusEntry(
                    page=snippet.page,
                    ordinal=snippet.ordinal,
                    language=snippet.language,
                    kind=classification.kind,
                    rule=classification.rule,
                    coverage=classification.coverage,
                    risk=classification.risk,
                    body_sha256=snippet.body_sha256,
                )
            )
    return tuple(entries)


def load_census(path: Path = CENSUS_PATH) -> tuple[CensusEntry, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        fail_redacted("CENSUS: census file must be a list")
    entries: list[CensusEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            fail_redacted("CENSUS: census entry must be an object")
        entries.append(
            CensusEntry(
                page=str(item["page"]),
                ordinal=int(item["ordinal"]),
                language=str(item["language"]),
                kind=str(item["kind"]),
                rule=str(item["rule"]),
                coverage=str(item["coverage"]),
                risk=str(item["risk"]),
                body_sha256=str(item["body_sha256"]),
            )
        )
    return tuple(entries)


def write_census_file(path: Path, entries: tuple[CensusEntry, ...]) -> None:
    path.write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compare_census(
    current: tuple[CensusEntry, ...],
    locked: tuple[CensusEntry, ...],
) -> list[str]:
    findings: list[str] = []
    if len(current) != len(locked):
        findings.append(f"CENSUS: entry count {len(current)} != {len(locked)}")
    for got, want in zip(current, locked):
        if got.body_sha256 != want.body_sha256:
            findings.append(f"CENSUS: hash drift on {got.page}#{got.ordinal}")
        identity = (
            got.page,
            got.ordinal,
            got.language,
            got.kind,
            got.rule,
            got.coverage,
            got.risk,
        )
        expected = (
            want.page,
            want.ordinal,
            want.language,
            want.kind,
            want.rule,
            want.coverage,
            want.risk,
        )
        if identity != expected:
            findings.append(f"CENSUS: metadata drift on {got.page}#{got.ordinal}")
    return findings


def _bash_n(body: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as handle:
        handle.write(body)
        path = handle.name
    try:
        completed = subprocess.run(
            ["bash", "-n", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            fail_redacted(f"SHELL: bash -n failed: {completed.stderr.strip()}")
    finally:
        os.unlink(path)


def _powershell_shape(body: str, classification: Classification) -> None:
    if classification.kind == "powershell-safe-get":
        if "Invoke-RestMethod" not in body or "api.fastino.ai" not in body:
            fail_redacted("POWERSHELL: expected Invoke-RestMethod to Fastino")
        return
    if classification.kind == "powershell-env" and "$env:" not in body:
        fail_redacted("POWERSHELL: expected $env: assignment")
        return
    if classification.kind == "powershell-install" and "pip " not in body:
        fail_redacted("POWERSHELL: expected pip install")


def _check_python(snippet: FencedSnippet, classification: Classification) -> None:
    if classification.kind == "python-incomplete":
        try:
            ast.parse(snippet.body)
        except SyntaxError:
            if "continue" not in snippet.body and "break" not in snippet.body:
                fail_redacted(f"PYTHON: unexpected syntax error in {snippet.page}#{snippet.ordinal}")
        return
    try:
        compile(snippet.body, f"{snippet.page}#{snippet.ordinal}", "exec")
    except SyntaxError as error:
        fail_redacted(f"PYTHON: compile failed: {error}")


def static_check(snippet: FencedSnippet, classification: Classification) -> None:
    language = snippet.language
    if language == "jsonl":
        for line in snippet.body.splitlines():
            if line.strip():
                json.loads(line)
        return
    if language == "json":
        json.loads(snippet.body)
        return
    if language.startswith("bash"):
        _bash_n(snippet.body)
        return
    if language.startswith("powershell"):
        _powershell_shape(snippet.body, classification)
        return
    if language == "python":
        _check_python(snippet, classification)
        return
    if language == "text":
        if not snippet.body.strip():
            fail_redacted("TEXT: empty diagram")
        return
    fail_redacted(f"UNCLASSIFIED: {snippet.page}#{snippet.ordinal}")


def substitute_job(path: str, job_id: str) -> str:
    quoted = urllib.parse.quote(job_id, safe="")
    return JOB_ID_IN_PATH.sub(f"/v1/training/jobs/{quoted}", path)


def substitute_dataset(path: str, dataset: str) -> str:
    quoted = urllib.parse.quote(dataset, safe="")
    query = ""
    concrete = path
    if "?" in path:
        concrete, query = path.split("?", maxsplit=1)
        query = f"?{query}"
    return DATASET_IN_PATH.sub(f"/v1/datasets/{quoted}", concrete) + query


def plan_live(
    snippet: FencedSnippet,
    classification: Classification,
    env: dict[str, str],
) -> LivePlan:
    if classification.coverage != "live" or classification.method != "GET" or not classification.path:
        return LivePlan(
            action="static-only",
            method=classification.method,
            path=classification.path,
            payload=None,
            detail=f"static-only mapped to {classification.mapped_command or 'static'}",
            snippet=snippet,
            classification=classification,
        )
    path = classification.path
    template = match_safe_template(path)
    if template in JOB_GET_TEMPLATES:
        job_id = env.get("FASTINO_DOCS_CANARY_TRAINING_JOB_ID", "")
        if not job_id:
            return LivePlan(
                action="skipped",
                method="GET",
                path=None,
                payload=None,
                detail=(
                    f"SKIP: {snippet.page}#{snippet.ordinal} needs "
                    "FASTINO_DOCS_CANARY_TRAINING_JOB_ID"
                ),
                snippet=snippet,
                classification=classification,
            )
        return LivePlan(
            action="get",
            method="GET",
            path=substitute_job(path, job_id),
            payload=None,
            detail="",
            snippet=snippet,
            classification=classification,
        )
    if template == "/v1/datasets/{dataset_name}":
        dataset = env.get("FASTINO_DOCS_CANARY_DATASET", "")
        if not dataset:
            return LivePlan(
                action="skipped",
                method="GET",
                path=None,
                payload=None,
                detail=(
                    f"SKIP: {snippet.page}#{snippet.ordinal} needs "
                    "FASTINO_DOCS_CANARY_DATASET"
                ),
                snippet=snippet,
                classification=classification,
            )
        return LivePlan(
            action="get",
            method="GET",
            path=substitute_dataset(path, dataset),
            payload=None,
            detail="",
            snippet=snippet,
            classification=classification,
        )
    return LivePlan(
        action="get",
        method="GET",
        path=path,
        payload=None,
        detail="",
        snippet=snippet,
        classification=classification,
    )


def report_live_plans(plans: list[LivePlan]) -> None:
    for plan in plans:
        if plan.action == "skipped" and plan.detail:
            print(redact_secrets(plan.detail), flush=True)


def execute_safe_get(
    method: str,
    path: str,
    *,
    api_key: str,
    payload: object | None = None,
    request_fn: Callable[..., HARNESS.Response] | None = None,
) -> HARNESS.Response:
    if method != "GET":
        fail_redacted("LIVE: only GET is allowed")
    if payload is not None:
        fail_redacted("LIVE: body is not allowed")
    if match_safe_template(path) is None:
        fail_redacted("LIVE: path is outside SAFE_GET_TEMPLATES")
    url = f"{HARNESS.API_ORIGIN}{path if path.startswith('/') else f'/{path}'}"
    headers = HARNESS._auth_headers(api_key)
    if request_fn is None:
        response = HARNESS._request_with_retry("GET", url, headers=headers, payload=None)
    else:
        response = request_fn(
            "GET",
            url,
            headers=headers,
            payload=None,
            follow_redirects=False,
        )
    if 300 <= response.status < 400:
        fail_redacted("LIVE: redirect refused")
    return response


def run_checkpoint_parser(body: str, log_path: str) -> object:
    namespace: dict[str, object] = {"json": json, "Path": Path}
    exec(compile(body, "<get_last_checkpoint>", "exec"), namespace, namespace)
    func = namespace.get("get_last_checkpoint")
    if not callable(func):
        fail_redacted("PYTHON: get_last_checkpoint was not defined")
    return func(log_path)


def _validate_live_get(path: str, response: HARNESS.Response) -> None:
    excerpt = redact_secrets(response.text()[:300])
    if response.status != 200:
        fail_redacted(f"LIVE: GET {path} returned HTTP {response.status}: {excerpt}")
    try:
        payload = response.json()
    except json.JSONDecodeError as error:
        fail_redacted(f"LIVE: GET {path} is not JSON: {error}")
    document = HARNESS._load_openapi()
    operations = HARNESS._operations(document)
    found = HARNESS._operation_for_path(operations, "get", path)
    if found is None:
        return
    HARNESS._validate_operation_response(
        document,
        operations,
        "get",
        found[0],
        response,
        payload,
    )


def run_training_snippets(*, live: bool = False, write_census: bool = False) -> None:
    pages = collect_pages()
    snippets = collect_snippets()
    classifications = [classify(snippet) for snippet in snippets]
    for snippet, classification in zip(snippets, classifications):
        static_check(snippet, classification)
        if live or classification.kind != "python-checkpoint-parser":
            continue
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "loop.jsonl"
            log_path.write_text('{"state_path": "ok", "step": 1}\n', encoding="utf-8")
            record = run_checkpoint_parser(snippet.body, str(log_path))
            if not isinstance(record, dict) or record.get("state_path") != "ok":
                fail_redacted("PYTHON: get_last_checkpoint fixture failed")
    entries = census_entries(pages, snippets)
    if write_census:
        write_census_file(CENSUS_PATH, entries)
        print(f"Wrote census {CENSUS_PATH.relative_to(HARNESS.ROOT)}")
        return
    if not CENSUS_PATH.is_file():
        fail_redacted("CENSUS: missing; regenerate with --write-census")
    findings = compare_census(entries, load_census(CENSUS_PATH))
    if findings:
        fail_redacted("\n".join(findings))
    for snippet, classification in zip(snippets, classifications):
        if classification.risk == "preview-mismatch":
            print(
                f"BLOCKED: {snippet.page}#{snippet.ordinal} preview "
                "type/model is not live coverage; Pioneer requires base_model",
                flush=True,
            )
    if live:
        plans = [
            plan_live(snippet, classification, dict(os.environ))
            for snippet, classification in zip(snippets, classifications)
        ]
        report_live_plans(plans)
        if any(plan.action == "get" for plan in plans):
            api_key = HARNESS._require_api_key()
            for plan in plans:
                if plan.action != "get" or plan.path is None:
                    continue
                response = execute_safe_get("GET", plan.path, api_key=api_key)
                _validate_live_get(plan.path, response)
    print(
        f"Training snippets passed: {len(snippets)} fences across {len(pages)} pages"
        + ("; live reads executed" if live else "")
    )
