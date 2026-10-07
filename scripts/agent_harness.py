#!/usr/bin/env python3
"""Exercise Fastino documentation the way a coding agent consumes it."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NoReturn

from check_api_docs import local_docs_findings
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

ROOT = Path(__file__).resolve().parents[1]
DOCS_ORIGIN = os.getenv("FASTINO_DOCS_ORIGIN", "https://docs.fastino.ai").rstrip("/")
API_ORIGIN = os.getenv("FASTINO_API_ORIGIN", "https://api.fastino.ai").rstrip("/")
USER_AGENT = "FastinoDocsCanary/1.0 (+https://docs.fastino.ai)"
HTTP_METHODS = frozenset({"delete", "get", "patch", "post", "put"})
RETRYABLE_STATUSES = frozenset({425, 429, 503})
LOCALES = frozenset({"cn", "de", "es", "fr"})
TERMINAL_TRAINING_STATUSES = frozenset(
    {
        "complete",
        "artifact_ready",
        "deployed",
        "failed",
        "cancelled",
        "errored",
        "stopped",
        "terminated",
    }
)

REQUIRED_DISCOVERY_LINKS = {
    "decision models": "/concepts/decision-models.md",
    "GLiDE inference": "/inference/systemone.md",
    "GLiNER models": "/concepts/gliner.md",
    "GLiNER inference": "/inference/chat-completions.md",
    "datasets": "/concepts/datasets.md",
    "training": "/training.md",
    "training creation": "/api-reference/training-jobs/create.md",
}
REQUIRED_OPERATIONS = {
    ("get", "/v1/base-models"),
    ("post", "/v1/chat/completions"),
    ("post", "/v1/systemone"),
    ("get", "/v1/datasets/{name}"),
    ("post", "/v1/datasets/upload/process"),
    ("post", "/v1/datasets/upload/url"),
    ("get", "/v1/training-jobs"),
    ("post", "/v1/training-jobs"),
    ("get", "/v1/training-jobs/{job_id}"),
    ("get", "/v1/training-jobs/{job_id}/billing"),
    ("get", "/v1/training-jobs/{job_id}/checkpoints"),
    ("get", "/v1/training-jobs/{job_id}/logs"),
}
JOURNEY_PAGES = (
    "concepts/decision-models.mdx",
    "concepts/models.mdx",
    "concepts/gliner.mdx",
    "concepts/datasets.mdx",
    "inference/systemone.mdx",
    "inference/chat-completions.mdx",
    "training.mdx",
    "api-reference/training-jobs/create.mdx",
)

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((?P<target>/[^)\s]+)\)")
HREF_LINK = re.compile(r'href=["\'](?P<target>/[^"\']+)["\']')
LLMS_URL = re.compile(r"https://docs\.fastino\.ai(?P<path>/[^)\s]+)")
API_URL = re.compile(r"https://api\.fastino\.ai(?P<path>/v1/[^\"'`\s<\\]+)")
METHOD_PATH = re.compile(r"\b(?P<method>GET|POST|PATCH|PUT|DELETE)\s+(?P<path>/v1/[^`\s\"'|,)]+)")
CURL_BODY_START = re.compile(r"-d\s+'")
FRONTMATTER = re.compile(r"\A---\s*\n.*?^---\s*$", re.DOTALL | re.MULTILINE)


class HarnessFailure(RuntimeError):
    """A deterministic documentation or API journey failure."""


@dataclass(frozen=True)
class Response:
    """Small HTTP response value with no credential-bearing request metadata."""

    status: int
    url: str
    headers: dict[str, str]
    body: bytes

    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> object:
        return json.loads(self.body)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def fail(message: str) -> NoReturn:
    raise HarnessFailure(message)


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: object | None = None,
    follow_redirects: bool = True,
    timeout: float = 30,
) -> Response:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    opener = urllib.request.build_opener() if follow_redirects else urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return Response(
                status=response.status,
                url=response.geturl(),
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as error:
        return Response(
            status=error.code,
            url=error.geturl(),
            headers={key.lower(): value for key, value in error.headers.items()},
            body=error.read(),
        )


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: object | None = None,
    attempts: int = 2,
    timeout: float = 120,
) -> Response:
    response: Response | None = None
    for attempt in range(attempts):
        try:
            response = _request(
                method,
                url,
                headers=headers,
                payload=payload,
                follow_redirects=False,
                timeout=timeout,
            )
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 15))
            continue
        if response.status not in RETRYABLE_STATUSES or attempt == attempts - 1:
            return response
        retry_after = response.headers.get("retry-after", "")
        delay = min(float(retry_after), 60.0) if retry_after.isdigit() else min(2**attempt, 15)
        time.sleep(delay)
    if response is None:
        fail(f"request did not run: {method} {url}")
    return response


def _load_openapi(path: Path = ROOT / "openapi.json") -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        fail("CONTRACT: openapi.json must contain an object")
    return document


def _resolve_pointer(document: object, reference: str) -> object:
    if not reference.startswith("#/"):
        fail(f"CONTRACT: external OpenAPI reference is not self-contained: {reference}")
    current = document
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            fail(f"CONTRACT: unresolved OpenAPI reference: {reference}")
        current = current[token]
    return current


def _walk(value: object) -> list[object]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk(child))
    return values


def _operations(document: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    paths = document.get("paths")
    if not isinstance(paths, dict):
        fail("CONTRACT: OpenAPI has no paths object")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for path, path_item in paths.items():
        if not isinstance(path, str) or not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method in HTTP_METHODS and isinstance(operation, dict):
                result[(method, path)] = operation
    return result


def _validate_openapi(document: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    version = document.get("openapi")
    if not isinstance(version, str) or not version.startswith("3."):
        fail(f"CONTRACT: unsupported OpenAPI version: {version!r}")
    servers = document.get("servers")
    if not isinstance(servers, list) or not any(
        isinstance(server, dict) and server.get("url") == API_ORIGIN for server in servers
    ):
        fail(f"CONTRACT: OpenAPI does not advertise {API_ORIGIN}")
    serialized = json.dumps(document)
    if "x-fastino-visibility" in serialized:
        fail("CONTRACT: public OpenAPI leaks x-fastino-visibility")
    for value in _walk(document):
        if isinstance(value, dict) and isinstance(value.get("$ref"), str):
            _resolve_pointer(document, value["$ref"])
    operations = _operations(document)
    missing = REQUIRED_OPERATIONS - set(operations)
    if missing:
        fail(f"CONTRACT: required operations are missing: {sorted(missing)}")
    components = document.get("components")
    schemes = components.get("securitySchemes") if isinstance(components, dict) else None
    if not isinstance(schemes, dict) or not {"ApiKeyAuth", "BearerAuth"} <= set(schemes):
        fail("CONTRACT: X-API-Key and Bearer security schemes must both be published")
    return operations


def _validate_value(
    value: object,
    schema: object,
    document: dict[str, object],
    *,
    location: str,
) -> None:
    if not isinstance(schema, dict) or not schema:
        return
    base_uri = "urn:fastino:agent-harness"
    validation_document = {**document, "$id": base_uri, "x-agent-schema": schema}
    resource = Resource.from_contents(
        validation_document,
        default_specification=DRAFT202012,
    )
    registry = Registry().with_resource(base_uri, resource)
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(
            {"$ref": f"{base_uri}#/x-agent-schema"},
            registry=registry,
        )
    except SchemaError as error:
        fail(f"CONTRACT: invalid schema for {location}: {error.message}")
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        suffix = "".join(f"[{part!r}]" for part in error.path)
        fail(f"CONTRACT: {location}{suffix}: {error.message}")


def _operation_for_path(
    operations: dict[tuple[str, str], dict[str, object]],
    method: str,
    concrete_path: str,
) -> tuple[str, dict[str, object]] | None:
    path = concrete_path.split("?", maxsplit=1)[0].rstrip(".,)")
    for (candidate_method, template), operation in operations.items():
        if candidate_method != method.lower():
            continue
        expression = re.sub(r"\\\{[^}]+\\\}", r"[^/]+", re.escape(template))
        if re.fullmatch(expression, path):
            return template, operation
    return None


def _request_schema(document: dict[str, object], operation: dict[str, object]) -> object:
    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return {}
    content = request_body.get("content")
    media = content.get("application/json") if isinstance(content, dict) else None
    return media.get("schema", {}) if isinstance(media, dict) else {}


def _response_schema(
    operation: dict[str, object],
    status: int,
) -> object:
    responses = operation.get("responses")
    response = responses.get(str(status)) if isinstance(responses, dict) else None
    content = response.get("content") if isinstance(response, dict) else None
    media = content.get("application/json") if isinstance(content, dict) else None
    return media.get("schema", {}) if isinstance(media, dict) else {}


def _curl_bodies(text: str) -> list[tuple[int, object | None, str | None]]:
    """Decode single-quoted JSON bodies without truncating nested objects."""
    decoder = json.JSONDecoder()
    bodies: list[tuple[int, object | None, str | None]] = []
    for match in CURL_BODY_START.finditer(text):
        candidate = text[match.end() :].lstrip()
        try:
            body, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError as error:
            bodies.append((match.start(), None, str(error)))
            continue
        if candidate[end:].lstrip().startswith("'"):
            bodies.append((match.start(), body, None))
        else:
            bodies.append((match.start(), None, "JSON body has no closing shell quote"))
    return bodies


def _english_docs() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.mdx")
        if path.relative_to(ROOT).parts[0] not in LOCALES
    ]


def _local_link_findings() -> list[str]:
    config = json.loads((ROOT / "docs.json").read_text(encoding="utf-8"))
    redirects = config.get("redirects")
    redirect_sources = {
        redirect["source"].strip("/")
        for redirect in redirects or []
        if isinstance(redirect, dict) and isinstance(redirect.get("source"), str)
    }
    findings: list[str] = []
    for page in _english_docs():
        text = page.read_text(encoding="utf-8")
        targets = [match.group("target") for match in MARKDOWN_LINK.finditer(text)]
        targets.extend(match.group("target") for match in HREF_LINK.finditer(text))
        for target in targets:
            parsed = urllib.parse.urlsplit(target)
            relative = parsed.path.strip("/")
            if not relative or relative.startswith(("v1/", "_mintlify/")):
                continue
            if relative in redirect_sources:
                continue
            candidate = ROOT / f"{relative}.mdx"
            if not candidate.is_file():
                findings.append(
                    f"LINK: {page.relative_to(ROOT)} points to missing /{relative}"
                )
    return findings


def _journey_discovery_findings() -> list[str]:
    text = (ROOT / "llms.txt").read_text(encoding="utf-8")
    indexed_paths = {match.group("path") for match in LLMS_URL.finditer(text)}
    findings = [
        f"DISCOVERY: llms.txt cannot lead an agent to {name}: {path}"
        for name, path in REQUIRED_DISCOVERY_LINKS.items()
        if path not in indexed_paths
    ]
    for required in ("/llms-full.txt", "/openapi.json"):
        if required not in indexed_paths and required not in text:
            findings.append(f"DISCOVERY: llms.txt does not advertise {required}")
    return findings


def _journey_findings(
    document: dict[str, object],
    operations: dict[tuple[str, str], dict[str, object]],
) -> list[str]:
    findings: list[str] = []
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    if "GET /v1/base-models" not in agents:
        findings.append("DISCOVERY: AGENTS.md must use GET /v1/base-models for model availability")
    if "GET /v1/models" in agents:
        findings.append("DISCOVERY: AGENTS.md conflicts by naming GET /v1/models as the model catalog")
    for relative in JOURNEY_PAGES:
        path = ROOT / relative
        if not path.is_file():
            findings.append(f"DISCOVERY: journey page is missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        if FRONTMATTER.match(text) is None:
            findings.append(f"DISCOVERY: journey page has invalid frontmatter: {relative}")
        for match in METHOD_PATH.finditer(text):
            method = match.group("method").lower()
            path_text = match.group("path")
            if _operation_for_path(operations, method, path_text) is None:
                findings.append(
                    f"CONTRACT: {relative} documents absent operation "
                    f"{method.upper()} {path_text}"
                )
        for match in API_URL.finditer(text):
            if _operation_for_path(operations, "get", match.group("path")) is None and not any(
                _operation_for_path(operations, method, match.group("path"))
                for method in HTTP_METHODS
            ):
                findings.append(
                    f"CONTRACT: {relative} uses absent API URL {match.group(0)}"
                )
        for body_start, body, body_error in _curl_bodies(text):
            prefix = text[max(0, body_start - 1_000) : body_start]
            curl_start = prefix.rfind("curl ")
            if curl_start >= 0:
                prefix = prefix[curl_start:]
            url_matches = list(API_URL.finditer(prefix))
            if not url_matches:
                continue
            path_text = url_matches[-1].group("path")
            method_match = re.search(r"-X\s+(GET|POST|PATCH|PUT|DELETE)", prefix)
            method = method_match.group(1).lower() if method_match else "post"
            found = _operation_for_path(operations, method, path_text)
            if found is None:
                continue
            if body_error is not None:
                findings.append(f"EXAMPLE: {relative} has malformed cURL JSON: {body_error}")
                continue
            _validate_value(
                body,
                _request_schema(document, found[1]),
                document,
                location=f"{relative} request",
            )
    return findings


def run_static() -> None:
    document = _load_openapi()
    operations = _validate_openapi(document)
    findings = local_docs_findings(ROOT, document)
    findings.extend(_journey_discovery_findings())
    findings.extend(_local_link_findings())
    try:
        findings.extend(_journey_findings(document, operations))
    except HarnessFailure as error:
        findings.append(str(error))
    if findings:
        fail("\n".join(findings))
    print(
        "Agent docs static harness passed: "
        f"{len(operations)} operations, {len(_english_docs())} English pages"
    )


def _require_status(response: Response, expected: set[int], label: str) -> None:
    if response.status not in expected:
        excerpt = response.text()[:500].replace("\n", " ")
        fail(f"{label}: expected HTTP {sorted(expected)}, got {response.status}: {excerpt}")


def _published_link(url: str) -> tuple[str, int]:
    response = _request("GET", url, headers={"Accept": "text/markdown"})
    return url, response.status


def run_published() -> None:
    root = _request("GET", f"{DOCS_ORIGIN}/", follow_redirects=False)
    if root.status not in {301, 302, 307, 308}:
        fail(f"DISCOVERY: docs root should redirect, got HTTP {root.status}")
    link_header = root.headers.get("link", "")
    for relation in ("llms-txt", "llms-full-txt"):
        if f'rel="{relation}"' not in link_header:
            fail(f"DISCOVERY: docs root Link header is missing {relation}")
    if root.headers.get("x-llms-txt") != "/llms.txt":
        fail("DISCOVERY: docs root is missing x-llms-txt: /llms.txt")

    human = _request("GET", f"{DOCS_ORIGIN}/", headers={"Accept": "text/html"})
    _require_status(human, {200}, "DISCOVERY human homepage")
    if "text/html" not in human.headers.get("content-type", ""):
        fail("DISCOVERY: a human browser did not receive HTML")

    agent = _request("GET", f"{DOCS_ORIGIN}/", headers={"Accept": "text/markdown"})
    _require_status(agent, {200}, "DISCOVERY markdown homepage")
    if not agent.url.endswith(".md"):
        fail(f"DISCOVERY: markdown negotiation ended at unexpected URL {agent.url}")
    for marker in ("llms.txt", "llms-full.txt", "openapi.json"):
        if marker not in agent.text():
            fail(f"DISCOVERY: agent markdown does not mention {marker}")

    llms = _request("GET", f"{DOCS_ORIGIN}/llms.txt")
    _require_status(llms, {200}, "DISCOVERY llms.txt")
    indexed_paths = {match.group("path") for match in LLMS_URL.finditer(llms.text())}
    for name, path in REQUIRED_DISCOVERY_LINKS.items():
        if path not in indexed_paths:
            fail(f"DISCOVERY: published llms.txt cannot lead an agent to {name}")

    full = _request("GET", f"{DOCS_ORIGIN}/llms-full.txt")
    _require_status(full, {200}, "DISCOVERY llms-full.txt")
    if len(full.body) < 10_000:
        fail("DISCOVERY: llms-full.txt is unexpectedly small")
    for marker in ("Decision Models", "GLiNER", "Training"):
        if marker.casefold() not in full.text().casefold():
            fail(f"DISCOVERY: llms-full.txt is missing {marker}")

    links = sorted(
        {
            match.group(0)
            for match in re.finditer(r"https://docs\.fastino\.ai/[^)\s]+", llms.text())
            if not match.group(0).endswith("/llms.txt")
        }
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = [
            f"LINK: {url} returned HTTP {status}"
            for url, status in pool.map(_published_link, links)
            if status != 200
        ]
    if failures:
        fail("\n".join(failures))

    published_openapi = _request("GET", f"{DOCS_ORIGIN}/openapi.json")
    _require_status(published_openapi, {200}, "CONTRACT published OpenAPI")
    published_document = published_openapi.json()
    if published_document != _load_openapi():
        fail("CONTRACT: published docs OpenAPI differs from repository openapi.json")
    if not isinstance(published_document, dict):
        fail("CONTRACT: published OpenAPI must be an object")
    _validate_openapi(published_document)
    print(f"Published agent journey passed: {len(links)} indexed resources")


def _auth_headers(api_key: str, *, bearer: bool = False) -> dict[str, str]:
    if bearer:
        return {"Authorization": f"Bearer {api_key}"}
    return {"X-API-Key": api_key}


def _json_response(
    response: Response,
    *,
    label: str,
    expected: set[int] = frozenset({200}),
) -> object:
    _require_status(response, expected, label)
    try:
        return response.json()
    except json.JSONDecodeError as error:
        fail(f"{label}: response is not JSON: {error}")


def _validate_operation_response(
    document: dict[str, object],
    operations: dict[tuple[str, str], dict[str, object]],
    method: str,
    template: str,
    response: Response,
    payload: object,
) -> None:
    operation = operations[(method, template)]
    schema = _response_schema(operation, response.status)
    _validate_value(payload, schema, document, location=f"{method.upper()} {template} response")


def _catalog_models(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        fail("INFERENCE: base-model catalog response is not an object")
    for key in ("models", "base_models", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    fail("INFERENCE: base-model catalog has no model list")


def _model_identifier(model: dict[str, object]) -> str | None:
    for key in ("id", "model_id", "name"):
        value = model.get(key)
        if isinstance(value, str):
            return value
    return None


def _require_api_key() -> str:
    api_key = os.getenv("FASTINO_DOCS_CANARY_API_KEY") or os.getenv("FASTINO_API_KEY", "")
    if not api_key.startswith("fast_sk_"):
        fail(
            "AUTH: FASTINO_DOCS_CANARY_API_KEY (or FASTINO_API_KEY locally) "
            "must be a dedicated fast_sk_ canary key"
        )
    return api_key


def _discover_models(api_key: str) -> tuple[str, str]:
    catalog_response = _request_with_retry(
        "GET",
        f"{API_ORIGIN}/v1/base-models?supports_inference=true",
        headers=_auth_headers(api_key),
    )
    catalog = _json_response(catalog_response, label="INFERENCE model catalog")
    models = _catalog_models(catalog)
    identifiers = {_model_identifier(model) for model in models}
    glide = "fastino/GLiDE"
    configured_encoder = os.getenv("FASTINO_DOCS_CANARY_GLINER_MODEL", "")
    encoder_candidates = [
        configured_encoder,
        "fastino/gliner2.5-multi-v1",
        "fastino/gliner2.5-base-v1",
        "fastino/gliner2-base-v1",
    ]
    encoder = next((candidate for candidate in encoder_candidates if candidate in identifiers), "")
    if glide not in identifiers:
        fail("INFERENCE: live catalog does not expose documented fastino/GLiDE")
    if not encoder:
        fail("INFERENCE: live catalog exposes none of the documented GLiNER canary models")
    return glide, encoder


def _discover_training_model(api_key: str) -> str:
    response = _request_with_retry(
        "GET",
        f"{API_ORIGIN}/v1/base-models?supports_training=true",
        headers=_auth_headers(api_key),
    )
    catalog = _json_response(response, label="TRAINING model catalog")
    identifiers = {
        identifier
        for model in _catalog_models(catalog)
        if (identifier := _model_identifier(model)) is not None
    }
    configured = os.getenv("FASTINO_DOCS_CANARY_BASE_MODEL", "")
    candidates = (
        configured,
        "fastino/gliner2-base-v1",
        "fastino/gliner2.5-base-v1",
        "fastino/gliner2.5-multi-v1",
    )
    selected = next((candidate for candidate in candidates if candidate in identifiers), "")
    if not selected:
        fail("TRAINING: live catalog exposes no supported GLiNER training model")
    if configured and selected != configured:
        fail(f"TRAINING: configured base model is not trainable: {configured}")
    return selected


def _run_glide(
    api_key: str,
    model: str,
    document: dict[str, object],
    operations: dict[tuple[str, str], dict[str, object]],
) -> None:
    body = {
        "model": model,
        "state": "The receipt is attached and the purchase was 10 days ago.",
        "questions": {
            "refund_requested": {
                "type": "noul",
                "instructions": "Is the customer requesting a refund?",
                "criteria": {"true": "Refund requested", "false": "No refund requested"},
            }
        },
    }
    response = _request_with_retry(
        "POST",
        f"{API_ORIGIN}/v1/systemone",
        headers=_auth_headers(api_key),
        payload=body,
    )
    payload = _json_response(response, label="INFERENCE GLiDE")
    _validate_operation_response(document, operations, "post", "/v1/systemone", response, payload)
    answers = payload.get("answers") if isinstance(payload, dict) else None
    answer = answers.get("refund_requested") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or not isinstance(answer.get("noul"), (int, float)):
        fail("INFERENCE: GLiDE did not return the documented Noul answer")


def _run_gliner(
    api_key: str,
    model: str,
    document: dict[str, object],
    operations: dict[tuple[str, str], dict[str, object]],
) -> None:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Ada Lovelace worked with Charles Babbage in London.",
            }
        ],
        "schema": {
            "entities": [
                {"name": "person", "description": "a named person"},
                {"name": "location", "description": "a city or place"},
            ]
        },
        "threshold": 0.3,
        "include_confidence": False,
        "include_spans": False,
        "store": False,
    }
    response = _request_with_retry(
        "POST",
        f"{API_ORIGIN}/v1/chat/completions",
        headers=_auth_headers(api_key, bearer=True),
        payload=body,
    )
    payload = _json_response(response, label="INFERENCE GLiNER")
    _validate_gliner_response(
        document,
        operations,
        response,
        payload,
        label="INFERENCE GLiNER",
        expected_entities={"person": "Ada Lovelace", "location": "London"},
    )


def _validate_gliner_response(
    document: dict[str, object],
    operations: dict[tuple[str, str], dict[str, object]],
    response: Response,
    payload: object,
    *,
    label: str,
    expected_entities: dict[str, str],
) -> None:
    _validate_operation_response(
        document,
        operations,
        "post",
        "/v1/chat/completions",
        response,
        payload,
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        fail(f"{label}: response has no choices[0].message.content")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        fail(f"{label}: content is not serialized JSON: {error}")
    entities = result.get("entities") if isinstance(result, dict) else None
    if not isinstance(entities, dict):
        fail(f"{label}: content does not contain an entities object")
    missing = set(expected_entities) - set(entities)
    if missing:
        fail(f"{label}: content is missing requested entities: {sorted(missing)}")
    if any(not isinstance(entities[name], list) for name in expected_entities):
        fail(f"{label}: requested entity values must be arrays")
    wrong_values = {
        name: expected
        for name, expected in expected_entities.items()
        if expected not in entities[name]
    }
    if wrong_values:
        fail(f"{label}: content is missing expected values: {wrong_values}")


def _ready_dataset_reference(api_key: str) -> dict[str, str]:
    dataset_name = os.getenv("FASTINO_DOCS_CANARY_DATASET", "")
    dataset_version = os.getenv("FASTINO_DOCS_CANARY_DATASET_VERSION", "")
    if not dataset_name or not dataset_version:
        fail(
            "TRAINING: FASTINO_DOCS_CANARY_DATASET and "
            "FASTINO_DOCS_CANARY_DATASET_VERSION are required"
        )
    response = _request_with_retry(
        "GET",
        f"{API_ORIGIN}/v1/datasets/{urllib.parse.quote(dataset_name, safe='')}",
        headers=_auth_headers(api_key),
    )
    payload = _json_response(response, label="TRAINING canary dataset")
    versions = payload.get("versions") if isinstance(payload, dict) else None
    if not isinstance(versions, list):
        fail("TRAINING: dataset response has no versions list")
    selected = next(
        (
            version
            for version in versions
            if isinstance(version, dict)
            and version.get("version_number") == dataset_version
        ),
        None,
    )
    if not isinstance(selected, dict):
        fail(f"TRAINING: dataset {dataset_name} has no version {dataset_version}")
    if selected.get("status") != "ready":
        fail(
            f"TRAINING: dataset {dataset_name} version {dataset_version} "
            f"is not ready: {selected.get('status')!r}"
        )
    if selected.get("dataset_type") != "ner" or selected.get("type") != "training":
        fail(
            "TRAINING: canary dataset version must be an NER training dataset, got "
            f"dataset_type={selected.get('dataset_type')!r}, type={selected.get('type')!r}"
        )
    labels = selected.get("labels")
    if not isinstance(labels, list) or not {"person", "location"} <= set(labels):
        fail(
            "TRAINING: canary dataset must contain person and location labels, got "
            f"{labels!r}"
        )
    return {"name": dataset_name, "version": dataset_version}


def _run_training_reads(api_key: str) -> None:
    jobs = _request_with_retry(
        "GET",
        f"{API_ORIGIN}/v1/training-jobs?limit=1",
        headers=_auth_headers(api_key),
    )
    _json_response(jobs, label="TRAINING list jobs")
    _ready_dataset_reference(api_key)


def _run_training_replay(api_key: str) -> None:
    job_id = os.getenv("FASTINO_DOCS_CANARY_TRAINING_JOB_ID", "")
    idempotency_key = os.getenv("FASTINO_DOCS_CANARY_IDEMPOTENCY_KEY", "")
    if not job_id or not idempotency_key:
        fail(
            "TRAINING: replay requires FASTINO_DOCS_CANARY_TRAINING_JOB_ID, "
            "and FASTINO_DOCS_CANARY_IDEMPOTENCY_KEY"
        )
    dataset_reference = _ready_dataset_reference(api_key)
    existing = _request_with_retry(
        "GET",
        f"{API_ORIGIN}/v1/training-jobs/{job_id}",
        headers=_auth_headers(api_key),
    )
    existing_payload = _json_response(existing, label="TRAINING existing canary job")
    existing_status, existing_terminal = _training_status(existing_payload)
    if not existing_terminal:
        _stop_and_confirm(api_key, job_id)
        fail(
            f"TRAINING: replay fixture {job_id} was still active "
            f"with status {existing_status!r} and was stopped"
        )
    if existing_status not in {"complete", "artifact_ready", "deployed"}:
        fail(
            f"TRAINING: replay fixture {job_id} is not a successful job: "
            f"{existing_status!r}"
        )
    body = {
        "model_name": "docs-agent-canary-replay",
        "base_model": os.getenv(
            "FASTINO_DOCS_CANARY_BASE_MODEL",
            "fastino/gliner2-base-v1",
        )
        or "fastino/gliner2-base-v1",
        "datasets": [dataset_reference],
        "training_type": "lora",
        "nr_epochs": 1,
        "validation_data_percentage": 0.2,
    }
    headers = {
        **_auth_headers(api_key),
        "Idempotency-Key": idempotency_key,
    }
    replay = _request_with_retry(
        "POST",
        f"{API_ORIGIN}/v1/training-jobs",
        headers=headers,
        payload=body,
    )
    payload = _json_response(replay, label="TRAINING idempotent create replay")
    replay_id = payload.get("id") if isinstance(payload, dict) else None
    if replay_id != job_id:
        if isinstance(replay_id, str):
            _stop_and_confirm(api_key, replay_id)
            _cleanup_training_job(api_key, replay_id)
        fail(
            "TRAINING: idempotent replay returned a different job; "
            "the unexpected job was stopped and deleted"
        )


def run_api(*, replay_training: bool) -> None:
    api_key = _require_api_key()
    document = _load_openapi()
    operations = _validate_openapi(document)
    invalid = _request(
        "GET",
        f"{API_ORIGIN}/v1/training-jobs?limit=1",
        headers={"X-API-Key": "fast_sk_invalid_docs_canary"},
        follow_redirects=False,
    )
    if invalid.status != 401:
        fail(f"AUTH: invalid X-API-Key expected 401, got {invalid.status}")
    glide, gliner = _discover_models(api_key)
    training_model = _discover_training_model(api_key)
    _run_glide(api_key, glide, document, operations)
    _run_gliner(api_key, gliner, document, operations)
    _run_training_reads(api_key)
    if replay_training:
        _run_training_replay(api_key)
    print(
        "Authenticated agent journey passed: "
        f"GLiDE={glide}, GLiNER={gliner}, trainable={training_model}, "
        f"training_replay={replay_training}"
    )


def _training_status(payload: object) -> tuple[str, bool]:
    if not isinstance(payload, dict):
        fail("TRAINING: job response is not an object")
    status = payload.get("normalized_status") or payload.get("status")
    terminal = payload.get("is_terminal_status")
    if not isinstance(status, str):
        fail("TRAINING: job response has no status")
    if isinstance(terminal, bool):
        if terminal and status not in TERMINAL_TRAINING_STATUSES:
            fail(
                "TRAINING: job response is inconsistent: "
                f"status={status!r}, is_terminal_status=true"
            )
        return status, terminal
    return status, status in TERMINAL_TRAINING_STATUSES


def _cleanup_training_job(api_key: str, job_id: str) -> None:
    response = _request_with_retry(
        "DELETE",
        f"{API_ORIGIN}/v1/training-jobs/{job_id}",
        headers=_auth_headers(api_key),
    )
    _require_status(response, {200}, "TRAINING cleanup")


def _error_code(response: Response) -> str | None:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _training_job_is_terminal(api_key: str, job_id: str) -> bool:
    poll = _request_with_retry(
        "GET",
        f"{API_ORIGIN}/v1/training-jobs/{job_id}",
        headers=_auth_headers(api_key),
        timeout=20,
    )
    payload = _json_response(poll, label=f"TRAINING confirm stop {job_id}")
    _, terminal = _training_status(payload)
    return terminal


def _stop_and_confirm(api_key: str, job_id: str, *, timeout_seconds: int = 300) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status = "unknown"
    while True:
        stop = _request_with_retry(
            "POST",
            f"{API_ORIGIN}/v1/training-jobs/{job_id}/stop",
            headers=_auth_headers(api_key),
            timeout=20,
        )
        if stop.status in {200, 202}:
            break
        if stop.status != 409:
            _require_status(stop, {200, 202}, f"TRAINING stop {job_id}")
        code = _error_code(stop)
        if code == "training_job_already_finished":
            break
        if code != "cancellation_unconfirmed":
            fail(f"TRAINING stop {job_id}: unexpected 409 error code {code!r}")
        if _training_job_is_terminal(api_key, job_id):
            return
        if time.monotonic() >= deadline:
            fail(f"TRAINING: cancellation remained unconfirmed for {job_id}")
        time.sleep(5)

    while time.monotonic() < deadline:
        if _training_job_is_terminal(api_key, job_id):
            return
        last_status = "non-terminal"
        time.sleep(10)
    fail(
        f"TRAINING: could not confirm stop for {job_id} within "
        f"{timeout_seconds} seconds; last status was {last_status}"
    )


def _list_training_jobs(api_key: str) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    offset = 0
    while True:
        response = _request_with_retry(
            "GET",
            f"{API_ORIGIN}/v1/training-jobs?limit=200&offset={offset}",
            headers=_auth_headers(api_key),
            timeout=30,
        )
        payload = _json_response(response, label="TRAINING list canary jobs")
        page = payload.get("training_jobs") if isinstance(payload, dict) else None
        if not isinstance(page, list):
            fail("TRAINING: job-list response has no training_jobs list")
        typed_page = [job for job in page if isinstance(job, dict)]
        if len(typed_page) != len(page):
            fail("TRAINING: job-list response contains a non-object job")
        jobs.extend(typed_page)
        has_more = payload.get("has_more") if isinstance(payload, dict) else None
        if has_more is not True:
            return jobs
        if not page:
            fail("TRAINING: job-list pagination reports has_more with an empty page")
        offset += len(page)


def _recover_training_job(api_key: str, model_name: str) -> dict[str, object]:
    """Recover an idempotent create whose HTTP response was lost."""
    matches = [
        job
        for job in _list_training_jobs(api_key)
        if job.get("model_name") == model_name
    ]
    if len(matches) != 1:
        fail(
            f"TRAINING: could not uniquely recover model_name={model_name!r}; "
            f"found {len(matches)} matches"
        )
    return matches[0]


def _create_or_recover_training_job(
    api_key: str,
    marker: uuid.UUID,
    model_name: str,
    body: dict[str, object],
) -> dict[str, object]:
    try:
        response = _request_with_retry(
            "POST",
            f"{API_ORIGIN}/v1/training-jobs",
            headers={
                **_auth_headers(api_key),
                "Idempotency-Key": str(marker),
            },
            payload=body,
        )
    except (OSError, TimeoutError, urllib.error.URLError):
        return _recover_training_job(api_key, model_name)
    if response.status == 200:
        try:
            payload = _json_response(response, label="TRAINING lifecycle create")
        except HarnessFailure:
            return _recover_training_job(api_key, model_name)
        if isinstance(payload, dict):
            return payload
        return _recover_training_job(api_key, model_name)
    if response.status == 409 or response.status >= 500:
        return _recover_training_job(api_key, model_name)
    _json_response(response, label="TRAINING lifecycle create")
    fail(f"TRAINING: unexpected create response HTTP {response.status}")


def _lifecycle_identity(run_id: str, run_attempt: int) -> tuple[uuid.UUID, str]:
    marker = (
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"fastino-docs-training:{run_id}:attempt:{run_attempt}",
        )
        if run_id
        else uuid.uuid4()
    )
    return marker, f"docs-agent-lifecycle-{marker.hex[:12]}"


def _is_lifecycle_canary_name(value: object) -> bool:
    return isinstance(value, str) and (
        value.startswith("docs-agent-lifecycle-")
        or (
            value.startswith("docs-agent-canary-")
            and value != "docs-agent-canary-replay"
        )
    )


def _reconcile_canary_jobs(
    api_key: str,
    current_model_name: str,
) -> None:
    for job in _list_training_jobs(api_key):
        model_name = job.get("model_name")
        if model_name == current_model_name or not _is_lifecycle_canary_name(model_name):
            continue
        job_id = job.get("id")
        if not isinstance(job_id, str):
            fail("TRAINING: prior canary attempt has no job id")
        _, terminal = _training_status(job)
        if not terminal:
            _stop_and_confirm(api_key, job_id)
        _cleanup_training_job(api_key, job_id)


def _billing_minutes(payload: object) -> Decimal | None:
    if not isinstance(payload, dict):
        fail("TRAINING: billing response is not an object")
    if payload.get("billed") is not True:
        return None
    raw_minutes = payload.get("gpu_minutes")
    if raw_minutes is None:
        fail("TRAINING: billed job has no gpu_minutes")
    try:
        minutes = Decimal(str(raw_minutes))
    except InvalidOperation:
        fail(f"TRAINING: invalid gpu_minutes value: {raw_minutes!r}")
    if not minutes.is_finite() or minutes < 0:
        fail(f"TRAINING: invalid gpu_minutes value: {raw_minutes!r}")
    return minutes


def _wait_for_billing(
    api_key: str,
    job_id: str,
    *,
    timeout_seconds: int,
) -> Decimal:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = _request_with_retry(
            "GET",
            f"{API_ORIGIN}/v1/training-jobs/{job_id}/billing",
            headers=_auth_headers(api_key),
        )
        payload = _json_response(response, label="TRAINING lifecycle billing")
        minutes = _billing_minutes(payload)
        if minutes is not None:
            return minutes
        if time.monotonic() >= deadline:
            fail(
                f"TRAINING: billing for {job_id} was not finalized within "
                f"{timeout_seconds} seconds"
            )
        time.sleep(10)


def _verify_completed_training_job(
    api_key: str,
    job_id: str,
    final_payload: dict[str, object],
    document: dict[str, object],
    operations: dict[tuple[str, str], dict[str, object]],
) -> None:
    if final_payload.get("is_deployable") is not True:
        fail(
            "TRAINING: successful canary is not deployable: "
            f"{final_payload.get('deployability_reason')!r}"
        )
    for suffix, label in (("logs", "logs"), ("checkpoints", "checkpoints")):
        result = _request_with_retry(
            "GET",
            f"{API_ORIGIN}/v1/training-jobs/{job_id}/{suffix}",
            headers=_auth_headers(api_key),
        )
        _json_response(result, label=f"TRAINING lifecycle {label}")

    billing_timeout = int(
        os.getenv("FASTINO_DOCS_CANARY_BILLING_TIMEOUT_SECONDS", "300")
    )
    minutes = _wait_for_billing(
        api_key,
        job_id,
        timeout_seconds=billing_timeout,
    )
    raw_maximum = os.getenv("FASTINO_DOCS_CANARY_MAX_GPU_MINUTES", "28")
    try:
        maximum = Decimal(raw_maximum)
    except InvalidOperation:
        fail(
            "TRAINING: FASTINO_DOCS_CANARY_MAX_GPU_MINUTES "
            f"is invalid: {raw_maximum!r}"
        )
    if not maximum.is_finite() or maximum < 0:
        fail("TRAINING: FASTINO_DOCS_CANARY_MAX_GPU_MINUTES must be non-negative")
    if minutes > maximum:
        fail(
            f"TRAINING: canary used {minutes} GPU minutes, "
            f"above the configured {maximum}"
        )

    inference = _request_with_retry(
        "POST",
        f"{API_ORIGIN}/v1/chat/completions",
        headers=_auth_headers(api_key),
        payload={
            "model": job_id,
            "messages": [
                {"role": "user", "content": "Ada Lovelace worked in London."}
            ],
            "schema": {
                "entities": [{"name": "person"}, {"name": "location"}]
            },
            "include_confidence": False,
            "include_spans": False,
            "store": False,
        },
    )
    inference_payload = _json_response(
        inference,
        label="TRAINING trained-model inference",
    )
    _validate_gliner_response(
        document,
        operations,
        inference,
        inference_payload,
        label="TRAINING trained-model inference",
        expected_entities={"person": "Ada Lovelace", "location": "London"},
    )


def run_training_lifecycle() -> None:
    api_key = _require_api_key()
    document = _load_openapi()
    operations = _validate_openapi(document)
    dataset_reference = _ready_dataset_reference(api_key)

    model = _discover_training_model(api_key)
    run_id = os.getenv("GITHUB_RUN_ID", "")
    run_attempt = int(os.getenv("GITHUB_RUN_ATTEMPT", "1"))
    if run_attempt < 1:
        fail("TRAINING: GITHUB_RUN_ATTEMPT must be at least 1")
    marker, model_name = _lifecycle_identity(run_id, run_attempt)
    _reconcile_canary_jobs(api_key, model_name)
    body = {
        "model_name": model_name,
        "base_model": model,
        "datasets": [dataset_reference],
        "training_type": "lora",
        "nr_epochs": 1,
        "validation_data_percentage": 0.2,
        "auto_data_sizing": True,
        "max_samples_per_dataset": 100,
    }
    payload = _create_or_recover_training_job(
        api_key,
        marker,
        model_name,
        body,
    )
    job_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(job_id, str):
        fail("TRAINING: create response has no job id")

    timeout_seconds = int(os.getenv("FASTINO_DOCS_CANARY_TIMEOUT_SECONDS", "7200"))
    deadline = time.monotonic() + timeout_seconds
    final_payload = payload
    try:
        while True:
            status, terminal = _training_status(final_payload)
            print(f"Training canary {job_id}: {status}")
            if terminal:
                break
            if time.monotonic() >= deadline:
                fail(f"TRAINING: job {job_id} exceeded {timeout_seconds} seconds")
            time.sleep(30)
            poll = _request_with_retry(
                "GET",
                f"{API_ORIGIN}/v1/training-jobs/{job_id}",
                headers=_auth_headers(api_key),
            )
            final_payload = _json_response(poll, label="TRAINING lifecycle poll")
    except (HarnessFailure, OSError, urllib.error.URLError, ValueError):
        _stop_and_confirm(api_key, job_id)
        raise

    status, _ = _training_status(final_payload)
    if status not in {"complete", "artifact_ready", "deployed"}:
        fail(f"TRAINING: canary ended unsuccessfully with status {status}")

    delete_successful = (
        os.getenv("FASTINO_DOCS_CANARY_DELETE_SUCCESSFUL", "true").lower() == "true"
    )
    try:
        _verify_completed_training_job(
            api_key,
            job_id,
            final_payload,
            document,
            operations,
        )
    finally:
        active_error = sys.exc_info()[0] is not None
        if delete_successful:
            try:
                _cleanup_training_job(api_key, job_id)
            except HarnessFailure as cleanup_error:
                if not active_error:
                    raise
                print(f"warning: {cleanup_error}", file=sys.stderr)
    print(f"Full training lifecycle passed: job={job_id}, status={status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("static", "published", "api", "training-replay", "training-lifecycle"),
    )
    args = parser.parse_args()
    try:
        if args.command == "static":
            run_static()
        elif args.command == "published":
            run_published()
        elif args.command == "api":
            run_api(replay_training=False)
        elif args.command == "training-replay":
            _run_training_replay(_require_api_key())
            print("Training idempotency replay passed")
        else:
            run_training_lifecycle()
    except (
        HarnessFailure,
        json.JSONDecodeError,
        OSError,
        urllib.error.URLError,
        ValueError,
    ) as error:
        print(f"agent docs harness failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
