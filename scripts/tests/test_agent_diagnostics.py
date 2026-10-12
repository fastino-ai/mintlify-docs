from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime
from pathlib import Path
from threading import Barrier
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import agent_diagnostics as DIAGNOSTICS

HARNESS_SPEC = importlib.util.spec_from_file_location(
    "diagnostics_test_harness",
    SCRIPTS / "agent_harness.py",
)
if HARNESS_SPEC is None or HARNESS_SPEC.loader is None:
    raise RuntimeError("cannot load agent_harness.py")
HARNESS = importlib.util.module_from_spec(HARNESS_SPEC)
sys.modules[HARNESS_SPEC.name] = HARNESS
HARNESS_SPEC.loader.exec_module(HARNESS)


def _target(**overrides: object) -> DIAGNOSTICS.DiscoveryTarget:
    values = {
        "name": "agent card",
        "url": "https://agent.fastino.ai/.well-known/agent-card.json",
        "content_types": ("application/json",),
        "max_cache_seconds": 300,
        "required_fragments": ('"fastino-gliner"',),
    }
    values.update(overrides)
    return DIAGNOSTICS.DiscoveryTarget(**values)


def _response(**overrides: object) -> HARNESS.Response:
    values = {
        "status": 200,
        "url": _target().url,
        "headers": {
            "cache-control": "public, max-age=300",
            "content-type": "application/json; charset=utf-8",
            "age": "42",
        },
        "body": b'{"skills":[{"id":"fastino-gliner"}]}',
    }
    values.update(overrides)
    return HARNESS.Response(**values)


def _valid_index(*, artifact: bytes = b"skill") -> dict[str, object]:
    digest = f"sha256:{hashlib.sha256(artifact).hexdigest()}"
    return {
        "skills": [
            {
                "name": name,
                "url": f"/.well-known/agent-skills/{name}/skill.md",
                "digest": digest,
            }
            for name in sorted(DIAGNOSTICS.EXPECTED_SKILLS)
        ]
    }


class AgentDiagnosticsTests(unittest.TestCase):
    def test_fetches_default_and_googlebot_concurrently_without_redirects(self) -> None:
        barrier = Barrier(2, timeout=1)
        user_agents: list[str] = []

        def request(
            method: str,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            payload: object | None = None,
            follow_redirects: bool = True,
            timeout: float = 30,
        ) -> HARNESS.Response:
            self.assertEqual((method, url), ("GET", _target().url))
            self.assertIsNone(payload)
            self.assertEqual(timeout, 30)
            self.assertFalse(follow_redirects)
            assert headers is not None
            user_agents.append(headers["User-Agent"])
            barrier.wait()
            return _response()

        default, googlebot, findings = DIAGNOSTICS.fetch_discovery_responses(
            (_target(),),
            request,
        )

        self.assertEqual(findings, [])
        self.assertEqual(default[_target().url], _response())
        self.assertEqual(googlebot[_target().url], _response())
        self.assertEqual(
            set(user_agents),
            {DIAGNOSTICS.USER_AGENT, DIAGNOSTICS.GOOGLEBOT_USER_AGENT},
        )

    def test_partial_fetch_failure_is_named_and_keeps_success(self) -> None:
        def request(
            method: str,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            payload: object | None = None,
            follow_redirects: bool = True,
            timeout: float = 30,
        ) -> HARNESS.Response:
            del method, url, payload, follow_redirects, timeout
            assert headers is not None
            if headers["User-Agent"] == DIAGNOSTICS.GOOGLEBOT_USER_AGENT:
                raise urllib.error.URLError("timed out")
            return _response()

        default, googlebot, findings = DIAGNOSTICS.fetch_discovery_responses(
            (_target(),),
            request,
        )

        self.assertEqual(default, {_target().url: _response()})
        self.assertEqual(googlebot, {})
        self.assertEqual(
            findings,
            ["agent card: Googlebot fetch failed: <urlopen error timed out>"],
        )

    def test_malformed_json_response_becomes_a_named_finding(self) -> None:
        response = _response(body=b"<html>upstream error</html>")

        self.assertEqual(
            DIAGNOSTICS.agent_card_skill_findings("agent card", response),
            ["agent card: response is not valid JSON"],
        )

    def test_cache_budget_includes_stale_window_and_requires_freshness(self) -> None:
        combined = _response(
            headers={
                "cache-control": "public, max-age=3000, stale-while-revalidate=1000",
                "content-type": "application/json",
            }
        )
        missing = _response(
            headers={
                "cache-control": "public",
                "content-type": "application/json",
            }
        )

        self.assertEqual(
            DIAGNOSTICS.discovery_response_findings(
                _target(max_cache_seconds=3600),
                combined,
            ),
            ["agent card: cache lifetime plus stale window 4000s exceeds 3600s"],
        )
        self.assertEqual(
            DIAGNOSTICS.discovery_response_findings(_target(), missing),
            ["agent card: missing cache freshness directive"],
        )

    def test_cache_age_uses_advertised_freshness_window(self) -> None:
        response = _response(
            headers={
                "cache-control": "public, max-age=300, stale-while-revalidate=100",
                "content-type": "application/json",
                "age": "401",
            }
        )

        self.assertEqual(
            DIAGNOSTICS.discovery_response_findings(
                _target(max_cache_seconds=1000),
                response,
            ),
            ["agent card: Age 401s exceeds advertised freshness window 400s"],
        )

    def test_malformed_cache_headers_are_findings_and_do_not_abort_validation(
        self,
    ) -> None:
        response = _response(
            headers={
                "cache-control": (
                    "public, max-age=forever, stale-while-revalidate=eventually"
                ),
                "content-type": "application/json",
                "age": "old",
            },
            body=b"{}",
        )

        self.assertEqual(
            DIAGNOSTICS.discovery_response_findings(_target(), response),
            [
                "agent card: invalid max-age cache directive: 'forever'",
                (
                    "agent card: invalid stale-while-revalidate cache directive: "
                    "'eventually'"
                ),
                "agent card: invalid Age header 'old'",
                "agent card: response is missing '\"fastino-gliner\"'",
            ],
        )

    def test_googlebot_must_receive_default_status_body_and_policy(self) -> None:
        googlebot = _response(
            status=403,
            headers={
                "cache-control": "no-store",
                "content-type": "text/plain",
            },
            body=b"bot blocked",
        )

        self.assertEqual(
            DIAGNOSTICS.googlebot_parity_findings(
                _target(),
                _response(),
                googlebot,
            ),
            [
                "agent card: Googlebot status 403 differs from default status 200",
                "agent card: Googlebot body differs from default response",
                "agent card: Googlebot cache-control differs from default response",
                "agent card: Googlebot content-type differs from default response",
            ],
        )

    def test_skill_index_requires_exact_unique_inventory(self) -> None:
        valid = _valid_index()
        DIAGNOSTICS.parse_skill_index(valid)
        mutations: list[tuple[str, dict[str, object], str]] = []

        duplicate_name = json.loads(json.dumps(valid))
        duplicate_name["skills"].append(dict(duplicate_name["skills"][0]))
        mutations.append(("duplicate name", duplicate_name, "duplicate skill names"))

        duplicate_url = json.loads(json.dumps(valid))
        duplicate_url["skills"][1]["url"] = duplicate_url["skills"][0]["url"]
        mutations.append(("duplicate URL", duplicate_url, "duplicate skill URLs"))

        wrong_path = json.loads(json.dumps(valid))
        wrong_path["skills"][0]["url"] = "/.well-known/agent-skills/wrong/skill.md"
        mutations.append(("wrong path", wrong_path, "URL must be"))

        malformed_digest = json.loads(json.dumps(valid))
        malformed_digest["skills"][0]["digest"] = "sha256:stale"
        mutations.append(("malformed digest", malformed_digest, "invalid SHA-256"))

        missing = json.loads(json.dumps(valid))
        missing["skills"].pop()
        mutations.append(("missing entry", missing, "exactly one entry"))

        for name, payload, message in mutations:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(DIAGNOSTICS.DiagnosticsFailure, message),
            ):
                DIAGNOSTICS.parse_skill_index(payload)

    def test_skill_digest_covers_every_validated_entry(self) -> None:
        artifact = b"hello"
        entries = DIAGNOSTICS.parse_skill_index(_valid_index(artifact=artifact))
        artifacts = {entry.name: artifact for entry in entries}
        artifacts[entries[0].name] = b"stale"

        self.assertEqual(
            DIAGNOSTICS.skill_digest_findings(entries, artifacts),
            [f"{entries[0].name}: artifact digest does not match skill index"],
        )

    def test_agent_card_rejects_duplicate_skill_ids(self) -> None:
        skill_ids = sorted(DIAGNOSTICS.EXPECTED_SKILLS)
        response = _response(
            body=json.dumps(
                {"skills": [{"id": name} for name in [*skill_ids, skill_ids[0]]]}
            ).encode()
        )

        self.assertIn(
            "expected one entry for each skill",
            DIAGNOSTICS.agent_card_skill_findings("agent card", response)[0],
        )

    def test_timestamp_requires_an_explicit_offset(self) -> None:
        self.assertEqual(
            DIAGNOSTICS.parse_timestamp("2026-10-08T12:00:00Z"),
            datetime(2026, 10, 8, 12, tzinfo=DIAGNOSTICS.UTC),
        )
        with self.assertRaisesRegex(ValueError, "must include a UTC offset"):
            DIAGNOSTICS.parse_timestamp("2026-10-08T12:00:00")

    def test_gsc_no_crawl_reports_coverage_reason(self) -> None:
        findings = DIAGNOSTICS.gsc_inspection_findings(
            url="https://docs.fastino.ai/openapi.json",
            index_status={
                "verdict": "NEUTRAL",
                "coverageState": "URL is unknown to Google",
                "robotsTxtState": "ALLOWED",
                "pageFetchState": "UNKNOWN",
            },
            last_modified=datetime(2026, 10, 8, tzinfo=DIAGNOSTICS.UTC),
            now=datetime(2026, 10, 9, tzinfo=DIAGNOSTICS.UTC),
            max_crawl_age_days=7,
        )

        self.assertEqual(
            findings,
            [
                (
                    "https://docs.fastino.ai/openapi.json: GSC has no "
                    "lastCrawlTime (verdict=NEUTRAL, coverage=URL is unknown "
                    "to Google, robots=ALLOWED, fetch=UNKNOWN)"
                )
            ],
        )

    def test_gsc_reports_fetch_blocker_even_with_recent_crawl(self) -> None:
        findings = DIAGNOSTICS.gsc_inspection_findings(
            url="https://docs.fastino.ai/llms.txt",
            index_status={
                "lastCrawlTime": "2026-10-08T12:00:00Z",
                "verdict": "FAIL",
                "coverageState": "Blocked by robots.txt",
                "robotsTxtState": "DISALLOWED",
                "pageFetchState": "BLOCKED_ROBOTS_TXT",
            },
            last_modified=datetime(2026, 10, 7, tzinfo=DIAGNOSTICS.UTC),
            now=datetime(2026, 10, 9, tzinfo=DIAGNOSTICS.UTC),
            max_crawl_age_days=7,
        )

        self.assertEqual(
            findings,
            [
                (
                    "https://docs.fastino.ai/llms.txt: GSC indexing blocker "
                    "(verdict=FAIL, coverage=Blocked by robots.txt, "
                    "robots=DISALLOWED, fetch=BLOCKED_ROBOTS_TXT)"
                )
            ],
        )

    def test_gsc_reports_each_stale_crawl_reason_and_accepts_freshness(self) -> None:
        clean_status = {
            "lastCrawlTime": "2026-10-08T12:00:00Z",
            "verdict": "PASS",
            "coverageState": "Submitted and indexed",
            "robotsTxtState": "ALLOWED",
            "pageFetchState": "SUCCESSFUL",
        }
        self.assertEqual(
            DIAGNOSTICS.gsc_inspection_findings(
                url="https://docs.fastino.ai/llms.txt",
                index_status=clean_status,
                last_modified=datetime(2026, 10, 8, 11, tzinfo=DIAGNOSTICS.UTC),
                now=datetime(2026, 10, 9, tzinfo=DIAGNOSTICS.UTC),
                max_crawl_age_days=7,
            ),
            [],
        )
        self.assertEqual(
            DIAGNOSTICS.gsc_inspection_findings(
                url="https://docs.fastino.ai/llms.txt",
                index_status=clean_status,
                last_modified=datetime(2026, 10, 8, 13, tzinfo=DIAGNOSTICS.UTC),
                now=datetime(2026, 10, 20, tzinfo=DIAGNOSTICS.UTC),
                max_crawl_age_days=7,
            ),
            [
                (
                    "https://docs.fastino.ai/llms.txt: GSC crawl "
                    "2026-10-08T12:00:00Z predates content update "
                    "2026-10-08T13:00:00+00:00"
                ),
                (
                    "https://docs.fastino.ai/llms.txt: GSC crawl "
                    "2026-10-08T12:00:00Z exceeds the 7-day freshness limit"
                ),
            ],
        )

    def test_gsc_requires_clean_submitted_sitemap(self) -> None:
        self.assertEqual(
            DIAGNOSTICS.gsc_sitemap_findings("https://docs.fastino.ai/", []),
            [
                (
                    "https://docs.fastino.ai/: GSC has no submitted sitemap "
                    "https://docs.fastino.ai/sitemap.xml"
                )
            ],
        )
        self.assertEqual(
            DIAGNOSTICS.gsc_sitemap_findings(
                "https://docs.fastino.ai/",
                [
                    {
                        "path": "https://docs.fastino.ai/sitemap.xml",
                        "errors": "2",
                        "warnings": "1",
                        "isPending": False,
                        "lastDownloaded": "2026-10-09T00:00:00Z",
                    }
                ],
            ),
            [
                (
                    "https://docs.fastino.ai/sitemap.xml: GSC reports "
                    "2 errors and 1 warnings"
                )
            ],
        )

    def test_gsc_rejects_domain_properties_and_invalid_manifests_before_fetch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {"https://docs.fastino.ai/llms.txt": "2026-10-08T12:00:00"}
                )
            )
            with self.assertRaisesRegex(
                DIAGNOSTICS.DiagnosticsFailure,
                "URL-prefix property",
            ):
                DIAGNOSTICS.prepare_gsc(
                    site_url="sc-domain:fastino.ai",
                    modified_manifest=manifest,
                )
            with self.assertRaisesRegex(
                DIAGNOSTICS.DiagnosticsFailure,
                "must include a UTC offset",
            ):
                DIAGNOSTICS.prepare_gsc(
                    site_url="https://docs.fastino.ai/",
                    modified_manifest=manifest,
                )

    def test_gsc_inspection_sends_expected_requests(self) -> None:
        calls: list[tuple[str, object]] = []
        crawl_time = datetime.now(DIAGNOSTICS.UTC)

        class Request:
            def __init__(self, payload: object) -> None:
                self.payload = payload

            def execute(self) -> object:
                return self.payload

        class Sitemaps:
            def list(self, *, siteUrl: str) -> Request:
                calls.append(("sitemap", siteUrl))
                return Request(
                    {
                        "sitemap": [
                            {
                                "path": f"{siteUrl.rstrip('/')}/sitemap.xml",
                                "errors": "0",
                                "warnings": "0",
                                "isPending": False,
                                "lastDownloaded": crawl_time.isoformat(),
                            }
                        ]
                    }
                )

        class UrlInspection:
            def index(self) -> UrlInspection:
                return self

            def inspect(self, *, body: dict[str, str]) -> Request:
                calls.append(("inspection", body))
                return Request(
                    {
                        "inspectionResult": {
                            "indexStatusResult": {
                                "lastCrawlTime": crawl_time.isoformat(),
                                "verdict": "PASS",
                                "coverageState": "Submitted and indexed",
                                "robotsTxtState": "ALLOWED",
                                "pageFetchState": "SUCCESSFUL",
                            }
                        }
                    }
                )

        class Service:
            def sitemaps(self) -> Sitemaps:
                return Sitemaps()

            def urlInspection(self) -> UrlInspection:
                return UrlInspection()

        url = "https://docs.fastino.ai/llms.txt"
        context = DIAGNOSTICS.GscContext(
            "https://docs.fastino.ai/",
            ((url, crawl_time - DIAGNOSTICS.timedelta(days=1)),),
            Service(),
            (RuntimeError,),
        )

        self.assertEqual(
            DIAGNOSTICS.inspect_gsc(context, max_crawl_age_days=7),
            [],
        )
        self.assertEqual(
            calls,
            [
                ("sitemap", "https://docs.fastino.ai/"),
                (
                    "inspection",
                    {
                        "inspectionUrl": url,
                        "siteUrl": "https://docs.fastino.ai/",
                        "languageCode": "en-US",
                    },
                ),
            ],
        )

    def test_gsc_transport_failures_become_findings(self) -> None:
        class BrokenRequest:
            def execute(self) -> object:
                raise RuntimeError("network unavailable")

        class BrokenResource:
            def list(self, **_: object) -> BrokenRequest:
                return BrokenRequest()

            def index(self) -> BrokenResource:
                return self

            def inspect(self, **_: object) -> BrokenRequest:
                return BrokenRequest()

        class BrokenService:
            def sitemaps(self) -> BrokenResource:
                return BrokenResource()

            def urlInspection(self) -> BrokenResource:
                return BrokenResource()

        url = "https://docs.fastino.ai/llms.txt"
        context = DIAGNOSTICS.GscContext(
            "https://docs.fastino.ai/",
            ((url, datetime.now(DIAGNOSTICS.UTC)),),
            BrokenService(),
            (RuntimeError,),
        )

        self.assertEqual(
            DIAGNOSTICS.inspect_gsc(context, max_crawl_age_days=7),
            [
                (
                    "https://docs.fastino.ai/: GSC sitemap inspection failed: "
                    "network unavailable"
                ),
                f"{url}: GSC URL inspection failed: network unavailable",
            ],
        )

    def test_gsc_malformed_responses_become_findings(self) -> None:
        class Request:
            def __init__(self, payload: object) -> None:
                self.payload = payload

            def execute(self) -> object:
                return self.payload

        class Resource:
            def __init__(self, payload: object) -> None:
                self.payload = payload

            def list(self, **_: object) -> Request:
                return Request(self.payload)

            def index(self) -> Resource:
                return self

            def inspect(self, **_: object) -> Request:
                return Request(self.payload)

        class Service:
            def __init__(self, sitemap: object, inspection: object) -> None:
                self.sitemap = sitemap
                self.inspection = inspection

            def sitemaps(self) -> Resource:
                return Resource(self.sitemap)

            def urlInspection(self) -> Resource:
                return Resource(self.inspection)

        url = "https://docs.fastino.ai/llms.txt"
        modified_urls = ((url, datetime.now(DIAGNOSTICS.UTC)),)
        cases = (
            (
                None,
                None,
                [
                    "https://docs.fastino.ai/: GSC sitemap response is invalid",
                    f"{url}: GSC URL inspection response is invalid",
                ],
            ),
            (
                {"sitemap": "invalid"},
                {"inspectionResult": None},
                [
                    "https://docs.fastino.ai/: GSC sitemap response is invalid",
                    f"{url}: GSC returned an invalid inspectionResult",
                ],
            ),
            (
                {"sitemap": []},
                {"inspectionResult": {"indexStatusResult": []}},
                [
                    (
                        "https://docs.fastino.ai/: GSC has no submitted sitemap "
                        "https://docs.fastino.ai/sitemap.xml"
                    ),
                    f"{url}: GSC returned an invalid indexStatusResult",
                ],
            ),
        )

        for sitemap, inspection, expected in cases:
            with self.subTest(sitemap=sitemap, inspection=inspection):
                context = DIAGNOSTICS.GscContext(
                    "https://docs.fastino.ai/",
                    modified_urls,
                    Service(sitemap, inspection),
                    (RuntimeError,),
                )
                self.assertEqual(
                    DIAGNOSTICS.inspect_gsc(context, max_crawl_age_days=7),
                    expected,
                )

    def test_max_gsc_age_must_be_positive_before_network_access(self) -> None:
        with (
            mock.patch.object(DIAGNOSTICS, "fetch_discovery_responses") as fetch,
            self.assertRaisesRegex(
                DIAGNOSTICS.DiagnosticsFailure,
                "greater than zero",
            ),
        ):
            DIAGNOSTICS.run_diagnostics(
                request=mock.Mock(),
                validate_openapi=mock.Mock(),
                max_gsc_crawl_age_days=0,
            )
        fetch.assert_not_called()

    def test_half_specified_gsc_flags_fail_before_network_access(self) -> None:
        with (
            mock.patch.object(DIAGNOSTICS, "fetch_discovery_responses") as fetch,
            self.assertRaisesRegex(
                DIAGNOSTICS.DiagnosticsFailure,
                "requires --gsc-site-url and --gsc-modified-manifest",
            ),
        ):
            DIAGNOSTICS.run_diagnostics(
                request=mock.Mock(),
                validate_openapi=mock.Mock(),
                gsc_site_url="https://docs.fastino.ai/",
            )
        fetch.assert_not_called()

    def test_orchestration_surfaces_index_googlebot_cache_and_digest_mutations(
        self,
    ) -> None:
        index_url = "https://docs.fastino.ai/.well-known/agent-skills/index.json"
        index = _valid_index(artifact=b"skill")
        index["skills"][0]["digest"] = f"sha256:{'0' * 64}"
        index_body = json.dumps(index).encode()
        base_target = DIAGNOSTICS.DiscoveryTarget(
            "canonical skill index",
            index_url,
            ("application/json",),
            3600,
        )

        def request(
            method: str,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            payload: object | None = None,
            follow_redirects: bool = True,
            timeout: float = 30,
        ) -> HARNESS.Response:
            del method, payload, follow_redirects, timeout
            assert headers is not None
            is_googlebot = headers["User-Agent"] == DIAGNOSTICS.GOOGLEBOT_USER_AGENT
            if url == index_url:
                return HARNESS.Response(
                    200,
                    url,
                    {
                        "cache-control": "public, max-age=forever",
                        "content-type": "application/json",
                    },
                    b'{"bot":true}' if is_googlebot else index_body,
                )
            return HARNESS.Response(
                200,
                url,
                {
                    "cache-control": "public, max-age=300",
                    "content-type": "text/markdown",
                },
                b"skill",
            )

        with (
            mock.patch.object(
                DIAGNOSTICS,
                "base_discovery_targets",
                return_value=(base_target,),
            ),
            self.assertRaises(DIAGNOSTICS.DiagnosticsFailure) as raised,
        ):
            DIAGNOSTICS.run_diagnostics(
                request=request,
                validate_openapi=mock.Mock(),
            )

        message = str(raised.exception)
        self.assertIn("invalid max-age cache directive", message)
        self.assertIn("Googlebot body differs", message)
        self.assertIn("artifact digest does not match", message)


if __name__ == "__main__":
    unittest.main()
