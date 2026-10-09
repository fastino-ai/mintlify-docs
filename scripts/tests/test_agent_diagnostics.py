"""Unit tests for public agent-discovery diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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

    def test_timestamp_without_timezone_is_normalized_to_utc(self) -> None:
        self.assertIs(
            DIAGNOSTICS.parse_timestamp("2026-10-08T12:00:00").tzinfo,
            DIAGNOSTICS.UTC,
        )

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
                gsc_site_url=None,
                gsc_modified_manifest=None,
                max_gsc_crawl_age_days=0,
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
                gsc_site_url=None,
                gsc_modified_manifest=None,
                max_gsc_crawl_age_days=7,
            )

        message = str(raised.exception)
        self.assertIn("invalid max-age cache directive", message)
        self.assertIn("Googlebot body differs", message)
        self.assertIn("artifact digest does not match", message)

    def test_diagnostics_workflow_is_manual_only_while_published_stays_daily(
        self,
    ) -> None:
        workflow = (
            SCRIPTS.parent / ".github/workflows/agent-docs-harness.yml"
        ).read_text()
        _, remainder = workflow.split("\n  published:", maxsplit=1)
        published_job, remainder = remainder.split("\n  diagnostics:", maxsplit=1)
        diagnostics_job, _ = remainder.split("\n  authenticated-api:", maxsplit=1)

        self.assertIn(
            "github.event_name == 'schedule' || "
            "github.event_name == 'workflow_dispatch'",
            published_job,
        )
        self.assertIn("python scripts/agent_harness.py published", published_job)
        self.assertIn("if: github.event_name == 'workflow_dispatch'", diagnostics_job)
        self.assertNotIn("github.event_name == 'schedule'", diagnostics_job)
        self.assertIn("python scripts/agent_harness.py diagnostics", diagnostics_job)


if __name__ == "__main__":
    unittest.main()
