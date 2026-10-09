"""Unit tests for the agent-facing documentation harness."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from decimal import Decimal
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "agent_harness.py"
sys.path.insert(0, str(MODULE_PATH.parent))
import check_api_docs as CHECKS

SPEC = importlib.util.spec_from_file_location("agent_harness", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class AgentHarnessTests(unittest.TestCase):
    def test_secret_bearing_workflows_are_pinned_to_main(self) -> None:
        daily = (HARNESS.ROOT / ".github/workflows/agent-docs-harness.yml").read_text()
        lifecycle = (
            HARNESS.ROOT / ".github/workflows/agent-docs-training-lifecycle.yml"
        ).read_text()

        self.assertEqual(daily.count("github.ref == 'refs/heads/main'"), 2)
        self.assertEqual(daily.count("ref: main"), 2)
        self.assertIn("github.ref == 'refs/heads/main'", lifecycle)
        self.assertIn("ref: main", lifecycle)

    def test_openapi_resolves_required_agent_operations(self) -> None:
        document = HARNESS._load_openapi()
        operations = HARNESS._validate_openapi(document)

        self.assertEqual(HARNESS.REQUIRED_OPERATIONS, set(operations))

    def test_every_openapi_operation_has_one_visible_reference_binding(self) -> None:
        document = HARNESS._load_openapi()

        self.assertEqual(
            CHECKS._api_reference_findings(HARNESS.ROOT, document),
            [],
        )

    def test_api_reference_has_only_inference_and_training_groups(self) -> None:
        config = json.loads((HARNESS.ROOT / "docs.json").read_text())
        english = next(
            language
            for language in config["navigation"]["languages"]
            if language["language"] == "en"
        )
        reference = next(
            tab for tab in english["tabs"] if tab["tab"] == "API Reference"
        )

        self.assertEqual(
            [group["group"] for group in reference["pages"]],
            ["Inference", "Training"],
        )

    def test_method_path_stops_at_full_width_punctuation(self) -> None:
        line = "调用 GLiDE 端点 POST /v1/systemone：发送 state，GET /v1/inferences，列出记录。"

        self.assertEqual(
            CHECKS.METHOD_PATH.findall(line),
            [("POST", "/v1/systemone"), ("GET", "/v1/inferences")],
        )

    def test_localized_key_retirement_notices_are_migration_lines(self) -> None:
        for line in (
            "Ältere `pio_sk_`-Schlüssel sind außer Betrieb genommen.",
            "veraltete `pio_sk_`-Schlüssel wurden eingestellt.",
            "Les anciennes clés `pio_sk_` sont retirées.",
            "Las claves heredadas `pio_sk_` están retiradas.",
        ):
            with self.subTest(line=line):
                self.assertTrue(CHECKS._is_migration_line(line))
        self.assertFalse(CHECKS._is_migration_line("Use a `pio_sk_` key."))

    def test_openapi_config_uses_current_mintlify_api_shape(self) -> None:
        config = json.loads((HARNESS.ROOT / "docs.json").read_text())

        self.assertNotIn("openapi", config)
        self.assertEqual(config["api"]["openapi"], "openapi.json")

    def test_visible_authentication_page_is_not_shadowed_by_redirect(self) -> None:
        config = {
            "navigation": {
                "languages": [
                    {
                        "language": "en",
                        "hidden": True,
                        "tabs": [{"tab": "Documentation", "pages": ["authentication"]}],
                    }
                ]
            },
            "redirects": [
                {
                    "source": "/authentication",
                    "destination": "/quickstart",
                }
            ],
        }

        self.assertIn(
            "REDIRECT: visible navigation page /authentication is a redirect source",
            HARNESS._redirect_findings(config),
        )

    def test_localized_navigation_rejects_unprefixed_page(self) -> None:
        config = {
            "navigation": {
                "languages": [
                    {
                        "language": "en",
                        "hidden": True,
                        "tabs": [{"tab": "Documentation", "pages": ["quickstart"]}],
                    },
                    {
                        "language": "cn",
                        "hidden": True,
                        "tabs": [{"tab": "文档", "pages": ["quickstart"]}],
                    },
                ]
            },
            "redirects": [],
        }

        self.assertTrue(
            any(
                "cn navigation has unlocalized pages" in finding
                for finding in HARNESS._locale_parity_findings(config, {})
            )
        )

    def test_redirect_chains_are_rejected(self) -> None:
        config = {
            "navigation": {"languages": []},
            "redirects": [
                {"source": "/old", "destination": "/older"},
                {"source": "/older", "destination": "/quickstart"},
            ],
        }

        self.assertTrue(
            any("chains through /older" in finding for finding in HARNESS._redirect_findings(config))
        )

    def test_locales_match_english_topology_bindings_and_links(self) -> None:
        config = json.loads((HARNESS.ROOT / "docs.json").read_text())
        operations = HARNESS._validate_openapi(HARNESS._load_openapi())

        self.assertEqual(HARNESS._locale_parity_findings(config, operations), [])

    def test_agent_index_orders_guides_before_contracts(self) -> None:
        self.assertEqual(CHECKS._journey_order_findings(HARNESS.ROOT), [])

    def test_operation_matching_handles_path_parameters_and_queries(self) -> None:
        operations = {
            ("get", "/v1/training-jobs/{job_id}"): {"responses": {}},
            ("get", "/v1/base-models"): {"responses": {}},
        }

        training = HARNESS._operation_for_path(
            operations,
            "GET",
            "/v1/training-jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6",
        )
        catalog = HARNESS._operation_for_path(
            operations,
            "GET",
            "/v1/base-models?supports_training=true",
        )

        self.assertEqual(training[0], "/v1/training-jobs/{job_id}")
        self.assertEqual(catalog[0], "/v1/base-models")

    def test_schema_validation_rejects_missing_required_field(self) -> None:
        document = {
            "components": {
                "schemas": {
                    "Request": {
                        "type": "object",
                        "required": ["model", "state"],
                        "properties": {
                            "model": {"type": "string"},
                            "state": {"type": "string"},
                        },
                    }
                }
            }
        }

        with self.assertRaisesRegex(HARNESS.HarnessFailure, "state"):
            HARNESS._validate_value(
                {"model": "fastino/GLiDE"},
                {"$ref": "#/components/schemas/Request"},
                document,
                location="request",
            )

    def test_schema_validation_accepts_documented_object(self) -> None:
        document = {"components": {"schemas": {}}}

        HARNESS._validate_value(
            {"model": "fastino/GLiDE", "store": False},
            {
                "type": "object",
                "required": ["model"],
                "additionalProperties": False,
                "properties": {
                    "model": {"type": "string"},
                    "store": {"type": "boolean"},
                },
            },
            document,
            location="request",
        )

    def test_curl_body_parser_preserves_nested_json(self) -> None:
        source = """curl https://api.fastino.ai/v1/systemone \\
  -d '{
    "state": {"customer": {"tier": "pro"}},
    "questions": {"route": {"type": "choice", "options": ["a", "b"]}}
  }'
"""

        bodies = HARNESS._curl_bodies(source)

        self.assertEqual(len(bodies), 1)
        self.assertIsNone(bodies[0][2])
        self.assertEqual(bodies[0][1]["state"]["customer"]["tier"], "pro")

    def test_documented_response_reads_first_json_block_after_heading(self) -> None:
        page = """---
title: Example
---

```json
{"not": "this one"}
```

## Response

Prose before the block.

```json Response
{"answers": {"route": {"type": "choice", "choice": "a"}}}
```

```json
{"also": "ignored"}
```
"""

        self.assertEqual(
            HARNESS._documented_response(page),
            {"answers": {"route": {"type": "choice", "choice": "a"}}},
        )

    def test_documented_response_requires_heading_and_valid_json(self) -> None:
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "no '## Response' heading"):
            HARNESS._documented_response('```json\n{"a": 1}\n```\n')
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "has no ```json block"):
            HARNESS._documented_response("## Response\n\n```bash\necho hi\n```\n")
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "does not parse"):
            HARNESS._documented_response("## Response\n\n```json\n{oops}\n```\n")

    def test_cookbook_request_parses_single_curl_and_auth_header(self) -> None:
        page = """```bash
export FASTINO_API_KEY=fast_sk_example
```

```bash
curl https://api.fastino.ai/v1/chat/completions \\
  -H "Authorization: Bearer $FASTINO_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "fastino/gliner2.5-base-v1", "messages": [{"role": "user", "content": "Hi"}]}'
```
"""

        request = HARNESS._cookbook_request(page)

        self.assertEqual(request.path, "/v1/chat/completions")
        self.assertTrue(request.bearer)
        self.assertEqual(request.body["model"], "fastino/gliner2.5-base-v1")

    def test_cookbook_request_rejects_wrong_auth_header_and_duplicate_curls(self) -> None:
        glide_with_bearer = """```bash
curl https://api.fastino.ai/v1/systemone \\
  -H "Authorization: Bearer $FASTINO_API_KEY" \\
  -d '{"model": "fastino/GLiDE"}'
```
"""
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "X-API-Key"):
            HARNESS._cookbook_request(glide_with_bearer)
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "found 2"):
            HARNESS._cookbook_request(glide_with_bearer * 2)

    def test_cookbook_findings_reject_empty_directory(self) -> None:
        self.assertEqual(
            HARNESS._cookbook_findings({}, {}, []),
            [f"COOKBOOK: {HARNESS.COOKBOOK_DIR}/ exists but has no cookbook pages"],
        )

    GLIDE_DOCUMENTED = {
        "model": "fastino/GLiDE",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.91,
                "probabilities": {"billing": 0.91, "tech": 0.09},
            },
            "urgency": {
                "type": "score",
                "score": 3,
                "expected_level": 2.8,
                "confidence": 0.7,
                "probabilities": {"3": 0.7},
                "legend": {"3": "high"},
            },
            "refund": {"type": "noul", "noul": 0.82, "confidence": 0.64},
        },
        "usage": {"input_tokens": 100, "output_tokens": 3},
    }

    def _glide_live(self, **overrides: dict[str, object]) -> dict[str, object]:
        live = json.loads(json.dumps(self.GLIDE_DOCUMENTED))
        live["answers"]["route"]["confidence"] = 0.55
        live["answers"]["urgency"]["expected_level"] = 2.1
        live["answers"]["refund"]["confidence"] = 0.2
        live["usage"] = {"input_tokens": 999, "output_tokens": 9}
        for name, fields in overrides.items():
            live["answers"][name].update(fields)
        return live

    def test_compare_glide_answers_ignores_probabilities_and_usage(self) -> None:
        self.assertEqual(
            HARNESS.compare_glide_answers(self.GLIDE_DOCUMENTED, self._glide_live(refund={"noul": 0.51})),
            [],
        )

    def test_compare_glide_answers_reports_choice_mismatch(self) -> None:
        mismatches = HARNESS.compare_glide_answers(
            self.GLIDE_DOCUMENTED, self._glide_live(route={"choice": "tech"})
        )

        self.assertEqual(len(mismatches), 1)
        self.assertIn("answers.route.choice", mismatches[0])

    def test_compare_glide_answers_reports_score_mismatch(self) -> None:
        mismatches = HARNESS.compare_glide_answers(
            self.GLIDE_DOCUMENTED, self._glide_live(urgency={"score": 2})
        )

        self.assertEqual(len(mismatches), 1)
        self.assertIn("answers.urgency.score", mismatches[0])

    def test_compare_glide_answers_reports_noul_flipped_across_half(self) -> None:
        mismatches = HARNESS.compare_glide_answers(
            self.GLIDE_DOCUMENTED, self._glide_live(refund={"noul": 0.31})
        )

        self.assertEqual(len(mismatches), 1)
        self.assertIn("opposite sides of 0.5", mismatches[0])

    def test_compare_glide_answers_reports_different_answer_set(self) -> None:
        live = self._glide_live()
        del live["answers"]["refund"]

        self.assertIn(
            "answers differ",
            HARNESS.compare_glide_answers(self.GLIDE_DOCUMENTED, live)[0],
        )

    GLINER_DOCUMENTED = {
        "entities": {
            "person": [{"text": "Ada Lovelace", "confidence": 0.98, "start": 0, "end": 12}],
            "location": [{"text": "London", "confidence": 0.95, "start": 44, "end": 50}],
        },
        "sentiment": {"label": "positive", "confidence": 0.9},
        "relation_extraction": {
            "works_with": [
                {"head": {"text": "Ada Lovelace"}, "tail": {"text": "Charles Babbage"}}
            ]
        },
    }

    def test_compare_gliner_content_matches_and_allows_extra_live_entities(self) -> None:
        live = json.loads(json.dumps(self.GLINER_DOCUMENTED))
        live["entities"]["person"][0]["confidence"] = 0.6
        live["entities"]["person"].append(
            {"text": "Charles Babbage", "confidence": 0.9, "start": 23, "end": 38}
        )
        live["sentiment"]["confidence"] = 0.51
        live["relation_extraction"]["works_with"].append(
            {"head": {"text": "Charles Babbage"}, "tail": {"text": "Ada Lovelace"}}
        )

        self.assertEqual(HARNESS.compare_gliner_content(self.GLINER_DOCUMENTED, live), [])

    def test_compare_gliner_content_reports_missing_entity(self) -> None:
        live = json.loads(json.dumps(self.GLINER_DOCUMENTED))
        live["entities"]["location"] = []

        mismatches = HARNESS.compare_gliner_content(self.GLINER_DOCUMENTED, live)

        self.assertEqual(mismatches, ["entities.location: live is missing documented ['London']"])

    def test_compare_gliner_content_reports_missing_relation_pair(self) -> None:
        live = json.loads(json.dumps(self.GLINER_DOCUMENTED))
        live["relation_extraction"]["works_with"] = [
            {"head": {"text": "Charles Babbage"}, "tail": {"text": "Ada Lovelace"}}
        ]

        mismatches = HARNESS.compare_gliner_content(self.GLINER_DOCUMENTED, live)

        self.assertEqual(len(mismatches), 1)
        self.assertIn("relation_extraction.works_with", mismatches[0])
        self.assertIn("('Ada Lovelace', 'Charles Babbage')", mismatches[0])

    def test_compare_gliner_content_reports_label_and_key_mismatch(self) -> None:
        live = json.loads(json.dumps(self.GLINER_DOCUMENTED))
        live["sentiment"]["label"] = "negative"
        del live["relation_extraction"]

        mismatches = HARNESS.compare_gliner_content(self.GLINER_DOCUMENTED, live)

        self.assertEqual(len(mismatches), 2)
        self.assertIn("top-level keys differ", mismatches[0])
        self.assertIn("sentiment.label", mismatches[1])

    def test_run_cookbooks_prints_pass_and_fail_lines(self) -> None:
        page_text = """---
title: Refunds
---

```bash
curl https://api.fastino.ai/v1/systemone \\
  -H "X-API-Key: $FASTINO_API_KEY" \\
  -d '{"model": "fastino/GLiDE"}'
```

## Response

```json
{"model": "fastino/GLiDE", "answers": {"refund": {"type": "noul", "noul": 0.8}}}
```
"""
        live_yes = {"model": "fastino/GLiDE", "answers": {"refund": {"type": "noul", "noul": 0.7}}}
        live_no = {"model": "fastino/GLiDE", "answers": {"refund": {"type": "noul", "noul": 0.2}}}
        responses = [
            HARNESS.Response(200, "https://api.fastino.ai/v1/systemone", {}, json.dumps(body).encode())
            for body in (live_yes, live_no)
        ]
        with tempfile.TemporaryDirectory() as directory:
            pages = [Path(directory) / "a.mdx", Path(directory) / "b.mdx"]
            for page in pages:
                page.write_text(page_text)
            with (
                mock.patch.object(HARNESS, "_require_api_key", return_value="fast_sk_test"),
                mock.patch.object(HARNESS, "_load_openapi", return_value={}),
                mock.patch.object(HARNESS, "_validate_openapi", return_value={}),
                mock.patch.object(HARNESS, "_validate_operation_response"),
                mock.patch.object(HARNESS, "_request_with_retry", side_effect=responses) as request,
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
                self.assertRaisesRegex(HARNESS.HarnessFailure, "1 of 2 cookbook pages failed"),
            ):
                HARNESS.run_cookbooks(pages)

        lines = stdout.getvalue().splitlines()
        self.assertRegex(lines[0], r"^PASS .*a\.mdx \d+ms$")
        self.assertRegex(lines[1], r"^FAIL .*b\.mdx: .*opposite sides of 0\.5")
        self.assertEqual(request.call_args.kwargs["headers"], {"X-API-Key": "fast_sk_test"})
        self.assertNotIn("fast_sk_test", stdout.getvalue())

    def test_training_terminal_fallback_handles_canonical_statuses(self) -> None:
        status, terminal = HARNESS._training_status(
            {"normalized_status": "artifact_ready", "status": "complete"}
        )

        self.assertEqual(status, "artifact_ready")
        self.assertTrue(terminal)

    def test_training_status_rejects_active_terminal_mismatch(self) -> None:
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "inconsistent"):
            HARNESS._training_status(
                {"status": "running", "is_terminal_status": True}
            )

    def test_schema_validation_enforces_multiple_of(self) -> None:
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "multiple"):
            HARNESS._validate_value(
                0.205,
                {"type": "number", "multipleOf": 0.01},
                {},
                location="validation_data_percentage",
            )

    def test_schema_validation_enforces_one_of_exclusivity(self) -> None:
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "valid under each"):
            HARNESS._validate_value(
                1,
                {"oneOf": [{"type": "number"}, {"minimum": 0}]},
                {},
                location="ambiguous",
            )

    def test_request_retry_retries_transport_failure(self) -> None:
        response = HARNESS.Response(200, "https://api.fastino.ai/v1/base-models", {}, b"{}")
        with (
            mock.patch.object(
                HARNESS,
                "_request",
                side_effect=[urllib.error.URLError("temporary"), response],
            ) as request,
            mock.patch.object(HARNESS.time, "sleep"),
        ):
            actual = HARNESS._request_with_retry(
                "GET",
                response.url,
                headers={},
            )

        self.assertIs(actual, response)
        self.assertEqual(request.call_count, 2)

    def test_billing_requires_finalized_numeric_minutes(self) -> None:
        self.assertIsNone(HARNESS._billing_minutes({"billed": False}))
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "no gpu_minutes"):
            HARNESS._billing_minutes({"billed": True, "gpu_minutes": None})
        self.assertEqual(
            HARNESS._billing_minutes({"billed": True, "gpu_minutes": "12.5"}),
            Decimal("12.5"),
        )

    def test_wait_for_billing_polls_until_finalized(self) -> None:
        pending = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs/job/billing",
            {},
            json.dumps({"billed": False, "gpu_minutes": None}).encode(),
        )
        billed = HARNESS.Response(
            200,
            pending.url,
            {},
            json.dumps({"billed": True, "gpu_minutes": "3.5"}).encode(),
        )
        with (
            mock.patch.object(
                HARNESS,
                "_request_with_retry",
                side_effect=[pending, billed],
            ),
            mock.patch.object(HARNESS.time, "monotonic", side_effect=[0, 1]),
            mock.patch.object(HARNESS.time, "sleep") as sleep,
        ):
            minutes = HARNESS._wait_for_billing("fast_sk_test", "job", timeout_seconds=30)

        self.assertEqual(minutes, Decimal("3.5"))
        sleep.assert_called_once_with(10)

    def test_wait_for_billing_fails_when_record_never_finalizes(self) -> None:
        pending = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs/job/billing",
            {},
            json.dumps({"billed": False, "gpu_minutes": None}).encode(),
        )
        with (
            mock.patch.object(HARNESS, "_request_with_retry", return_value=pending),
            mock.patch.object(HARNESS.time, "monotonic", side_effect=[0, 31]),
            self.assertRaisesRegex(HARNESS.HarnessFailure, "not finalized"),
        ):
            HARNESS._wait_for_billing("fast_sk_test", "job", timeout_seconds=30)

    def test_gliner_response_requires_requested_entity_arrays(self) -> None:
        response = HARNESS.Response(200, "https://api.fastino.ai/v1/chat/completions", {}, b"")
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"entities": {"person": ["Ada Lovelace"]}}
                        )
                    }
                }
            ]
        }
        with (
            mock.patch.object(HARNESS, "_validate_operation_response"),
            self.assertRaisesRegex(HARNESS.HarnessFailure, "location"),
        ):
            HARNESS._validate_gliner_response(
                {},
                {("post", "/v1/chat/completions"): {}},
                response,
                payload,
                label="test",
                expected_entities={"person": "Ada Lovelace", "location": "London"},
            )

    def test_gliner_response_accepts_nonempty_requested_entities(self) -> None:
        response = HARNESS.Response(200, "https://api.fastino.ai/v1/chat/completions", {}, b"")
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": {
                                    "person": ["Ada Lovelace"],
                                    "location": ["London"],
                                }
                            }
                        )
                    }
                }
            ]
        }
        with mock.patch.object(HARNESS, "_validate_operation_response"):
            HARNESS._validate_gliner_response(
                {},
                {("post", "/v1/chat/completions"): {}},
                response,
                payload,
                label="test",
                expected_entities={"person": "Ada Lovelace", "location": "London"},
            )

    def test_gliner_response_rejects_unrelated_entity_values(self) -> None:
        response = HARNESS.Response(200, "https://api.fastino.ai/v1/chat/completions", {}, b"")
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": {
                                    "person": ["Ada Lovelace"],
                                    "location": ["Paris"],
                                }
                            }
                        )
                    }
                }
            ]
        }
        with (
            mock.patch.object(HARNESS, "_validate_operation_response"),
            self.assertRaisesRegex(HARNESS.HarnessFailure, "expected values"),
        ):
            HARNESS._validate_gliner_response(
                {},
                {("post", "/v1/chat/completions"): {}},
                response,
                payload,
                label="test",
                expected_entities={"person": "Ada Lovelace", "location": "London"},
            )

    def test_stop_retries_cancellation_unconfirmed_until_terminal(self) -> None:
        stop_unconfirmed = HARNESS.Response(
            409,
            "https://api.fastino.ai/v1/training-jobs/job-1/stop",
            {},
            json.dumps({"error": {"code": "cancellation_unconfirmed"}}).encode(),
        )
        stop_accepted = HARNESS.Response(202, stop_unconfirmed.url, {}, b"{}")
        running = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs/job-1",
            {},
            json.dumps(
                {"status": "running", "is_terminal_status": False}
            ).encode(),
        )
        stopped = HARNESS.Response(
            200,
            running.url,
            {},
            json.dumps(
                {"status": "stopped", "is_terminal_status": True}
            ).encode(),
        )
        with (
            mock.patch.object(
                HARNESS,
                "_request_with_retry",
                side_effect=[stop_unconfirmed, running, stop_accepted, stopped],
            ) as request,
            mock.patch.object(HARNESS.time, "monotonic", side_effect=[0, 1, 2]),
            mock.patch.object(HARNESS.time, "sleep"),
        ):
            HARNESS._stop_and_confirm("fast_sk_test", "job-1", timeout_seconds=30)

        self.assertEqual(request.call_count, 4)

    def test_create_recovers_after_ambiguous_server_response(self) -> None:
        unavailable = HARNESS.Response(
            503,
            "https://api.fastino.ai/v1/training-jobs",
            {},
            b"temporarily unavailable",
        )
        recovered = {"id": "job-1", "model_name": "canary"}
        with (
            mock.patch.object(HARNESS, "_request_with_retry", return_value=unavailable),
            mock.patch.object(
                HARNESS,
                "_recover_training_job",
                return_value=recovered,
            ) as recover,
        ):
            actual = HARNESS._create_or_recover_training_job(
                "fast_sk_test",
                HARNESS.uuid.uuid4(),
                "canary",
                {"model_name": "canary"},
            )

        self.assertEqual(actual, recovered)
        recover.assert_called_once_with("fast_sk_test", "canary")

    def test_training_job_listing_follows_pagination(self) -> None:
        first = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs?limit=200&offset=0",
            {},
            json.dumps(
                {
                    "training_jobs": [{"id": "job-1"}],
                    "has_more": True,
                }
            ).encode(),
        )
        second = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs?limit=200&offset=1",
            {},
            json.dumps(
                {
                    "training_jobs": [{"id": "job-2"}],
                    "has_more": False,
                }
            ).encode(),
        )
        with mock.patch.object(
            HARNESS,
            "_request_with_retry",
            side_effect=[first, second],
        ) as request:
            jobs = HARNESS._list_training_jobs("fast_sk_test")

        self.assertEqual([job["id"] for job in jobs], ["job-1", "job-2"])
        self.assertIn("offset=1", request.call_args_list[1].args[1])

    def test_lifecycle_rerun_uses_new_identity_and_stops_active_prior_job(self) -> None:
        first_marker, first_name = HARNESS._lifecycle_identity("run-1", 1)
        repeated_marker, repeated_name = HARNESS._lifecycle_identity("run-1", 1)
        second_marker, _ = HARNESS._lifecycle_identity("run-1", 2)
        self.assertEqual((first_marker, first_name), (repeated_marker, repeated_name))
        self.assertNotEqual(first_marker, second_marker)

        prior_job = {
            "id": "job-1",
            "model_name": first_name,
            "status": "running",
            "is_terminal_status": False,
        }
        with (
            mock.patch.object(HARNESS, "_list_training_jobs", return_value=[prior_job]),
            mock.patch.object(HARNESS, "_stop_and_confirm") as stop,
            mock.patch.object(HARNESS, "_cleanup_training_job") as cleanup,
        ):
            HARNESS._reconcile_canary_jobs("fast_sk_test", "current-canary")

        stop.assert_called_once_with("fast_sk_test", "job-1")
        cleanup.assert_called_once_with("fast_sk_test", "job-1")

    def test_recover_training_job_requires_unique_model_name(self) -> None:
        job = {"id": "job-1", "model_name": "canary"}
        with mock.patch.object(HARNESS, "_list_training_jobs", return_value=[job]):
            self.assertEqual(HARNESS._recover_training_job("fast_sk_test", "canary"), job)
        with (
            mock.patch.object(HARNESS, "_list_training_jobs", return_value=[]),
            self.assertRaisesRegex(HARNESS.HarnessFailure, "uniquely recover"),
        ):
            HARNESS._recover_training_job("fast_sk_test", "canary")

    def test_replay_stops_and_deletes_unexpected_job(self) -> None:
        existing = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs/expected-job",
            {},
            json.dumps(
                {
                    "id": "expected-job",
                    "status": "artifact_ready",
                    "is_terminal_status": True,
                }
            ).encode(),
        )
        unexpected = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs",
            {},
            json.dumps({"id": "unexpected-job"}).encode(),
        )
        with (
            mock.patch.dict(
                HARNESS.os.environ,
                {
                    "FASTINO_DOCS_CANARY_TRAINING_JOB_ID": "expected-job",
                    "FASTINO_DOCS_CANARY_IDEMPOTENCY_KEY": "fixed-key",
                },
                clear=False,
            ),
            mock.patch.object(
                HARNESS,
                "_ready_dataset_reference",
                return_value={"name": "canary", "version": "1"},
            ),
            mock.patch.object(
                HARNESS,
                "_request_with_retry",
                side_effect=[existing, unexpected],
            ),
            mock.patch.object(
                HARNESS,
                "_cleanup_unexpected_replay_jobs",
            ) as cleanup,
            self.assertRaisesRegex(HARNESS.HarnessFailure, "stopped and deleted"),
        ):
            HARNESS._run_training_replay("fast_sk_test")

        self.assertEqual(cleanup.call_count, 2)
        cleanup.assert_has_calls(
            [mock.call("fast_sk_test", "expected-job")] * 2
        )

    def test_replay_ambiguous_responses_trigger_reconciliation(self) -> None:
        existing = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs/expected-job",
            {},
            json.dumps(
                {
                    "id": "expected-job",
                    "status": "artifact_ready",
                    "is_terminal_status": True,
                }
            ).encode(),
        )
        ambiguous = (
            urllib.error.URLError("lost response"),
            HARNESS.Response(503, "https://api.fastino.ai/v1/training-jobs", {}, b""),
            HARNESS.Response(200, "https://api.fastino.ai/v1/training-jobs", {}, b"{}"),
        )
        for outcome in ambiguous:
            with (
                self.subTest(outcome=type(outcome).__name__),
                mock.patch.dict(
                    HARNESS.os.environ,
                    {
                        "FASTINO_DOCS_CANARY_TRAINING_JOB_ID": "expected-job",
                        "FASTINO_DOCS_CANARY_IDEMPOTENCY_KEY": "fixed-key",
                    },
                    clear=False,
                ),
                mock.patch.object(
                    HARNESS,
                    "_ready_dataset_reference",
                    return_value={"name": "canary", "version": "1"},
                ),
                mock.patch.object(
                    HARNESS,
                    "_request_with_retry",
                    side_effect=[existing, outcome],
                ),
                mock.patch.object(
                    HARNESS,
                    "_cleanup_unexpected_replay_jobs",
                ) as cleanup,
                self.assertRaises(HARNESS.HarnessFailure),
            ):
                HARNESS._run_training_replay("fast_sk_test")
            self.assertEqual(cleanup.call_count, 2)
            cleanup.assert_has_calls(
                [mock.call("fast_sk_test", "expected-job")] * 2
            )

    def test_successful_replay_reconciles_before_and_after_request(self) -> None:
        expected = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs/expected-job",
            {},
            json.dumps(
                {
                    "id": "expected-job",
                    "status": "artifact_ready",
                    "is_terminal_status": True,
                }
            ).encode(),
        )
        replay = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training-jobs",
            {},
            json.dumps({"id": "expected-job"}).encode(),
        )
        with (
            mock.patch.dict(
                HARNESS.os.environ,
                {
                    "FASTINO_DOCS_CANARY_TRAINING_JOB_ID": "expected-job",
                    "FASTINO_DOCS_CANARY_IDEMPOTENCY_KEY": "fixed-key",
                },
                clear=False,
            ),
            mock.patch.object(
                HARNESS,
                "_ready_dataset_reference",
                return_value={"name": "canary", "version": "1"},
            ),
            mock.patch.object(
                HARNESS,
                "_request_with_retry",
                side_effect=[expected, replay],
            ),
            mock.patch.object(
                HARNESS,
                "_cleanup_unexpected_replay_jobs",
            ) as cleanup,
        ):
            HARNESS._run_training_replay("fast_sk_test")

        self.assertEqual(cleanup.call_count, 2)
        cleanup.assert_has_calls(
            [mock.call("fast_sk_test", "expected-job")] * 2
        )

    def test_replay_reconciliation_stops_and_deletes_nonfixture_jobs(self) -> None:
        jobs = [
            {
                "id": "expected-job",
                "model_name": "docs-agent-canary-replay",
                "status": "artifact_ready",
                "is_terminal_status": True,
            },
            {
                "id": "unexpected-active",
                "model_name": "docs-agent-canary-replay_2",
                "status": "running",
                "is_terminal_status": False,
            },
            {
                "id": "unrelated",
                "model_name": "customer-model",
                "status": "running",
                "is_terminal_status": False,
            },
        ]
        with (
            mock.patch.object(HARNESS, "_list_training_jobs", return_value=jobs),
            mock.patch.object(HARNESS, "_stop_and_confirm") as stop,
            mock.patch.object(HARNESS, "_cleanup_training_job") as cleanup,
        ):
            HARNESS._cleanup_unexpected_replay_jobs(
                "fast_sk_test",
                "expected-job",
            )

        stop.assert_called_once_with("fast_sk_test", "unexpected-active")
        cleanup.assert_called_once_with("fast_sk_test", "unexpected-active")

    def test_lifecycle_cleans_up_when_post_training_verification_fails(self) -> None:
        created = {
            "id": "job-1",
            "status": "artifact_ready",
            "is_terminal_status": True,
            "is_deployable": True,
        }
        with (
            mock.patch.dict(
                HARNESS.os.environ,
                {
                    "FASTINO_DOCS_CANARY_DELETE_SUCCESSFUL": "true",
                    "GITHUB_RUN_ID": "run-1",
                    "GITHUB_RUN_ATTEMPT": "1",
                },
                clear=False,
            ),
            mock.patch.object(HARNESS, "_require_api_key", return_value="fast_sk_test"),
            mock.patch.object(HARNESS, "_load_openapi", return_value={}),
            mock.patch.object(HARNESS, "_validate_openapi", return_value={}),
            mock.patch.object(
                HARNESS,
                "_ready_dataset_reference",
                return_value={"name": "canary", "version": "1"},
            ),
            mock.patch.object(
                HARNESS,
                "_discover_training_model",
                return_value="fastino/gliner2-base-v1",
            ),
            mock.patch.object(HARNESS, "_reconcile_canary_jobs"),
            mock.patch.object(
                HARNESS,
                "_create_or_recover_training_job",
                return_value=created,
            ),
            mock.patch.object(
                HARNESS,
                "_verify_completed_training_job",
                side_effect=HARNESS.HarnessFailure("verification failed"),
            ),
            mock.patch.object(HARNESS, "_cleanup_training_job") as cleanup,
            self.assertRaisesRegex(HARNESS.HarnessFailure, "verification failed"),
        ):
            HARNESS.run_training_lifecycle()

        cleanup.assert_called_once_with("fast_sk_test", "job-1")

    def test_completed_job_enforces_gpu_minute_ceiling(self) -> None:
        ok = HARNESS.Response(200, "https://api.fastino.ai/test", {}, b"{}")
        with (
            mock.patch.dict(
                HARNESS.os.environ,
                {"FASTINO_DOCS_CANARY_MAX_GPU_MINUTES": "0"},
                clear=False,
            ),
            mock.patch.object(HARNESS, "_request_with_retry", return_value=ok),
            mock.patch.object(HARNESS, "_wait_for_billing", return_value=Decimal("1")),
            self.assertRaisesRegex(HARNESS.HarnessFailure, "above the configured"),
        ):
            HARNESS._verify_completed_training_job(
                "fast_sk_test",
                "job-1",
                {"is_deployable": True},
                {},
                {},
            )


if __name__ == "__main__":
    unittest.main()
