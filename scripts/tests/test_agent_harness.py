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
import agent_diagnostics as DIAGNOSTICS
import check_api_docs as CHECKS
import generate_localized_llms as LOCALIZED

SPEC = importlib.util.spec_from_file_location("agent_harness", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)

CONFIG_CREATE_PAGES = (
    "training/encoder-quickstart.mdx",
    "concepts/datasets.mdx",
    "guides/fine-tune-gliner.mdx",
    "api-reference/training-jobs/create.mdx",
    "api-reference/overview.mdx",
    "concepts/gliner-2-5-decide-thresholds.mdx",
)


def _json_objects(text: str) -> list[object]:
    decoder = json.JSONDecoder()
    objects: list[object] = []
    index = 0
    while index < len(text):
        if text[index] == "{":
            try:
                obj, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                index += 1
                continue
            objects.append(obj)
            index = end
        else:
            index += 1
    return objects


def _is_config_create_request(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    if "model_name" not in obj or "datasets" not in obj:
        return False
    if "id" in obj or "status" in obj:
        return False
    return True


class AgentHarnessTests(unittest.TestCase):
    def test_localized_agent_instructions_cover_current_public_routes(self) -> None:
        for locale, copy in LOCALIZED.LOCALE_COPY.items():
            with self.subTest(locale=locale):
                self.assertIn("/v1/systemone", copy.instructions)
                self.assertIn("fastino/GLiDE", copy.instructions)
                self.assertIn("base_model", copy.instructions)
                self.assertIn("/v1/training/jobs", copy.instructions)
                self.assertIn("/v1/training/usage", copy.instructions)

    def test_api_reference_and_training_pages_are_indexed_in_every_locale(self) -> None:
        config = json.loads((HARNESS.ROOT / "docs.json").read_text(encoding="utf-8"))
        languages = {
            language["language"]: language
            for language in config["navigation"]["languages"]
        }
        redirects = {
            redirect["source"].strip("/"): redirect["destination"].strip("/")
            for redirect in config["redirects"]
        }
        pages = sorted(
            {
                path.relative_to(HARNESS.ROOT).as_posix().removesuffix(".mdx")
                for path in (HARNESS.ROOT / "api-reference").rglob("*.mdx")
            }
            | {"training"}
            | {
                path.relative_to(HARNESS.ROOT).as_posix().removesuffix(".mdx")
                for path in (HARNESS.ROOT / "training").rglob("*.mdx")
            }
        )

        for locale in ("en", "cn", "de", "es", "fr"):
            prefix = "" if locale == "en" else f"{locale}/"
            visible = HARNESS._visible_navigation_pages(languages[locale])
            llms = (HARNESS.ROOT / f"{prefix}llms.txt").read_text(encoding="utf-8")
            for page in pages:
                localized = f"{prefix}{page}"
                with self.subTest(locale=locale, page=page):
                    self.assertTrue((HARNESS.ROOT / f"{localized}.mdx").is_file())
                    if localized in redirects:
                        self.assertNotIn(localized, visible)
                        self.assertIn(f"/{redirects[localized]}.md", llms)
                    else:
                        self.assertIn(localized, visible)
                        self.assertIn(f"/{localized}.md", llms)

    def test_secret_bearing_workflows_are_pinned_to_main(self) -> None:
        daily = (HARNESS.ROOT / ".github/workflows/agent-docs-harness.yml").read_text()
        lifecycle = (
            HARNESS.ROOT / ".github/workflows/agent-docs-training-lifecycle.yml"
        ).read_text()

        self.assertEqual(daily.count("github.ref == 'refs/heads/main'"), 2)
        self.assertEqual(daily.count("ref: main"), 2)
        self.assertIn("github.ref == 'refs/heads/main'", lifecycle)
        self.assertIn("ref: main", lifecycle)

    def test_diagnostics_command_runs_diagnostics_module_and_reports_targets(self) -> None:
        with (
            mock.patch.object(
                DIAGNOSTICS,
                "run_diagnostics",
                return_value=12,
            ) as diagnostics,
            mock.patch.object(HARNESS, "_request") as request,
            mock.patch.object(HARNESS, "_validate_openapi") as validate_openapi,
            mock.patch.object(HARNESS.sys, "argv", ["agent_harness.py", "diagnostics"]),
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = HARNESS.main()

        self.assertEqual(exit_code, 0)
        diagnostics.assert_called_once_with(
            request=request,
            validate_openapi=validate_openapi,
        )
        self.assertEqual(
            stdout.getvalue(),
            "Agent diagnostics passed: 12 discovery targets\n",
        )

    def test_training_job_create_matches_deployed_request_contract(self) -> None:
        document = HARNESS._load_openapi()
        create = document["components"]["schemas"]["TrainingJobCreate"]
        properties = create["properties"]
        response = document["components"]["schemas"]["TrainingJobResponse"]

        self.assertEqual(create["required"], ["model_name", "datasets", "base_model"])
        self.assertIn("base_model", properties)
        self.assertNotIn("type", properties)
        self.assertNotIn("model", properties)
        self.assertNotIn("x-fastino-preview", create)
        self.assertNotIn("preview", create.get("description", "").lower())
        self.assertIn("type", response["properties"])

    def test_config_create_excludes_interactive_decoder_contract(self) -> None:
        document = HARNESS._load_openapi()
        properties = document["components"]["schemas"]["TrainingJobCreate"]["properties"]
        retired_decoder_fields = {
            "packing",
            "mask_history",
            "system_prompt",
            "rl_config",
        }

        self.assertTrue(retired_decoder_fields.isdisjoint(properties))
        self.assertEqual(properties["training_algorithm"]["const"], "sft")
        self.assertNotIn("enum", properties["training_algorithm"])

    def test_english_config_create_requests_use_base_model_and_omit_type(self) -> None:
        found = 0
        for page in CONFIG_CREATE_PAGES:
            text = (HARNESS.ROOT / page).read_text(encoding="utf-8")
            requests = [
                obj for obj in _json_objects(text) if _is_config_create_request(obj)
            ]
            self.assertTrue(requests, f"{page} has no config-create request example")
            for obj in requests:
                self.assertIn("base_model", obj, page)
                self.assertNotIn("type", obj, page)
                self.assertNotIn("model", obj, page)
            found += len(requests)
        self.assertGreaterEqual(found, len(CONFIG_CREATE_PAGES))

    def test_public_openapi_has_no_compatibility_paths(self) -> None:
        operations = CHECKS._operations(HARNESS._load_openapi())

        self.assertFalse(
            any(path.startswith("/v1/training/compat/tinker") for _, path in operations)
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

    def test_run_static_uses_explicit_root_and_spec(self) -> None:
        spec = {"openapi": "3.1.0", "paths": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs.json").write_text('{"navigation": {}}')
            with (
                mock.patch.object(HARNESS, "_validate_openapi", return_value={}),
                mock.patch.object(HARNESS, "local_docs_findings", return_value=[]) as local,
                mock.patch.object(HARNESS, "_journey_discovery_findings", return_value=[]) as discovery,
                mock.patch.object(HARNESS, "_training_navigation_findings", return_value=[]),
                mock.patch.object(HARNESS, "_private_training_route_findings", return_value=[]),
                mock.patch.object(HARNESS, "_local_link_findings", return_value=[]),
                mock.patch.object(HARNESS, "_redirect_findings", return_value=[]),
                mock.patch.object(HARNESS, "_locale_parity_findings", return_value=[]),
                mock.patch.object(HARNESS, "_journey_findings", return_value=[]),
                mock.patch.object(HARNESS, "_cookbook_pages", return_value=None) as cookbooks,
            ):
                findings = HARNESS.run_static(root, spec)

        self.assertEqual(findings, [])
        local.assert_called_once_with(root, spec)
        discovery.assert_called_once_with(root)
        cookbooks.assert_called_once_with(root)

    def test_journey_findings_collect_every_schema_failure(self) -> None:
        document = {"components": {"schemas": {}}}
        operation = {
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["required"],
                        }
                    }
                }
            }
        }
        page = """---
title: Example
---
```bash
curl -X POST https://api.fastino.ai/v1/test -d '{}'
```
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AGENTS.md").write_text("Use GET /v1/base-models.\n")
            for name in ("a.mdx", "b.mdx"):
                (root / name).write_text(page)
            with mock.patch.object(HARNESS, "JOURNEY_PAGES", ("a.mdx", "b.mdx")):
                findings = HARNESS._journey_findings(
                    document,
                    {("post", "/v1/test"): operation},
                    root,
                )

        schema_findings = [
            finding for finding in findings if "is a required property" in finding
        ]
        self.assertEqual(len(schema_findings), 2)
        self.assertTrue(any("a.mdx request" in finding for finding in schema_findings))
        self.assertTrue(any("b.mdx request" in finding for finding in schema_findings))

    def test_cookbook_findings_collect_independent_page_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "broken.mdx"
            page.write_text("no frontmatter, curl, or response")

            findings = HARNESS._cookbook_findings({}, {}, [page], root)

        self.assertEqual(len(findings), 3)
        self.assertTrue(any("invalid frontmatter" in finding for finding in findings))
        self.assertTrue(any("cURL block" in finding for finding in findings))
        self.assertTrue(any("Response" in finding for finding in findings))

    def test_agent_index_orders_guides_before_contracts(self) -> None:
        self.assertEqual(CHECKS._journey_order_findings(HARNESS.ROOT), [])

    def test_training_navigation_matches_the_unified_flow(self) -> None:
        config = json.loads((HARNESS.ROOT / "docs.json").read_text())

        self.assertEqual(HARNESS._training_navigation_findings(config), [])

    def test_private_training_routes_are_not_published(self) -> None:
        self.assertEqual(HARNESS._private_training_route_findings(), [])

    def test_operation_matching_handles_path_parameters_and_queries(self) -> None:
        operations = {
            ("get", "/v1/training/jobs/{job_id}"): {"responses": {}},
            ("get", "/v1/base-models"): {"responses": {}},
        }

        training = HARNESS._operation_for_path(
            operations,
            "GET",
            "/v1/training/jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6",
        )
        catalog = HARNESS._operation_for_path(
            operations,
            "GET",
            "/v1/base-models?supports_training=true",
        )

        self.assertEqual(training[0], "/v1/training/jobs/{job_id}")
        self.assertEqual(catalog[0], "/v1/base-models")

    def test_sdk_compatibility_base_is_an_allowed_non_openapi_url(self) -> None:
        self.assertTrue(
            CHECKS._path_matches_route("/v1/training/compat/tinker", {})
        )
        self.assertFalse(CHECKS._path_matches_route("/tinker", {}))
        self.assertFalse(CHECKS._path_matches_route("/not-a-public-route", {}))

    def test_skill_routes_do_not_accept_the_sdk_compatibility_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = root / ".mintlify/skills/example/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("Call GET /v1/training/compat/tinker.\n")

            self.assertEqual(
                CHECKS._skill_route_findings(root, {"paths": {}}),
                [
                    ".mintlify/skills/example/SKILL.md:1: "
                    "GET /v1/training/compat/tinker is not a public OpenAPI operation"
                ],
            )

    def test_training_routes_require_the_canonical_deployed_route(self) -> None:
        deployed = {"GET": [CHECKS.re.compile(r"^/v1/training/jobs/?$")]}

        self.assertTrue(
            CHECKS._path_matches_route("/v1/training/jobs", deployed, "GET")
        )
        self.assertFalse(
            CHECKS._path_matches_route("/v1/training/jobs", {}, "GET")
        )
        self.assertFalse(
            CHECKS._path_matches_route(
                "/v1/training/jobs/job-1/forward",
                {},
                "POST",
            )
        )

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
            "https://api.fastino.ai/v1/training/jobs/job/billing",
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
            "https://api.fastino.ai/v1/training/jobs/job/billing",
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
            "https://api.fastino.ai/v1/training/jobs/job-1/stop",
            {},
            json.dumps({"error": {"code": "cancellation_unconfirmed"}}).encode(),
        )
        stop_accepted = HARNESS.Response(202, stop_unconfirmed.url, {}, b"{}")
        running = HARNESS.Response(
            200,
            "https://api.fastino.ai/v1/training/jobs/job-1",
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
            "https://api.fastino.ai/v1/training/jobs",
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
            "https://api.fastino.ai/v1/training/jobs?limit=200&offset=0",
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
            "https://api.fastino.ai/v1/training/jobs?limit=200&offset=1",
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
            "https://api.fastino.ai/v1/training/jobs/expected-job",
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
            "https://api.fastino.ai/v1/training/jobs",
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
            "https://api.fastino.ai/v1/training/jobs/expected-job",
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
            HARNESS.Response(503, "https://api.fastino.ai/v1/training/jobs", {}, b""),
            HARNESS.Response(200, "https://api.fastino.ai/v1/training/jobs", {}, b"{}"),
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
            "https://api.fastino.ai/v1/training/jobs/expected-job",
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
            "https://api.fastino.ai/v1/training/jobs",
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
