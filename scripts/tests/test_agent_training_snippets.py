from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

HARNESS_SPEC = importlib.util.spec_from_file_location(
    "agent_harness",
    SCRIPTS / "agent_harness.py",
)
if HARNESS_SPEC is None or HARNESS_SPEC.loader is None:
    raise RuntimeError("cannot load agent_harness.py")
HARNESS = importlib.util.module_from_spec(HARNESS_SPEC)
sys.modules[HARNESS_SPEC.name] = HARNESS
HARNESS_SPEC.loader.exec_module(HARNESS)

SPEC = importlib.util.spec_from_file_location(
    "training_snippets",
    SCRIPTS / "training_snippets.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPTS / 'training_snippets.py'}")
SNIPPETS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SNIPPETS
SPEC.loader.exec_module(SNIPPETS)


EXPECTED_PAGES = (
    "training.mdx",
    "training/quickstart.mdx",
    "training/migrate-from-tinker.mdx",
    "concepts/datasets.mdx",
    "guides/fine-tune-gliner.mdx",
    "training/training-loop.mdx",
    "training/verifiable-rewards.mdx",
    "training/checkpoints.mdx",
    "training/auto.mdx",
)
EXPECTED_EMPTY_PAGES = (
    "training.mdx",
    "training/training-loop.mdx",
    "training/auto.mdx",
)
EXPECTED_FENCE_COUNTS = {
    "training.mdx": 0,
    "training/quickstart.mdx": 16,
    "training/migrate-from-tinker.mdx": 4,
    "concepts/datasets.mdx": 8,
    "guides/fine-tune-gliner.mdx": 8,
    "training/training-loop.mdx": 0,
    "training/verifiable-rewards.mdx": 7,
    "training/checkpoints.mdx": 4,
    "training/auto.mdx": 0,
}
EXPECTED_LIVE_GETS_WITH_FIXTURES = 9
PARITY_PAGES = (
    "training/quickstart.mdx",
    "training/migrate-from-tinker.mdx",
    "training/training-loop.mdx",
    "training/verifiable-rewards.mdx",
    "training/checkpoints.mdx",
    "api-reference/training-jobs/download-checkpoint.mdx",
)
ONE_FUTURE_CLAIMS = {
    "": "Requests cancellation of one future",
    "cn": "请求取消一个 future",
    "de": "Fordert den Abbruch eines Future an",
    "es": "Solicita la cancelación de un future",
    "fr": "Demande l'annulation d'une future",
}
RUN_KEY_EXPORT_CLAIMS = (
    "export external weights",
    "导出外部权重",
    "keine externen Gewichte exportieren",
    "exportar pesos externos",
    "exporter des poids externes",
)


@dataclasses.dataclass(frozen=True)
class TinkerSdkSupport:
    package: str
    rest_client_methods: frozenset[str]
    unsupported_calls: frozenset[str]
    retired_text: frozenset[str]


TINKER_SDK_SUPPORT = TinkerSdkSupport(
    package="tinker==0.32.0",
    rest_client_methods=frozenset({
        "list_training_runs",
        "get_training_run",
        "list_checkpoints",
        "list_user_checkpoints",
        "publish_checkpoint_from_tinker_path",
        "unpublish_checkpoint_from_tinker_path",
        "set_checkpoint_ttl_from_tinker_path",
        "get_weights_info_by_tinker_path",
        "delete_checkpoint_from_tinker_path",
        "get_checkpoint_archive_url",
        "get_current_checkpoint_storage_usage",
        "get_billing_usage",
    }),
    unsupported_calls=frozenset({
        "retrieve_futures",
        "save_weights_external",
        "external_weights_urls",
    }),
    retired_text=frozenset({
        "tinker==0.33.1",
        "training_runs_api_enabled",
        "/api/v1/{operation}",
        "/v1/training/compat/tinker/api/",
    }),
)


def retired_claim_hits(text: str) -> set[str]:
    return {token for token in TINKER_SDK_SUPPORT.retired_text if token in text}


def _between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        raise AssertionError(f"missing section markers {start!r} .. {end!r}")
    return text.split(start, 1)[1].split(end, 1)[0]


def fence_bodies(mdx: str) -> list[str]:
    return re.findall(r"```[^\n]*\n(.*?)```", mdx, flags=re.DOTALL)


def code_spans(mdx: str) -> set[str]:
    without_fences = re.sub(r"```.*?```", "", mdx, flags=re.DOTALL)
    return set(re.findall(r"`([^`\n]+)`", without_fences))


def _claim_texts() -> list[tuple[Path, str]]:
    paths = list(ROOT.rglob("*.mdx"))
    paths.append(ROOT / "AGENTS.md")
    paths.extend((ROOT / ".mintlify" / "skills").rglob("SKILL.md"))
    return [(path, path.read_text(encoding="utf-8")) for path in paths]


def _future_cancel_row(text: str) -> str:
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("|") and "`sample`" in line
    ]
    if len(rows) != 1:
        raise AssertionError(
            f"expected one Future cancel row containing `sample`, found {rows}"
        )
    return rows[0]


def _snippet(page: str, ordinal: int) -> object:
    matches = [
        snippet
        for snippet in SNIPPETS.collect_snippets()
        if snippet.page == page and snippet.ordinal == ordinal
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one snippet {page}#{ordinal}, found {len(matches)}")
    return matches[0]


class AgentTrainingSnippetsTests(unittest.TestCase):
    def test_sdk_contract_examples_do_not_use_stale_fields(self) -> None:
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in ROOT.rglob("*.mdx")
        )
        self.assertNotIn("model_data.model_id", corpus)
        self.assertNotIn("plans to rename that request field", corpus)
        self.assertNotIn("compatibility adapter", corpus.lower())

        skill = (ROOT / ".mintlify/skills/fastino-fine-tune/SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn('"base_model": "fastino/gliner2-base-v1"', skill)
        self.assertNotIn('"type": "config"', skill)
        self.assertNotIn('"model": "fastino/gliner2-base-v1"', skill)
        self.assertNotIn("upcoming unified Training API", skill)

    def test_checkpoint_download_examples_match_openapi_response_fields(self) -> None:
        expected = {
            "success",
            "run_id",
            "checkpoint_id",
            "download_url",
            "expires_in_seconds",
            "file_name",
        }
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / f"{prefix}api-reference/training-jobs/download-checkpoint.mdx"
            text = path.read_text(encoding="utf-8")
            match = re.search(r"```json Response\n(?P<body>\{.*?\})\n```", text, re.DOTALL)
            self.assertIsNotNone(match, path)
            payload = json.loads(match.group("body"))
            self.assertEqual(set(payload), expected, path)
            self.assertNotIn("url", payload, path)

    def test_checkpoint_list_documents_config_and_interactive_shapes(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / f"{prefix}api-reference/training-jobs/checkpoints.mdx"
            text = path.read_text(encoding="utf-8")
            self.assertIn('"job_id":', text, path)
            self.assertIn('"run_id":', text, path)
            self.assertIn('"checkpoint_id": "sampler_weights/', text, path)

    def test_stop_separates_interactive_and_config_behavior(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            stop_path = ROOT / f"{prefix}api-reference/training-jobs/stop.mdx"
            stop = stop_path.read_text(encoding="utf-8")
            for field in ("`success`", "`job_id`", "`status`"):
                self.assertIn(field, stop, stop_path)
            self.assertIn("`202`", stop, stop_path)
            self.assertRegex(stop, r"(?:config|Config|配置)", stop_path)
            self.assertNotRegex(stop, r"(?:interactive|交互).{0,30}UUID", stop_path)
            self.assertRegex(
                stop,
                r"\| `422` \|[^\n]+(?:JSON|corps|cuerpo|请求体|Anfragetext)",
                stop_path,
            )

            terminate_path = ROOT / f"{prefix}api-reference/training-jobs/terminate.mdx"
            terminate = terminate_path.read_text(encoding="utf-8")
            self.assertRegex(terminate, r"\| `400` \|[^\n]+UUID", terminate_path)
            self.assertRegex(
                terminate,
                r"\| `422` \|[^\n]+(?:JSON|corps|cuerpo|请求体|Anfragetext)",
                terminate_path,
            )

    def test_sampler_checkpoint_examples_set_an_explicit_ttl(self) -> None:
        for path in ROOT.glob("**/training/*.mdx"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"save_weights_for_sampler\(", text):
                call = text[match.start():match.start() + 180]
                self.assertIn("ttl_seconds=", call, f"{path}: {call}")

    def test_interactive_billing_and_sdk_limits_are_documented_in_every_locale(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            overview = (ROOT / f"{prefix}training.mdx").read_text(encoding="utf-8")
            migration = (ROOT / f"{prefix}training/migrate-from-tinker.mdx").read_text(
                encoding="utf-8"
            )

            for marker in ("forward", "checkpoint"):
                self.assertIn(marker, overview.lower(), f"{prefix}training.mdx")
            for marker in (
                "/v1/training/jobs",
                "target_prompt_logprobs",
                "prompt_alt_tokens_k",
                "encoded_text",
                "AdamW",
                "`403`",
            ):
                self.assertIn(marker, migration, f"{prefix}training/migrate-from-tinker.mdx")

        english = (ROOT / "training/migrate-from-tinker.mdx").read_text(encoding="utf-8")
        english_overview = (ROOT / "training.mdx").read_text(encoding="utf-8")
        self.assertIn("GB-month", english_overview)
        self.assertIn("Reads interactive runs only", english)
        self.assertIn("weight-download access", english)
        self.assertIn("does not make the path loadable by another team", english)
        self.assertNotIn("Reads canonical Fastino jobs", english)

    def test_tinker_sdk_support_matches_pioneer_contract(self) -> None:
        support = TINKER_SDK_SUPPORT
        for path, text in _claim_texts():
            self.assertEqual(retired_claim_hits(text), set(), path)

        english = (ROOT / "training/migrate-from-tinker.mdx").read_text(encoding="utf-8")
        install_fences = [body for body in fence_bodies(english) if "tinker==" in body]
        self.assertEqual(len(install_fences), 2)
        for body in install_fences:
            self.assertIn(support.package, body)
        self.assertIn(
            support.package,
            _between(english, "## Supported client behavior", "### Training and sampling"),
        )
        self.assertIn(support.package, english.split("## Migration checklist", 1)[1])

        futures = _between(english, "### Futures and administration", "### Unsupported calls")
        future_spans = code_spans(futures)
        self.assertTrue(support.rest_client_methods <= future_spans, future_spans)
        supported = _between(english, "## Supported client behavior", "### Unsupported calls")
        unsupported = _between(english, "### Unsupported calls", "## Troubleshoot the connection")
        for name in support.unsupported_calls:
            self.assertIn(f"`{name}`", unsupported)
            self.assertNotIn(name, supported)
        self.assertIn("Session trace export", unsupported)
        self.assertIn("Audit-log entries", unsupported)

        cancel = _future_cancel_row(english)
        self.assertIn("`sample`", cancel)
        self.assertIn("unfinished sampling future", cancel)
        self.assertIn("Training-operation futures are not cancelled", cancel)
        self.assertNotIn("one future", cancel)
        for locale, claim in ONE_FUTURE_CLAIMS.items():
            prefix = f"{locale}/" if locale else ""
            page = ROOT / f"{prefix}training/migrate-from-tinker.mdx"
            text = page.read_text(encoding="utf-8")
            row = _future_cancel_row(text)
            self.assertIn("`sample`", row, page)
            self.assertNotIn(claim, text, page)
            for phrase in RUN_KEY_EXPORT_CLAIMS:
                self.assertNotIn(phrase, text, page)

        loop = (ROOT / "training/training-loop.mdx").read_text(encoding="utf-8")
        self.assertIn("Only sample operations can be cancelled", loop)
        self.assertNotIn("Resolve or cancel futures individually", loop)

    def test_native_interactive_semantics_are_documented(self) -> None:
        interactive = ROOT / "api-reference/training-jobs/interactive"
        create = (interactive / "create.mdx").read_text(encoding="utf-8")
        operation = (interactive / "get-operation.mdx").read_text(encoding="utf-8")
        cancel = (interactive / "cancel-operation.mdx").read_text(encoding="utf-8")
        sample = (interactive / "sample.mdx").read_text(encoding="utf-8")
        usage = (interactive / "storage-usage.mdx").read_text(encoding="utf-8")

        self.assertIn("`base_model`", create)
        self.assertIn("`from_checkpoint`", create)
        self.assertIn("`load_optimizer`", create)
        self.assertIn("do not send `seq_id` values or heartbeats", create)
        self.assertIn("at most 25 seconds", operation)
        for status in ("`completed`", "`failed`", "`cancelled`"):
            self.assertIn(status, operation)
        self.assertIn("sample operation", cancel.lower())
        self.assertIn("Training operations cannot be cancelled", cancel)
        self.assertIn("after the interactive job is stopped", sample)
        self.assertIn("checkpoint storage", usage.lower())
        self.assertIn("team-scoped", usage)
        self.assertIn("does not report compute usage or total billing", usage)

    def test_localized_training_pages_keep_english_code_and_identifiers(self) -> None:
        for page in PARITY_PAGES:
            english = (ROOT / page).read_text(encoding="utf-8")
            english_fences = fence_bodies(english)
            english_spans = code_spans(english)
            for locale in ("cn", "de", "es", "fr"):
                path = ROOT / locale / page
                text = path.read_text(encoding="utf-8")
                self.assertEqual(
                    fence_bodies(text),
                    english_fences,
                    f"{path} fence bodies differ",
                )
                local_spans = code_spans(text)
                self.assertEqual(
                    local_spans,
                    english_spans,
                    f"{path} code spans differ: {sorted(local_spans ^ english_spans)}",
                )

    def test_retired_claim_guard_rejects_banned_text(self) -> None:
        cases = (
            ('uv add "tinker==0.33.1"\n', {"tinker==0.33.1"}),
            ("training_runs_api_enabled\n", {"training_runs_api_enabled"}),
            ("/api/v1/{operation}", {"/api/v1/{operation}"}),
            (
                "/v1/training/compat/tinker/api/v1/cancel_future\n",
                {"/v1/training/compat/tinker/api/"},
            ),
            ("https://api.fastino.ai/v1/training/compat/tinker\n", set()),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(retired_claim_hits(text), expected)

    def test_inventory_is_exactly_47_fences_across_9_training_api_pages(self) -> None:
        pages = SNIPPETS.collect_pages()
        snippets = SNIPPETS.collect_snippets()
        counts = {page: 0 for page in pages}
        for snippet in snippets:
            counts[snippet.page] += 1

        self.assertEqual(pages, EXPECTED_PAGES)
        self.assertEqual(len(snippets), 47)
        self.assertEqual(len(pages), 9)
        self.assertEqual(counts, EXPECTED_FENCE_COUNTS)
        self.assertEqual(
            tuple(page for page, count in counts.items() if count == 0),
            EXPECTED_EMPTY_PAGES,
        )

    def test_unclassified_snippet_fails(self) -> None:
        snippet = SNIPPETS.FencedSnippet(
            page="probe.mdx",
            ordinal=0,
            language="rust",
            body='fn main() { println!("hi"); }\n',
            body_sha256="0" * 64,
        )

        with self.assertRaisesRegex(HARNESS.HarnessFailure, r"UNCLASSIFIED: probe\.mdx#0"):
            SNIPPETS.classify(snippet)

    def test_census_hash_drift_fails_until_rewritten(self) -> None:
        entries = SNIPPETS.census_entries(SNIPPETS.collect_pages(), SNIPPETS.collect_snippets())
        drifted = [dataclasses.replace(entries[0], body_sha256="a" * 64), *entries[1:]]

        findings = SNIPPETS.compare_census(entries, drifted)

        self.assertTrue(any("CENSUS:" in finding and "hash" in finding for finding in findings))
        self.assertEqual(SNIPPETS.compare_census(entries, entries), [])

    def test_live_transport_is_get_only(self) -> None:
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "only GET"):
            SNIPPETS.execute_safe_get("POST", "/v1/base-models", api_key="fast_sk_test")
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "body is not allowed"):
            SNIPPETS.execute_safe_get(
                "GET",
                "/v1/base-models",
                api_key="fast_sk_test",
                payload={"unused": True},
            )
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "SAFE_GET_TEMPLATES"):
            SNIPPETS.execute_safe_get("GET", "/v1/training/jobs", api_key="fast_sk_test")
        with self.assertRaisesRegex(HARNESS.HarnessFailure, "only GET"):
            SNIPPETS.execute_safe_get("DELETE", "/v1/datasets/docs-example-ner", api_key="fast_sk_test")

        captured: list[tuple[str, str, object, bool]] = []

        def request(
            method: str,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            payload: object | None = None,
            follow_redirects: bool = True,
            timeout: float = 30,
        ) -> HARNESS.Response:
            captured.append((method, url, payload, follow_redirects))
            return HARNESS.Response(200, url, {}, b'{"models":[]}')

        response = SNIPPETS.execute_safe_get(
            "GET",
            "/v1/base-models?supports_training=true",
            api_key="fast_sk_test",
            request_fn=request,
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(captured, [
            (
                "GET",
                f"{HARNESS.API_ORIGIN}/v1/base-models?supports_training=true",
                None,
                False,
            )
        ])

    def test_live_transport_refuses_redirects(self) -> None:
        def request(
            method: str,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            payload: object | None = None,
            follow_redirects: bool = True,
            timeout: float = 30,
        ) -> HARNESS.Response:
            self.assertFalse(follow_redirects)
            return HARNESS.Response(
                302,
                url,
                {"location": "https://evil.example/steal"},
                b"",
            )

        with self.assertRaisesRegex(HARNESS.HarnessFailure, "redirect"):
            SNIPPETS.execute_safe_get(
                "GET",
                "/v1/base-models",
                api_key="fast_sk_test",
                request_fn=request,
            )

    def test_real_corpus_plans_no_live_post_put_or_delete(self) -> None:
        env = {
            "FASTINO_DOCS_CANARY_TRAINING_JOB_ID": "job-canary",
            "FASTINO_DOCS_CANARY_DATASET": "canary-dataset",
        }
        planned = [
            SNIPPETS.plan_live(snippet, SNIPPETS.classify(snippet), env)
            for snippet in SNIPPETS.collect_snippets()
        ]
        gets = [plan for plan in planned if plan.action == "get"]
        mutating_live = [
            plan
            for plan in planned
            if plan.action == "get" and plan.method != "GET"
        ]

        self.assertEqual(mutating_live, [])
        self.assertEqual(len(gets), EXPECTED_LIVE_GETS_WITH_FIXTURES)
        self.assertTrue(all(plan.method == "GET" for plan in gets))
        self.assertTrue(all(plan.payload is None for plan in gets))
        self.assertFalse(
            any(
                plan.action == "get"
                and any(token in plan.path for token in ("/upload", "/deploy", "/chat/completions"))
                for plan in planned
            )
        )

        create = SNIPPETS.classify(_snippet("training/quickstart.mdx", 3))
        self.assertEqual(create.coverage, "static")
        self.assertEqual(create.mapped_command, "training-replay")
        self.assertEqual(create.risk, "mutating")

        deploy = SNIPPETS.classify(_snippet("guides/fine-tune-gliner.mdx", 6))
        self.assertEqual(deploy.coverage, "static")
        self.assertEqual(deploy.mapped_command, "training-lifecycle")
        self.assertIn(deploy.risk, {"mutating", "blocker"})

        inference = SNIPPETS.classify(_snippet("guides/fine-tune-gliner.mdx", 7))
        self.assertEqual(inference.coverage, "static")
        self.assertEqual(inference.mapped_command, "blocker")

        upload = SNIPPETS.classify(_snippet("concepts/datasets.mdx", 3))
        self.assertEqual(upload.coverage, "static")
        self.assertEqual(upload.mapped_command, "blocker")

    def test_missing_placeholders_print_skip_and_do_not_guess(self) -> None:
        job = _snippet("training/quickstart.mdx", 4)
        dataset = _snippet("concepts/datasets.mdx", 6)
        catalog = _snippet("training/quickstart.mdx", 2)

        skipped_job = SNIPPETS.plan_live(job, SNIPPETS.classify(job), {})
        skipped_dataset = SNIPPETS.plan_live(dataset, SNIPPETS.classify(dataset), {})
        catalog_plan = SNIPPETS.plan_live(catalog, SNIPPETS.classify(catalog), {})

        self.assertEqual(skipped_job.action, "skipped")
        self.assertIn("SKIP:", skipped_job.detail)
        self.assertIn("FASTINO_DOCS_CANARY_TRAINING_JOB_ID", skipped_job.detail)
        self.assertNotIn("JOB_ID", skipped_job.path or "")
        self.assertEqual(skipped_dataset.action, "skipped")
        self.assertIn("FASTINO_DOCS_CANARY_DATASET", skipped_dataset.detail)
        self.assertNotIn("docs-example-ner", skipped_dataset.path or "")
        self.assertEqual(catalog_plan.action, "get")
        self.assertEqual(catalog_plan.path, "/v1/base-models?supports_training=true")

        buffer = io.StringIO()
        with mock.patch.object(sys, "stdout", buffer):
            SNIPPETS.report_live_plans([skipped_job, skipped_dataset, catalog_plan])
        printed = buffer.getvalue()
        self.assertIn("SKIP:", printed)
        self.assertIn("FASTINO_DOCS_CANARY_TRAINING_JOB_ID", printed)
        self.assertIn("FASTINO_DOCS_CANARY_DATASET", printed)

    def test_live_run_skips_missing_fixtures_and_runs_catalog_reads(self) -> None:
        response = HARNESS.Response(
            200,
            f"{HARNESS.API_ORIGIN}/v1/base-models?supports_training=true",
            {},
            b'{"models":[]}',
        )
        with (
            mock.patch.dict(
                SNIPPETS.os.environ,
                {
                    "FASTINO_API_KEY": "fast_sk_test",
                    "FASTINO_DOCS_CANARY_TRAINING_JOB_ID": "",
                    "FASTINO_DOCS_CANARY_DATASET": "",
                },
                clear=False,
            ),
            mock.patch.object(
                SNIPPETS,
                "execute_safe_get",
                return_value=response,
            ) as request,
            mock.patch.object(SNIPPETS, "_validate_live_get") as validate,
            mock.patch.object(SNIPPETS, "run_checkpoint_parser") as checkpoint_parser,
        ):
            SNIPPETS.run_training_snippets(live=True)

        self.assertEqual(request.call_count, 2)
        self.assertTrue(
            all(call.args[:2] == ("GET", "/v1/base-models?supports_training=true") for call in request.call_args_list)
        )
        self.assertEqual(validate.call_count, 2)
        checkpoint_parser.assert_not_called()

    def test_redacts_fast_sk_tokens_from_errors_and_output(self) -> None:
        leaked = (
            "AUTH failed for fast_sk_abc123SECRET "
            "header=X-API-Key: fast_sk_abc123SECRET"
        )

        self.assertEqual(
            SNIPPETS.redact_secrets(leaked),
            "AUTH failed for fast_sk_[REDACTED] header=X-API-Key: fast_sk_[REDACTED]",
        )
        self.assertNotIn("abc123SECRET", SNIPPETS.redact_secrets(leaked))

        with self.assertRaisesRegex(HARNESS.HarnessFailure, r"fast_sk_\[REDACTED\]") as raised:
            SNIPPETS.fail_redacted("upstream said fast_sk_livekey999")
        self.assertNotIn("livekey999", str(raised.exception))

    def test_get_last_checkpoint_reads_newest_complete_record(self) -> None:
        snippet = _snippet("training/checkpoints.mdx", 1)
        self.assertIn("def get_last_checkpoint", snippet.body)
        classification = SNIPPETS.classify(snippet)
        self.assertEqual(classification.kind, "python-checkpoint-parser")
        self.assertEqual(classification.rule, "python-checkpoint-exec")

        with tempfile.TemporaryDirectory() as raw_directory:
            log_path = Path(raw_directory) / "loop.jsonl"
            log_path.write_text(
                '{"state_path": "s1", "step": 1}\n'
                "not-json\n"
                '{"state_path": "s2", "step": 2}\n'
                "{malformed\n",
                encoding="utf-8",
            )

            record = SNIPPETS.run_checkpoint_parser(snippet.body, str(log_path))

        self.assertEqual(record, {"state_path": "s2", "step": 2})

    def test_config_create_snippets_are_static_mutating_without_preview_blocker(self) -> None:
        snippets = SNIPPETS.collect_snippets()
        classifications = [SNIPPETS.classify(snippet) for snippet in snippets]
        creates = [
            classification
            for classification in classifications
            if classification.rule in {"http-post-create", "json-config-create"}
        ]

        self.assertFalse(
            any(classification.risk == "preview-mismatch" for classification in classifications)
        )
        self.assertEqual(len(creates), 3)
        for classification in creates:
            self.assertEqual(classification.coverage, "static")
            self.assertEqual(classification.risk, "mutating")
            self.assertNotEqual(classification.coverage, "live")

    def test_preview_model_type_is_an_explicit_blocker_not_live_coverage(self) -> None:
        snippet = SNIPPETS.FencedSnippet(
            page="probe.mdx",
            ordinal=0,
            language="json",
            body='{"type": "config", "model": "fastino/gliner2-base-v1"}\n',
            body_sha256="0" * 64,
        )
        classification = SNIPPETS.classify(snippet)

        self.assertEqual(classification.risk, "preview-mismatch")
        self.assertEqual(classification.coverage, "static")
        self.assertNotEqual(classification.coverage, "live")

    def test_harness_cli_accepts_training_snippets_flags_only_for_that_command(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["agent_harness.py", "static", "--live"],
        ):
            self.assertEqual(HARNESS.main(), 1)

        buffer = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["agent_harness.py", "training-snippets"]),
            mock.patch.object(SNIPPETS, "run_training_snippets", return_value=None) as runner,
            mock.patch.object(sys, "stdout", buffer),
            mock.patch.object(sys, "stderr", buffer),
        ):
            with mock.patch.dict(sys.modules, {"training_snippets": SNIPPETS}):
                code = HARNESS.main()

        self.assertEqual(code, 0)
        runner.assert_called_once_with(live=False, write_census=False)

    def test_workflow_runs_secret_free_static_and_guarded_live(self) -> None:
        daily = (ROOT / ".github/workflows/agent-docs-harness.yml").read_text()
        lifecycle = (ROOT / ".github/workflows/agent-docs-training-lifecycle.yml").read_text()
        static_job = daily.split("\n  published:", maxsplit=1)[0]
        authenticated_job = daily.split("\n  authenticated-api:", maxsplit=1)[1].split(
            "\n  training-replay:",
            maxsplit=1,
        )[0]

        self.assertRegex(
            static_job,
            r"python scripts/agent_harness\.py training-snippets\n",
        )
        self.assertNotIn("training-snippets --live", static_job)
        self.assertNotIn("FASTINO_DOCS_CANARY_API_KEY", static_job)
        self.assertIn(
            "python scripts/agent_harness.py training-snippets --live",
            authenticated_job,
        )
        self.assertIn("github.ref == 'refs/heads/main'", authenticated_job)
        self.assertIn("ref: main", authenticated_job)
        self.assertIn("FASTINO_DOCS_CANARY_API_KEY", authenticated_job)
        self.assertIn("FASTINO_DOCS_CANARY_TRAINING_JOB_ID", authenticated_job)
        self.assertIn("FASTINO_DOCS_CANARY_DATASET", authenticated_job)
        self.assertNotIn("training-snippets", lifecycle)
        self.assertEqual(daily.count("github.ref == 'refs/heads/main'"), 2)
        self.assertEqual(daily.count("ref: main"), 2)


if __name__ == "__main__":
    unittest.main()
