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
    "training/encoder-quickstart.mdx",
    "training/decoder-quickstart.mdx",
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
    "training/encoder-quickstart.mdx": 5,
    "training/decoder-quickstart.mdx": 13,
    "training/migrate-from-tinker.mdx": 3,
    "concepts/datasets.mdx": 11,
    "guides/fine-tune-gliner.mdx": 8,
    "training/training-loop.mdx": 0,
    "training/verifiable-rewards.mdx": 8,
    "training/checkpoints.mdx": 6,
    "training/auto.mdx": 0,
}
EXPECTED_LIVE_GETS_WITH_FIXTURES = 10
PARITY_PAGES = (
    "training/encoder-quickstart.mdx",
    "training/decoder-quickstart.mdx",
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


def retired_claim_hits(text: str, *, tinker_context: bool = True) -> set[str]:
    return {
        token
        for token in TINKER_SDK_SUPPORT.retired_text
        if token in text and (token != "/api/v1/{operation}" or tinker_context)
    }


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
    def test_dataset_skill_uses_public_upload_contract(self) -> None:
        skill = (ROOT / ".mintlify/skills/fastino-datasets/SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn('"filename": "data.jsonl"', skill)
        self.assertIn('"dataset_type": "ner"', skill)
        self.assertIn("`presigned_url`", skill)
        self.assertIn("`version_number` exactly matches", skill)
        self.assertIn('`status` is `ready` or `failed`', skill)
        self.assertIn("`processing_error`", skill)
        self.assertNotIn('"file_name": "data.jsonl"', skill)
        self.assertNotIn("Returns `upload_url`", skill)
        self.assertNotIn("DPO uses `prompt`, `chosen`, and `rejected`", skill)
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            guide = ROOT / f"{prefix}guides/fine-tune-gliner.mdx"
            text = guide.read_text(encoding="utf-8")
            self.assertIn("`ready`", text, guide)
            self.assertIn("`failed`", text, guide)
            self.assertIn("`processing_error`", text, guide)

    def test_dataset_references_document_live_upload_behavior(self) -> None:
        expected_markers = {
            "upload-url.mdx": (
                "application/octet-stream",
                "`403`",
                "/api-reference/datasets/process-upload",
            ),
            "process-upload.mdx": (
                "`202`",
                "`ready`",
                "`failed`",
                "`processing_error`",
                "`503`",
                "/api-reference/datasets/versions",
            ),
            "list.mdx": (
                "`include_all_versions=true`",
                "`include_failed=true`",
                "`limit`",
                "`offset`",
                "200",
            ),
            "versions.mdx": (
                "`success`",
                "`versions`",
                "`count`",
                "`version_number`",
                "`processing_error`",
            ),
        }
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            for filename, markers in expected_markers.items():
                path = ROOT / f"{prefix}api-reference/datasets/{filename}"
                text = path.read_text(encoding="utf-8")
                for marker in markers:
                    localized_marker = marker.replace(
                        "/api-reference/", f"/{prefix}api-reference/"
                    )
                    self.assertIn(localized_marker, text, path)

    def test_agent_checkpoint_kinds_match_public_contract(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("`weights` or `sampler_weights`", agents)
        self.assertNotIn("`state` or `sampler_weights`", agents)

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

    def test_inference_skill_uses_deployed_interactive_model_id(self) -> None:
        skill = (ROOT / ".mintlify/skills/fastino-inference/SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"model": "YOUR_TRAINING_JOB_ID"', skill)
        self.assertIn('"model": "YOUR_CONFIG_TRAINING_JOB_ID"', skill)
        self.assertIn('"model": "YOUR_DEPLOYED_MODEL_ID"', skill)
        self.assertIn(
            "/training/jobs/YOUR_INTERACTIVE_JOB_ID/checkpoints/"
            "sampler_weights/YOUR_CHECKPOINT_NAME/deploy",
            skill,
        )
        deployed_example = skill[skill.index('"model": "YOUR_DEPLOYED_MODEL_ID"'):]
        deployed_example = deployed_example[:deployed_example.index("```")]
        self.assertNotIn('"schema"', deployed_example)
        self.assertIn("A `400`", skill)
        self.assertIn("bounded backoff", skill)

        spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
        catalog_parameters = {
            parameter["name"]
            for parameter in spec["paths"]["/v1/base-models"]["get"]["parameters"]
        }
        self.assertEqual(
            catalog_parameters,
            {"supports_training", "supports_inference", "task_type"},
        )
        for query in catalog_parameters:
            self.assertRegex(skill, rf"/v1/base-models\?{query}=[^\s]+")
        self.assertNotIn("Filterable by `training`, `inference`", skill)

    def test_glide_skill_lists_optional_request_fields_and_retention(self) -> None:
        skill = (ROOT / ".mintlify/skills/fastino-glide/SKILL.md").read_text(
            encoding="utf-8"
        )
        spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
        request = spec["components"]["schemas"]["SystemOneRequest"]
        optional_fields = set(request["properties"]) - set(request["required"])
        self.assertEqual(
            optional_fields,
            {"provider", "user", "session_id", "trace", "thinking", "store"},
        )
        for field in optional_fields:
            self.assertIn(f"`{field}`", skill)
        self.assertIn("`store` defaults to `true`", skill)
        self.assertIn("`store: false`", skill)
        self.assertIn("billing metadata", skill)
        self.assertNotIn(
            "Any other top-level request key returns",
            skill,
        )

    def test_gliner_training_examples_match_development_contract(self) -> None:
        canonical_classification = (
            '"classifications":[{"task":"action","labels":["jump","wait"],'
            '"true_label":["jump"]},{"task":"urgency","labels":["now","later"],'
            '"true_label":["now"]}]'
        )
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            paths = (
                ROOT / f"{prefix}training/encoder-quickstart.mdx",
                ROOT / f"{prefix}api-reference/training-jobs/create.mdx",
                ROOT / f"{prefix}concepts/datasets.mdx",
            )
            for path in paths:
                text = path.read_text(encoding="utf-8")
                self.assertIn("fastino/gliner2-base-v1", text, path)
                self.assertNotIn("fastino/gliner2-multi-v1", text, path)

            datasets = paths[-1].read_text(encoding="utf-8")
            self.assertIn(canonical_classification, datasets)
            self.assertIn('"output":{"entities":', datasets)

            models_path = ROOT / f"{prefix}concepts/models.mdx"
            models = models_path.read_text(encoding="utf-8")
            trainable = models.split(
                "GET /v1/base-models?supports_inference=true", 1
            )[0]
            self.assertIn("fastino/gliner2-base-v1", trainable, models_path)
            self.assertNotIn("fastino/gliner2-multi-v1", trainable, models_path)

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

    def test_checkpoint_deploy_examples_use_canonical_response(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / f"{prefix}api-reference/training-jobs/deploy-checkpoint.mdx"
            text = path.read_text(encoding="utf-8")
            match = re.search(r"```json Response\n(?P<body>\{.*?\})\n```", text, re.DOTALL)
            self.assertIsNotNone(match, path)
            payload = json.loads(match.group("body"))
            self.assertEqual(payload["message"], "Checkpoint deployed successfully", path)
            self.assertIn("deployment_path", payload, path)
            self.assertIn("mme_path", payload, path)
            self.assertRegex(text, r"`mme_path` .*(?:deprecated|已弃用|veraltet|obsoleto|obsolète)", path)

    def test_mutation_references_document_minimum_roles(self) -> None:
        editor_pages = (
            "api-reference/datasets/upload-url.mdx",
            "api-reference/datasets/process-upload.mdx",
            "api-reference/datasets/delete.mdx",
            "api-reference/training-jobs/create.mdx",
            "api-reference/training-jobs/delete.mdx",
            "api-reference/training-jobs/stop.mdx",
            "api-reference/training-jobs/terminate.mdx",
            "api-reference/training-jobs/update-project.mdx",
            "api-reference/training-jobs/update-model-name.mdx",
        )
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            for relative in editor_pages:
                path = ROOT / f"{prefix}{relative}"
                text = path.read_text(encoding="utf-8")
                self.assertIn("Owner", text, path)
                self.assertIn("Admin", text, path)
                self.assertIn("Editor", text, path)
                self.assertIn("`403 Forbidden`", text, path)

            deploy = ROOT / f"{prefix}api-reference/training-jobs/deploy-checkpoint.mdx"
            text = deploy.read_text(encoding="utf-8")
            self.assertIn("Owner", text, deploy)
            self.assertIn("Admin", text, deploy)
            self.assertNotIn("Editor", text, deploy)
            self.assertIn("`403 Forbidden`", text, deploy)

    def test_status_filter_includes_interactive_status_mappings(self) -> None:
        tie_phrases = {
            "": "same timestamp",
            "cn": "时间戳相同",
            "de": "denselben Zeitstempel",
            "es": "misma marca de tiempo",
            "fr": "même horodatage",
        }
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / f"{prefix}api-reference/training-jobs/list.mdx"
            text = path.read_text(encoding="utf-8")
            status = _between(
                text,
                '<ParamField query="status" type="string">',
                "</ParamField>",
            )
            self.assertRegex(status, r"(?:config|Config|配置)")
            self.assertRegex(
                status,
                r"(?i:interactive|interaktiv|interactivo|interactif)|交互式",
            )
            for value in (
                "creating",
                "active",
                "stopping",
                "stopped",
                "failed",
                "running",
                "errored",
                "error",
            ):
                self.assertIn(f"`{value}`", status, path)
            self.assertIn("`created_at`", text, path)
            self.assertIn(tie_phrases[locale], " ".join(text.split()), path)
            self.assertNotRegex(
                text,
                r"config jobs \(newest first\) before interactive",
                path,
            )
            for field in ("`id`", "`created_at`", "`job_id`"):
                self.assertIn(field, text, path)
            self.assertIn('`source: "interactive_checkpoint"`', text, path)
            self.assertIn("`null`", text, path)

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

    def test_checkpoint_ttl_update_requires_ttl_seconds(self) -> None:
        spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
        ttl_schema = spec["components"]["schemas"]["SetTtlRequest"]
        self.assertIn("ttl_seconds", ttl_schema["required"])
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / (
                f"{prefix}api-reference/training-jobs/interactive/"
                "set-checkpoint-ttl.mdx"
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                '<ParamField body="ttl_seconds" type="integer | null" required>',
                text,
                path,
            )
            self.assertRegex(text, r"\| `422` \|[^\n]+`ttl_seconds`", path)

    def test_delete_documents_config_and_interactive_jobs(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / f"{prefix}api-reference/training-jobs/delete.mdx"
            text = path.read_text(encoding="utf-8")
            job_id = _between(
                text,
                '<ParamField path="job_id" type="string" required>',
                "</ParamField>",
            )
            self.assertRegex(job_id, r"(?:config|Config|配置)")
            self.assertRegex(job_id, r"(?:interactive|interaktiv|interactivo|interactif|交互式)")
            self.assertIn("future", text.lower(), path)
            self.assertRegex(text, r"(?i:sampling)|采样", path)
            self.assertRegex(text, r"(?:billed|abgerechnet|factur|费用)", path)
            self.assertRegex(
                text,
                r"(?:separate|separat|independiente|distinct|独立)",
                path,
            )
            self.assertRegex(
                text,
                r"\| `409 cancellation_unconfirmed` \|[^\n]+(?:config|Config|配置)",
                path,
            )

    def test_config_only_operations_refuse_interactive_job_ids(self) -> None:
        pages = {
            "logs.mdx": "future",
            "billing.mdx": "`GET /v1/training/usage`",
            "download.mdx": (
                "`GET /v1/training/jobs/{job_id}/checkpoints/"
                "{checkpoint_id}/download`"
            ),
            "deployments.mdx": ("`model_id`", "`status`"),
            "sync.mdx": "`GET /v1/training/jobs/{job_id}`",
            "terminate.mdx": "`POST /v1/training/jobs/{job_id}/stop`",
            "update-model-name.mdx": "`model_name`",
            "update-project.mdx": "Tinker",
        }
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            for page, alternative in pages.items():
                path = ROOT / f"{prefix}api-reference/training-jobs/{page}"
                text = path.read_text(encoding="utf-8")
                self.assertRegex(text, r"(?:config|Config|配置)", path)
                self.assertRegex(
                    text,
                    r"(?:interactive|interaktiv|interactivo|interactif|交互式)",
                    path,
                )
                self.assertIn("`400`", text, path)
                markers = alternative if isinstance(alternative, tuple) else (alternative,)
                for marker in markers:
                    self.assertIn(marker.lower(), text.lower(), path)

            push_path = ROOT / f"{prefix}api-reference/training-jobs/push-to-hub.mdx"
            push = push_path.read_text(encoding="utf-8")
            self.assertRegex(push, r"(?:config|Config|配置)", push_path)
            self.assertRegex(
                push,
                r"(?:interactive|interaktiv|interactivo|interactif|交互式)",
                push_path,
            )
            self.assertIn("`400`", push, push_path)
            self.assertRegex(
                push,
                r"(?i:(?:no|keinen|keine|ninguna|aucune|n[’']ont|没有).{0,30}push)",
                push_path,
            )

    def test_tinker_quickstarts_require_python_311(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / f"{prefix}training/decoder-quickstart.mdx"
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("Python 3.10", text, path)
            self.assertGreaterEqual(text.count("Python 3.11"), 1, path)

    def test_windows_quickstarts_set_the_current_process_environment(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            path = ROOT / f"{prefix}quickstart.mdx"
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                '$env:FASTINO_API_KEY = "your_api_key_here"',
                text,
                path,
            )
            self.assertNotIn("setx FASTINO_API_KEY", text, path)

    def test_interactive_billing_and_sdk_limits_are_documented_in_every_locale(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            overview = (ROOT / f"{prefix}training.mdx").read_text(encoding="utf-8")
            quickstart = (ROOT / f"{prefix}training/decoder-quickstart.mdx").read_text(
                encoding="utf-8"
            )
            loop = (ROOT / f"{prefix}training/training-loop.mdx").read_text(
                encoding="utf-8"
            )
            migration = (ROOT / f"{prefix}training/migrate-from-tinker.mdx").read_text(
                encoding="utf-8"
            )

            for marker in ("forward", "checkpoint"):
                self.assertIn(marker, overview.lower(), f"{prefix}training.mdx")
            self.assertIn("/api-reference/training-jobs/interactive/storage-usage", overview)
            self.assertIn('f"{jobs}/forward-backward"', quickstart)
            self.assertIn('f"{jobs}/save-weights-for-sampler"', quickstart)
            self.assertIn(f"/{prefix}training/migrate-from-tinker", overview)
            self.assertIn("Tinker", overview)
            self.assertRegex(loop, r"(?:canonical|kanonische|canónico|canonique|规范)")
            self.assertRegex(migration, r"(?:explicit TTL|expliziten TTL|TTL explícito|TTL explicite|明确的 TTL)")
            for marker in (
                "/v1/training/jobs",
                "target_prompt_logprobs",
                "prompt_alt_tokens_k",
                "encoded_text",
                "AdamW",
                "PEFT",
            ):
                self.assertIn(marker, migration, f"{prefix}training/migrate-from-tinker.mdx")

        english = (ROOT / "training/migrate-from-tinker.mdx").read_text(encoding="utf-8")
        english_overview = (ROOT / "training.mdx").read_text(encoding="utf-8")
        self.assertIn("GB-month", english_overview)
        self.assertIn("reads interactive runs only", english.lower())
        self.assertIn("enabled by default", english)
        self.assertIn("stock PEFT", english)
        self.assertIn("does not make the path loadable by another team", english)
        self.assertNotIn("Reads canonical Fastino jobs", english)

    def test_tinker_sdk_support_matches_pioneer_contract(self) -> None:
        support = TINKER_SDK_SUPPORT
        for path, text in _claim_texts():
            self.assertEqual(
                retired_claim_hits(
                    text,
                    tinker_context=(
                        "tinker" in path.as_posix().lower()
                        or "tinker" in text.lower()
                    ),
                ),
                set(),
                path,
            )

        english = (ROOT / "training/migrate-from-tinker.mdx").read_text(encoding="utf-8")
        install_fences = [body for body in fence_bodies(english) if "tinker==" in body]
        self.assertEqual(install_fences, [])
        self.assertIn(support.package, english.split("## 1.", 1)[0])
        compatibility = english.split("## Tinker SDK compatibility", 1)[1]
        self.assertIn(support.package, compatibility)

        compatibility_spans = code_spans(compatibility)
        self.assertTrue(
            support.rest_client_methods <= compatibility_spans,
            compatibility_spans,
        )
        supported, unsupported = compatibility.split(
            '<Accordion title="Unsupported calls">', 1
        )
        for name in support.unsupported_calls:
            self.assertIn(f"`{name}`", unsupported)
            self.assertNotIn(name, supported)
        self.assertIn("Session trace export", unsupported)
        self.assertIn("Audit-log entries", unsupported)

        native_conversion_headings = {
            "": "## Switch to the native routes",
            "cn": "## 切换到原生路由",
            "de": "## Zu den nativen Routen wechseln",
            "es": "## Cambia a las rutas nativas",
            "fr": "## Passer aux routes natives",
        }
        for locale, heading in native_conversion_headings.items():
            prefix = f"{locale}/" if locale else ""
            migration = (
                ROOT / f"{prefix}training/migrate-from-tinker.mdx"
            ).read_text(encoding="utf-8")
            self.assertNotIn(heading, migration)

        self.assertIn("`sample`", compatibility)
        self.assertIn("unfinished sampling futures", compatibility)
        self.assertIn("Training-operation futures cannot be cancelled", compatibility)
        self.assertNotIn("one future", compatibility)
        for locale, claim in ONE_FUTURE_CLAIMS.items():
            prefix = f"{locale}/" if locale else ""
            page = ROOT / f"{prefix}training/migrate-from-tinker.mdx"
            text = page.read_text(encoding="utf-8")
            self.assertIn("`sample`", text, page)
            self.assertNotIn(claim, text, page)
            for phrase in RUN_KEY_EXPORT_CLAIMS:
                self.assertNotIn(phrase, text, page)

        loop = (ROOT / "training/training-loop.mdx").read_text(encoding="utf-8")
        self.assertIn("Only sample futures can be cancelled", loop)
        self.assertNotIn("Resolve or cancel futures individually", loop)

    def test_native_interactive_semantics_are_documented(self) -> None:
        interactive = ROOT / "api-reference/training-jobs/interactive"
        create = (interactive / "create-model.mdx").read_text(encoding="utf-8")
        future = (interactive / "get-future.mdx").read_text(encoding="utf-8")
        cancel = (interactive / "cancel-future.mdx").read_text(encoding="utf-8")
        save = (interactive / "save-weights.mdx").read_text(encoding="utf-8")
        save_sampler = (interactive / "save-weights-for-sampler.mdx").read_text(
            encoding="utf-8"
        )
        load = (interactive / "load-weights.mdx").read_text(encoding="utf-8")
        sample = (interactive / "sample.mdx").read_text(encoding="utf-8")
        usage = (interactive / "storage-usage.mdx").read_text(encoding="utf-8")

        self.assertIn("POST /v1/training/jobs/create-model", create)
        self.assertIn("`base_model`", create)
        self.assertIn("`from_checkpoint`", create)
        self.assertIn("`load_optimizer`", create)
        self.assertIn("do not send `seq_id` values or heartbeats", create)
        self.assertIn("{future_id}", future)
        self.assertIn("`wait=true`", future)
        self.assertIn("at most 25 seconds", future)
        self.assertIn("nonterminal", future)
        for status in ("`completed`", "`failed`", "`cancelled`"):
            self.assertIn(status, future)
        self.assertIn("Cancellation is best effort", cancel)
        self.assertIn("training future", cancel)
        self.assertIn("successful no-op", cancel)
        self.assertIn("POST /v1/training/jobs/{job_id}/save-weights", save)
        self.assertIn(
            "POST /v1/training/jobs/{job_id}/save-weights-for-sampler",
            save_sampler,
        )
        self.assertIn("POST /v1/training/jobs/{job_id}/load-weights", load)
        self.assertIn("state name", load)
        self.assertIn("this job's own `tinker://", load)
        self.assertIn("POST /v1/training/jobs/{job_id}/sample", sample)
        self.assertIn("sampler checkpoint name", sample)
        self.assertIn("after the interactive job is stopped", sample)
        self.assertIn("checkpoint storage", usage.lower())
        self.assertIn("team-scoped", usage)
        self.assertIn("does not report compute usage or total billing", usage)

    def test_corrected_training_contract_is_documented_in_every_locale(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            create = (
                ROOT / f"{prefix}api-reference/training-jobs/create.mdx"
            ).read_text(encoding="utf-8")
            future = (
                ROOT
                / f"{prefix}api-reference/training-jobs/interactive/get-future.mdx"
            ).read_text(encoding="utf-8")
            get_job = (
                ROOT / f"{prefix}api-reference/training-jobs/get.mdx"
            ).read_text(encoding="utf-8")
            list_jobs = (
                ROOT / f"{prefix}api-reference/training-jobs/list.mdx"
            ).read_text(encoding="utf-8")

            for marker in (
                "warmup_ratio",
                "unsupported_encoder_scheduling",
                "linear",
                "weight_decay",
                '"error"',
            ):
                self.assertIn(marker, create, f"{prefix} create page")
            self.assertNotRegex(create, r'(?m)^\s*"detail"\s*:')
            for marker in ("`404`", "`410`", "/training/checkpoints"):
                self.assertIn(marker, future, f"{prefix} future page")
            if not locale:
                self.assertIn("one-hour", future)
            for marker in (
                "job_id",
                "training_run_id",
                "status",
                "stop_reason",
                "error_message",
                "is_terminal_status",
            ):
                self.assertIn(marker, get_job, f"{prefix} get-job page")
                self.assertIn(marker, list_jobs, f"{prefix} list-jobs page")

    def test_part_four_training_contract_is_documented_in_every_locale(self) -> None:
        for locale in ("", "cn", "de", "es", "fr"):
            prefix = f"{locale}/" if locale else ""
            base = ROOT / prefix
            create = (base / "api-reference/training-jobs/create.mdx").read_text()
            create_model = (
                base / "api-reference/training-jobs/interactive/create-model.mdx"
            ).read_text()
            sample = (
                base / "api-reference/training-jobs/interactive/sample.mdx"
            ).read_text()
            list_jobs = (
                base / "api-reference/training-jobs/list.mdx"
            ).read_text()
            download = (
                base / "api-reference/training-jobs/download.mdx"
            ).read_text()
            logs = (base / "api-reference/training-jobs/logs.mdx").read_text()
            quickstart = (base / "training/decoder-quickstart.mdx").read_text()
            guide = (base / "guides/fine-tune-gliner.mdx").read_text()

            for marker in ("60", "<name>_2", "idempotency_key_reused", "`429`", "`503`"):
                self.assertIn(marker, create, f"{prefix} create page")
            for marker in ("`16`", "`512`", "`429`"):
                self.assertIn(marker, create_model, f"{prefix} create-model page")
            for marker in (
                "`1`",
                "`512`",
                'default="1024"',
                "`65,536`",
                "`32,768`",
                "`2,048`",
                "`429`",
            ):
                self.assertIn(marker, sample, f"{prefix} sample page")
            for marker in ("status=complete", "artifact_ready", "deployed"):
                self.assertIn(marker, list_jobs, f"{prefix} list page")
            self.assertIn("gliner2-model-a1b2c3d4.tar.gz", download)
            for marker in ("`stdout`", "`stderr`", "`system`"):
                self.assertIn(marker, logs, f"{prefix} logs page")
            self.assertIn("future", quickstart, f"{prefix} decoder quickstart")
            self.assertNotIn("6.5", quickstart, f"{prefix} decoder quickstart")
            self.assertIn("deployed", guide, f"{prefix} GLiNER guide")
            self.assertNotIn("/checkpoints/$CHECKPOINT_ID/deploy", guide)

        agents = (ROOT / "AGENTS.md").read_text()
        dataset_skill = (
            ROOT / ".mintlify/skills/fastino-datasets/SKILL.md"
        ).read_text()
        fine_tune_skill = (
            ROOT / ".mintlify/skills/fastino-fine-tune/SKILL.md"
        ).read_text()
        self.assertNotIn("training/quickstart#custom-training-loop", agents)
        self.assertNotIn("supported-client-behavior", agents)
        self.assertIn("training/encoder-quickstart", agents)
        self.assertIn("training/decoder-quickstart", agents)
        self.assertNotIn("`task_type` (`ner`", dataset_skill)
        for marker in ("artifact_ready", "paused"):
            self.assertIn(marker, fine_tune_skill)

    def test_openapi_exposes_interactive_job_lifecycle(self) -> None:
        spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
        schema = spec["components"]["schemas"]["InteractiveJobView"]
        properties = schema["properties"]
        for field in (
            "job_id",
            "training_run_id",
            "status",
            "stop_reason",
            "error_message",
            "is_terminal_status",
        ):
            self.assertIn(field, properties)

    def test_checkpoint_download_and_deployment_guidance_matches_contract(self) -> None:
        download = (
            ROOT / "api-reference/training-jobs/download-checkpoint.mdx"
        ).read_text(encoding="utf-8")
        deploy = (
            ROOT
            / "api-reference/training-jobs/interactive/deploy-checkpoint.mdx"
        ).read_text(encoding="utf-8")
        checkpoints = (ROOT / "training/checkpoints.mdx").read_text(encoding="utf-8")

        self.assertIn("presigned object-storage URL", download)
        self.assertNotIn("/v1/training/archives/", download)
        self.assertNotIn("contact sales", download.lower())
        self.assertIn("stock PEFT", download)
        self.assertIn("Qwen3.6-35B-A3B", deploy)
        self.assertIn("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", deploy)
        self.assertIn("returns `400`", deploy)
        self.assertIn("`425 model_warming`", checkpoints)
        self.assertIn('"reasoning": {"enabled": false}', checkpoints)

    def test_dataset_and_catalog_distinctions_are_explicit(self) -> None:
        datasets = (ROOT / "concepts/datasets.mdx").read_text(encoding="utf-8")
        models = (ROOT / "concepts/models.mdx").read_text(encoding="utf-8")
        pricing = (ROOT / "pricing.mdx").read_text(encoding="utf-8")
        usage = (
            ROOT
            / "api-reference/training-jobs/interactive/storage-usage.mdx"
        ).read_text(encoding="utf-8")

        self.assertIn("matching is case-insensitive", datasets)
        self.assertIn('"entities":[],"label":"other"', datasets)
        self.assertIn("Interactive jobs don't accept a dataset name", datasets)
        self.assertIn(
            "| Nemotron 3.5 Super VL | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` |",
            models,
        )
        self.assertIn(
            "`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`",
            models,
        )
        self.assertIn("| Nemotron 3.5 Super VL |", pricing)
        self.assertIn('"size_gigabytes": 0.805306368', usage)
        self.assertIn('"estimated_monthly_cost_usd": 0.0402653184', usage)

    def test_interactive_token_arrays_are_marked_as_placeholders(self) -> None:
        interactive = ROOT / "api-reference/training-jobs/interactive"
        for name in ("forward.mdx", "forward-backward.mdx", "sample.mdx", "get-future.mdx"):
            text = (interactive / name).read_text(encoding="utf-8")
            self.assertIn("illustrative placeholder", text, name)
            self.assertIn("`base_model`", text, name)

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

    def test_chat_templates_disable_thinking_before_supervised_answers(self) -> None:
        pages = [
            ROOT / f"{prefix}{page}"
            for prefix in ("", "cn/", "de/", "es/", "fr/")
            for page in (
                "training/decoder-quickstart.mdx",
                "training/verifiable-rewards.mdx",
            )
        ]
        for page in pages:
            text = page.read_text(encoding="utf-8")
            calls = re.findall(
                r"apply_chat_template\((.*?)\n\s*\)",
                text,
                flags=re.DOTALL,
            )
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertTrue(calls)
                for call in calls:
                    self.assertIn("enable_thinking=False", call)

        for prefix in ("", "cn/", "de/", "es/", "fr/"):
            loop = (ROOT / f"{prefix}training/training-loop.mdx").read_text(
                encoding="utf-8"
            )
            self.assertIn("`enable_thinking=False`", loop)

    def test_encoder_and_decoder_quickstarts_are_separate(self) -> None:
        titles = {
            "": ("Encoder Quickstart", "Decoder Quickstart"),
            "cn/": ("编码器快速入门", "解码器快速入门"),
            "de/": ("Encoder-Schnellstart", "Decoder-Schnellstart"),
            "es/": ("Inicio rápido del codificador", "Inicio rápido del decodificador"),
            "fr/": ("Démarrage rapide de l'encodeur", "Démarrage rapide du décodeur"),
        }
        docs_config = json.loads((ROOT / "docs.json").read_text(encoding="utf-8"))
        navigation = json.dumps(docs_config["navigation"], ensure_ascii=False)
        redirects = {
            (redirect["source"], redirect["destination"], redirect["permanent"])
            for redirect in docs_config["redirects"]
        }
        for prefix, (encoder_title, decoder_title) in titles.items():
            encoder_path = f"{prefix}training/encoder-quickstart"
            decoder_path = f"{prefix}training/decoder-quickstart"
            legacy_path = f"/{prefix}training/quickstart"
            encoder = (ROOT / f"{encoder_path}.mdx").read_text(encoding="utf-8")
            decoder = (ROOT / f"{decoder_path}.mdx").read_text(encoding="utf-8")
            llms = (ROOT / f"{prefix}llms.txt").read_text(encoding="utf-8")
            with self.subTest(locale=prefix or "en"):
                self.assertIn(f'title: "{encoder_title}"', encoder)
                self.assertIn(f'title: "{decoder_title}"', decoder)
                self.assertNotIn("/forward-backward", encoder)
                self.assertIn('f"{jobs}/forward-backward"', decoder)
                self.assertIn(f'"{encoder_path}"', navigation)
                self.assertIn(f'"{decoder_path}"', navigation)
                self.assertIn(f"/{encoder_path}.md", llms)
                self.assertIn(f"/{decoder_path}.md", llms)
                self.assertIn(
                    (legacy_path, f"/{decoder_path}", True),
                    redirects,
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

        self.assertEqual(
            retired_claim_hits(
                "/api/v1/{operation}\n",
                tinker_context=False,
            ),
            set(),
        )

    def test_inventory_is_exactly_54_fences_across_10_training_api_pages(self) -> None:
        pages = SNIPPETS.collect_pages()
        snippets = SNIPPETS.collect_snippets()
        counts = {page: 0 for page in pages}
        for snippet in snippets:
            counts[snippet.page] += 1

        self.assertEqual(pages, EXPECTED_PAGES)
        self.assertEqual(len(snippets), 54)
        self.assertEqual(len(pages), 10)
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

        create = SNIPPETS.classify(_snippet("training/encoder-quickstart.mdx", 3))
        self.assertEqual(create.coverage, "static")
        self.assertEqual(create.mapped_command, "training-replay")
        self.assertEqual(create.risk, "mutating")

        deployment_status = SNIPPETS.classify(
            _snippet("guides/fine-tune-gliner.mdx", 6)
        )
        self.assertEqual(deployment_status.coverage, "live")
        self.assertEqual(deployment_status.risk, "none")

        inference = SNIPPETS.classify(_snippet("guides/fine-tune-gliner.mdx", 7))
        self.assertEqual(inference.coverage, "static")
        self.assertEqual(inference.mapped_command, "blocker")

        upload = SNIPPETS.classify(_snippet("concepts/datasets.mdx", 6))
        self.assertEqual(upload.coverage, "static")
        self.assertEqual(upload.mapped_command, "blocker")

    def test_missing_placeholders_print_skip_and_do_not_guess(self) -> None:
        job = _snippet("training/encoder-quickstart.mdx", 4)
        dataset = _snippet("concepts/datasets.mdx", 9)
        catalog = _snippet("training/encoder-quickstart.mdx", 2)

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
        self.assertIn(
            'python -m unittest discover -s scripts/tests -p "test_*.py"',
            static_job,
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
