"""Unit tests for the agent-facing documentation harness."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import urllib.error
from decimal import Decimal
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "agent_harness.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("agent_harness", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class AgentHarnessTests(unittest.TestCase):
    def test_openapi_resolves_required_agent_operations(self) -> None:
        document = HARNESS._load_openapi()
        operations = HARNESS._validate_openapi(document)

        self.assertTrue(HARNESS.REQUIRED_OPERATIONS <= set(operations))

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
            mock.patch.object(HARNESS, "_stop_and_confirm") as stop,
            mock.patch.object(HARNESS, "_cleanup_training_job") as cleanup,
            self.assertRaisesRegex(HARNESS.HarnessFailure, "stopped and deleted"),
        ):
            HARNESS._run_training_replay("fast_sk_test")

        stop.assert_called_once_with("fast_sk_test", "unexpected-job")
        cleanup.assert_called_once_with("fast_sk_test", "unexpected-job")

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
