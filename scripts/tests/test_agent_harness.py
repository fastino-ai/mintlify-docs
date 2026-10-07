"""Unit tests for the agent-facing documentation harness."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "agent_harness.py"
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


if __name__ == "__main__":
    unittest.main()
