"""Tests for the stable API-doc checker interface."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_api_docs as CHECKS


class CheckApiDocsInterfaceTests(unittest.TestCase):
    def _root(self, directory: str) -> Path:
        root = Path(directory)
        (root / "openapi.json").write_text(
            json.dumps({"openapi": "3.1.0", "paths": {}}),
            encoding="utf-8",
        )
        return root

    def test_hygiene_does_not_require_or_read_pioneer_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            missing = root / "does-not-exist"
            with (
                mock.patch.object(CHECKS, "hygiene_findings", return_value=[]) as hygiene,
                mock.patch.object(
                    CHECKS,
                    "contract_findings",
                    side_effect=AssertionError("contract checker must not run"),
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "check_api_docs.py",
                        "hygiene",
                        "--root",
                        str(root),
                        "--pioneer-root",
                        str(missing),
                    ],
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(CHECKS.main(), 0)
            hygiene.assert_called_once()

    def test_contract_subcommand_reads_route_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            page = root / "guide.mdx"
            page.write_text("Call POST /v1/missing.\n", encoding="utf-8")
            pioneer = root / "pioneer"
            (pioneer / "docs").mkdir(parents=True)
            (pioneer / "docs/route_manifest.json").write_text(
                json.dumps({"routes": []}),
                encoding="utf-8",
            )

            findings = CHECKS.contract_findings(
                root,
                {"paths": {}},
                pioneer,
            )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "route_absent")
        self.assertEqual(findings[0].operation, "POST /v1/missing")
        self.assertEqual(findings[0].file, "guide.mdx")

    def test_json_format_has_versioned_exhaustive_stable_objects(self) -> None:
        finding = CHECKS.finding_from_message(
            "guide.mdx:12: documented operation is absent from the "
            "production route manifest: POST /v1/missing"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            with (
                mock.patch.object(CHECKS, "hygiene_findings", return_value=[finding]),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "check_api_docs.py",
                        "hygiene",
                        "--root",
                        str(root),
                        "--format",
                        "json",
                    ],
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                self.assertEqual(CHECKS.main(), 1)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["findings"]), 1)
        item = payload["findings"][0]
        self.assertEqual(
            set(item),
            {"key", "class", "code", "operation", "file", "detail", "message"},
        )
        self.assertEqual(
            item["key"],
            "|".join(item[field] for field in ("code", "operation", "file", "detail")),
        )

    def test_human_format_preserves_finding_message(self) -> None:
        finding = CHECKS.finding_from_message("guide.mdx:3: stale example")
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            with (
                mock.patch.object(CHECKS, "hygiene_findings", return_value=[finding]),
                mock.patch.object(
                    sys,
                    "argv",
                    ["check_api_docs.py", "hygiene", "--root", str(root)],
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                CHECKS.main()
        self.assertEqual(stdout.getvalue(), f"{finding.message}\n")

    def test_stable_key_ignores_line_and_column_numbers(self) -> None:
        first = CHECKS.finding_from_message("guide.mdx:3:4: stale example")
        second = CHECKS.finding_from_message("guide.mdx:99:12: stale example")
        self.assertEqual(first.key, second.key)
        self.assertRegex(first.detail, r"^[0-9a-f]{64}$")

    def test_known_bracket_field_path_becomes_json_pointer(self) -> None:
        first = CHECKS.finding_from_message(
            "guide.mdx:3: CONTRACT: request['questions'][0]['type']: is invalid"
        )
        second = CHECKS.finding_from_message(
            "guide.mdx:99: CONTRACT: request['questions'][0]['type']: is invalid"
        )

        self.assertEqual(first.detail, "/questions/0/type")
        self.assertEqual(first.key, second.key)

    def test_explicit_json_pointer_is_canonicalized(self) -> None:
        finding = CHECKS.finding_from_message(
            "CONTRACT: JSON pointer #/paths/~1v1~1widgets/get/~0metadata is invalid"
        )

        self.assertEqual(
            finding.detail,
            "/paths/~1v1~1widgets/get/~0metadata",
        )

    def test_param_and_response_field_names_are_known_details(self) -> None:
        param = CHECKS.finding_from_message(
            'guide.mdx:8: unsupported <ParamField body="store">'
        )
        response = CHECKS.finding_from_message(
            'guide.mdx:9: unsupported <ResponseField name="job_id">'
        )

        self.assertEqual(param.detail, "body:store")
        self.assertEqual(response.detail, "job_id")

    def test_duplicate_keys_are_emitted_once(self) -> None:
        first = CHECKS.finding_from_message("guide.mdx:3: stale example")
        second = CHECKS.finding_from_message("guide.mdx:9: stale example")
        self.assertEqual(len(CHECKS.unique_findings([first, second])), 1)

    def test_all_combines_contract_and_hygiene_findings(self) -> None:
        one = CHECKS.finding_from_message("one.mdx:1: first")
        two = CHECKS.finding_from_message("two.mdx:2: second")
        with (
            mock.patch.object(CHECKS, "hygiene_findings", return_value=[one]),
            mock.patch.object(CHECKS, "contract_findings", return_value=[two]),
        ):
            findings = CHECKS.all_findings(Path("."), {}, Path("."))
        self.assertEqual({finding.key for finding in findings}, {one.key, two.key})

    def test_stale_bindings_include_locale_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fr/api-reference").mkdir(parents=True)
            (root / "fr/api-reference/widget.mdx").write_text(
                '---\napi: "GET /v1/retired"\n---\n',
                encoding="utf-8",
            )
            (root / "docs.json").write_text(
                json.dumps(
                    {
                        "navigation": {
                            "languages": [{"language": "en", "tabs": []}]
                        }
                    }
                ),
                encoding="utf-8",
            )
            messages = CHECKS._api_reference_findings(root, {"paths": {}})
            findings = CHECKS.findings_from_messages(messages)

        stale = [finding for finding in findings if finding.code == "stale_binding"]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].operation, "GET /v1/retired")
        self.assertEqual(stale[0].file, "fr/api-reference/widget")


if __name__ == "__main__":
    unittest.main()
