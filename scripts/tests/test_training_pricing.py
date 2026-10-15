"""Contract tests for interactive Training API pricing."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCALES = ("cn", "de", "es", "fr")
PRICING_PAGES = (ROOT / "pricing.mdx",) + tuple(
    ROOT / locale / "pricing.mdx" for locale in LOCALES
)
MODEL_CATALOG_PAGES = (ROOT / "concepts" / "models.mdx",) + tuple(
    ROOT / locale / "concepts" / "models.mdx" for locale in LOCALES
)
STORAGE_USAGE_PAGES = (
    ROOT / "api-reference" / "training-jobs" / "interactive" / "storage-usage.mdx",
) + tuple(
    ROOT
    / locale
    / "api-reference"
    / "training-jobs"
    / "interactive"
    / "storage-usage.mdx"
    for locale in LOCALES
)
SECTION_HEADINGS = {
    ROOT / "pricing.mdx": ("## Training", "## Inference"),
    ROOT / "cn" / "pricing.mdx": ("## 训练", "## 推理"),
    ROOT / "de" / "pricing.mdx": ("## Training", "## Inferenz"),
    ROOT / "es" / "pricing.mdx": ("## Entrenamiento", "## Inferencia"),
    ROOT / "fr" / "pricing.mdx": ("## Entraînement", "## Inférence"),
}
MODEL_SECTION_HEADINGS = {
    ROOT / "concepts" / "models.mdx": ("## Trainable models", "## Inference models"),
    ROOT / "cn" / "concepts" / "models.mdx": ("## 可训练模型", "## 推理模型"),
    ROOT / "de" / "concepts" / "models.mdx": ("## Trainierbare Modelle", "## Inferenzmodelle"),
    ROOT / "es" / "concepts" / "models.mdx": ("## Modelos entrenables", "## Modelos de inferencia"),
    ROOT / "fr" / "concepts" / "models.mdx": ("## Modèles entraînables", "## Modèles d'inférence"),
}

EXPECTED_RATES = {
    "Nemotron 3.5 Super VL": (
        r"\$0.16",
        r"\$0.032",
        r"\$0.65",
        r"\$0.65",
    ),
    "Nemotron 3.5 Lightning": (
        r"\$0.05",
        r"\$0.01",
        r"\$0.35",
        r"\$0.30",
    ),
    "Qwen3.6-35B-A3B": (
        r"\$0.08",
        r"\$0.016",
        r"\$0.44",
        r"\$0.44",
    ),
}
CALLABLE_IDS = {
    "Nemotron 3.5 Super VL": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "Nemotron 3.5 Lightning": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
}
ENCODER_IDS = (
    "fastino/gliner2-base-v1",
    "fastino/GLiNER-2.5-Decide",
)
INFERENCE_SUPER_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"


def _interactive_rates(page: Path) -> dict[str, tuple[str, ...]]:
    text = page.read_text(encoding="utf-8")
    start_heading, end_heading = (
        SECTION_HEADINGS.get(page) or MODEL_SECTION_HEADINGS[page]
    )
    interactive_section = text[text.index(start_heading) : text.index(end_heading)]
    rows: dict[str, tuple[str, ...]] = {}
    for line in interactive_section.splitlines():
        if not line.startswith("|"):
            continue
        for display_name in EXPECTED_RATES:
            if display_name not in line:
                continue
            cells = tuple(cell.strip() for cell in line.split("|")[1:-1])
            current_rates = []
            for cell in cells[1:]:
                without_previous_rate = re.sub(r"~~.*?~~", "", cell)
                match = re.search(r"\\\$\d+(?:\.\d+)?", without_previous_rate)
                if match:
                    current_rates.append(match.group())
            rows[display_name] = rows.get(display_name, ()) + tuple(current_rates)
    return rows


class TrainingPricingTests(unittest.TestCase):
    def test_pricing_and_model_catalogs_publish_the_same_interactive_rates(self) -> None:
        for page in PRICING_PAGES + MODEL_CATALOG_PAGES:
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertEqual(_interactive_rates(page), EXPECTED_RATES)

    def test_config_and_interactive_units_are_distinct(self) -> None:
        for page in PRICING_PAGES:
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertIn(r"\$0.07", text)
                self.assertIn("GPU", text)
                self.assertEqual(text.count("per 1M tokens"), 1 if page == ROOT / "pricing.mdx" else 0)

    def test_training_super_uses_product_display_name(self) -> None:
        for page in PRICING_PAGES:
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertIn("Nemotron 3.5 Super VL", page.read_text(encoding="utf-8"))

    def test_callable_ids_live_in_model_catalogs_not_pricing_tables(self) -> None:
        for page in PRICING_PAGES:
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                for model_id in CALLABLE_IDS.values():
                    self.assertNotIn(model_id, text)

        for page in MODEL_CATALOG_PAGES:
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                for display_name, model_id in CALLABLE_IDS.items():
                    self.assertIn(display_name, text)
                    self.assertIn(f"`{model_id}`", text)
                self.assertIn(f"`{INFERENCE_SUPER_ID}`", text)

    def test_inference_super_has_distinct_pricing(self) -> None:
        for page in PRICING_PAGES:
            lines = page.read_text(encoding="utf-8").splitlines()
            with self.subTest(page=page.relative_to(ROOT)):
                rows = [line for line in lines if "Nemotron 3.5 Super VL" in line]
                self.assertEqual(len(rows), 3)
                self.assertTrue(
                    any(r"\$0.08" in row and r"\$0.45" in row for row in rows)
                )
                self.assertTrue(
                    any(r"\$0.16" in row and r"\$0.032" in row for row in rows)
                )
                self.assertTrue(
                    any(r"\$0.65" in row and row.count(r"\$0.65") == 2 for row in rows)
                )

    def test_model_catalogs_publish_encoder_training_rates(self) -> None:
        for page in MODEL_CATALOG_PAGES:
            lines = page.read_text(encoding="utf-8").splitlines()
            with self.subTest(page=page.relative_to(ROOT)):
                for model_id in ENCODER_IDS:
                    rows = [line for line in lines if f"`{model_id}`" in line]
                    self.assertTrue(
                        any(r"\$0.07" in row and r"\$4.20" in row for row in rows)
                    )

    def test_training_precedes_inference(self) -> None:
        for page, (training_heading, inference_heading) in (
            SECTION_HEADINGS | MODEL_SECTION_HEADINGS
        ).items():
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertLess(text.index(training_heading), text.index(inference_heading))

    def test_interactive_dollar_rows_live_only_on_pricing_and_catalog_pages(self) -> None:
        approved_pages = {
            page.resolve() for page in PRICING_PAGES + MODEL_CATALOG_PAGES
        }
        for page in ROOT.glob("**/*.mdx"):
            if page.resolve() in approved_pages:
                continue
            for line in page.read_text(encoding="utf-8").splitlines():
                if any(name in line for name in EXPECTED_RATES):
                    self.assertNotRegex(
                        line,
                        r"\\\$\d",
                        f"interactive rates must remain on approved pages: {page}",
                    )

    def test_promotional_markers_are_absent(self) -> None:
        for page in PRICING_PAGES + MODEL_CATALOG_PAGES:
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertNotIn("Limited-time discount", text)
                self.assertNotIn("~~", text)

    def test_storage_usage_examples_match_the_billed_rate(self) -> None:
        for page in STORAGE_USAGE_PAGES:
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertIn('"effective_rate_usd_per_gigabyte_month": 0.05', text)
                self.assertIn('"size_gigabytes": 0.805306368', text)
                self.assertIn('"estimated_monthly_cost_usd": 0.0402653184', text)


if __name__ == "__main__":
    unittest.main()
