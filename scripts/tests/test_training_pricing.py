"""Contract tests for interactive Training API pricing."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCALES = ("cn", "de", "es", "fr")
PRICING_PAGES = (ROOT / "pricing.mdx",) + tuple(
    ROOT / locale / "pricing.mdx" for locale in LOCALES
)
SECTION_HEADINGS = {
    ROOT / "pricing.mdx": ("## Training", "## Inference"),
    ROOT / "cn" / "pricing.mdx": ("## 训练", "## 推理"),
    ROOT / "de" / "pricing.mdx": ("## Training", "## Inferenz"),
    ROOT / "es" / "pricing.mdx": ("## Entrenamiento", "## Inferencia"),
    ROOT / "fr" / "pricing.mdx": ("## Entraînement", "## Inférence"),
}

EXPECTED_RATES = {
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": (
        r"\$0.96",
        r"\$0.16",
        r"\$0.032",
        r"\$0.72",
    ),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (
        r"\$0.30",
        r"\$0.05",
        r"\$0.01",
        r"\$0.35",
    ),
    "Qwen/Qwen3.6-35B-A3B": (
        r"\$0.35",
        r"\$0.08",
        r"\$0.016",
        r"\$0.35",
    ),
}


def _interactive_rates(page: Path) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for line in page.read_text(encoding="utf-8").splitlines():
        for model_id in EXPECTED_RATES:
            if f"`{model_id}`" not in line:
                continue
            cells = tuple(cell.strip() for cell in line.split("|")[1:-1])
            rows[model_id] = cells[1:]
    return rows


class TrainingPricingTests(unittest.TestCase):
    def test_all_locales_publish_the_same_interactive_rates(self) -> None:
        for page in PRICING_PAGES:
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertEqual(_interactive_rates(page), EXPECTED_RATES)

    def test_config_and_interactive_units_are_distinct(self) -> None:
        for page in PRICING_PAGES:
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertIn(r"\$0.07", text)
                self.assertIn("GPU", text)
                self.assertEqual(text.count("per 1M tokens"), 1 if page == ROOT / "pricing.mdx" else 0)

    def test_training_precedes_inference(self) -> None:
        for page, (training_heading, inference_heading) in SECTION_HEADINGS.items():
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertLess(text.index(training_heading), text.index(inference_heading))

    def test_interactive_dollar_rows_live_only_on_pricing_pages(self) -> None:
        pricing_pages = {page.resolve() for page in PRICING_PAGES}
        for page in ROOT.glob("**/*.mdx"):
            if page.resolve() in pricing_pages:
                continue
            for line in page.read_text(encoding="utf-8").splitlines():
                if any(f"`{model_id}`" in line for model_id in EXPECTED_RATES):
                    self.assertNotRegex(
                        line,
                        r"\\\$\d",
                        f"interactive rates must remain on pricing pages: {page}",
                    )


if __name__ == "__main__":
    unittest.main()
