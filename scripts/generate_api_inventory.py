"""Generate localized API inventories from the canonical OpenAPI document."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTTP_METHODS = {"delete", "get", "patch", "post", "put"}
TAG_ORDER = (
    "model-catalog",
    "inference",
    "gliner",
    "inference-history",
    "anthropic-compat",
    "openai-compat",
    "systemone",
    "datasets",
    "training-jobs",
    "deprecated",
)
TAG_DISPLAY_NAMES = {
    "model-catalog": "Model Catalog",
    "inference": "Inference",
    "gliner": "GLiNER",
    "inference-history": "Inference History",
    "anthropic-compat": "Anthropic Compatibility",
    "openai-compat": "OpenAI Compatibility",
    "systemone": "SystemOne",
    "datasets": "Datasets",
    "training-jobs": "Training Jobs",
    "deprecated": "Deprecated",
}


@dataclass(frozen=True)
class InventoryCopy:
    title: str
    sidebar_title: str
    description: str
    introduction: str
    operation: str
    summary: str
    deprecated: str


COPY = {
    "": InventoryCopy(
        "Fastino API inventory",
        "API Inventory",
        "Complete method-and-path inventory generated from the Fastino OpenAPI specification.",
        "This inventory mirrors every operation in `openapi.json`. Use the linked task guides "
        "for intent and examples; use OpenAPI for the canonical wire contract.",
        "Operation",
        "Summary",
        "deprecated",
    ),
    "cn": InventoryCopy(
        "Fastino API 清单",
        "API 清单",
        "根据 Fastino OpenAPI 规范生成的完整方法和路径清单。",
        "此清单与 `openapi.json` 中的所有操作保持一致。任务意图和示例请参阅相关指南；"
        "规范的请求和响应契约以 OpenAPI 为准。",
        "操作",
        "摘要",
        "已弃用",
    ),
    "es": InventoryCopy(
        "Inventario de la API de Fastino",
        "Inventario de API",
        "Inventario completo de métodos y rutas generado desde la especificación OpenAPI de Fastino.",
        "Este inventario refleja todas las operaciones de `openapi.json`. Usa las guías para "
        "la intención y los ejemplos; OpenAPI es el contrato canónico.",
        "Operación",
        "Resumen",
        "obsoleto",
    ),
    "fr": InventoryCopy(
        "Inventaire de l'API Fastino",
        "Inventaire API",
        "Inventaire complet des méthodes et chemins généré depuis la spécification OpenAPI de Fastino.",
        "Cet inventaire reflète toutes les opérations de `openapi.json`. Utilisez les guides "
        "pour l'intention et les exemples ; OpenAPI est le contrat canonique.",
        "Opération",
        "Résumé",
        "obsolète",
    ),
    "de": InventoryCopy(
        "Fastino API-Inventar",
        "API-Inventar",
        "Vollständiges Methoden- und Pfadinventar aus der Fastino-OpenAPI-Spezifikation.",
        "Dieses Inventar spiegelt alle Operationen in `openapi.json` wider. Nutzen Sie die "
        "Anleitungen für Absicht und Beispiele; OpenAPI ist der verbindliche Vertrag.",
        "Operation",
        "Zusammenfassung",
        "veraltet",
    ),
}


def _operations(root: Path) -> dict[str, list[tuple[str, str, str, bool]]]:
    spec = json.loads((root / "openapi.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[str, str, str, bool]]] = {}
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            tags = operation.get("tags", [])
            tag = tags[0] if isinstance(tags, list) and tags else "other"
            summary = operation.get("summary", "")
            deprecated = (
                operation.get("deprecated") is True
                or tag == "deprecated"
                or (isinstance(summary, str) and "[Deprecated]" in summary)
            )
            grouped.setdefault(tag, []).append(
                (method.upper(), path, summary if isinstance(summary, str) else "", deprecated)
            )
    return grouped


def render_inventory(root: Path, locale: str) -> str:
    copy = COPY[locale]
    grouped = _operations(root)
    lines = [
        "---",
        f'title: "{copy.title}"',
        f'sidebarTitle: "{copy.sidebar_title}"',
        f'description: "{copy.description}"',
        "---",
        "",
        copy.introduction,
    ]
    ordered_tags = list(TAG_ORDER) + sorted(set(grouped) - set(TAG_ORDER))
    for tag in ordered_tags:
        if tag not in grouped:
            continue
        lines.extend(
            [
                "",
                f"## {TAG_DISPLAY_NAMES.get(tag, tag.replace('-', ' ').title())}",
                "",
                f"| {copy.operation} | {copy.summary} |",
                "| --- | --- |",
            ]
        )
        for method, path, summary, deprecated in grouped[tag]:
            suffix = f" ({copy.deprecated})" if deprecated else ""
            lines.append(f"| `{method} {path}` | {summary}{suffix} |")
    return "\n".join(lines) + "\n"


def inventory_findings(root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    for locale in COPY:
        destination = root / locale / "api-reference" / "inventory.mdx"
        expected = render_inventory(root, locale)
        if not destination.is_file() or destination.read_text(encoding="utf-8") != expected:
            findings.append(f"{destination.relative_to(root)} is stale")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        findings = inventory_findings()
        if findings:
            print("API inventory validation failed:")
            for finding in findings:
                print(f"- {finding}")
            print("Run: python3 scripts/generate_api_inventory.py")
            return 1
        print(f"API inventories are current: {len(COPY)}")
        return 0

    for locale in COPY:
        destination = ROOT / locale / "api-reference" / "inventory.mdx"
        destination.write_text(render_inventory(ROOT, locale), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
