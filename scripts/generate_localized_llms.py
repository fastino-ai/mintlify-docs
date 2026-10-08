"""Generate localized llms.txt indexes from Mintlify navigation."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://docs.fastino.ai"
FRONTMATTER_FIELD = re.compile(r"^(title|description):\s*(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class LocaleCopy:
    """Localized labels used around page metadata."""

    title: str
    summary: str
    instructions_heading: str
    instructions: str
    resources_heading: str
    root_label: str
    root_description: str
    openapi_label: str
    openapi_description: str
    skills_label: str
    skills_description: str


LOCALE_COPY = {
    "cn": LocaleCopy(
        title="Fastino Labs 中文文档",
        summary="使用 GLiDE 构建结构化决策，使用 GLiNER 进行基于模式的抽取，并通过 Fastino 微调专用模型。",
        instructions_heading="代理说明",
        instructions=(
            f"使用 {BASE_URL}/openapi.json 作为面向客户的 API 路由的事实来源。"
            "GLiDE 使用 `POST /v1/systemone`；GLiNER 和解码器推理使用 "
            "`POST /v1/chat/completions`；兼容接口使用 `POST /v1/responses` 或 "
            "`POST /v1/messages`。大型 GLiNER-2 输入使用 `POST /v1/gliner-2/async`，"
            "然后轮询 `GET /v1/gliner-2/jobs/{job_id}`。使用 "
            "`GET /v1/base-models?supports_inference=true` 或 "
            "`GET /v1/base-models?supports_training=true` 查询实时能力。"
            "不要推断未记录的路由。请从 `FASTINO_API_KEY` 读取 API 密钥，"
            "切勿在代码、日志或报告中嵌入凭据。"
        ),
        resources_heading="代理资源",
        root_label="英文文档索引",
        root_description="浏览 Fastino 英文文档。",
        openapi_label="OpenAPI 规范",
        openapi_description="查看面向客户的路由、身份验证要求以及请求和响应模式。",
        skills_label="代理技能目录",
        skills_description="发现适用于数据集、微调、GLiDE、GLiNER 和推理的可安装技能。",
    ),
    "es": LocaleCopy(
        title="Documentación de Fastino Labs en español",
        summary=(
            "Crea decisiones estructuradas con GLiDE, ejecuta extracción basada en esquemas "
            "con GLiNER y ajusta modelos especializados con Fastino."
        ),
        instructions_heading="Instrucciones para agentes",
        instructions=(
            f"Usa {BASE_URL}/openapi.json como fuente de verdad para las rutas públicas. "
            "GLiDE usa `POST /v1/systemone`; GLiNER y los decodificadores usan "
            "`POST /v1/chat/completions`; las interfaces compatibles usan "
            "`POST /v1/responses` o `POST /v1/messages`. Para entradas GLiNER-2 grandes, "
            "envía `POST /v1/gliner-2/async` y consulta "
            "`GET /v1/gliner-2/jobs/{job_id}`. Usa "
            "`GET /v1/base-models?supports_inference=true` o "
            "`GET /v1/base-models?supports_training=true` para capacidades actuales. "
            "No deduzcas rutas no documentadas. Lee las claves de API desde `FASTINO_API_KEY` "
            "y nunca incluyas credenciales en código, registros o informes."
        ),
        resources_heading="Recursos para agentes",
        root_label="Índice de documentación en inglés",
        root_description="Consulta la documentación de Fastino en inglés.",
        openapi_label="Especificación OpenAPI",
        openapi_description=(
            "Consulta las rutas públicas, los requisitos de autenticación y los esquemas."
        ),
        skills_label="Catálogo de habilidades para agentes",
        skills_description=(
            "Descubre habilidades instalables para datasets, ajuste fino, GLiDE, GLiNER e inferencia."
        ),
    ),
    "fr": LocaleCopy(
        title="Documentation Fastino Labs en français",
        summary=(
            "Créez des décisions structurées avec GLiDE, effectuez une extraction basée sur "
            "des schémas avec GLiNER et affinez des modèles spécialisés avec Fastino."
        ),
        instructions_heading="Instructions pour les agents",
        instructions=(
            f"Utilisez {BASE_URL}/openapi.json comme source de vérité pour les routes publiques. "
            "GLiDE utilise `POST /v1/systemone` ; GLiNER et les décodeurs utilisent "
            "`POST /v1/chat/completions` ; les interfaces compatibles utilisent "
            "`POST /v1/responses` ou `POST /v1/messages`. Pour les grandes entrées GLiNER-2, "
            "envoyez `POST /v1/gliner-2/async`, puis interrogez "
            "`GET /v1/gliner-2/jobs/{job_id}`. Utilisez "
            "`GET /v1/base-models?supports_inference=true` ou "
            "`GET /v1/base-models?supports_training=true` pour les capacités actuelles. "
            "N'inférez pas de routes non documentées. Lisez les clés API depuis `FASTINO_API_KEY` "
            "et n'intégrez jamais d'identifiants dans le code, les journaux ou les rapports."
        ),
        resources_heading="Ressources pour les agents",
        root_label="Index de la documentation anglaise",
        root_description="Consultez la documentation Fastino en anglais.",
        openapi_label="Spécification OpenAPI",
        openapi_description=(
            "Consultez les routes publiques, les exigences d'authentification et les schémas."
        ),
        skills_label="Catalogue de compétences pour agents",
        skills_description=(
            "Découvrez des compétences installables pour les jeux de données, l'affinage, "
            "GLiDE, GLiNER et l'inférence."
        ),
    ),
    "de": LocaleCopy(
        title="Fastino Labs Dokumentation auf Deutsch",
        summary=(
            "Erstellen Sie strukturierte Entscheidungen mit GLiDE, führen Sie schema-basierte "
            "Extraktion mit GLiNER aus und optimieren Sie spezialisierte Modelle mit Fastino."
        ),
        instructions_heading="Anweisungen für Agenten",
        instructions=(
            f"Verwenden Sie {BASE_URL}/openapi.json als verbindliche Quelle für öffentliche Routen. "
            "GLiDE verwendet `POST /v1/systemone`; GLiNER und Decoder verwenden "
            "`POST /v1/chat/completions`; kompatible Schnittstellen verwenden "
            "`POST /v1/responses` oder `POST /v1/messages`. Senden Sie große GLiNER-2-Eingaben "
            "an `POST /v1/gliner-2/async` und fragen Sie anschließend "
            "`GET /v1/gliner-2/jobs/{job_id}` ab. Verwenden Sie "
            "`GET /v1/base-models?supports_inference=true` oder "
            "`GET /v1/base-models?supports_training=true` für aktuelle Fähigkeiten. "
            "Leiten Sie keine undokumentierten Routen ab. Lesen Sie API-Schlüssel aus "
            "`FASTINO_API_KEY` und betten Sie Anmeldedaten niemals in Code, Protokolle oder Berichte ein."
        ),
        resources_heading="Ressourcen für Agenten",
        root_label="Englischer Dokumentationsindex",
        root_description="Lesen Sie die englische Fastino-Dokumentation.",
        openapi_label="OpenAPI-Spezifikation",
        openapi_description=(
            "Lesen Sie öffentliche Routen, Authentifizierungsanforderungen und Schemas."
        ),
        skills_label="Katalog der Agenten-Skills",
        skills_description=(
            "Entdecken Sie installierbare Skills für Datensätze, Fine-Tuning, GLiDE, GLiNER und Inferenz."
        ),
    ),
}


def _visible_pages(node: object) -> list[str]:
    """Return visible page paths in navigation order."""
    if isinstance(node, str):
        return [node]
    if not isinstance(node, dict) or node.get("hidden") is True:
        return []
    pages: list[str] = []
    for key in ("groups", "pages"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                pages.extend(_visible_pages(child))
    return pages


def _navigation_groups(language: dict[str, object]) -> list[dict[str, object]]:
    """Return visible groups from either language-level groups or tabs."""
    containers = language.get("tabs", language.get("groups"))
    if not isinstance(containers, list):
        raise ValueError(f"{language.get('language')} navigation has no groups or tabs")

    groups: list[dict[str, object]] = []
    deferred_sdk_groups: list[dict[str, object]] = []
    for container in containers:
        if not isinstance(container, dict) or container.get("hidden") is True:
            continue
        if "tab" in container:
            children = container.get("pages")
            if not isinstance(children, list):
                raise ValueError(
                    f"{language.get('language')} navigation tab has no pages"
                )
            direct_pages = [child for child in children if isinstance(child, str)]
            if direct_pages:
                groups.append({"group": container["tab"], "pages": direct_pages})
            for child in children:
                if (
                    not isinstance(child, dict)
                    or not isinstance(child.get("group"), str)
                    or child.get("hidden") is True
                ):
                    continue
                child_pages = _visible_pages(child)
                if any(
                    page.endswith("api-reference/inference/openai-compatible")
                    for page in child_pages
                ):
                    deferred_sdk_groups.append(child)
                else:
                    groups.append(child)
        elif isinstance(container.get("group"), str):
            groups.append(container)
    return groups + deferred_sdk_groups


def _frontmatter(root: Path, path: Path) -> tuple[str, str]:
    """Return a page's title and description."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.relative_to(root)} has no frontmatter")
    parts = text.split("---", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"{path.relative_to(root)} has unterminated frontmatter")
    frontmatter = parts[1]
    fields: dict[str, str] = {}
    for name, raw_value in FRONTMATTER_FIELD.findall(frontmatter):
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value.strip()
        if isinstance(value, str):
            fields[name] = value
    missing = {"title", "description"} - fields.keys()
    if missing:
        raise ValueError(f"{path.relative_to(root)} is missing frontmatter fields: {sorted(missing)}")
    return fields["title"], fields["description"]


def _render_index(root: Path, language: dict[str, object]) -> str:
    """Render one localized index."""
    locale = language["language"]
    if not isinstance(locale, str) or locale not in LOCALE_COPY:
        raise ValueError(f"unsupported localized navigation language: {locale!r}")
    copy = LOCALE_COPY[locale]
    lines = [
        f"# {copy.title}",
        "",
        f"> {copy.summary}",
        "",
        f"> ## {copy.instructions_heading}",
        f"> {copy.instructions}",
        "",
        f"## {copy.resources_heading}",
        "",
        f"- [{copy.root_label}]({BASE_URL}/llms.txt): {copy.root_description}",
        (
            f"- [{copy.openapi_label}]({BASE_URL}/openapi.json): "
            f"{copy.openapi_description}"
        ),
        (
            f"- [{copy.skills_label}]({BASE_URL}/.well-known/agent-skills/index.json): "
            f"{copy.skills_description}"
        ),
    ]
    for group in _navigation_groups(language):
        heading = group.get("group")
        if not isinstance(heading, str):
            raise ValueError(f"{locale} navigation contains a group without a name")
        pages = _visible_pages(group)
        if not pages:
            continue
        lines.extend(["", f"## {heading}", ""])
        for page in pages:
            title, description = _frontmatter(root, root / f"{page}.mdx")
            lines.append(f"- [{title}]({BASE_URL}/{page}.md): {description}")
    return "\n".join(lines) + "\n"


def _localized_languages(root: Path) -> list[dict[str, object]]:
    """Load localized navigation entries."""
    config = json.loads((root / "docs.json").read_text(encoding="utf-8"))
    languages = config.get("navigation", {}).get("languages", [])
    if not isinstance(languages, list):
        raise ValueError("docs.json navigation.languages must be a list")
    localized = [
        language
        for language in languages
        if isinstance(language, dict) and language.get("language") != "en"
    ]
    actual = {language.get("language") for language in localized}
    if actual != set(LOCALE_COPY):
        raise ValueError(
            f"localized navigation languages {sorted(actual)} do not match "
            f"generator languages {sorted(LOCALE_COPY)}"
        )
    return localized


def localized_index_findings(root: Path = ROOT) -> list[str]:
    """Return drift findings for localized indexes and root discovery links."""
    findings: list[str] = []
    root_index = (root / "llms.txt").read_text(encoding="utf-8")
    for language in _localized_languages(root):
        locale = language["language"]
        if not isinstance(locale, str):
            raise TypeError("localized navigation language must be a string")
        destination = root / locale / "llms.txt"
        if not destination.is_file() or destination.read_text(encoding="utf-8") != _render_index(
            root, language
        ):
            findings.append(f"{destination.relative_to(root)} is stale")
        index_url = f"{BASE_URL}/{locale}/llms.txt"
        if f"]({index_url})" not in root_index:
            findings.append(f"llms.txt is missing localized index: {index_url}")
    return findings


def main() -> int:
    """Generate indexes or verify that checked-in indexes are current."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        for language in _localized_languages(ROOT):
            locale = language["language"]
            if not isinstance(locale, str):
                raise TypeError("localized navigation language must be a string")
            (ROOT / locale / "llms.txt").write_text(
                _render_index(ROOT, language),
                encoding="utf-8",
            )
        return 0

    findings = localized_index_findings()
    if findings:
        print("Localized llms.txt index validation failed:")
        for finding in findings:
            print(f"- {finding}")
        print("Run: python3 scripts/generate_localized_llms.py")
        return 1
    print(f"Localized llms.txt indexes are current: {len(LOCALE_COPY)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
