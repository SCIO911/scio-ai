"""
SCIO Finance & Economics Knowledge Loader

Laedt Wirtschafts- und Finanzwissen in die Knowledge Base.
"""

import json
from pathlib import Path
from typing import Optional

from scio.core.logging import get_logger
from scio.knowledge.base import (
    KnowledgeBase,
    KnowledgeType,
    ConfidenceLevel,
)

logger = get_logger(__name__)


# Mapping von JSON-Typen zu KnowledgeType
TYPE_MAPPING = {
    "fact": KnowledgeType.FACT,
    "concept": KnowledgeType.CONCEPT,
    "procedure": KnowledgeType.PROCEDURE,
    "rule": KnowledgeType.RULE,
    "experience": KnowledgeType.EXPERIENCE,
    "observation": KnowledgeType.OBSERVATION,
}

CONFIDENCE_MAPPING = {
    "verified": ConfidenceLevel.VERIFIED,
    "high": ConfidenceLevel.HIGH,
    "medium": ConfidenceLevel.MEDIUM,
    "low": ConfidenceLevel.LOW,
}


def load_finance_knowledge(
    kb: KnowledgeBase,
    knowledge_dir: Optional[Path] = None,
) -> int:
    """
    Laedt alle Finanz- und Wirtschaftswissen-Dateien.

    Args:
        kb: Die KnowledgeBase-Instanz
        knowledge_dir: Verzeichnis mit JSON-Wissensdateien

    Returns:
        Anzahl der geladenen Eintraege
    """
    if knowledge_dir is None:
        knowledge_dir = Path(__file__).parent.parent.parent / "data" / "knowledge"

    if not knowledge_dir.exists():
        logger.warning("Knowledge directory not found", path=str(knowledge_dir))
        return 0

    total_loaded = 0

    # Lade alle JSON-Dateien im Verzeichnis
    for json_file in knowledge_dir.glob("*.json"):
        try:
            loaded = _load_knowledge_file(kb, json_file)
            total_loaded += loaded
            logger.info(
                "Loaded knowledge file",
                file=json_file.name,
                entries=loaded,
            )
        except Exception as e:
            logger.error(
                "Failed to load knowledge file",
                file=json_file.name,
                error=str(e),
            )

    return total_loaded


def _sanitize_for_fts(text: str) -> str:
    """Bereinigt Text fuer FTS5 (entfernt problematische Zeichen)."""
    import re
    # Entferne Sonderzeichen die FTS5 stoeren
    text = re.sub(r'[\'\"():;/\\.,!?<>=\-\+\*\[\]{}]', ' ', text)
    # Mehrfache Leerzeichen entfernen
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _load_knowledge_file(kb: KnowledgeBase, file_path: Path) -> int:
    """Laedt eine einzelne Wissensdatei."""
    data = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        data = [data]

    loaded = 0
    for item in data:
        try:
            # Pruefe ob bereits vorhanden (basierend auf content-hash)
            content = item.get("content", "")

            # Skip FTS search bei Problemen - direkt hinzufuegen
            # existing = kb.keyword_search(content[:50], limit=1)
            # if existing and existing[0].content == content:
            #     continue  # Bereits vorhanden

            # Erstelle neuen Eintrag
            knowledge_type = TYPE_MAPPING.get(
                item.get("knowledge_type", "fact"),
                KnowledgeType.FACT,
            )
            confidence = CONFIDENCE_MAPPING.get(
                item.get("confidence", "medium"),
                ConfidenceLevel.MEDIUM,
            )

            # Tags bereinigen fuer FTS5
            raw_tags = item.get("tags", [])
            clean_tags = [_sanitize_for_fts(t) for t in raw_tags]

            kb.add(
                content=content,
                knowledge_type=knowledge_type,
                confidence=confidence,
                source=item.get("source"),
                tags=clean_tags,
                metadata={
                    "original_id": item.get("id"),
                    "category": _detect_category(raw_tags),
                },
            )
            loaded += 1

        except Exception as e:
            logger.warning(
                "Failed to load knowledge entry",
                id=item.get("id"),
                error=str(e),
            )

    return loaded


def _detect_category(tags: list[str]) -> str:
    """Erkennt die Kategorie basierend auf Tags."""
    tag_set = set(t.lower() for t in tags)

    if tag_set & {"trading", "technische-analyse", "chartmuster", "indikatoren"}:
        return "trading"
    elif tag_set & {"krypto", "bitcoin", "ethereum", "defi"}:
        return "crypto"
    elif tag_set & {"wirtschaft", "makro", "bip", "inflation", "konjunktur"}:
        return "economics"
    elif tag_set & {"finanzen", "portfolio", "investieren", "aktien", "dividenden"}:
        return "finance"
    elif tag_set & {"steuern", "recht"}:
        return "tax"
    elif tag_set & {"einkommen", "gpu", "api", "monetarisierung"}:
        return "income"
    elif tag_set & {"risiko", "sicherheit"}:
        return "risk"
    else:
        return "general"


class FinanceKnowledgeQuery:
    """Spezialisierte Abfragen fuer Finanz- und Wirtschaftswissen."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def get_trading_rules(self) -> list:
        """Gibt alle Trading-Regeln zurueck."""
        return self.kb.get_by_tags(["trading", "regel", "rule"], match_all=False)

    def get_risk_management(self) -> list:
        """Gibt Risikomanagement-Wissen zurueck."""
        return self.kb.get_by_tags(["risiko", "risk", "sicherheit"])

    def get_passive_income_strategies(self) -> list:
        """Gibt Strategien fuer passives Einkommen zurueck."""
        return self.kb.get_by_tags(["passives-einkommen", "dividenden", "einkommen"])

    def get_tax_knowledge(self) -> list:
        """Gibt Steuerwissen zurueck."""
        return self.kb.get_by_tags(["steuern", "tax", "kapitalerträge"])

    def get_crypto_knowledge(self) -> list:
        """Gibt Krypto-Wissen zurueck."""
        return self.kb.get_by_tags(["krypto", "bitcoin", "ethereum", "defi"])

    def get_technical_indicators(self) -> list:
        """Gibt technische Indikatoren zurueck."""
        return self.kb.get_by_tags(["indikatoren", "rsi", "macd", "moving-average"])

    def search_finance_topic(self, query: str, limit: int = 5) -> list:
        """Semantische Suche in Finanzwissen."""
        from scio.knowledge.base import KnowledgeQuery

        return self.kb.search(KnowledgeQuery(
            query=query,
            tags=["finanzen", "wirtschaft", "trading", "krypto", "einkommen"],
            limit=limit,
            threshold=0.5,
        ))


def init_finance_knowledge() -> tuple[KnowledgeBase, FinanceKnowledgeQuery]:
    """
    Initialisiert die Knowledge Base mit Finanzwissen.

    Returns:
        Tuple aus (KnowledgeBase, FinanceKnowledgeQuery)
    """
    kb = KnowledgeBase()

    # Lade Finanzwissen
    loaded = load_finance_knowledge(kb)
    logger.info("Finance knowledge initialized", entries=loaded)

    return kb, FinanceKnowledgeQuery(kb)


if __name__ == "__main__":
    # Test-Ausfuehrung
    kb, query = init_finance_knowledge()

    print(f"Knowledge Base Stats: {kb.stats()}")
    print("\n--- Trading Rules ---")
    for entry in query.get_trading_rules()[:3]:
        print(f"- {entry.content[:100]}...")

    print("\n--- Risk Management ---")
    for entry in query.get_risk_management()[:3]:
        print(f"- {entry.content[:100]}...")
