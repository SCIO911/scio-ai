#!/usr/bin/env python3
"""
SCIO - Capability Engine
Intelligente Tool-Auswahl und Orchestrierung
"""

import json
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .tool_registry import ToolRegistry, Tool, ToolCategory, get_tool_registry


class TaskComplexity(str, Enum):
    """Komplexität einer Aufgabe"""
    SIMPLE = "simple"      # Einzelnes Tool
    MODERATE = "moderate"  # 2-3 Tools
    COMPLEX = "complex"    # 4+ Tools, Verzweigungen
    EXPERT = "expert"      # Erfordert ML/Reasoning


@dataclass
class CapabilityMatch:
    """Ein passender Tool-Vorschlag"""
    tool: Tool
    confidence: float
    reasoning: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    """Ein Plan zur Aufgabenausführung"""
    id: str
    description: str
    complexity: TaskComplexity
    steps: List[Dict[str, Any]]
    estimated_time_ms: float
    requires_gpu: bool
    confidence: float


class CapabilityEngine:
    """
    SCIO Capability Engine

    Funktionen:
    - Intelligente Tool-Auswahl basierend auf Aufgabenbeschreibung
    - Multi-Tool Orchestrierung
    - Automatische Pipeline-Erstellung
    - Capability-Matching
    """

    def __init__(self):
        self.registry = get_tool_registry()
        self._initialized = False

        # Keyword-Mappings für schnelle Suche
        self.keyword_mappings: Dict[str, List[str]] = {}

        # Capability Chains (häufige Tool-Kombinationen)
        self.chains: Dict[str, List[str]] = {}

    def initialize(self) -> bool:
        """Initialisiert die Capability Engine"""
        try:
            if not self.registry._initialized:
                self.registry.initialize()

            self._build_keyword_index()
            self._define_chains()
            self._initialized = True
            print(f"[OK] Capability Engine initialisiert ({self.registry._tool_count} Tools)")
            return True
        except Exception as e:
            print(f"[ERROR] Capability Engine Fehler: {e}")
            return False

    def _build_keyword_index(self):
        """Erstellt Keyword-Index für schnelle Suche"""
        # Task-Typen zu Tool-Kategorien
        self.keyword_mappings = {
            # NLP Tasks
            "übersetzen": ["nlp_translate_auto", "nlp_translate_de_en", "nlp_translate_en_de"],
            "translate": ["nlp_translate_auto", "nlp_translate_de_en", "nlp_translate_en_de"],
            "zusammenfassen": ["nlp_text_summarize", "doc_summarize"],
            "summarize": ["nlp_text_summarize", "doc_summarize"],
            "schreiben": ["gen_code_generate", "nlp_text_generate", "doc_email"],
            "write": ["gen_code_generate", "nlp_text_generate", "doc_email"],
            "analysieren": ["nlp_sentiment_analyze", "data_analyze", "code_analyze"],
            "analyze": ["nlp_sentiment_analyze", "data_analyze", "code_analyze"],

            # Image Tasks
            "bild erstellen": ["gen_img_txt2img", "gen_logo", "gen_banner"],
            "generate image": ["gen_img_txt2img", "gen_logo", "gen_banner"],
            "bild bearbeiten": ["img_resize", "img_crop", "img_filter", "gen_img_inpaint"],
            "edit image": ["img_resize", "img_crop", "img_filter", "gen_img_inpaint"],
            "hintergrund entfernen": ["gen_img_remove_bg"],
            "remove background": ["gen_img_remove_bg"],
            "upscale": ["gen_img_upscale", "video_upscale"],
            "hochskalieren": ["gen_img_upscale", "video_upscale"],

            # Audio Tasks
            "transkribieren": ["audio_stt_transcribe", "audio_stt_realtime"],
            "transcribe": ["audio_stt_transcribe", "audio_stt_realtime"],
            "vorlesen": ["audio_tts_synthesize", "audio_tts_emotional"],
            "text to speech": ["audio_tts_synthesize", "audio_tts_emotional"],
            "musik": ["audio_music_generate", "audio_music_remix"],
            "music": ["audio_music_generate", "audio_music_remix"],

            # Video Tasks
            "video erstellen": ["video_generate", "video_img2vid"],
            "create video": ["video_generate", "video_img2vid"],
            "video schneiden": ["video_cut", "video_merge", "video_trim"],
            "edit video": ["video_cut", "video_merge", "video_trim"],
            "untertitel": ["video_subtitle_generate", "video_subtitle_add"],
            "subtitles": ["video_subtitle_generate", "video_subtitle_add"],

            # Code Tasks
            "code schreiben": ["gen_code_generate", "gen_code_complete"],
            "write code": ["gen_code_generate", "gen_code_complete"],
            "code erklären": ["gen_code_explain"],
            "explain code": ["gen_code_explain"],
            "code testen": ["gen_code_test", "code_python_lint"],
            "test code": ["gen_code_test", "code_python_lint"],
            "debugging": ["gen_code_debug", "code_analyze"],
            "refactoring": ["gen_code_refactor", "gen_code_optimize"],

            # Data Tasks
            "daten bereinigen": ["data_clean", "data_dedupe", "data_impute"],
            "clean data": ["data_clean", "data_dedupe", "data_impute"],
            "daten konvertieren": ["csv_parse", "json_parse", "data_transform"],
            "convert data": ["csv_parse", "json_parse", "data_transform"],
            "statistik": ["stats_describe", "stats_correlate", "stats_test"],
            "statistics": ["stats_describe", "stats_correlate", "stats_test"],

            # Document Tasks
            "pdf": ["doc_pdf_read", "doc_pdf_create", "doc_pdf_merge"],
            "word": ["doc_docx_read", "doc_docx_create"],
            "excel": ["doc_xlsx_read", "doc_xlsx_create"],
            "powerpoint": ["doc_pptx_read", "doc_pptx_create"],
            "ocr": ["vision_ocr_extract", "vision_ocr_document"],

            # Web Tasks
            "scrapen": ["web_scrape", "web_crawl", "web_extract"],
            "scrape": ["web_scrape", "web_crawl", "web_extract"],
            "api": ["http_get", "http_post", "api_test"],
            "screenshot": ["web_screenshot"],

            # Security Tasks
            "verschlüsseln": ["sec_encrypt_aes", "sec_encrypt_rsa"],
            "encrypt": ["sec_encrypt_aes", "sec_encrypt_rsa"],
            "hash": ["sec_hash_sha256", "sec_hash_bcrypt"],
            "password": ["sec_password_generate", "sec_password_check"],

            # ML Tasks
            "trainieren": ["ml_train", "ml_finetune"],
            "train": ["ml_train", "ml_finetune"],
            "vorhersagen": ["stats_forecast", "ml_evaluate"],
            "predict": ["stats_forecast", "ml_evaluate"],

            # 3D Tasks
            "3d": ["gen_3d_txt2mesh", "gen_3d_img2mesh", "gen_3d_texture"],
            "mesh": ["gen_3d_txt2mesh", "gen_3d_img2mesh"],

            # Business Tasks
            "rechnung": ["biz_finance_invoice"],
            "invoice": ["biz_finance_invoice"],
            "email": ["biz_email_send", "biz_email_template"],
            "kalender": ["biz_calendar_create", "biz_calendar_list"],
            "calendar": ["biz_calendar_create", "biz_calendar_list"],
        }

    def _define_chains(self):
        """Definiert häufige Tool-Ketten"""
        self.chains = {
            # Document Processing Pipeline
            "pdf_to_text": [
                "doc_pdf_read",
                "vision_ocr_document",
                "nlp_text_summarize"
            ],

            # Image Enhancement Pipeline
            "image_enhance": [
                "gen_img_upscale",
                "gen_img_face_restore",
                "img_sharpen"
            ],

            # Video Production Pipeline
            "video_production": [
                "video_generate",
                "audio_tts_synthesize",
                "video_add_audio",
                "video_subtitle_generate"
            ],

            # Data Analysis Pipeline
            "data_analysis": [
                "csv_parse",
                "data_clean",
                "stats_describe",
                "viz_chart_bar"
            ],

            # Code Review Pipeline
            "code_review": [
                "code_analyze",
                "code_security",
                "gen_code_review",
                "gen_code_document"
            ],

            # Translation Pipeline
            "document_translation": [
                "doc_pdf_read",
                "nlp_translate_auto",
                "doc_pdf_create"
            ],

            # SEO Pipeline
            "seo_optimization": [
                "web_scrape",
                "seo_analyze",
                "nlp_keyword_extract",
                "nlp_text_generate"
            ],

            # ML Training Pipeline
            "ml_pipeline": [
                "data_clean",
                "ml_preprocess",
                "ml_feature_eng",
                "ml_split",
                "ml_train",
                "ml_evaluate"
            ],

            # Content Creation
            "content_creation": [
                "nlp_text_generate",
                "gen_img_txt2img",
                "audio_tts_synthesize"
            ],
        }

    def find_tools(self, task_description: str, limit: int = 5) -> List[CapabilityMatch]:
        """
        Findet passende Tools für eine Aufgabenbeschreibung

        Args:
            task_description: Beschreibung der Aufgabe
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Liste von CapabilityMatch mit passenden Tools
        """
        task_lower = task_description.lower()
        matches = []
        seen_tools = set()

        # 1. Keyword-basierte Suche
        for keyword, tool_ids in self.keyword_mappings.items():
            if keyword in task_lower:
                for tool_id in tool_ids:
                    if tool_id not in seen_tools:
                        tool = self.registry.get_tool(tool_id)
                        if tool:
                            matches.append(CapabilityMatch(
                                tool=tool,
                                confidence=0.9,
                                reasoning=f"Keyword match: '{keyword}'",
                                alternatives=tool_ids[:3]
                            ))
                            seen_tools.add(tool_id)

        # 2. Registry-Suche für nicht-gematchte Begriffe
        if len(matches) < limit:
            registry_results = self.registry.search(query=task_description, limit=limit * 2)
            for tool in registry_results:
                if tool.id not in seen_tools:
                    matches.append(CapabilityMatch(
                        tool=tool,
                        confidence=0.6,
                        reasoning="Registry search match"
                    ))
                    seen_tools.add(tool.id)

        # Sortiere nach Confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        return matches[:limit]

    def create_plan(self, task_description: str) -> TaskPlan:
        """
        Erstellt einen Ausführungsplan für eine Aufgabe

        Args:
            task_description: Beschreibung der Aufgabe

        Returns:
            TaskPlan mit Schritten
        """
        import uuid

        # Finde passende Tools
        matches = self.find_tools(task_description, limit=10)

        if not matches:
            return TaskPlan(
                id=f"plan_{uuid.uuid4().hex[:8]}",
                description=task_description,
                complexity=TaskComplexity.SIMPLE,
                steps=[],
                estimated_time_ms=0,
                requires_gpu=False,
                confidence=0
            )

        # Analysiere Komplexität
        task_lower = task_description.lower()
        complexity = TaskComplexity.SIMPLE

        # Suche nach passender Chain
        selected_chain = None
        for chain_name, chain_tools in self.chains.items():
            chain_keywords = chain_name.replace("_", " ")
            if any(kw in task_lower for kw in chain_keywords.split()):
                selected_chain = chain_tools
                complexity = TaskComplexity.COMPLEX if len(chain_tools) > 3 else TaskComplexity.MODERATE
                break

        # Erstelle Schritte
        steps = []
        requires_gpu = False
        estimated_time = 0

        if selected_chain:
            for i, tool_id in enumerate(selected_chain):
                tool = self.registry.get_tool(tool_id)
                if tool:
                    steps.append({
                        "step": i + 1,
                        "tool_id": tool_id,
                        "tool_name": tool.name,
                        "description": tool.description,
                        "input_types": tool.input_types,
                        "output_types": tool.output_types
                    })
                    if tool.requires_gpu:
                        requires_gpu = True
                    estimated_time += tool.avg_execution_time_ms or 1000
        else:
            # Nutze Top-Matches
            for i, match in enumerate(matches[:3]):
                steps.append({
                    "step": i + 1,
                    "tool_id": match.tool.id,
                    "tool_name": match.tool.name,
                    "description": match.tool.description,
                    "confidence": match.confidence,
                    "reasoning": match.reasoning
                })
                if match.tool.requires_gpu:
                    requires_gpu = True
                estimated_time += match.tool.avg_execution_time_ms or 1000

            if len(steps) > 1:
                complexity = TaskComplexity.MODERATE

        return TaskPlan(
            id=f"plan_{uuid.uuid4().hex[:8]}",
            description=task_description,
            complexity=complexity,
            steps=steps,
            estimated_time_ms=estimated_time,
            requires_gpu=requires_gpu,
            confidence=matches[0].confidence if matches else 0
        )

    def get_capabilities_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Gibt alle Capabilities einer Domain zurück"""
        domain_lower = domain.lower()

        # Mapping Domain zu Kategorien
        domain_categories = {
            "text": [ToolCategory.NLP],
            "nlp": [ToolCategory.NLP],
            "sprache": [ToolCategory.NLP, ToolCategory.AUDIO],
            "language": [ToolCategory.NLP, ToolCategory.AUDIO],
            "bild": [ToolCategory.VISION, ToolCategory.IMAGE, ToolCategory.GENERATIVE],
            "image": [ToolCategory.VISION, ToolCategory.IMAGE, ToolCategory.GENERATIVE],
            "video": [ToolCategory.VIDEO],
            "audio": [ToolCategory.AUDIO, ToolCategory.AUDIO_PROCESSING],
            "code": [ToolCategory.CODE],
            "programmierung": [ToolCategory.CODE],
            "daten": [ToolCategory.DATA_PROCESSING, ToolCategory.ANALYTICS],
            "data": [ToolCategory.DATA_PROCESSING, ToolCategory.ANALYTICS],
            "dokument": [ToolCategory.DOCUMENT, ToolCategory.PDF, ToolCategory.OFFICE],
            "document": [ToolCategory.DOCUMENT, ToolCategory.PDF, ToolCategory.OFFICE],
            "web": [ToolCategory.WEB_SCRAPING, ToolCategory.WEB_AUTOMATION],
            "sicherheit": [ToolCategory.SECURITY, ToolCategory.ENCRYPTION],
            "security": [ToolCategory.SECURITY, ToolCategory.ENCRYPTION],
            "cloud": [ToolCategory.CLOUD, ToolCategory.CONTAINER],
            "ml": [ToolCategory.ML_OPS],
            "ki": [ToolCategory.GENERATIVE, ToolCategory.ML_OPS],
            "ai": [ToolCategory.GENERATIVE, ToolCategory.ML_OPS],
            "3d": [ToolCategory.THREE_D],
            "business": [ToolCategory.CRM, ToolCategory.FINANCE, ToolCategory.HR],
            "wissenschaft": [ToolCategory.MATH, ToolCategory.STATISTICS, ToolCategory.RESEARCH],
            "science": [ToolCategory.MATH, ToolCategory.STATISTICS, ToolCategory.RESEARCH],
        }

        categories = domain_categories.get(domain_lower, [])
        capabilities = []

        for category in categories:
            tools = self.registry.search(category=category, limit=100)
            for tool in tools:
                capabilities.append({
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category.value,
                    "tags": tool.tags
                })

        return capabilities

    def get_all_capabilities_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung aller Capabilities zurück"""
        stats = self.registry.get_statistics()
        categories = self.registry.get_categories()

        # Top Tags
        top_tags = sorted(
            [(tag, len(tools)) for tag, tools in self.registry.tags_index.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]

        return {
            "total_tools": stats["total_tools"],
            "total_categories": stats["categories"],
            "categories": categories,
            "top_tags": dict(top_tags),
            "chains_available": list(self.chains.keys()),
            "gpu_tools": stats["gpu_required_tools"],
            "network_tools": stats["network_required_tools"]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        return {
            "registry": self.registry.get_statistics(),
            "keyword_mappings": len(self.keyword_mappings),
            "chains_defined": len(self.chains),
            "initialized": self._initialized
        }

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Gibt alle registrierten Tools zurueck"""
        tools = []
        for tool_id, tool in self.registry.tools.items():
            tools.append({
                "id": tool_id,
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value if hasattr(tool.category, 'value') else str(tool.category),
                "tags": list(tool.tags) if tool.tags else [],
                "requires_gpu": tool.requires_gpu,
                "requires_network": tool.requires_network
            })
        return tools


# Singleton
_capability_engine: Optional[CapabilityEngine] = None

def get_capability_engine() -> CapabilityEngine:
    """Gibt Singleton-Instanz zurück"""
    global _capability_engine
    if _capability_engine is None:
        _capability_engine = CapabilityEngine()
    return _capability_engine
