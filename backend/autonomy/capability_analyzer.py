#!/usr/bin/env python3
"""
SCIO - Capability Analyzer
Analysiert vorhandene Fähigkeiten und erkennt Lücken

Features:
- Analyse aller Worker-Fähigkeiten
- Erkennung fehlender Fähigkeiten
- Vergleich mit Ideal-Zustand
- Priorisierung von Erweiterungen
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class CapabilityCategory(str, Enum):
    """Kategorien von Fähigkeiten"""
    LLM = "llm"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    DOCUMENT = "document"
    THREE_D = "3d"
    AUTOMATION = "automation"
    INTEGRATION = "integration"


@dataclass
class Capability:
    """Einzelne Fähigkeit"""
    name: str
    category: CapabilityCategory
    description: str
    implemented: bool
    worker: Optional[str] = None
    quality_score: float = 0.0  # 0-1
    priority: int = 5  # 1-10, higher = more important


# Alle möglichen Fähigkeiten die SCIO haben könnte
IDEAL_CAPABILITIES = {
    # LLM
    "llm_inference": Capability("llm_inference", CapabilityCategory.LLM, "Text-Generierung mit LLMs", False, priority=10),
    "llm_training": Capability("llm_training", CapabilityCategory.LLM, "LLM Fine-Tuning", False, priority=9),
    "llm_rag": Capability("llm_rag", CapabilityCategory.LLM, "Retrieval Augmented Generation", False, priority=8),
    "llm_agents": Capability("llm_agents", CapabilityCategory.LLM, "Autonome LLM-Agenten", False, priority=9),
    "llm_function_calling": Capability("llm_function_calling", CapabilityCategory.LLM, "Function Calling", False, priority=8),

    # Image
    "image_generation": Capability("image_generation", CapabilityCategory.IMAGE, "Bild-Generierung", False, priority=9),
    "image_editing": Capability("image_editing", CapabilityCategory.IMAGE, "Bild-Bearbeitung", False, priority=7),
    "image_upscale": Capability("image_upscale", CapabilityCategory.IMAGE, "Bild-Upscaling", False, priority=6),
    "image_inpainting": Capability("image_inpainting", CapabilityCategory.IMAGE, "Bild-Inpainting", False, priority=6),
    "image_style_transfer": Capability("image_style_transfer", CapabilityCategory.IMAGE, "Stil-Transfer", False, priority=5),
    "image_background_removal": Capability("image_background_removal", CapabilityCategory.IMAGE, "Hintergrund-Entfernung", False, priority=6),

    # Audio
    "audio_stt": Capability("audio_stt", CapabilityCategory.AUDIO, "Speech-to-Text", False, priority=8),
    "audio_tts": Capability("audio_tts", CapabilityCategory.AUDIO, "Text-to-Speech", False, priority=8),
    "audio_music": Capability("audio_music", CapabilityCategory.AUDIO, "Musik-Generierung", False, priority=6),
    "audio_voice_clone": Capability("audio_voice_clone", CapabilityCategory.AUDIO, "Stimmen-Klonen", False, priority=5),
    "audio_separation": Capability("audio_separation", CapabilityCategory.AUDIO, "Audio-Trennung", False, priority=4),

    # Video
    "video_generation": Capability("video_generation", CapabilityCategory.VIDEO, "Video-Generierung", False, priority=7),
    "video_image_to_video": Capability("video_image_to_video", CapabilityCategory.VIDEO, "Bild-zu-Video", False, priority=6),
    "video_editing": Capability("video_editing", CapabilityCategory.VIDEO, "Video-Bearbeitung", False, priority=5),
    "video_interpolation": Capability("video_interpolation", CapabilityCategory.VIDEO, "Frame-Interpolation", False, priority=4),

    # Code
    "code_generation": Capability("code_generation", CapabilityCategory.CODE, "Code-Generierung", False, priority=10),
    "code_completion": Capability("code_completion", CapabilityCategory.CODE, "Code-Vervollständigung", False, priority=9),
    "code_review": Capability("code_review", CapabilityCategory.CODE, "Code-Review", False, priority=8),
    "code_fix": Capability("code_fix", CapabilityCategory.CODE, "Bug-Fixing", False, priority=9),
    "code_translate": Capability("code_translate", CapabilityCategory.CODE, "Code-Übersetzung", False, priority=6),
    "code_explain": Capability("code_explain", CapabilityCategory.CODE, "Code-Erklärung", False, priority=7),
    "code_test_generation": Capability("code_test_generation", CapabilityCategory.CODE, "Test-Generierung", False, priority=7),
    "code_refactor": Capability("code_refactor", CapabilityCategory.CODE, "Code-Refactoring", False, priority=6),

    # Vision
    "vision_ocr": Capability("vision_ocr", CapabilityCategory.VISION, "OCR", False, priority=7),
    "vision_caption": Capability("vision_caption", CapabilityCategory.VISION, "Bild-Beschreibung", False, priority=7),
    "vision_object_detection": Capability("vision_object_detection", CapabilityCategory.VISION, "Objekt-Erkennung", False, priority=6),
    "vision_face_detection": Capability("vision_face_detection", CapabilityCategory.VISION, "Gesichts-Erkennung", False, priority=5),
    "vision_segmentation": Capability("vision_segmentation", CapabilityCategory.VISION, "Bild-Segmentierung", False, priority=5),

    # Embeddings
    "embeddings_text": Capability("embeddings_text", CapabilityCategory.EMBEDDINGS, "Text-Embeddings", False, priority=8),
    "embeddings_image": Capability("embeddings_image", CapabilityCategory.EMBEDDINGS, "Bild-Embeddings", False, priority=6),
    "embeddings_search": Capability("embeddings_search", CapabilityCategory.EMBEDDINGS, "Semantische Suche", False, priority=8),
    "embeddings_clustering": Capability("embeddings_clustering", CapabilityCategory.EMBEDDINGS, "Clustering", False, priority=5),

    # Document
    "document_pdf": Capability("document_pdf", CapabilityCategory.DOCUMENT, "PDF-Verarbeitung", False, priority=7),
    "document_parse": Capability("document_parse", CapabilityCategory.DOCUMENT, "Dokument-Parsing", False, priority=7),
    "document_chunk": Capability("document_chunk", CapabilityCategory.DOCUMENT, "Text-Chunking", False, priority=6),
    "document_summarize": Capability("document_summarize", CapabilityCategory.DOCUMENT, "Zusammenfassung", False, priority=7),
    "document_translate": Capability("document_translate", CapabilityCategory.DOCUMENT, "Dokument-Übersetzung", False, priority=6),

    # 3D
    "3d_text_to_3d": Capability("3d_text_to_3d", CapabilityCategory.THREE_D, "Text-zu-3D", False, priority=5),
    "3d_image_to_3d": Capability("3d_image_to_3d", CapabilityCategory.THREE_D, "Bild-zu-3D", False, priority=5),
    "3d_mesh_generation": Capability("3d_mesh_generation", CapabilityCategory.THREE_D, "Mesh-Generierung", False, priority=4),

    # Automation
    "automation_scheduling": Capability("automation_scheduling", CapabilityCategory.AUTOMATION, "Task-Scheduling", False, priority=8),
    "automation_workflow": Capability("automation_workflow", CapabilityCategory.AUTOMATION, "Workflow-Automation", False, priority=7),
    "automation_monitoring": Capability("automation_monitoring", CapabilityCategory.AUTOMATION, "System-Monitoring", False, priority=8),
    "automation_self_healing": Capability("automation_self_healing", CapabilityCategory.AUTOMATION, "Selbst-Heilung", False, priority=9),
    "automation_self_programming": Capability("automation_self_programming", CapabilityCategory.AUTOMATION, "Selbst-Programmierung", False, priority=10),

    # Integration
    "integration_api": Capability("integration_api", CapabilityCategory.INTEGRATION, "REST API", False, priority=9),
    "integration_websocket": Capability("integration_websocket", CapabilityCategory.INTEGRATION, "WebSocket", False, priority=7),
    "integration_webhook": Capability("integration_webhook", CapabilityCategory.INTEGRATION, "Webhooks", False, priority=6),
    "integration_stripe": Capability("integration_stripe", CapabilityCategory.INTEGRATION, "Stripe-Zahlungen", False, priority=7),
    "integration_cloud_gpu": Capability("integration_cloud_gpu", CapabilityCategory.INTEGRATION, "Cloud-GPU (Vast.ai/RunPod)", False, priority=6),
}


class CapabilityAnalyzer:
    """
    SCIO Capability Analyzer

    Analysiert welche Fähigkeiten SCIO hat und welche fehlen.
    """

    def __init__(self):
        self.self_awareness = None
        self.capabilities: Dict[str, Capability] = {}
        self._initialized = False

    def initialize(self, self_awareness) -> bool:
        """Initialisiert mit Self-Awareness Referenz"""
        self.self_awareness = self_awareness

        # Copy ideal capabilities
        for name, cap in IDEAL_CAPABILITIES.items():
            self.capabilities[name] = Capability(
                name=cap.name,
                category=cap.category,
                description=cap.description,
                implemented=False,
                worker=None,
                quality_score=0.0,
                priority=cap.priority,
            )

        # Analyze actual implementation
        self._analyze_implementations()
        self._initialized = True

        return True

    def _analyze_implementations(self):
        """Analysiert welche Fähigkeiten implementiert sind"""
        if not self.self_awareness:
            return

        workers = self.self_awareness.get_workers()

        # Map worker methods to capabilities
        capability_mapping = {
            # CodeWorker
            "generate_code": "code_generation",
            "complete_code": "code_completion",
            "review_code": "code_review",
            "fix_code": "code_fix",
            "translate_code": "code_translate",
            "explain_code": "code_explain",

            # LLM Worker
            "generate": "llm_inference",
            "inference": "llm_inference",
            "train": "llm_training",
            "fine_tune": "llm_training",

            # Image Worker
            "generate_image": "image_generation",
            "upscale": "image_upscale",
            "inpaint": "image_inpainting",

            # Audio Worker
            "transcribe": "audio_stt",
            "speech_to_text": "audio_stt",
            "text_to_speech": "audio_tts",
            "synthesize": "audio_tts",
            "generate_music": "audio_music",

            # Video Worker
            "generate_video": "video_generation",
            "image_to_video": "video_image_to_video",

            # Vision Worker
            "ocr": "vision_ocr",
            "extract_text": "vision_ocr",
            "caption": "vision_caption",
            "describe": "vision_caption",
            "detect_objects": "vision_object_detection",

            # Embedding Worker
            "embed": "embeddings_text",
            "embed_text": "embeddings_text",
            "embed_image": "embeddings_image",
            "search": "embeddings_search",
            "similarity_search": "embeddings_search",

            # Document Worker
            "parse_pdf": "document_pdf",
            "extract_pdf": "document_pdf",
            "parse": "document_parse",
            "chunk": "document_chunk",
            "chunk_text": "document_chunk",

            # 3D Worker
            "text_to_3d": "3d_text_to_3d",
            "image_to_3d": "3d_image_to_3d",
        }

        for worker_name, worker_info in workers.items():
            for method in worker_info.methods:
                # Check direct mapping
                if method in capability_mapping:
                    cap_name = capability_mapping[method]
                    if cap_name in self.capabilities:
                        self.capabilities[cap_name].implemented = True
                        self.capabilities[cap_name].worker = worker_name
                        self.capabilities[cap_name].quality_score = 0.8

                # Check partial matches
                for method_pattern, cap_name in capability_mapping.items():
                    if method_pattern in method.lower():
                        if cap_name in self.capabilities:
                            self.capabilities[cap_name].implemented = True
                            self.capabilities[cap_name].worker = worker_name
                            self.capabilities[cap_name].quality_score = 0.7

        # Check for automation capabilities
        routes = self.self_awareness.get_routes()

        if "admin" in routes:
            self.capabilities["automation_monitoring"].implemented = True
            self.capabilities["automation_monitoring"].quality_score = 0.8

        if "api" in routes:
            self.capabilities["integration_api"].implemented = True
            self.capabilities["integration_api"].quality_score = 0.9

        if "webhooks" in routes:
            self.capabilities["integration_webhook"].implemented = True
            self.capabilities["integration_stripe"].implemented = True
            self.capabilities["integration_webhook"].quality_score = 0.8
            self.capabilities["integration_stripe"].quality_score = 0.8

        # Check for scheduler
        code_files = self.self_awareness._code_files
        for path, file_info in code_files.items():
            if "scheduler" in path.lower():
                self.capabilities["automation_scheduling"].implemented = True
                self.capabilities["automation_scheduling"].quality_score = 0.8

    def analyze_all(self) -> dict:
        """Gibt Analyse aller Fähigkeiten zurück"""
        implemented = []
        missing = []

        for name, cap in self.capabilities.items():
            cap_dict = {
                "name": name,
                "category": cap.category.value,
                "description": cap.description,
                "worker": cap.worker,
                "quality_score": cap.quality_score,
                "priority": cap.priority,
            }

            if cap.implemented:
                implemented.append(cap_dict)
            else:
                missing.append(cap_dict)

        # Sort by priority
        implemented.sort(key=lambda x: x["priority"], reverse=True)
        missing.sort(key=lambda x: x["priority"], reverse=True)

        return {
            "total": len(self.capabilities),
            "implemented_count": len(implemented),
            "missing_count": len(missing),
            "coverage_percent": round(len(implemented) / len(self.capabilities) * 100, 1),
            "implemented": implemented,
            "missing": missing,
        }

    def find_gaps(self) -> List[dict]:
        """Findet fehlende Fähigkeiten, sortiert nach Priorität"""
        gaps = []

        for name, cap in self.capabilities.items():
            if not cap.implemented:
                gaps.append({
                    "name": name,
                    "category": cap.category.value,
                    "description": cap.description,
                    "priority": cap.priority,
                    "suggested_worker": self._suggest_worker(cap),
                })

        # Sort by priority (highest first)
        gaps.sort(key=lambda x: x["priority"], reverse=True)

        return gaps

    def _suggest_worker(self, cap: Capability) -> str:
        """Schlägt einen Worker für fehlende Fähigkeit vor"""
        worker_suggestions = {
            CapabilityCategory.LLM: "llm_inference.py / llm_training.py",
            CapabilityCategory.IMAGE: "image_gen.py",
            CapabilityCategory.AUDIO: "audio_worker.py",
            CapabilityCategory.VIDEO: "video_worker.py",
            CapabilityCategory.CODE: "code_worker.py",
            CapabilityCategory.VISION: "vision_worker.py",
            CapabilityCategory.EMBEDDINGS: "embedding_worker.py",
            CapabilityCategory.DOCUMENT: "document_worker.py",
            CapabilityCategory.THREE_D: "threed_worker.py",
            CapabilityCategory.AUTOMATION: "automation/ (neuer Service)",
            CapabilityCategory.INTEGRATION: "integrations/ (neue Integration)",
        }

        return worker_suggestions.get(cap.category, "Neuer Worker benötigt")

    def get_by_category(self, category: CapabilityCategory) -> List[Capability]:
        """Gibt Fähigkeiten nach Kategorie zurück"""
        return [cap for cap in self.capabilities.values() if cap.category == category]

    def get_implemented(self) -> List[Capability]:
        """Gibt implementierte Fähigkeiten zurück"""
        return [cap for cap in self.capabilities.values() if cap.implemented]

    def get_missing(self) -> List[Capability]:
        """Gibt fehlende Fähigkeiten zurück"""
        return [cap for cap in self.capabilities.values() if not cap.implemented]

    def get_priority_improvements(self, limit: int = 5) -> List[dict]:
        """Gibt die wichtigsten Verbesserungen zurück"""
        gaps = self.find_gaps()
        return gaps[:limit]


# Singleton
_capability_analyzer: Optional[CapabilityAnalyzer] = None

def get_capability_analyzer() -> CapabilityAnalyzer:
    """Gibt Singleton-Instanz zurück"""
    global _capability_analyzer
    if _capability_analyzer is None:
        _capability_analyzer = CapabilityAnalyzer()
    return _capability_analyzer
