"""
SCIO Unified MultiModal Engine

Einheitliche Verarbeitung aller Modalitäten:
- Text
- Bild
- Audio
- Video
- Code
- 3D

Jede Modalität kann verstanden, generiert und transformiert werden.
"""

import asyncio
import base64
import hashlib
import io
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ModalityType(str, Enum):
    """Verfügbare Modalitäten."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    THREE_D = "3d"


class OperationType(str, Enum):
    """Verfügbare Operationen."""
    UNDERSTAND = "understand"
    GENERATE = "generate"
    TRANSFORM = "transform"
    TRANSLATE = "translate"
    EDIT = "edit"
    ANALYZE = "analyze"


@dataclass
class MultiModalResult:
    """Ergebnis einer MultiModal-Operation."""

    operation: OperationType
    input_modality: ModalityType
    output_modality: ModalityType
    content: Any  # bytes, str, or dict
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation.value,
            "input_modality": self.input_modality.value,
            "output_modality": self.output_modality.value,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class Understanding:
    """Verständnis eines Inputs."""

    description: str
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalConfig:
    """Konfiguration für MultiModal Engine."""

    # Modelle
    text_model: str = "gpt-4"
    image_model: str = "flux-dev"
    audio_model: str = "whisper-large"
    video_model: str = "runway-gen3"
    code_model: str = "codellama-34b"
    three_d_model: str = "point-e"

    # Limits
    max_text_length: int = 100000
    max_image_size: int = 4096
    max_audio_duration_seconds: float = 3600.0
    max_video_duration_seconds: float = 300.0

    # Qualität
    default_image_quality: str = "high"  # low, medium, high
    default_audio_quality: str = "high"
    default_video_quality: str = "720p"


# ============================================================================
# PROCESSOR BASE CLASS
# ============================================================================

class ModalityProcessor(ABC):
    """Abstrakte Basisklasse für Modalitäts-Prozessoren."""

    @property
    @abstractmethod
    def modality(self) -> ModalityType:
        """Gibt den Modalitätstyp zurück."""
        pass

    @abstractmethod
    async def understand(self, content: Any) -> Understanding:
        """Versteht den Inhalt."""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Any:
        """Generiert Inhalt basierend auf Prompt."""
        pass


# ============================================================================
# TEXT PROCESSOR
# ============================================================================

class TextProcessor(ModalityProcessor):
    """Prozessor für Text-Modalität."""

    @property
    def modality(self) -> ModalityType:
        return ModalityType.TEXT

    def __init__(self, llm_callback: Optional[Callable] = None):
        self.llm_callback = llm_callback or self._default_llm

    async def _default_llm(self, prompt: str, **kwargs) -> str:
        """Standard LLM Callback."""
        return f"Generated text for: {prompt[:50]}..."

    async def understand(self, text: str) -> Understanding:
        """Analysiert und versteht Text."""
        prompt = f"""Analyze this text and provide:
1. A brief description
2. Key entities mentioned
3. Main concepts
4. Overall sentiment

Text: {text[:2000]}

Analysis:"""

        response = await self.llm_callback(prompt)

        return Understanding(
            description=response,
            entities=self._extract_entities(text),
            concepts=self._extract_concepts(text),
            sentiment=self._analyze_sentiment(text),
            confidence=0.85,
        )

    def _extract_entities(self, text: str) -> List[str]:
        """Extrahiert Entitäten (vereinfacht)."""
        # In echter Implementierung würde man NER verwenden
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 2]
        return list(set(entities))[:10]

    def _extract_concepts(self, text: str) -> List[str]:
        """Extrahiert Konzepte (vereinfacht)."""
        # Keywords basierend auf Häufigkeit
        words = text.lower().split()
        word_count = {}
        for w in words:
            if len(w) > 4:
                word_count[w] = word_count.get(w, 0) + 1
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:10]]

    def _analyze_sentiment(self, text: str) -> str:
        """Analysiert Sentiment (vereinfacht)."""
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "happy", "love"}
        negative_words = {"bad", "terrible", "awful", "horrible", "sad", "hate", "angry"}

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generiert Text."""
        return await self.llm_callback(prompt, **kwargs)

    async def translate(self, text: str, target_language: str) -> str:
        """Übersetzt Text."""
        prompt = f"Translate the following text to {target_language}:\n\n{text}\n\nTranslation:"
        return await self.llm_callback(prompt)

    async def summarize(self, text: str, max_length: int = 200) -> str:
        """Fasst Text zusammen."""
        prompt = f"Summarize the following text in about {max_length} words:\n\n{text}\n\nSummary:"
        return await self.llm_callback(prompt)

    async def expand(self, text: str, target_length: int = 500) -> str:
        """Erweitert Text."""
        prompt = f"Expand the following text to about {target_length} words while maintaining the meaning:\n\n{text}\n\nExpanded:"
        return await self.llm_callback(prompt)

    async def paraphrase(self, text: str) -> str:
        """Paraphrasiert Text."""
        prompt = f"Paraphrase the following text while keeping the same meaning:\n\n{text}\n\nParaphrase:"
        return await self.llm_callback(prompt)


# ============================================================================
# IMAGE PROCESSOR
# ============================================================================

class ImageProcessor(ModalityProcessor):
    """Prozessor für Bild-Modalität."""

    @property
    def modality(self) -> ModalityType:
        return ModalityType.IMAGE

    def __init__(
        self,
        vision_callback: Optional[Callable] = None,
        generation_callback: Optional[Callable] = None,
    ):
        self.vision_callback = vision_callback or self._default_vision
        self.generation_callback = generation_callback or self._default_generation

    async def _default_vision(self, image: bytes, prompt: str) -> str:
        """Standard Vision Callback."""
        return f"Image analysis: A picture related to {prompt[:50]}"

    async def _default_generation(self, prompt: str, **kwargs) -> bytes:
        """Standard Generation Callback."""
        # Platzhalter - gibt leeres 1x1 PNG zurück
        return b'\x89PNG\r\n\x1a\n'

    async def understand(self, image: bytes) -> Understanding:
        """Analysiert und versteht ein Bild."""
        prompt = "Describe this image in detail. Include objects, colors, composition, and mood."

        description = await self.vision_callback(image, prompt)

        return Understanding(
            description=description,
            entities=self._extract_objects(description),
            concepts=["visual", "image"],
            confidence=0.8,
        )

    def _extract_objects(self, description: str) -> List[str]:
        """Extrahiert Objekte aus Beschreibung."""
        # Vereinfacht - in echt würde man Object Detection verwenden
        common_objects = [
            "person", "car", "dog", "cat", "tree", "building",
            "sky", "water", "road", "grass", "flower"
        ]
        found = [obj for obj in common_objects if obj in description.lower()]
        return found

    async def generate(self, prompt: str, **kwargs) -> bytes:
        """Generiert ein Bild."""
        return await self.generation_callback(prompt, **kwargs)

    async def caption(self, image: bytes) -> str:
        """Erstellt eine Bildunterschrift."""
        prompt = "Write a concise caption for this image in one sentence."
        return await self.vision_callback(image, prompt)

    async def ocr(self, image: bytes) -> str:
        """Extrahiert Text aus Bild."""
        prompt = "Extract and return all text visible in this image."
        return await self.vision_callback(image, prompt)

    async def edit(self, image: bytes, instruction: str) -> bytes:
        """Bearbeitet ein Bild nach Anweisung."""
        # In echter Implementierung würde man Inpainting verwenden
        return image  # Platzhalter

    async def variation(self, image: bytes, count: int = 1) -> List[bytes]:
        """Erstellt Variationen eines Bildes."""
        variations = []
        for _ in range(count):
            # Platzhalter
            variations.append(image)
        return variations


# ============================================================================
# AUDIO PROCESSOR
# ============================================================================

class AudioProcessor(ModalityProcessor):
    """Prozessor für Audio-Modalität."""

    @property
    def modality(self) -> ModalityType:
        return ModalityType.AUDIO

    def __init__(
        self,
        transcription_callback: Optional[Callable] = None,
        tts_callback: Optional[Callable] = None,
        music_callback: Optional[Callable] = None,
    ):
        self.transcription_callback = transcription_callback or self._default_transcription
        self.tts_callback = tts_callback or self._default_tts
        self.music_callback = music_callback or self._default_music

    async def _default_transcription(self, audio: bytes) -> str:
        return "Transcribed audio content..."

    async def _default_tts(self, text: str, voice: str = "default") -> bytes:
        return b''  # Empty audio

    async def _default_music(self, prompt: str, duration: float = 30.0) -> bytes:
        return b''

    async def understand(self, audio: bytes) -> Understanding:
        """Analysiert und versteht Audio."""
        transcription = await self.transcription_callback(audio)

        return Understanding(
            description=transcription,
            entities=[],
            concepts=["audio", "speech"],
            confidence=0.85,
        )

    async def generate(self, prompt: str, **kwargs) -> bytes:
        """Generiert Audio/Musik."""
        if kwargs.get("type") == "music":
            return await self.music_callback(prompt, kwargs.get("duration", 30.0))
        else:
            return await self.tts_callback(prompt, kwargs.get("voice", "default"))

    async def transcribe(self, audio: bytes, language: str = "auto") -> str:
        """Transkribiert Audio zu Text."""
        return await self.transcription_callback(audio)

    async def speak(self, text: str, voice: str = "default") -> bytes:
        """Konvertiert Text zu Sprache."""
        return await self.tts_callback(text, voice)

    async def generate_music(self, prompt: str, duration: float = 30.0) -> bytes:
        """Generiert Musik."""
        return await self.music_callback(prompt, duration)


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class VideoProcessor(ModalityProcessor):
    """Prozessor für Video-Modalität."""

    @property
    def modality(self) -> ModalityType:
        return ModalityType.VIDEO

    def __init__(
        self,
        understanding_callback: Optional[Callable] = None,
        generation_callback: Optional[Callable] = None,
    ):
        self.understanding_callback = understanding_callback or self._default_understanding
        self.generation_callback = generation_callback or self._default_generation

    async def _default_understanding(self, video: bytes) -> str:
        return "Video content description..."

    async def _default_generation(self, prompt: str, duration: float = 5.0) -> bytes:
        return b''

    async def understand(self, video: bytes) -> Understanding:
        """Analysiert und versteht Video."""
        description = await self.understanding_callback(video)

        return Understanding(
            description=description,
            entities=[],
            concepts=["video", "motion"],
            confidence=0.75,
        )

    async def generate(self, prompt: str, **kwargs) -> bytes:
        """Generiert Video."""
        duration = kwargs.get("duration", 5.0)
        return await self.generation_callback(prompt, duration)

    async def summarize(self, video: bytes) -> str:
        """Erstellt Video-Zusammenfassung."""
        return await self.understanding_callback(video)

    async def extract_frames(self, video: bytes, fps: int = 1) -> List[bytes]:
        """Extrahiert Frames aus Video."""
        # Platzhalter
        return []

    async def add_audio(self, video: bytes, audio: bytes) -> bytes:
        """Fügt Audio zu Video hinzu."""
        # Platzhalter
        return video


# ============================================================================
# CODE PROCESSOR
# ============================================================================

class CodeProcessor(ModalityProcessor):
    """Prozessor für Code-Modalität."""

    @property
    def modality(self) -> ModalityType:
        return ModalityType.CODE

    def __init__(self, llm_callback: Optional[Callable] = None):
        self.llm_callback = llm_callback or self._default_llm

    async def _default_llm(self, prompt: str) -> str:
        return f"# Generated code\nprint('Hello')"

    async def understand(self, code: str) -> Understanding:
        """Analysiert und versteht Code."""
        prompt = f"""Analyze this code and explain:
1. What it does
2. Key functions/classes
3. Potential issues

Code:
```
{code[:3000]}
```

Analysis:"""

        analysis = await self.llm_callback(prompt)

        return Understanding(
            description=analysis,
            entities=self._extract_symbols(code),
            concepts=["code", "programming"],
            confidence=0.9,
        )

    def _extract_symbols(self, code: str) -> List[str]:
        """Extrahiert Symbole aus Code."""
        import re
        # Funktionen und Klassen
        functions = re.findall(r'def\s+(\w+)', code)
        classes = re.findall(r'class\s+(\w+)', code)
        return functions + classes

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generiert Code."""
        language = kwargs.get("language", "python")
        full_prompt = f"Write {language} code for the following:\n\n{prompt}\n\nCode:"
        return await self.llm_callback(full_prompt)

    async def review(self, code: str) -> Dict[str, Any]:
        """Überprüft Code."""
        prompt = f"""Review this code for:
1. Bugs
2. Security issues
3. Performance problems
4. Style issues

Code:
```
{code[:3000]}
```

Provide structured feedback:"""

        review = await self.llm_callback(prompt)

        return {
            "review": review,
            "has_issues": "issue" in review.lower() or "bug" in review.lower(),
        }

    async def fix(self, code: str, error: str) -> str:
        """Repariert Code basierend auf Fehler."""
        prompt = f"""Fix this code based on the error:

Code:
```
{code[:2000]}
```

Error:
{error}

Fixed code:"""

        return await self.llm_callback(prompt)

    async def translate(self, code: str, source_lang: str, target_lang: str) -> str:
        """Übersetzt Code zwischen Sprachen."""
        prompt = f"Translate this {source_lang} code to {target_lang}:\n\n```{source_lang}\n{code}\n```\n\n{target_lang} code:"
        return await self.llm_callback(prompt)

    async def explain(self, code: str) -> str:
        """Erklärt Code."""
        prompt = f"Explain what this code does in simple terms:\n\n```\n{code}\n```\n\nExplanation:"
        return await self.llm_callback(prompt)

    async def optimize(self, code: str) -> str:
        """Optimiert Code für Performance."""
        prompt = f"Optimize this code for better performance:\n\n```\n{code}\n```\n\nOptimized code:"
        return await self.llm_callback(prompt)


# ============================================================================
# 3D PROCESSOR
# ============================================================================

class ThreeDProcessor(ModalityProcessor):
    """Prozessor für 3D-Modalität."""

    @property
    def modality(self) -> ModalityType:
        return ModalityType.THREE_D

    def __init__(self, generation_callback: Optional[Callable] = None):
        self.generation_callback = generation_callback or self._default_generation

    async def _default_generation(self, prompt: str) -> bytes:
        return b''  # Empty 3D model

    async def understand(self, model: bytes) -> Understanding:
        """Analysiert 3D-Modell."""
        return Understanding(
            description="3D model analysis",
            entities=["mesh", "vertices", "faces"],
            concepts=["3d", "geometry"],
            confidence=0.7,
        )

    async def generate(self, prompt: str, **kwargs) -> bytes:
        """Generiert 3D-Modell."""
        return await self.generation_callback(prompt)

    async def from_image(self, image: bytes) -> bytes:
        """Erstellt 3D aus Bild."""
        # Platzhalter - würde Point-E oder ähnlich verwenden
        return b''

    async def from_images(self, images: List[bytes]) -> bytes:
        """Erstellt 3D aus mehreren Bildern."""
        return b''


# ============================================================================
# UNIFIED MULTIMODAL ENGINE
# ============================================================================

class UnifiedMultiModal:
    """
    Einheitliche MultiModal Engine.

    Bietet eine konsistente API für alle Modalitäten:
    - Text
    - Bild
    - Audio
    - Video
    - Code
    - 3D
    """

    def __init__(
        self,
        config: Optional[MultiModalConfig] = None,
        llm_callback: Optional[Callable] = None,
    ):
        self.config = config or MultiModalConfig()
        self.llm_callback = llm_callback

        # Initialize processors
        self.text = TextProcessor(llm_callback)
        self.image = ImageProcessor()
        self.audio = AudioProcessor()
        self.video = VideoProcessor()
        self.code = CodeProcessor(llm_callback)
        self.three_d = ThreeDProcessor()

        self._processors: Dict[ModalityType, ModalityProcessor] = {
            ModalityType.TEXT: self.text,
            ModalityType.IMAGE: self.image,
            ModalityType.AUDIO: self.audio,
            ModalityType.VIDEO: self.video,
            ModalityType.CODE: self.code,
            ModalityType.THREE_D: self.three_d,
        }

        logger.info("UnifiedMultiModal initialized")

    # ========================================================================
    # UNIFIED API
    # ========================================================================

    async def understand(
        self,
        content: Any,
        modality: ModalityType,
    ) -> Understanding:
        """
        Versteht Inhalt beliebiger Modalität.

        Args:
            content: Der Inhalt (str, bytes)
            modality: Typ der Modalität

        Returns:
            Understanding Objekt
        """
        processor = self._processors.get(modality)
        if not processor:
            raise ValueError(f"Unknown modality: {modality}")

        return await processor.understand(content)

    async def generate(
        self,
        prompt: str,
        output_modality: ModalityType,
        **kwargs,
    ) -> Any:
        """
        Generiert Inhalt in beliebiger Modalität.

        Args:
            prompt: Beschreibung was generiert werden soll
            output_modality: Gewünschte Ausgabe-Modalität
            **kwargs: Zusätzliche Parameter

        Returns:
            Generierter Inhalt
        """
        processor = self._processors.get(output_modality)
        if not processor:
            raise ValueError(f"Unknown modality: {output_modality}")

        return await processor.generate(prompt, **kwargs)

    async def transform(
        self,
        content: Any,
        input_modality: ModalityType,
        output_modality: ModalityType,
        **kwargs,
    ) -> MultiModalResult:
        """
        Transformiert Inhalt zwischen Modalitäten.

        Args:
            content: Eingabe-Inhalt
            input_modality: Eingabe-Modalität
            output_modality: Ausgabe-Modalität

        Returns:
            MultiModalResult
        """
        start_time = time.time()

        # Verstehe zuerst die Eingabe
        understanding = await self.understand(content, input_modality)

        # Generiere in Ziel-Modalität
        output = await self.generate(
            understanding.description,
            output_modality,
            **kwargs,
        )

        return MultiModalResult(
            operation=OperationType.TRANSFORM,
            input_modality=input_modality,
            output_modality=output_modality,
            content=output,
            confidence=understanding.confidence * 0.9,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    # ========================================================================
    # TEXT OPERATIONS
    # ========================================================================

    async def understand_text(self, text: str) -> Understanding:
        """Versteht Text."""
        return await self.text.understand(text)

    async def generate_text(self, prompt: str) -> str:
        """Generiert Text."""
        return await self.text.generate(prompt)

    async def translate(self, text: str, target_language: str) -> str:
        """Übersetzt Text."""
        return await self.text.translate(text, target_language)

    async def summarize(self, text: str) -> str:
        """Fasst Text zusammen."""
        return await self.text.summarize(text)

    # ========================================================================
    # IMAGE OPERATIONS
    # ========================================================================

    async def understand_image(self, image: bytes) -> Understanding:
        """Versteht Bild."""
        return await self.image.understand(image)

    async def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generiert Bild."""
        return await self.image.generate(prompt, **kwargs)

    async def edit_image(self, image: bytes, instruction: str) -> bytes:
        """Bearbeitet Bild."""
        return await self.image.edit(image, instruction)

    async def image_to_text(self, image: bytes) -> str:
        """Konvertiert Bild zu Text (Caption)."""
        return await self.image.caption(image)

    # ========================================================================
    # AUDIO OPERATIONS
    # ========================================================================

    async def understand_audio(self, audio: bytes) -> Understanding:
        """Versteht Audio."""
        return await self.audio.understand(audio)

    async def generate_speech(self, text: str, voice: str = "default") -> bytes:
        """Generiert Sprache."""
        return await self.audio.speak(text, voice)

    async def transcribe(self, audio: bytes) -> str:
        """Transkribiert Audio."""
        return await self.audio.transcribe(audio)

    async def generate_music(self, prompt: str, duration: float = 30.0) -> bytes:
        """Generiert Musik."""
        return await self.audio.generate_music(prompt, duration)

    # ========================================================================
    # VIDEO OPERATIONS
    # ========================================================================

    async def understand_video(self, video: bytes) -> Understanding:
        """Versteht Video."""
        return await self.video.understand(video)

    async def generate_video(self, prompt: str, duration: float = 5.0) -> bytes:
        """Generiert Video."""
        return await self.video.generate(prompt, duration=duration)

    async def edit_video(self, video: bytes, instruction: str) -> bytes:
        """Bearbeitet Video."""
        # Platzhalter
        return video

    # ========================================================================
    # CODE OPERATIONS
    # ========================================================================

    async def understand_code(self, code: str) -> Understanding:
        """Versteht Code."""
        return await self.code.understand(code)

    async def generate_code(self, spec: str, language: str = "python") -> str:
        """Generiert Code."""
        return await self.code.generate(spec, language=language)

    async def review_code(self, code: str) -> Dict[str, Any]:
        """Überprüft Code."""
        return await self.code.review(code)

    async def fix_code(self, code: str, error: str) -> str:
        """Repariert Code."""
        return await self.code.fix(code, error)

    # ========================================================================
    # 3D OPERATIONS
    # ========================================================================

    async def generate_3d(self, prompt: str) -> bytes:
        """Generiert 3D-Modell."""
        return await self.three_d.generate(prompt)

    async def image_to_3d(self, image: bytes) -> bytes:
        """Konvertiert Bild zu 3D."""
        return await self.three_d.from_image(image)

    # ========================================================================
    # MULTI-MODAL CHAINS
    # ========================================================================

    async def text_to_image_to_3d(self, text: str) -> bytes:
        """Kettenverarbeitung: Text -> Bild -> 3D."""
        image = await self.generate_image(text)
        return await self.image_to_3d(image)

    async def audio_to_text_to_image(self, audio: bytes) -> bytes:
        """Kettenverarbeitung: Audio -> Text -> Bild."""
        text = await self.transcribe(audio)
        return await self.generate_image(text)

    async def image_to_text_to_speech(self, image: bytes) -> bytes:
        """Kettenverarbeitung: Bild -> Text -> Sprache."""
        caption = await self.image_to_text(image)
        return await self.generate_speech(caption)

    # ========================================================================
    # UTILITY
    # ========================================================================

    def get_supported_modalities(self) -> List[ModalityType]:
        """Gibt unterstützte Modalitäten zurück."""
        return list(self._processors.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        return {
            "supported_modalities": [m.value for m in self._processors.keys()],
            "config": {
                "text_model": self.config.text_model,
                "image_model": self.config.image_model,
                "audio_model": self.config.audio_model,
                "video_model": self.config.video_model,
            },
        }


# ============================================================================
# SINGLETON & CONVENIENCE
# ============================================================================

_default_multimodal: Optional[UnifiedMultiModal] = None


def get_multimodal(config: Optional[MultiModalConfig] = None) -> UnifiedMultiModal:
    """Gibt eine Singleton-Instanz zurück."""
    global _default_multimodal
    if _default_multimodal is None:
        _default_multimodal = UnifiedMultiModal(config)
    return _default_multimodal
