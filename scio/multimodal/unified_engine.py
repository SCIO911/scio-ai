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
        """
        Extrahiert Named Entities aus Text.

        Nutzt regelbasierte Erkennung für Eigennamen, Organisationen,
        Datumsangaben, Geldbeträge und URLs.
        """
        import re

        entities = set()

        # 1. Eigennamen (nicht am Satzanfang)
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if i == 0:
                    continue
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                    entities.add(clean_word)

        # 2. Multi-Wort Eigennamen
        multi_word_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        entities.update(re.findall(multi_word_pattern, text))

        # 3. Organisationen
        org_pattern = r'\b([A-Z][a-zA-Z]*(?:\s+(?:Inc|Corp|Ltd|GmbH|AG|LLC|Company|University|Institute))\b\.?)'
        entities.update(re.findall(org_pattern, text))

        # 4. Datumsangaben
        date_patterns = [
            r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        ]
        for pattern in date_patterns:
            entities.update(re.findall(pattern, text, re.IGNORECASE))

        # 5. Geldbeträge
        entities.update(re.findall(r'[\$€£]\s?\d+(?:[.,]\d+)?(?:\s?(?:million|billion))?', text, re.IGNORECASE))

        # 6. E-Mails und URLs
        entities.update(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        entities.update(re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text))

        # Filtere Stopwörter
        stopwords = {'The', 'This', 'That', 'These', 'Der', 'Die', 'Das'}
        return [e for e in entities if len(str(e)) > 2 and e not in stopwords][:20]

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extrahiert Schlüsselkonzepte mit TF-basierter Gewichtung.
        """
        import re

        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'mit', 'von'
        }

        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]{4,}\b', text.lower())
        words = [w for w in words if w not in stopwords]

        # Term-Frequenz
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1

        # Bigrams
        bigram_tf = {}
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigram_tf[bigram] = bigram_tf.get(bigram, 0) + 1

        concepts = [w for w, _ in sorted(tf.items(), key=lambda x: x[1], reverse=True)[:12]]
        concepts.extend([b for b, c in sorted(bigram_tf.items(), key=lambda x: x[1], reverse=True)[:3] if c >= 2])

        return concepts[:15]

    def _analyze_sentiment(self, text: str) -> str:
        """
        Analysiert Sentiment mit erweiterten Lexika und Negationserkennung.
        """
        import re

        positive = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'superb', 'perfect', 'happy', 'love',
            'success', 'positive', 'better', 'helpful', 'useful',
            'gut', 'toll', 'super', 'wunderbar', 'perfekt', 'glücklich', 'erfolg'
        }
        negative = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'wrong',
            'sad', 'hate', 'angry', 'disappointed', 'fail', 'problem', 'error',
            'schlecht', 'schrecklich', 'falsch', 'traurig', 'fehler', 'problem'
        }
        negations = {'not', 'no', "n't", 'never', 'nicht', 'kein', 'keine'}

        words = re.findall(r'\b\w+\b', text.lower())
        pos_score, neg_score = 0, 0
        negated = False

        for i, word in enumerate(words):
            if word in negations:
                negated = True
                continue

            if negated and i > 0 and i >= 3:
                negated = False

            if word in positive:
                neg_score += 1 if negated else 0
                pos_score += 0 if negated else 1
            elif word in negative:
                pos_score += 1 if negated else 0
                neg_score += 0 if negated else 1

        if pos_score > neg_score * 1.2:
            return "positive"
        elif neg_score > pos_score * 1.2:
            return "negative"
        elif pos_score > 0 and neg_score > 0:
            return "mixed"
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
        """
        Standard Generation Callback - Generiert einfaches Bild.

        Erstellt ein generatives Bild basierend auf dem Prompt.
        Unterstützt width, height, color_mode Parameter.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            import hashlib

            width = kwargs.get('width', 512)
            height = kwargs.get('height', 512)

            # Hash des Prompts für konsistente Farben
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

            # Farben aus Hash generieren
            r = int(prompt_hash[:2], 16)
            g = int(prompt_hash[2:4], 16)
            b = int(prompt_hash[4:6], 16)

            # Gradient-Bild erstellen
            img = Image.new('RGB', (width, height), (r, g, b))
            draw = ImageDraw.Draw(img)

            # Gradient-Effekt
            for y in range(height):
                factor = y / height
                line_r = int(r * (1 - factor) + 128 * factor)
                line_g = int(g * (1 - factor) + 128 * factor)
                line_b = int(b * (1 - factor) + 200 * factor)
                draw.line([(0, y), (width, y)], fill=(line_r, line_g, line_b))

            # Text hinzufügen
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            # Prompt-Text zentriert
            text = prompt[:50] + "..." if len(prompt) > 50 else prompt
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                # Schatten
                draw.text((x+1, y+1), text, fill=(0, 0, 0), font=font)
                draw.text((x, y), text, fill=(255, 255, 255), font=font)

            # Zu PNG Bytes
            output = io.BytesIO()
            img.save(output, format='PNG')
            return output.getvalue()

        except ImportError:
            # Fallback: Minimal gültiges PNG
            return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82'

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
        """
        Bearbeitet ein Bild nach Anweisung.

        Unterstützte Instruktionen:
        - "rotate:90" - Rotation um Grad
        - "resize:800x600" - Größenänderung
        - "grayscale" - Graustufen
        - "blur:5" - Weichzeichnen
        - "sharpen" - Schärfen
        - "contrast:1.5" - Kontrast anpassen
        - "brightness:1.2" - Helligkeit anpassen
        - "flip:horizontal/vertical" - Spiegeln
        - "crop:x,y,w,h" - Zuschneiden
        """
        try:
            from PIL import Image, ImageFilter, ImageEnhance
            import io

            # Bild laden
            img = Image.open(io.BytesIO(image))

            # Instruktion parsen
            instruction = instruction.lower().strip()

            if instruction.startswith("rotate:"):
                angle = int(instruction.split(":")[1])
                img = img.rotate(angle, expand=True)

            elif instruction.startswith("resize:"):
                size_str = instruction.split(":")[1]
                w, h = map(int, size_str.split("x"))
                img = img.resize((w, h), Image.Resampling.LANCZOS)

            elif instruction == "grayscale":
                img = img.convert("L").convert("RGB")

            elif instruction.startswith("blur:"):
                radius = float(instruction.split(":")[1])
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))

            elif instruction == "sharpen":
                img = img.filter(ImageFilter.SHARPEN)

            elif instruction.startswith("contrast:"):
                factor = float(instruction.split(":")[1])
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)

            elif instruction.startswith("brightness:"):
                factor = float(instruction.split(":")[1])
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)

            elif instruction.startswith("flip:"):
                direction = instruction.split(":")[1]
                if direction == "horizontal":
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                elif direction == "vertical":
                    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            elif instruction.startswith("crop:"):
                coords = list(map(int, instruction.split(":")[1].split(",")))
                if len(coords) == 4:
                    x, y, w, h = coords
                    img = img.crop((x, y, x + w, y + h))

            # Bild zurück in Bytes
            output = io.BytesIO()
            img.save(output, format="PNG")
            return output.getvalue()

        except ImportError:
            logger.warning("PIL nicht installiert - Bild unverändert")
            return image
        except Exception as e:
            logger.error(f"Bildbearbeitung fehlgeschlagen: {e}")
            return image

    async def variation(self, image: bytes, count: int = 1) -> List[bytes]:
        """
        Erstellt Variationen eines Bildes.

        Wendet zufällige Transformationen an:
        - Leichte Farbänderungen
        - Kleine Rotationen
        - Leichte Größenänderungen
        """
        try:
            from PIL import Image, ImageEnhance
            import io
            import random

            variations = []
            img = Image.open(io.BytesIO(image))

            for i in range(count):
                # Kopie erstellen
                variant = img.copy()

                # Zufällige Transformationen
                # Leichte Rotation (-5 bis +5 Grad)
                angle = random.uniform(-5, 5)
                variant = variant.rotate(angle, expand=False, fillcolor=(255, 255, 255))

                # Leichte Farbänderung
                color_factor = random.uniform(0.9, 1.1)
                enhancer = ImageEnhance.Color(variant)
                variant = enhancer.enhance(color_factor)

                # Leichte Kontraständerung
                contrast_factor = random.uniform(0.95, 1.05)
                enhancer = ImageEnhance.Contrast(variant)
                variant = enhancer.enhance(contrast_factor)

                # Leichte Helligkeitsänderung
                brightness_factor = random.uniform(0.95, 1.05)
                enhancer = ImageEnhance.Brightness(variant)
                variant = enhancer.enhance(brightness_factor)

                # Zu Bytes konvertieren
                output = io.BytesIO()
                variant.save(output, format="PNG")
                variations.append(output.getvalue())

            return variations

        except ImportError:
            logger.warning("PIL nicht installiert - Original zurückgegeben")
            return [image] * count
        except Exception as e:
            logger.error(f"Variation fehlgeschlagen: {e}")
            return [image] * count


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
        """
        Extrahiert Frames aus Video.

        Args:
            video: Video als Bytes
            fps: Frames pro Sekunde zu extrahieren

        Returns:
            Liste von Frame-Bildern als PNG-Bytes
        """
        try:
            import cv2
            import numpy as np
            import tempfile
            import os
            from PIL import Image
            import io

            # Video temporär speichern
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(video)
                temp_path = f.name

            frames = []
            cap = cv2.VideoCapture(temp_path)

            try:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = max(1, int(video_fps / fps))
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        # BGR zu RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)

                        # Zu PNG-Bytes
                        output = io.BytesIO()
                        img.save(output, format='PNG')
                        frames.append(output.getvalue())

                    frame_count += 1

            finally:
                cap.release()
                os.unlink(temp_path)

            logger.info(f"Extrahierte {len(frames)} Frames aus Video")
            return frames

        except ImportError:
            logger.warning("opencv-python nicht installiert - keine Frame-Extraktion möglich")
            return []
        except Exception as e:
            logger.error(f"Frame-Extraktion fehlgeschlagen: {e}")
            return []

    async def add_audio(self, video: bytes, audio: bytes) -> bytes:
        """
        Fügt Audio zu Video hinzu.

        Args:
            video: Video als Bytes
            audio: Audio als Bytes (WAV/MP3)

        Returns:
            Video mit Audio als Bytes
        """
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            import tempfile
            import os

            # Temporäre Dateien erstellen
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf:
                vf.write(video)
                video_path = vf.name

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as af:
                af.write(audio)
                audio_path = af.name

            output_path = tempfile.mktemp(suffix='.mp4')

            try:
                # Video und Audio laden
                video_clip = VideoFileClip(video_path)
                audio_clip = AudioFileClip(audio_path)

                # Audio auf Video-Länge anpassen
                if audio_clip.duration > video_clip.duration:
                    audio_clip = audio_clip.subclip(0, video_clip.duration)

                # Audio hinzufügen
                final_clip = video_clip.set_audio(audio_clip)

                # Speichern
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    logger=None
                )

                # Cleanup
                video_clip.close()
                audio_clip.close()
                final_clip.close()

                # Ergebnis lesen
                with open(output_path, 'rb') as f:
                    result = f.read()

                return result

            finally:
                # Temporäre Dateien löschen
                for path in [video_path, audio_path, output_path]:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except Exception:
                            pass

        except ImportError:
            logger.warning("moviepy nicht installiert - Video unverändert")
            return video
        except Exception as e:
            logger.error(f"Audio-Hinzufügung fehlgeschlagen: {e}")
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
        """
        Analysiert 3D-Modell.

        Unterstützte Formate: OBJ, STL, PLY, GLB, GLTF
        """
        try:
            import trimesh
            import tempfile
            import os

            # Temporär speichern
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
                f.write(model)
                temp_path = f.name

            try:
                mesh = trimesh.load(temp_path)

                # Mesh-Statistiken
                if hasattr(mesh, 'vertices'):
                    n_vertices = len(mesh.vertices)
                    n_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0
                    bounds = mesh.bounds.tolist() if hasattr(mesh, 'bounds') else []
                    is_watertight = mesh.is_watertight if hasattr(mesh, 'is_watertight') else False
                    volume = float(mesh.volume) if hasattr(mesh, 'volume') and mesh.is_watertight else 0

                    description = f"3D Mesh mit {n_vertices} Vertices und {n_faces} Faces. "
                    if is_watertight:
                        description += f"Wasserdicht mit Volumen {volume:.2f}. "
                    description += f"Bounding Box: {bounds}"

                    return Understanding(
                        description=description,
                        entities=[
                            f"vertices:{n_vertices}",
                            f"faces:{n_faces}",
                            "watertight" if is_watertight else "open_mesh"
                        ],
                        concepts=["3d", "geometry", "mesh"],
                        confidence=0.9,
                        metadata={
                            "vertices": n_vertices,
                            "faces": n_faces,
                            "bounds": bounds,
                            "volume": volume,
                            "is_watertight": is_watertight
                        }
                    )

            finally:
                os.unlink(temp_path)

        except ImportError:
            logger.warning("trimesh nicht installiert")
        except Exception as e:
            logger.error(f"3D-Analyse fehlgeschlagen: {e}")

        return Understanding(
            description="3D model (Analyse nicht möglich)",
            entities=["mesh"],
            concepts=["3d", "geometry"],
            confidence=0.5,
        )

    async def generate(self, prompt: str, **kwargs) -> bytes:
        """Generiert 3D-Modell."""
        return await self.generation_callback(prompt)

    async def from_image(self, image: bytes) -> bytes:
        """
        Erstellt 3D aus Bild mittels Tiefenschätzung.

        Generiert ein einfaches Relief-Mesh basierend auf der Tiefenkarte.
        """
        try:
            from PIL import Image
            import numpy as np
            import trimesh
            import io

            # Bild laden
            img = Image.open(io.BytesIO(image))
            img_gray = img.convert('L')
            width, height = img_gray.size

            # Tiefenkarte aus Grauwerten
            depth = np.array(img_gray, dtype=np.float32) / 255.0

            # Downsampling für Performance
            scale = max(1, max(width, height) // 200)
            if scale > 1:
                new_w, new_h = width // scale, height // scale
                depth = np.array(img_gray.resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                width, height = new_w, new_h

            # Vertices erstellen
            vertices = []
            for y in range(height):
                for x in range(width):
                    z = depth[y, x] * 10  # Skalierung für sichtbare Tiefe
                    vertices.append([x, height - y, z])

            vertices = np.array(vertices)

            # Faces erstellen (Quad-Mesh als Triangles)
            faces = []
            for y in range(height - 1):
                for x in range(width - 1):
                    i = y * width + x
                    # Zwei Triangles pro Quad
                    faces.append([i, i + width, i + 1])
                    faces.append([i + 1, i + width, i + width + 1])

            faces = np.array(faces)

            # Mesh erstellen
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Als OBJ exportieren
            obj_data = mesh.export(file_type='obj')
            return obj_data.encode() if isinstance(obj_data, str) else obj_data

        except ImportError:
            logger.warning("trimesh oder PIL nicht installiert")
            return b''
        except Exception as e:
            logger.error(f"3D aus Bild fehlgeschlagen: {e}")
            return b''

    async def from_images(self, images: List[bytes]) -> bytes:
        """
        Erstellt 3D aus mehreren Bildern (Multi-View Reconstruction).

        Bei nur einem Bild wird from_image verwendet.
        Bei mehreren Bildern wird versucht, ein einfaches
        kombiniertes Mesh zu erstellen.
        """
        if not images:
            return b''

        if len(images) == 1:
            return await self.from_image(images[0])

        try:
            import numpy as np
            import trimesh

            # Für jeden View ein Relief erstellen und kombinieren
            meshes = []
            for i, img_bytes in enumerate(images[:6]):  # Max 6 Views
                mesh_bytes = await self.from_image(img_bytes)
                if mesh_bytes:
                    try:
                        import tempfile
                        import os

                        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
                            f.write(mesh_bytes)
                            temp_path = f.name

                        mesh = trimesh.load(temp_path)
                        os.unlink(temp_path)

                        # Mesh um verschiedene Achsen rotieren basierend auf Index
                        if i == 1:  # Rechte Seite
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
                        elif i == 2:  # Hinten
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
                        elif i == 3:  # Linke Seite
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))
                        elif i == 4:  # Oben
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
                        elif i == 5:  # Unten
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))

                        meshes.append(mesh)
                    except Exception:
                        pass

            if not meshes:
                return b''

            # Meshes kombinieren
            combined = trimesh.util.concatenate(meshes)
            obj_data = combined.export(file_type='obj')
            return obj_data.encode() if isinstance(obj_data, str) else obj_data

        except ImportError:
            logger.warning("trimesh nicht installiert")
            return b''
        except Exception as e:
            logger.error(f"Multi-View 3D fehlgeschlagen: {e}")
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
        """
        Bearbeitet Video nach Anweisung.

        Unterstützte Instruktionen:
        - "trim:start,end" - Zuschneiden (Sekunden)
        - "speed:factor" - Geschwindigkeit (1.5 = 50% schneller)
        - "reverse" - Rückwärts abspielen
        - "grayscale" - Schwarz-Weiß
        - "resize:width,height" - Größe ändern
        - "rotate:90" - Rotation (Grad, 90/180/270)
        - "fadein:seconds" - Einblenden
        - "fadeout:seconds" - Ausblenden
        - "mirror" - Horizontal spiegeln
        """
        try:
            from moviepy.editor import VideoFileClip
            import tempfile
            import os

            # Video temporär speichern
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(video)
                video_path = f.name

            output_path = tempfile.mktemp(suffix='.mp4')

            try:
                clip = VideoFileClip(video_path)

                # Instruktion parsen
                instruction = instruction.lower().strip()

                if instruction.startswith("trim:"):
                    times = instruction.split(":")[1].split(",")
                    start = float(times[0])
                    end = float(times[1]) if len(times) > 1 else clip.duration
                    clip = clip.subclip(start, min(end, clip.duration))

                elif instruction.startswith("speed:"):
                    factor = float(instruction.split(":")[1])
                    clip = clip.speedx(factor)

                elif instruction == "reverse":
                    clip = clip.fx(lambda c: c.fl_time(lambda t: clip.duration - t, apply_to=['video', 'audio']))

                elif instruction == "grayscale":
                    clip = clip.fx(lambda c: c.fl_image(lambda frame: frame.mean(axis=2, keepdims=True).repeat(3, axis=2).astype('uint8')))

                elif instruction.startswith("resize:"):
                    size_str = instruction.split(":")[1]
                    w, h = map(int, size_str.split(","))
                    clip = clip.resize((w, h))

                elif instruction.startswith("rotate:"):
                    angle = int(instruction.split(":")[1])
                    clip = clip.rotate(angle)

                elif instruction.startswith("fadein:"):
                    duration = float(instruction.split(":")[1])
                    clip = clip.fadein(duration)

                elif instruction.startswith("fadeout:"):
                    duration = float(instruction.split(":")[1])
                    clip = clip.fadeout(duration)

                elif instruction == "mirror":
                    clip = clip.fx(lambda c: c.fl_image(lambda frame: frame[:, ::-1]))

                # Speichern
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    logger=None
                )

                clip.close()

                # Ergebnis lesen
                with open(output_path, 'rb') as f:
                    result = f.read()

                return result

            finally:
                for path in [video_path, output_path]:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except Exception:
                            pass

        except ImportError:
            logger.warning("moviepy nicht installiert - Video unverändert")
            return video
        except Exception as e:
            logger.error(f"Video-Bearbeitung fehlgeschlagen: {e}")
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
