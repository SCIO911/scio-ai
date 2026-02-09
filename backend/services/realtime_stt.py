#!/usr/bin/env python3
"""
SCIO - Real-Time Speech-to-Text Service (MEGA-UPGRADE)

WebSocket-basiertes Streaming STT.

Features:
- WebSocket Real-Time Streaming
- Voice Activity Detection (VAD)
- Speaker Diarization
- Noise Reduction Pre-Processing
- Multiple Language Support
- Live Transcription Feedback
"""

import asyncio
import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Generator
from pathlib import Path
import numpy as np

from backend.config import Config

logger = logging.getLogger(__name__)

# Optional Libraries
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

WEBRTCVAD_AVAILABLE = False
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    pass

NOISEREDUCE_AVAILABLE = False
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    pass

PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TranscriptionSegment:
    """Einzelnes Transkriptions-Segment."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    speaker: Optional[str] = None
    language: Optional[str] = None
    is_final: bool = True


@dataclass
class StreamingConfig:
    """Konfiguration für Streaming STT."""
    sample_rate: int = 16000
    chunk_duration_ms: int = 30  # VAD chunk size
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    language: str = "de"
    enable_noise_reduction: bool = True
    enable_diarization: bool = False
    model_id: str = "openai/whisper-large-v3"


class VoiceActivityDetector:
    """
    MEGA-UPGRADE: Voice Activity Detection (VAD)

    Erkennt Sprache vs. Stille im Audio-Stream.
    """

    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        if not WEBRTCVAD_AVAILABLE:
            logger.warning("webrtcvad nicht verfügbar - VAD deaktiviert")
            self._vad = None
            return

        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate
        self._frame_duration_ms = 30  # 10, 20, oder 30 ms

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Prüft ob Audio-Chunk Sprache enthält.

        Args:
            audio_chunk: PCM Audio (16-bit, mono)

        Returns:
            True wenn Sprache erkannt
        """
        if self._vad is None:
            return True  # Fallback: Immer Speech

        try:
            return self._vad.is_speech(audio_chunk, self._sample_rate)
        except Exception as e:
            logger.error(f"VAD Fehler: {e}")
            return True


class NoiseReducer:
    """
    MEGA-UPGRADE: Noise Reduction Pre-Processing

    Reduziert Hintergrundgeräusche vor der Transkription.
    """

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._noise_profile = None

    def reduce_noise(
        self,
        audio: np.ndarray,
        stationary: bool = True,
    ) -> np.ndarray:
        """
        Reduziert Rauschen im Audio.

        Args:
            audio: Audio als numpy array
            stationary: Stationäres Rauschen annehmen

        Returns:
            Bereinigtes Audio
        """
        if not NOISEREDUCE_AVAILABLE:
            return audio

        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=self._sample_rate,
                stationary=stationary,
                prop_decrease=0.8,
            )
            return reduced
        except Exception as e:
            logger.error(f"Noise Reduction Fehler: {e}")
            return audio


class SpeakerDiarizer:
    """
    MEGA-UPGRADE: Speaker Diarization

    Erkennt und unterscheidet verschiedene Sprecher.
    """

    def __init__(self, hf_token: str = None):
        self._pipeline = None
        self._hf_token = hf_token or Config.get("HF_TOKEN")

        if PYANNOTE_AVAILABLE and self._hf_token:
            try:
                self._pipeline = DiarizationPipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self._hf_token,
                )
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    self._pipeline = self._pipeline.to(torch.device("cuda"))
                logger.info("Speaker Diarization Pipeline geladen")
            except Exception as e:
                logger.warning(f"Diarization laden fehlgeschlagen: {e}")

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Führt Speaker Diarization durch.

        Args:
            audio_path: Pfad zur Audio-Datei

        Returns:
            Liste von Segmenten mit Speaker-IDs
        """
        if self._pipeline is None:
            return []

        try:
            diarization = self._pipeline(audio_path)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                })

            return segments

        except Exception as e:
            logger.error(f"Diarization Fehler: {e}")
            return []


class StreamingTranscriber:
    """
    MEGA-UPGRADE: Streaming Speech-to-Text

    Transkribiert Audio in Echtzeit mit minimaler Latenz.
    """

    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

        # Komponenten
        self._vad = VoiceActivityDetector(
            sample_rate=self.config.sample_rate,
            aggressiveness=self.config.vad_aggressiveness,
        )
        self._noise_reducer = NoiseReducer(sample_rate=self.config.sample_rate)
        self._diarizer = SpeakerDiarizer() if self.config.enable_diarization else None

        # Whisper Model
        self._model = None
        self._processor = None
        self._pipe = None

        # Streaming State
        self._audio_buffer = []
        self._is_speaking = False
        self._speech_start_time = 0.0
        self._callbacks: List[Callable] = []

    def initialize(self) -> bool:
        """Initialisiert das Modell."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers nicht installiert")
            return False

        try:
            logger.info(f"Lade STT Modell: {self.config.model_id}")

            # Verwende Pipeline für einfacheres Streaming
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.config.model_id,
                device=self._device,
                chunk_length_s=30,
                stride_length_s=5,
            )

            logger.info("STT Modell geladen")
            return True

        except Exception as e:
            logger.error(f"Modell laden fehlgeschlagen: {e}")
            return False

    def add_callback(self, callback: Callable[[TranscriptionSegment], None]):
        """Registriert Callback für neue Transkriptionen."""
        self._callbacks.append(callback)

    def _notify(self, segment: TranscriptionSegment):
        """Benachrichtigt alle Callbacks."""
        for cb in self._callbacks:
            try:
                cb(segment)
            except Exception as e:
                logger.error(f"Callback Fehler: {e}")

    def process_chunk(self, audio_chunk: bytes) -> Optional[TranscriptionSegment]:
        """
        Verarbeitet Audio-Chunk.

        Args:
            audio_chunk: PCM Audio (16-bit, mono, 16kHz)

        Returns:
            TranscriptionSegment wenn Segment abgeschlossen
        """
        # VAD Check
        is_speech = self._vad.is_speech(audio_chunk)

        # Konvertiere zu numpy
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        if is_speech:
            if not self._is_speaking:
                # Beginn der Sprache
                self._is_speaking = True
                self._speech_start_time = time.time()
                self._audio_buffer = []

            self._audio_buffer.append(audio_array)

        else:
            if self._is_speaking and len(self._audio_buffer) > 0:
                # Ende der Sprache - Transkribiere Buffer
                self._is_speaking = False

                combined_audio = np.concatenate(self._audio_buffer)
                self._audio_buffer = []

                # Noise Reduction
                if self.config.enable_noise_reduction:
                    combined_audio = self._noise_reducer.reduce_noise(combined_audio)

                # Transkribieren
                segment = self._transcribe(
                    combined_audio,
                    self._speech_start_time,
                )

                if segment:
                    self._notify(segment)
                    return segment

        return None

    def _transcribe(
        self,
        audio: np.ndarray,
        start_time: float,
    ) -> Optional[TranscriptionSegment]:
        """Transkribiert Audio-Segment."""
        if self._pipe is None:
            return None

        try:
            result = self._pipe(
                audio,
                generate_kwargs={"language": self.config.language},
            )

            text = result.get("text", "").strip()

            if not text:
                return None

            return TranscriptionSegment(
                text=text,
                start_time=start_time,
                end_time=time.time(),
                language=self.config.language,
                is_final=True,
            )

        except Exception as e:
            logger.error(f"Transkription Fehler: {e}")
            return None

    def transcribe_file(
        self,
        audio_path: str,
        with_diarization: bool = False,
    ) -> List[TranscriptionSegment]:
        """
        Transkribiert Audio-Datei.

        Args:
            audio_path: Pfad zur Audio-Datei
            with_diarization: Speaker Diarization aktivieren

        Returns:
            Liste von Transkriptions-Segmenten
        """
        if self._pipe is None:
            if not self.initialize():
                return []

        try:
            result = self._pipe(
                audio_path,
                return_timestamps=True,
                generate_kwargs={"language": self.config.language},
            )

            segments = []

            if "chunks" in result:
                for chunk in result["chunks"]:
                    segments.append(TranscriptionSegment(
                        text=chunk["text"],
                        start_time=chunk["timestamp"][0] or 0.0,
                        end_time=chunk["timestamp"][1] or 0.0,
                        language=self.config.language,
                    ))
            else:
                segments.append(TranscriptionSegment(
                    text=result.get("text", ""),
                    start_time=0.0,
                    end_time=0.0,
                    language=self.config.language,
                ))

            # Diarization
            if with_diarization and self._diarizer:
                diarization = self._diarizer.diarize(audio_path)
                segments = self._merge_diarization(segments, diarization)

            return segments

        except Exception as e:
            logger.error(f"Datei-Transkription Fehler: {e}")
            return []

    def _merge_diarization(
        self,
        transcription: List[TranscriptionSegment],
        diarization: List[Dict[str, Any]],
    ) -> List[TranscriptionSegment]:
        """Merged Transkription mit Diarization."""
        for segment in transcription:
            # Finde überlappenden Speaker
            for diar in diarization:
                if (segment.start_time >= diar['start'] and
                    segment.start_time < diar['end']):
                    segment.speaker = diar['speaker']
                    break

        return transcription

    def cleanup(self):
        """Gibt Ressourcen frei."""
        self._model = None
        self._processor = None
        self._pipe = None
        self._audio_buffer = []

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Streaming Transcriber bereinigt")


class WebSocketSTTHandler:
    """
    MEGA-UPGRADE: WebSocket Handler für Real-Time STT

    Verarbeitet WebSocket-Verbindungen für Streaming-Transkription.
    """

    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self._sessions: Dict[str, StreamingTranscriber] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> StreamingTranscriber:
        """Erstellt neue Streaming-Session."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            transcriber = StreamingTranscriber(self.config)
            transcriber.initialize()
            self._sessions[session_id] = transcriber

            logger.info(f"STT Session erstellt: {session_id}")
            return transcriber

    def process_audio(
        self,
        session_id: str,
        audio_data: bytes,
    ) -> Optional[TranscriptionSegment]:
        """
        Verarbeitet Audio für eine Session.

        Args:
            session_id: Session ID
            audio_data: Audio-Daten (PCM 16-bit mono 16kHz)

        Returns:
            TranscriptionSegment wenn verfügbar
        """
        with self._lock:
            if session_id not in self._sessions:
                self.create_session(session_id)

            transcriber = self._sessions[session_id]

        return transcriber.process_chunk(audio_data)

    def close_session(self, session_id: str):
        """Schließt eine Session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].cleanup()
                del self._sessions[session_id]
                logger.info(f"STT Session geschlossen: {session_id}")

    def get_active_sessions(self) -> List[str]:
        """Gibt aktive Session-IDs zurück."""
        with self._lock:
            return list(self._sessions.keys())


# Singleton Instance
_websocket_handler: Optional[WebSocketSTTHandler] = None


def get_websocket_stt_handler() -> WebSocketSTTHandler:
    """Gibt Singleton-Instanz zurück."""
    global _websocket_handler
    if _websocket_handler is None:
        _websocket_handler = WebSocketSTTHandler()
    return _websocket_handler
