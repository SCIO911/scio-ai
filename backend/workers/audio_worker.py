#!/usr/bin/env python3
"""
SCIO - Audio Worker
Text-to-Speech (TTS), Speech-to-Text (STT), Music Generation
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    TORCH_AVAILABLE = False

# Whisper (Speech-to-Text)
WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    try:
        from faster_whisper import WhisperModel
        WHISPER_AVAILABLE = True
    except ImportError:
        pass

# TTS Libraries
TTS_AVAILABLE = False
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    pass

BARK_AVAILABLE = False
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    BARK_AVAILABLE = True
except ImportError:
    pass

# Music Generation
MUSICGEN_AVAILABLE = False
try:
    from audiocraft.models import MusicGen
    MUSICGEN_AVAILABLE = True
except ImportError:
    pass

# Audio I/O
try:
    import soundfile as sf
    import numpy as np
    AUDIO_IO_AVAILABLE = True
except ImportError:
    AUDIO_IO_AVAILABLE = False


# Available Audio Models
AUDIO_MODELS = {
    # Speech-to-Text
    'whisper-tiny': {'name': 'Whisper Tiny', 'type': 'stt', 'vram_gb': 1},
    'whisper-base': {'name': 'Whisper Base', 'type': 'stt', 'vram_gb': 1},
    'whisper-small': {'name': 'Whisper Small', 'type': 'stt', 'vram_gb': 2},
    'whisper-medium': {'name': 'Whisper Medium', 'type': 'stt', 'vram_gb': 5},
    'whisper-large': {'name': 'Whisper Large V3', 'type': 'stt', 'vram_gb': 10},

    # Text-to-Speech
    'bark': {'name': 'Bark TTS', 'type': 'tts', 'vram_gb': 12},
    'xtts-v2': {'name': 'XTTS v2', 'type': 'tts', 'vram_gb': 6},
    'speecht5': {'name': 'SpeechT5', 'type': 'tts', 'vram_gb': 2},

    # Music Generation
    'musicgen-small': {'name': 'MusicGen Small', 'type': 'music', 'vram_gb': 4},
    'musicgen-medium': {'name': 'MusicGen Medium', 'type': 'music', 'vram_gb': 8},
    'musicgen-large': {'name': 'MusicGen Large', 'type': 'music', 'vram_gb': 16},
}


class AudioWorker(BaseWorker):
    """
    Audio Worker - Handles all audio AI tasks

    Features:
    - Speech-to-Text (Whisper)
    - Text-to-Speech (Bark, XTTS)
    - Music Generation (MusicGen)
    - Audio Processing
    """

    def __init__(self):
        super().__init__("Audio Processing")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._whisper_model = None
        self._tts_model = None
        self._music_model = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        available_features = []

        if WHISPER_AVAILABLE:
            available_features.append("STT")
        if TTS_AVAILABLE or BARK_AVAILABLE:
            available_features.append("TTS")
        if MUSICGEN_AVAILABLE:
            available_features.append("Music")

        if not available_features:
            self._error_message = "No audio libraries available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] Audio Worker bereit (Device: {self._device}, Features: {', '.join(available_features)})")
        return True

    def _load_whisper(self, model_size: str = "large"):
        """Load Whisper model for STT"""
        model_key = f"whisper_{model_size}"

        def loader():
            print(f"[LOAD] Lade Whisper {model_size}...")
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel(
                    model_size,
                    device=self._device,
                    compute_type="float16" if self._device == "cuda" else "int8"
                )
            except ImportError:
                import whisper
                model = whisper.load_model(model_size, device=self._device)
            return model

        self._whisper_model = model_manager.get_model(model_key, loader)
        print(f"[OK] Whisper {model_size} geladen")

    def _load_tts(self, model_id: str = "xtts-v2"):
        """Load TTS model"""
        def loader():
            print(f"[LOAD] Lade TTS Modell: {model_id}...")

            if model_id == "bark" and BARK_AVAILABLE:
                preload_models()
                return {"type": "bark"}
            elif TTS_AVAILABLE:
                if model_id == "xtts-v2":
                    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                else:
                    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
                tts.to(self._device)
                return {"type": "coqui", "model": tts}

            raise ValueError(f"TTS model {model_id} not available")

        self._tts_model = model_manager.get_model(f"tts_{model_id}", loader)
        self._current_model_id = model_id
        print(f"[OK] TTS {model_id} geladen")

    def _load_musicgen(self, model_size: str = "medium"):
        """Load MusicGen model"""
        def loader():
            print(f"[LOAD] Lade MusicGen {model_size}...")
            model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")
            model.set_generation_params(duration=30)
            return model

        self._music_model = model_manager.get_model(f"musicgen_{model_size}", loader)
        print(f"[OK] MusicGen {model_size} geladen")

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'de', 'fr')

        Returns:
            dict with text, language, duration, gpu_seconds

        Raises:
            ValueError: If audio file not found or invalid
        """
        # Validate input
        if not audio_path:
            raise ValueError("Audio path is required")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        # Validate file extension
        valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
        if audio_file.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid audio format: {audio_file.suffix}. Supported: {valid_extensions}")

        if not self._whisper_model:
            self._load_whisper("large")

        start_time = time.time()

        try:
            # Faster Whisper
            segments, info = self._whisper_model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join([segment.text for segment in segments])

            return {
                "text": text,
                "language": info.language,
                "duration": info.duration,
                "gpu_seconds": time.time() - start_time,
            }
        except AttributeError:
            # Original Whisper (different API)
            result = self._whisper_model.transcribe(audio_path, language=language)
            return {
                "text": result["text"],
                "language": result.get("language"),
                "gpu_seconds": time.time() - start_time,
            }

    def text_to_speech(
        self,
        text: str,
        output_path: str = None,
        voice: str = None,
        language: str = "en",
        model: str = "xtts-v2",
    ) -> dict:
        """
        Convert text to speech.

        Args:
            text: Text to convert (max 10000 characters)
            output_path: Optional output file path
            voice: Optional voice/speaker WAV file for cloning
            language: Language code (en, de, fr, es, etc.)
            model: TTS model (xtts-v2, bark)

        Returns:
            dict with output_path, text, model, gpu_seconds

        Raises:
            ValueError: If text is empty or too long
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text is required for TTS")

        text = text.strip()
        if len(text) > 10000:
            raise ValueError("Text too long (max 10000 characters)")

        # Validate model
        valid_models = ['xtts-v2', 'bark', 'speecht5']
        if model not in valid_models:
            raise ValueError(f"Unknown TTS model: {model}. Available: {valid_models}")

        # Validate voice file if provided
        if voice and isinstance(voice, str):
            if not Path(voice).exists():
                raise ValueError(f"Voice file not found: {voice}")

        if not self._tts_model or self._current_model_id != model:
            self._load_tts(model)

        start_time = time.time()

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"tts_{uuid.uuid4().hex[:8]}.wav")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self._tts_model.get("type") == "bark":
            audio_array = generate_audio(text)
            sf.write(output_path, audio_array, SAMPLE_RATE)
        else:
            tts = self._tts_model["model"]
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language,
                speaker_wav=voice,
            )

        return {
            "output_path": output_path,
            "text": text,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        output_path: str = None,
        model_size: str = "medium",
    ) -> dict:
        """
        Generate music from text prompt.

        Args:
            prompt: Text description of the music (max 1000 characters)
            duration: Duration in seconds (5-300)
            output_path: Optional output file path
            model_size: Model size (small, medium, large)

        Returns:
            dict with output_path, prompt, duration, model, gpu_seconds

        Raises:
            ValueError: If prompt is empty or parameters are invalid
        """
        if not MUSICGEN_AVAILABLE:
            raise ValueError("MusicGen not installed. Install with: pip install audiocraft")

        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required for music generation")

        prompt = prompt.strip()
        if len(prompt) > 1000:
            raise ValueError("Prompt too long (max 1000 characters)")

        # Validate duration
        duration = max(5, min(300, duration))

        # Validate model size
        valid_sizes = ['small', 'medium', 'large']
        if model_size not in valid_sizes:
            raise ValueError(f"Unknown model size: {model_size}. Available: {valid_sizes}")

        if not self._music_model:
            self._load_musicgen(model_size)

        start_time = time.time()

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"music_{uuid.uuid4().hex[:8]}.wav")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self._music_model.set_generation_params(duration=duration)
        wav = self._music_model.generate([prompt])

        # Save audio
        audio_data = wav[0].cpu().numpy()
        sf.write(output_path, audio_data.T, 32000)

        return {
            "output_path": output_path,
            "prompt": prompt,
            "duration": duration,
            "model": f"musicgen-{model_size}",
            "gpu_seconds": time.time() - start_time,
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process an audio job"""
        task_type = input_data.get("task", "transcribe")

        self.notify_progress(job_id, 0.1, f"Starting {task_type}")

        if task_type == "transcribe" or task_type == "stt":
            audio_path = input_data.get("audio_path") or input_data.get("file")
            language = input_data.get("language")
            result = self.transcribe(audio_path, language)

        elif task_type == "tts" or task_type == "text_to_speech":
            text = input_data.get("text")
            voice = input_data.get("voice")
            language = input_data.get("language", "en")
            model = input_data.get("model", "xtts-v2")
            result = self.text_to_speech(text, voice=voice, language=language, model=model)

        elif task_type == "music" or task_type == "generate_music":
            prompt = input_data.get("prompt")
            duration = input_data.get("duration", 30)
            model_size = input_data.get("model", "medium")
            result = self.generate_music(prompt, duration, model_size=model_size)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._whisper_model = None
        self._tts_model = None
        self._music_model = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Audio Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        available = {}
        for model_id, info in AUDIO_MODELS.items():
            if info["type"] == "stt" and WHISPER_AVAILABLE:
                available[model_id] = info
            elif info["type"] == "tts" and (TTS_AVAILABLE or BARK_AVAILABLE):
                available[model_id] = info
            elif info["type"] == "music" and MUSICGEN_AVAILABLE:
                available[model_id] = info
        return available


# Singleton Instance
_audio_worker: Optional[AudioWorker] = None


def get_audio_worker() -> AudioWorker:
    """Get singleton instance"""
    global _audio_worker
    if _audio_worker is None:
        _audio_worker = AudioWorker()
    return _audio_worker
