#!/usr/bin/env python3
"""
SCIO - Audio Worker
Die BESTEN Audio-Modelle 2024/2025
Optimiert für RTX 5090 mit 24GB VRAM

Unterstützte Features:

Speech-to-Text (STT):
- Whisper Large V3 Turbo - Schnellste Version
- Whisper Large V3 - Beste Genauigkeit
- Faster-Whisper - Optimiert für Geschwindigkeit
- Distil-Whisper - Kompakt und schnell

Text-to-Speech (TTS):
- XTTS v2 - Bestes Open Source TTS mit Voice Cloning
- F5-TTS - Neuestes hochqualitatives TTS
- Parler-TTS - Natürlich klingend
- Bark - Expressiv mit Sound Effects
- Fish-Speech - Exzellentes Voice Cloning
- Kokoro-TTS - Sehr natürlich
- StyleTTS 2 - Style-Kontrolle
- MeloTTS - Schnell multilingual

Music Generation:
- MusicGen Large/Melody - Meta's beste Musikgenerierung
- Stable Audio Open - Stability AI
- AudioCraft Suite - Meta's komplette Audio-Suite

Voice Cloning:
- OpenVoice v2 - Echtzeit Voice Cloning
- RVC v2 - Retrieval Voice Conversion

Audio Processing:
- Demucs v4 - Audio Source Separation
- Noise Reduction
"""

import os
import time
import uuid
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from io import BytesIO

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# PyTorch
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    TORCH_AVAILABLE = False

# NumPy und Audio I/O
try:
    import numpy as np
    import soundfile as sf
    AUDIO_IO_AVAILABLE = True
except ImportError:
    AUDIO_IO_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# SPEECH-TO-TEXT
# ═══════════════════════════════════════════════════════════════

# Faster-Whisper (Empfohlen)
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    pass

# Original Whisper
WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    pass

# Transformers Whisper
TRANSFORMERS_WHISPER_AVAILABLE = False
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    TRANSFORMERS_WHISPER_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH
# ═══════════════════════════════════════════════════════════════

# Coqui TTS (XTTS)
COQUI_AVAILABLE = False
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    pass

# Bark
BARK_AVAILABLE = False
try:
    from bark import SAMPLE_RATE as BARK_SAMPLE_RATE, generate_audio as bark_generate, preload_models as bark_preload
    BARK_AVAILABLE = True
except ImportError:
    pass

# F5-TTS
F5TTS_AVAILABLE = False
try:
    from f5_tts.api import F5TTS
    F5TTS_AVAILABLE = True
except ImportError:
    pass

# Parler-TTS
PARLER_AVAILABLE = False
try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    PARLER_AVAILABLE = True
except ImportError:
    pass

# Fish-Speech
FISH_AVAILABLE = False
try:
    from fish_speech.inference import TTSInference
    FISH_AVAILABLE = True
except ImportError:
    pass

# MeloTTS
MELO_AVAILABLE = False
try:
    from melo.api import TTS as MeloTTS
    MELO_AVAILABLE = True
except ImportError:
    pass

# StyleTTS 2
STYLETTS_AVAILABLE = False
try:
    from styletts2 import tts as styletts_tts
    STYLETTS_AVAILABLE = True
except ImportError:
    pass

# Kokoro-TTS
KOKORO_AVAILABLE = False
try:
    from kokoro import KokoroTTS
    KOKORO_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# MUSIC GENERATION
# ═══════════════════════════════════════════════════════════════

# MusicGen (Meta)
MUSICGEN_AVAILABLE = False
try:
    from audiocraft.models import MusicGen
    MUSICGEN_AVAILABLE = True
except ImportError:
    pass

# Stable Audio
STABLE_AUDIO_AVAILABLE = False
try:
    from stable_audio_tools import get_pretrained_model
    STABLE_AUDIO_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# VOICE CLONING
# ═══════════════════════════════════════════════════════════════

# OpenVoice
OPENVOICE_AVAILABLE = False
try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    OPENVOICE_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# AUDIO PROCESSING
# ═══════════════════════════════════════════════════════════════

# Demucs (Source Separation)
DEMUCS_AVAILABLE = False
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# VERFÜGBARE MODELLE - DIE BESTEN 2024/2025
# ═══════════════════════════════════════════════════════════════

AUDIO_MODELS = {
    # ═══════════════════════════════════════════════════════════
    # SPEECH-TO-TEXT (STT)
    # ═══════════════════════════════════════════════════════════
    'whisper-large-v3-turbo': {
        'name': 'Whisper Large V3 Turbo',
        'hf_id': 'openai/whisper-large-v3-turbo',
        'type': 'stt',
        'vram_gb': 6,
        'description': 'Schnellste Whisper-Version, 8x schneller',
        'recommended': True,
    },
    'whisper-large-v3': {
        'name': 'Whisper Large V3',
        'hf_id': 'openai/whisper-large-v3',
        'type': 'stt',
        'vram_gb': 10,
        'description': 'Beste Genauigkeit',
        'recommended': True,
    },
    'distil-whisper-large-v3': {
        'name': 'Distil-Whisper Large V3',
        'hf_id': 'distil-whisper/distil-large-v3',
        'type': 'stt',
        'vram_gb': 4,
        'description': 'Kompakt, 6x schneller',
    },
    'whisper-medium': {
        'name': 'Whisper Medium',
        'type': 'stt',
        'vram_gb': 5,
    },
    'whisper-small': {
        'name': 'Whisper Small',
        'type': 'stt',
        'vram_gb': 2,
    },

    # ═══════════════════════════════════════════════════════════
    # TEXT-TO-SPEECH (TTS)
    # ═══════════════════════════════════════════════════════════
    'xtts-v2': {
        'name': 'XTTS v2',
        'type': 'tts',
        'vram_gb': 6,
        'description': 'Bestes Open Source TTS mit Voice Cloning',
        'languages': ['en', 'de', 'fr', 'es', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'ko', 'hi'],
        'recommended': True,
    },
    'f5-tts': {
        'name': 'F5-TTS',
        'hf_id': 'SWivid/F5-TTS',
        'type': 'tts',
        'vram_gb': 4,
        'description': 'Neuestes hochqualitatives TTS',
        'recommended': True,
    },
    'parler-tts-large': {
        'name': 'Parler-TTS Large',
        'hf_id': 'parler-tts/parler-tts-large-v1',
        'type': 'tts',
        'vram_gb': 8,
        'description': 'Natürlich mit Style-Prompts',
    },
    'parler-tts-mini': {
        'name': 'Parler-TTS Mini',
        'hf_id': 'parler-tts/parler-tts-mini-v1',
        'type': 'tts',
        'vram_gb': 2,
        'description': 'Schnelle kompakte Version',
    },
    'bark': {
        'name': 'Bark',
        'type': 'tts',
        'vram_gb': 12,
        'description': 'Expressiv mit Sound Effects, Lachen, Musik',
    },
    'fish-speech': {
        'name': 'Fish-Speech',
        'hf_id': 'fishaudio/fish-speech-1.4',
        'type': 'tts',
        'vram_gb': 4,
        'description': 'Exzellentes Voice Cloning',
    },
    'melo-tts': {
        'name': 'MeloTTS',
        'type': 'tts',
        'vram_gb': 2,
        'description': 'Schnell, multilingual',
    },
    'styletts2': {
        'name': 'StyleTTS 2',
        'type': 'tts',
        'vram_gb': 4,
        'description': 'Style-Kontrolle',
    },
    'kokoro-tts': {
        'name': 'Kokoro-TTS',
        'hf_id': 'hexgrad/Kokoro-82M',
        'type': 'tts',
        'vram_gb': 1,
        'description': 'Sehr natürlich, kompakt',
    },

    # ═══════════════════════════════════════════════════════════
    # MUSIC GENERATION
    # ═══════════════════════════════════════════════════════════
    'musicgen-large': {
        'name': 'MusicGen Large',
        'hf_id': 'facebook/musicgen-large',
        'type': 'music',
        'vram_gb': 16,
        'description': 'Beste Musikqualität',
        'recommended': True,
    },
    'musicgen-melody-large': {
        'name': 'MusicGen Melody Large',
        'hf_id': 'facebook/musicgen-melody-large',
        'type': 'music',
        'vram_gb': 16,
        'description': 'Mit Melodie-Konditionierung',
    },
    'musicgen-medium': {
        'name': 'MusicGen Medium',
        'hf_id': 'facebook/musicgen-medium',
        'type': 'music',
        'vram_gb': 8,
    },
    'musicgen-small': {
        'name': 'MusicGen Small',
        'hf_id': 'facebook/musicgen-small',
        'type': 'music',
        'vram_gb': 4,
    },
    'stable-audio-open': {
        'name': 'Stable Audio Open',
        'type': 'music',
        'vram_gb': 8,
        'description': 'Stability AI, bis 47 Sekunden',
    },

    # ═══════════════════════════════════════════════════════════
    # VOICE CLONING
    # ═══════════════════════════════════════════════════════════
    'openvoice-v2': {
        'name': 'OpenVoice v2',
        'type': 'clone',
        'vram_gb': 4,
        'description': 'Echtzeit Voice Cloning',
        'recommended': True,
    },

    # ═══════════════════════════════════════════════════════════
    # AUDIO PROCESSING
    # ═══════════════════════════════════════════════════════════
    'demucs-htdemucs': {
        'name': 'Demucs HTDemucs',
        'type': 'separation',
        'vram_gb': 4,
        'description': 'Audio Source Separation (Vocals, Drums, Bass, Other)',
        'recommended': True,
    },
    'demucs-htdemucs_ft': {
        'name': 'Demucs HTDemucs Fine-tuned',
        'type': 'separation',
        'vram_gb': 4,
        'description': 'Verbesserte Vocals-Trennung',
    },
}


class AudioWorker(BaseWorker):
    """
    Audio Worker - Die BESTEN Audio-Modelle

    Features:
    - Speech-to-Text (Whisper Large V3 Turbo)
    - Text-to-Speech (XTTS v2, F5-TTS, Parler, Bark)
    - Music Generation (MusicGen, Stable Audio)
    - Voice Cloning (OpenVoice)
    - Audio Processing (Demucs Source Separation)
    """

    def __init__(self):
        super().__init__("Audio Processing")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._whisper = None
        self._tts = None
        self._music = None
        self._openvoice = None
        self._demucs = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        available = []

        if FASTER_WHISPER_AVAILABLE or WHISPER_AVAILABLE or TRANSFORMERS_WHISPER_AVAILABLE:
            available.append("STT")
        if COQUI_AVAILABLE or BARK_AVAILABLE or F5TTS_AVAILABLE or PARLER_AVAILABLE:
            available.append("TTS")
        if MUSICGEN_AVAILABLE or STABLE_AUDIO_AVAILABLE:
            available.append("Music")
        if OPENVOICE_AVAILABLE:
            available.append("Voice Cloning")
        if DEMUCS_AVAILABLE:
            available.append("Separation")

        if not available:
            self._error_message = "Keine Audio-Libraries verfügbar"
            self.status = WorkerStatus.ERROR
            return False

        print(f"[OK] Audio Worker bereit")
        print(f"    Device: {self._device}")
        print(f"    Features: {', '.join(available)}")
        print(f"    Empfohlen: whisper-large-v3-turbo, xtts-v2, musicgen-large")

        self.status = WorkerStatus.READY
        return True

    def _load_whisper(self, model_id: str = "whisper-large-v3"):
        """Load Whisper model for STT"""
        model_info = AUDIO_MODELS.get(model_id, AUDIO_MODELS['whisper-large-v3'])

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            # Faster-Whisper (empfohlen für Geschwindigkeit)
            if FASTER_WHISPER_AVAILABLE:
                size = model_id.replace('whisper-', '').replace('-turbo', '')
                if 'distil' in model_id:
                    size = 'distil-large-v3'
                elif 'v3' in model_id:
                    size = 'large-v3'
                model = WhisperModel(
                    size,
                    device=self._device,
                    compute_type="float16" if self._device == "cuda" else "int8"
                )
                return {"type": "faster", "model": model}

            # Transformers Whisper
            elif TRANSFORMERS_WHISPER_AVAILABLE and model_info.get('hf_id'):
                hf_id = model_info['hf_id']
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(self._device)
                processor = AutoProcessor.from_pretrained(hf_id)
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=self._device,
                )
                return {"type": "transformers", "pipe": pipe}

            # Original Whisper
            elif WHISPER_AVAILABLE:
                size = model_id.replace('whisper-', '').split('-')[0]
                model = whisper.load_model(size, device=self._device)
                return {"type": "original", "model": model}

            raise ValueError("Kein Whisper-Backend verfügbar")

        self._whisper = model_manager.get_model(f"whisper_{model_id}", loader)
        self._current_model_id = model_id
        print(f"[OK] {model_info['name']} geladen")

    def _load_tts(self, model_id: str = "xtts-v2"):
        """Load TTS model"""
        model_info = AUDIO_MODELS.get(model_id, AUDIO_MODELS['xtts-v2'])

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            if model_id == "xtts-v2" and COQUI_AVAILABLE:
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                tts.to(self._device)
                return {"type": "xtts", "model": tts}

            elif model_id == "f5-tts" and F5TTS_AVAILABLE:
                tts = F5TTS()
                return {"type": "f5", "model": tts}

            elif "parler" in model_id and PARLER_AVAILABLE:
                hf_id = model_info['hf_id']
                model = ParlerTTSForConditionalGeneration.from_pretrained(hf_id).to(self._device)
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                return {"type": "parler", "model": model, "tokenizer": tokenizer}

            elif model_id == "bark" and BARK_AVAILABLE:
                bark_preload()
                return {"type": "bark"}

            elif model_id == "fish-speech" and FISH_AVAILABLE:
                tts = TTSInference()
                return {"type": "fish", "model": tts}

            elif model_id == "melo-tts" and MELO_AVAILABLE:
                tts = MeloTTS(language="EN", device=self._device)
                return {"type": "melo", "model": tts}

            elif model_id == "styletts2" and STYLETTS_AVAILABLE:
                return {"type": "styletts"}

            elif model_id == "kokoro-tts" and KOKORO_AVAILABLE:
                tts = KokoroTTS()
                return {"type": "kokoro", "model": tts}

            # Fallback zu XTTS wenn verfügbar
            elif COQUI_AVAILABLE:
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                tts.to(self._device)
                return {"type": "xtts", "model": tts}

            raise ValueError(f"TTS-Modell {model_id} nicht verfügbar")

        self._tts = model_manager.get_model(f"tts_{model_id}", loader)
        self._current_model_id = model_id
        print(f"[OK] {model_info['name']} geladen")

    def _load_musicgen(self, model_id: str = "musicgen-large"):
        """Load MusicGen model"""
        model_info = AUDIO_MODELS.get(model_id, AUDIO_MODELS['musicgen-large'])

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            if MUSICGEN_AVAILABLE:
                hf_id = model_info.get('hf_id', 'facebook/musicgen-large')
                model = MusicGen.get_pretrained(hf_id)
                model.set_generation_params(duration=30)
                return {"type": "musicgen", "model": model}

            raise ValueError("MusicGen nicht verfügbar")

        self._music = model_manager.get_model(f"music_{model_id}", loader)
        print(f"[OK] {model_info['name']} geladen")

    def _load_demucs(self, model_id: str = "demucs-htdemucs"):
        """Load Demucs for source separation"""
        def loader():
            print(f"[LOAD] Lade Demucs...")
            model_name = model_id.replace('demucs-', '')
            model = pretrained.get_model(model_name)
            model.to(self._device)
            return {"model": model}

        self._demucs = model_manager.get_model(model_id, loader)
        print(f"[OK] Demucs geladen")

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        model: str = "whisper-large-v3-turbo",
    ) -> dict:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'de')
            model: Whisper model variant

        Returns:
            dict with text, language, segments, duration, gpu_seconds
        """
        if not audio_path:
            raise ValueError("Audio-Pfad erforderlich")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise ValueError(f"Audio-Datei nicht gefunden: {audio_path}")

        if self._current_model_id != model or self._whisper is None:
            self._load_whisper(model)

        start_time = time.time()

        if self._whisper["type"] == "faster":
            segments, info = self._whisper["model"].transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            segments_list = []
            for seg in segments:
                segments_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                })
            text = " ".join([s["text"] for s in segments_list])
            return {
                "text": text,
                "segments": segments_list,
                "language": info.language,
                "duration": info.duration,
                "model": model,
                "gpu_seconds": time.time() - start_time,
            }

        elif self._whisper["type"] == "transformers":
            result = self._whisper["pipe"](audio_path, return_timestamps=True)
            return {
                "text": result["text"],
                "chunks": result.get("chunks", []),
                "model": model,
                "gpu_seconds": time.time() - start_time,
            }

        else:  # original whisper
            result = self._whisper["model"].transcribe(audio_path, language=language)
            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language"),
                "model": model,
                "gpu_seconds": time.time() - start_time,
            }

    def text_to_speech(
        self,
        text: str,
        output_path: str = None,
        voice: str = None,
        language: str = "en",
        model: str = "xtts-v2",
        style_prompt: str = None,
    ) -> dict:
        """
        Convert text to speech.

        Args:
            text: Text to convert (max 10000 characters)
            output_path: Optional output file path
            voice: Speaker WAV file for voice cloning
            language: Language code
            model: TTS model
            style_prompt: Style description for Parler-TTS

        Returns:
            dict with output_path, duration, model, gpu_seconds
        """
        if not text or not text.strip():
            raise ValueError("Text erforderlich")

        text = text.strip()[:10000]

        if self._current_model_id != model or self._tts is None:
            self._load_tts(model)

        start_time = time.time()

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"tts_{uuid.uuid4().hex[:8]}.wav")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        tts_type = self._tts["type"]

        if tts_type == "xtts":
            tts = self._tts["model"]
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language,
                speaker_wav=voice,
            )

        elif tts_type == "f5":
            tts = self._tts["model"]
            audio = tts.infer(text, ref_audio=voice)
            sf.write(output_path, audio, 24000)

        elif tts_type == "parler":
            model = self._tts["model"]
            tokenizer = self._tts["tokenizer"]
            prompt = style_prompt or "A female speaker with a clear, natural voice."
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self._device)
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio = generation.cpu().numpy().squeeze()
            sf.write(output_path, audio, model.config.sampling_rate)

        elif tts_type == "bark":
            audio = bark_generate(text)
            sf.write(output_path, audio, BARK_SAMPLE_RATE)

        elif tts_type == "fish":
            tts = self._tts["model"]
            audio = tts.tts(text, reference_audio=voice)
            sf.write(output_path, audio, 44100)

        elif tts_type == "melo":
            tts = self._tts["model"]
            speaker_ids = tts.hps.data.spk2id
            speaker_id = list(speaker_ids.values())[0]
            tts.tts_to_file(text, speaker_id, output_path)

        elif tts_type == "kokoro":
            tts = self._tts["model"]
            audio = tts.generate(text)
            sf.write(output_path, audio, 24000)

        else:
            raise ValueError(f"Unbekannter TTS-Typ: {tts_type}")

        # Berechne Dauer
        duration = 0.0
        try:
            info = sf.info(output_path)
            duration = info.duration
        except Exception:
            pass

        return {
            "output_path": output_path,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "duration": duration,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def generate_music(
        self,
        prompt: str,
        duration: int = 30,
        output_path: str = None,
        model: str = "musicgen-large",
        melody_audio: str = None,
    ) -> dict:
        """
        Generate music from text prompt.

        Args:
            prompt: Music description (max 1000 characters)
            duration: Duration in seconds (5-300)
            output_path: Optional output file path
            model: MusicGen model variant
            melody_audio: Optional melody conditioning audio file

        Returns:
            dict with output_path, prompt, duration, model, gpu_seconds
        """
        if not MUSICGEN_AVAILABLE:
            raise ValueError("MusicGen nicht installiert")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt erforderlich")

        prompt = prompt.strip()[:1000]
        duration = max(5, min(300, duration))

        if self._music is None:
            self._load_musicgen(model)

        start_time = time.time()

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"music_{uuid.uuid4().hex[:8]}.wav")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        music_model = self._music["model"]
        music_model.set_generation_params(duration=duration)

        # Mit Melodie-Konditionierung
        if melody_audio and "melody" in model:
            melody, sr = torchaudio.load(melody_audio)
            wav = music_model.generate_with_chroma([prompt], melody[None].to(self._device), sr)
        else:
            wav = music_model.generate([prompt])

        audio = wav[0].cpu().numpy()
        sf.write(output_path, audio.T, 32000)

        return {
            "output_path": output_path,
            "prompt": prompt,
            "duration": duration,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def separate_audio(
        self,
        audio_path: str,
        output_dir: str = None,
        model: str = "demucs-htdemucs",
    ) -> dict:
        """
        Separate audio into stems (vocals, drums, bass, other).

        Args:
            audio_path: Path to audio file
            output_dir: Output directory for stems
            model: Demucs model variant

        Returns:
            dict with stems paths and gpu_seconds
        """
        if not DEMUCS_AVAILABLE:
            raise ValueError("Demucs nicht installiert")

        if not audio_path or not Path(audio_path).exists():
            raise ValueError(f"Audio-Datei nicht gefunden: {audio_path}")

        if self._demucs is None:
            self._load_demucs(model)

        start_time = time.time()

        if output_dir is None:
            output_dir = str(Config.DATA_DIR / "generated" / f"stems_{uuid.uuid4().hex[:8]}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load audio
        wav, sr = torchaudio.load(audio_path)
        wav = wav.to(self._device)

        # Apply model
        demucs_model = self._demucs["model"]
        sources = apply_model(demucs_model, wav[None], device=self._device)[0]

        # Save stems
        stem_names = ["drums", "bass", "other", "vocals"]
        stems = {}
        for i, name in enumerate(stem_names):
            stem_path = str(Path(output_dir) / f"{name}.wav")
            torchaudio.save(stem_path, sources[i].cpu(), sr)
            stems[name] = stem_path

        return {
            "stems": stems,
            "output_dir": output_dir,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def clone_voice(
        self,
        source_audio: str,
        reference_audio: str,
        output_path: str = None,
    ) -> dict:
        """
        Clone voice from reference to source audio.

        Args:
            source_audio: Audio to convert
            reference_audio: Reference voice to clone
            output_path: Output file path

        Returns:
            dict with output_path and gpu_seconds
        """
        if not OPENVOICE_AVAILABLE:
            raise ValueError("OpenVoice nicht installiert")

        if not Path(source_audio).exists():
            raise ValueError(f"Source-Audio nicht gefunden: {source_audio}")
        if not Path(reference_audio).exists():
            raise ValueError(f"Reference-Audio nicht gefunden: {reference_audio}")

        start_time = time.time()

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"clone_{uuid.uuid4().hex[:8]}.wav")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self._openvoice is None:
            self._openvoice = ToneColorConverter(device=self._device)
            self._openvoice.load_ckpt()

        # Extract speaker embedding from reference
        target_se = se_extractor.get_se(reference_audio, self._openvoice.model)

        # Convert voice
        self._openvoice.convert(
            audio_src_path=source_audio,
            src_se=None,
            tgt_se=target_se,
            output_path=output_path,
        )

        return {
            "output_path": output_path,
            "source": source_audio,
            "reference": reference_audio,
            "gpu_seconds": time.time() - start_time,
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process an audio job"""
        task_type = input_data.get("task", "transcribe")

        self.notify_progress(job_id, 0.1, f"Starte {task_type}")

        if task_type in ["transcribe", "stt"]:
            audio_path = input_data.get("audio_path") or input_data.get("file")
            language = input_data.get("language")
            model = input_data.get("model", "whisper-large-v3-turbo")
            result = self.transcribe(audio_path, language=language, model=model)

        elif task_type in ["tts", "text_to_speech"]:
            text = input_data.get("text")
            voice = input_data.get("voice")
            language = input_data.get("language", "en")
            model = input_data.get("model", "xtts-v2")
            style_prompt = input_data.get("style_prompt")
            result = self.text_to_speech(
                text, voice=voice, language=language,
                model=model, style_prompt=style_prompt
            )

        elif task_type in ["music", "generate_music"]:
            prompt = input_data.get("prompt")
            duration = input_data.get("duration", 30)
            model = input_data.get("model", "musicgen-large")
            melody = input_data.get("melody_audio")
            result = self.generate_music(prompt, duration=duration, model=model, melody_audio=melody)

        elif task_type in ["separate", "separation"]:
            audio_path = input_data.get("audio_path") or input_data.get("file")
            model = input_data.get("model", "demucs-htdemucs")
            result = self.separate_audio(audio_path, model=model)

        elif task_type in ["clone", "voice_clone"]:
            source = input_data.get("source_audio")
            reference = input_data.get("reference_audio")
            result = self.clone_voice(source, reference)

        else:
            raise ValueError(f"Unbekannter Task-Typ: {task_type}")

        self.notify_progress(job_id, 1.0, "Fertig")
        return result

    def cleanup(self):
        """Release resources"""
        self._whisper = None
        self._tts = None
        self._music = None
        self._openvoice = None
        self._demucs = None
        self._current_model_id = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Audio Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        return AUDIO_MODELS

    def get_recommended_models(self) -> List[str]:
        """Return recommended models"""
        return [k for k, v in AUDIO_MODELS.items() if v.get('recommended', False)]


# Singleton
_audio_worker: Optional[AudioWorker] = None


def get_audio_worker() -> AudioWorker:
    """Get singleton instance"""
    global _audio_worker
    if _audio_worker is None:
        _audio_worker = AudioWorker()
    return _audio_worker
