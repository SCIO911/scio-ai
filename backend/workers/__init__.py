#!/usr/bin/env python3
"""
SCIO - Workers

Zentrales Worker-System mit:
- BaseWorker: Basis-Klasse fuer alle Worker
- WorkerManager: Zentrales Management aller Worker
- Spezialisierte Worker fuer verschiedene AI-Tasks
"""

from .base_worker import BaseWorker
from .worker_manager import WorkerManager, WorkerType, get_worker_manager

# Lazy imports für ML-Worker (können fehlende Dependencies haben)
LLMInferenceWorker = None
LLMTrainingWorker = None
ImageGenerationWorker = None

def _lazy_import_workers():
    """Lazy import der ML-Worker bei Bedarf"""
    global LLMInferenceWorker, LLMTrainingWorker, ImageGenerationWorker

    try:
        from .llm_inference import LLMInferenceWorker as _LLMInf
        LLMInferenceWorker = _LLMInf
    except ImportError as e:
        print(f"[WARN] LLMInferenceWorker nicht verfügbar: {e}")

    try:
        from .llm_training import LLMTrainingWorker as _LLMTrain
        LLMTrainingWorker = _LLMTrain
    except ImportError as e:
        print(f"[WARN] LLMTrainingWorker nicht verfügbar: {e}")

    try:
        from .image_gen import ImageGenerationWorker as _ImgGen
        ImageGenerationWorker = _ImgGen
    except ImportError as e:
        print(f"[WARN] ImageGenerationWorker nicht verfügbar: {e}")

# Versuche Workers zu laden (nicht-kritisch wenn es fehlschlägt)
try:
    _lazy_import_workers()
except Exception as e:
    print(f"[WARN] ML-Workers konnten nicht geladen werden: {e}")

__all__ = [
    'BaseWorker',
    'LLMInferenceWorker',
    'LLMTrainingWorker',
    'ImageGenerationWorker',
    'WorkerManager',
    'WorkerType',
    'get_worker_manager',
]
