#!/usr/bin/env python3
"""
SCIO - Workers

Zentrales Worker-System mit:
- BaseWorker: Basis-Klasse fuer alle Worker
- WorkerManager: Zentrales Management aller Worker
- Spezialisierte Worker fuer verschiedene AI-Tasks
"""

from .base_worker import BaseWorker
from .llm_inference import LLMInferenceWorker
from .llm_training import LLMTrainingWorker
from .image_gen import ImageGenerationWorker
from .worker_manager import WorkerManager, WorkerType, get_worker_manager

__all__ = [
    'BaseWorker',
    'LLMInferenceWorker',
    'LLMTrainingWorker',
    'ImageGenerationWorker',
    'WorkerManager',
    'WorkerType',
    'get_worker_manager',
]
