#!/usr/bin/env python3
"""
SCIO - Base Worker
Basisklasse für alle Worker
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from backend.config import Config


class WorkerStatus(str, Enum):
    """Worker Status"""
    IDLE = 'idle'
    LOADING = 'loading'
    READY = 'ready'
    BUSY = 'busy'
    ERROR = 'error'


@dataclass
class WorkerStats:
    """Worker Statistiken"""
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    total_tokens_processed: int = 0
    total_gpu_seconds: float = 0
    average_latency_ms: float = 0


class BaseWorker(ABC):
    """
    Basis-Worker Klasse

    Alle Worker erben von dieser Klasse und implementieren:
    - initialize(): Lädt Modelle/Ressourcen
    - process(): Verarbeitet einen Job
    - cleanup(): Gibt Ressourcen frei
    """

    def __init__(self, name: str):
        self.name = name
        self.status = WorkerStatus.IDLE
        self.stats = WorkerStats()
        self._lock = threading.Lock()
        self._callbacks: list = []
        self._error_message: Optional[str] = None

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialisiert den Worker (lädt Modelle, etc.)

        Returns:
            True wenn erfolgreich, False sonst
        """
        pass

    @abstractmethod
    def process(self, job_id: str, input_data: dict) -> dict:
        """
        Verarbeitet einen Job

        Args:
            job_id: Eindeutige Job-ID
            input_data: Eingabedaten

        Returns:
            dict mit Ergebnis-Daten
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Gibt Ressourcen frei"""
        pass

    def add_callback(self, callback: Callable):
        """Registriert Progress-Callback"""
        self._callbacks.append(callback)

    def notify_progress(self, job_id: str, progress: float, message: str = None):
        """Benachrichtigt über Fortschritt"""
        for callback in self._callbacks:
            try:
                callback(job_id, progress, message)
            except Exception:
                pass  # Callback errors should not affect job processing

    def get_status(self) -> dict:
        """Gibt Worker-Status zurück"""
        return {
            'name': self.name,
            'status': self.status.value,
            'error': self._error_message,
            'stats': {
                'total_jobs': self.stats.total_jobs,
                'successful_jobs': self.stats.successful_jobs,
                'failed_jobs': self.stats.failed_jobs,
                'total_tokens': self.stats.total_tokens_processed,
                'total_gpu_seconds': round(self.stats.total_gpu_seconds, 2),
                'avg_latency_ms': round(self.stats.average_latency_ms, 2),
            },
        }

    def _update_stats(self, success: bool, tokens: int = 0, gpu_seconds: float = 0, latency_ms: float = 0):
        """Aktualisiert Statistiken"""
        with self._lock:
            self.stats.total_jobs += 1
            if success:
                self.stats.successful_jobs += 1
            else:
                self.stats.failed_jobs += 1

            self.stats.total_tokens_processed += tokens
            self.stats.total_gpu_seconds += gpu_seconds

            # Rolling average for latency
            if self.stats.total_jobs > 1:
                self.stats.average_latency_ms = (
                    (self.stats.average_latency_ms * (self.stats.total_jobs - 1) + latency_ms)
                    / self.stats.total_jobs
                )
            else:
                self.stats.average_latency_ms = latency_ms

    def __call__(self, job_id: str, input_data: dict) -> dict:
        """Wrapper für Job-Ausführung mit Timing und Error-Handling"""
        start_time = time.time()

        with self._lock:
            self.status = WorkerStatus.BUSY

        try:
            result = self.process(job_id, input_data)

            # Calculate stats
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            tokens = result.get('tokens_input', 0) + result.get('tokens_output', 0)
            gpu_seconds = result.get('gpu_seconds', end_time - start_time)

            self._update_stats(True, tokens, gpu_seconds, latency_ms)

            return result

        except Exception as e:
            self._error_message = str(e)
            self._update_stats(False)
            raise

        finally:
            with self._lock:
                self.status = WorkerStatus.READY if not self._error_message else WorkerStatus.ERROR


class ModelManager:
    """
    Model Manager

    Verwaltet das Laden/Entladen von Modellen für optimale VRAM-Nutzung.
    Optimiert für RTX 5090 mit 24GB VRAM.
    """

    def __init__(self, max_models: int = 4):
        # RTX 5090 kann 4+ Modelle gleichzeitig halten
        self.max_models = max_models
        self._loaded_models: Dict[str, Any] = {}
        self._model_usage: Dict[str, float] = {}  # model_id -> last_used_timestamp
        self._lock = threading.Lock()

    def get_model(self, model_id: str, loader_func: Callable) -> Any:
        """
        Gibt Modell zurück, lädt es bei Bedarf

        Args:
            model_id: Modell-Identifikator
            loader_func: Funktion zum Laden des Modells

        Returns:
            Geladenes Modell
        """
        with self._lock:
            # Check if already loaded
            if model_id in self._loaded_models:
                self._model_usage[model_id] = time.time()
                return self._loaded_models[model_id]

            # Evict old models if necessary
            while len(self._loaded_models) >= self.max_models:
                self._evict_oldest()

            # Load new model
            print(f"[RETRY] Lade Modell: {model_id}")
            model = loader_func()
            self._loaded_models[model_id] = model
            self._model_usage[model_id] = time.time()

            return model

    def _evict_oldest(self):
        """Entfernt das am längsten ungenutzte Modell"""
        if not self._model_usage:
            return

        oldest_id = min(self._model_usage, key=self._model_usage.get)
        print(f"🗑️  Entlade Modell: {oldest_id}")

        # Free memory
        model = self._loaded_models.pop(oldest_id, None)
        del self._model_usage[oldest_id]

        if model is not None:
            del model

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass  # torch not available or CUDA error

    def unload_all(self):
        """Entlädt alle Modelle"""
        with self._lock:
            for model_id in list(self._loaded_models.keys()):
                model = self._loaded_models.pop(model_id, None)
                if model is not None:
                    del model

            self._model_usage.clear()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass  # torch not available or CUDA error

            print("[OK] Alle Modelle entladen")

    def is_loaded(self, model_id: str) -> bool:
        """Prüft ob Modell geladen ist"""
        return model_id in self._loaded_models


# Global Model Manager
model_manager = ModelManager()
