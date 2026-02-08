#!/usr/bin/env python3
"""
SCIO Worker Manager

Zentrales Management aller Worker mit:
- Singleton-Pattern fuer jeden Worker-Typ
- Health Monitoring
- VRAM-Management
- Automatische Skalierung
"""

import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WorkerType(str, Enum):
    """Verfuegbare Worker-Typen"""
    LLM_INFERENCE = "llm_inference"
    LLM_TRAINING = "llm_training"
    IMAGE_GEN = "image_gen"
    AUDIO = "audio"
    VIDEO = "video"
    VISION = "vision"
    CODE = "code"
    EMBEDDING = "embedding"
    UPSCALE = "upscale"
    DOCUMENT = "document"
    THREED = "threed"


@dataclass
class WorkerInfo:
    """Information ueber einen Worker"""
    worker_type: WorkerType
    name: str
    instance: Any
    status: str = "idle"
    last_activity: datetime = field(default_factory=datetime.now)
    jobs_completed: int = 0
    jobs_failed: int = 0
    current_job: Optional[str] = None
    vram_usage_mb: int = 0


class WorkerManager:
    """
    Zentraler Worker Manager

    Features:
    - Lazy Loading: Worker werden erst bei Bedarf initialisiert
    - Health Checks: Automatische Ueberwachung
    - VRAM Management: Entlaedt Worker bei Speichermangel
    - Statistiken: Jobs, Fehler, Auslastung
    """

    def __init__(self):
        self._workers: Dict[WorkerType, WorkerInfo] = {}
        self._lock = threading.RLock()
        self._initialized = False

        # Worker Getter-Funktionen (Lazy Loading)
        self._worker_factories = {
            WorkerType.LLM_INFERENCE: self._get_llm_inference,
            WorkerType.LLM_TRAINING: self._get_llm_training,
            WorkerType.IMAGE_GEN: self._get_image_gen,
            WorkerType.AUDIO: self._get_audio,
            WorkerType.VIDEO: self._get_video,
            WorkerType.VISION: self._get_vision,
            WorkerType.CODE: self._get_code,
            WorkerType.EMBEDDING: self._get_embedding,
            WorkerType.UPSCALE: self._get_upscale,
            WorkerType.DOCUMENT: self._get_document,
            WorkerType.THREED: self._get_threed,
        }

    def initialize(self) -> bool:
        """Initialisiert den Worker Manager (ohne Worker zu laden)"""
        self._initialized = True
        return True

    def get_worker(self, worker_type: WorkerType) -> Optional[Any]:
        """
        Holt einen Worker (Lazy Loading)

        Args:
            worker_type: Typ des Workers

        Returns:
            Worker-Instanz oder None
        """
        with self._lock:
            # Bereits geladen?
            if worker_type in self._workers:
                info = self._workers[worker_type]
                info.last_activity = datetime.now()
                return info.instance

            # Lazy Load
            factory = self._worker_factories.get(worker_type)
            if factory:
                try:
                    instance = factory()
                    if instance:
                        self._workers[worker_type] = WorkerInfo(
                            worker_type=worker_type,
                            name=worker_type.value,
                            instance=instance
                        )
                        return instance
                except Exception as e:
                    logger.error(f"Worker {worker_type} konnte nicht geladen werden: {e}")

            return None

    def _get_llm_inference(self):
        from backend.workers.llm_inference import get_llm_worker
        return get_llm_worker()

    def _get_llm_training(self):
        from backend.workers.llm_training import get_training_worker
        return get_training_worker()

    def _get_image_gen(self):
        from backend.workers.image_gen import get_image_worker
        return get_image_worker()

    def _get_audio(self):
        from backend.workers.audio_worker import get_audio_worker
        return get_audio_worker()

    def _get_video(self):
        from backend.workers.video_worker import get_video_worker
        return get_video_worker()

    def _get_vision(self):
        from backend.workers.vision_worker import get_vision_worker
        return get_vision_worker()

    def _get_code(self):
        from backend.workers.code_worker import get_code_worker
        return get_code_worker()

    def _get_embedding(self):
        from backend.workers.embedding_worker import get_embedding_worker
        return get_embedding_worker()

    def _get_upscale(self):
        from backend.workers.upscale_worker import get_upscale_worker
        return get_upscale_worker()

    def _get_document(self):
        from backend.workers.document_worker import get_document_worker
        return get_document_worker()

    def _get_threed(self):
        from backend.workers.threed_worker import get_threed_worker
        return get_threed_worker()

    # Convenience Methods
    @property
    def llm(self):
        """LLM Inference Worker"""
        return self.get_worker(WorkerType.LLM_INFERENCE)

    @property
    def image(self):
        """Image Generation Worker"""
        return self.get_worker(WorkerType.IMAGE_GEN)

    @property
    def audio(self):
        """Audio Worker"""
        return self.get_worker(WorkerType.AUDIO)

    @property
    def video(self):
        """Video Worker"""
        return self.get_worker(WorkerType.VIDEO)

    @property
    def vision(self):
        """Vision Worker"""
        return self.get_worker(WorkerType.VISION)

    @property
    def code(self):
        """Code Worker"""
        return self.get_worker(WorkerType.CODE)

    @property
    def embedding(self):
        """Embedding Worker"""
        return self.get_worker(WorkerType.EMBEDDING)

    def get_status(self) -> Dict[str, Any]:
        """Gibt den Status aller Worker zurueck"""
        with self._lock:
            return {
                "initialized": self._initialized,
                "workers_loaded": len(self._workers),
                "workers_available": len(self._worker_factories),
                "workers": {
                    wt.value: {
                        "status": info.status,
                        "jobs_completed": info.jobs_completed,
                        "jobs_failed": info.jobs_failed,
                        "last_activity": info.last_activity.isoformat(),
                        "current_job": info.current_job,
                    }
                    for wt, info in self._workers.items()
                }
            }

    def get_loaded_workers(self) -> List[str]:
        """Liste der geladenen Worker"""
        with self._lock:
            return [wt.value for wt in self._workers.keys()]

    def record_job_completed(self, worker_type: WorkerType, job_id: str):
        """Zaehlt einen abgeschlossenen Job"""
        with self._lock:
            if worker_type in self._workers:
                info = self._workers[worker_type]
                info.jobs_completed += 1
                info.status = "idle"
                info.current_job = None
                info.last_activity = datetime.now()

    def record_job_failed(self, worker_type: WorkerType, job_id: str):
        """Zaehlt einen fehlgeschlagenen Job"""
        with self._lock:
            if worker_type in self._workers:
                info = self._workers[worker_type]
                info.jobs_failed += 1
                info.status = "idle"
                info.current_job = None
                info.last_activity = datetime.now()

    def record_job_started(self, worker_type: WorkerType, job_id: str):
        """Markiert einen Worker als beschaeftigt"""
        with self._lock:
            if worker_type in self._workers:
                info = self._workers[worker_type]
                info.status = "busy"
                info.current_job = job_id
                info.last_activity = datetime.now()

    def unload_worker(self, worker_type: WorkerType) -> bool:
        """
        Entlaedt einen Worker um VRAM freizugeben

        Args:
            worker_type: Typ des Workers

        Returns:
            True wenn erfolgreich entladen
        """
        with self._lock:
            if worker_type in self._workers:
                info = self._workers[worker_type]

                # Worker entladen wenn moeglich
                if hasattr(info.instance, 'unload'):
                    try:
                        info.instance.unload()
                    except Exception as e:
                        logger.warning(f"Fehler beim Entladen von {worker_type}: {e}")

                del self._workers[worker_type]

                # VRAM aufraeumen
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                logger.info(f"Worker {worker_type} entladen")
                return True

            return False

    def unload_all(self):
        """Entlaedt alle Worker"""
        with self._lock:
            for worker_type in list(self._workers.keys()):
                self.unload_worker(worker_type)

    def health_check(self) -> Dict[str, bool]:
        """Fuehrt Health Check fuer alle geladenen Worker durch"""
        results = {}

        with self._lock:
            for worker_type, info in self._workers.items():
                try:
                    if hasattr(info.instance, 'health_check'):
                        results[worker_type.value] = info.instance.health_check()
                    elif hasattr(info.instance, 'status'):
                        results[worker_type.value] = info.instance.status != "error"
                    else:
                        results[worker_type.value] = True
                except Exception:
                    results[worker_type.value] = False

        return results


# Singleton
_worker_manager: Optional[WorkerManager] = None
_manager_lock = threading.Lock()


def get_worker_manager() -> WorkerManager:
    """Gibt die Singleton-Instanz des Worker Managers zurueck"""
    global _worker_manager

    if _worker_manager is None:
        with _manager_lock:
            if _worker_manager is None:
                _worker_manager = WorkerManager()
                _worker_manager.initialize()

    return _worker_manager
