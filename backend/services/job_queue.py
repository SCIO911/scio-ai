#!/usr/bin/env python3
"""
SCIO - Job Queue Service
Verwaltet Job-Queue mit Priority, Retry-Logik und Timeouts

Features:
- Priority-basierte Job-Verarbeitung
- Automatische Retries bei Fehlern
- Job-Timeouts (verhindert hängende Jobs)
- Concurrent Job Limits
- Persistenz in Datenbank
"""

import uuid
import time
import threading
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Dict, Any, Generator
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from sqlalchemy.orm import Session
from backend.models import SessionLocal, Job
from backend.models.job import JobStatus, JobType
from backend.config import Config

# Configure logging
logger = logging.getLogger(__name__)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Ensures proper cleanup and rollback on errors.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@dataclass(order=True)
class QueuedJob:
    """Job in der Queue mit Priority"""
    priority: int
    created_at: float = field(compare=False)
    job_id: str = field(compare=False)
    job_type: JobType = field(compare=False)
    input_data: dict = field(compare=False, default_factory=dict)


class JobQueue:
    """
    Job Queue Service

    Verwaltet Jobs mit:
    - Priority Queue (höhere Priorität = frühere Bearbeitung)
    - Persistenz in Datenbank
    - Retry bei Fehlern
    - Concurrent Job Limits
    """

    def __init__(self, max_concurrent: int = None):
        self.max_concurrent = max_concurrent or Config.MAX_CONCURRENT_JOBS
        self.job_timeout = getattr(Config, 'JOB_TIMEOUT_SECONDS', 86400)  # Default 24h
        self._queue: PriorityQueue = PriorityQueue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._timeout_thread: Optional[threading.Thread] = None
        self._workers: Dict[str, Callable] = {}
        self._active_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> {thread, started_at}
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent or Config.MAX_CONCURRENT_JOBS)

        # Load pending jobs from database
        self._load_pending_jobs()

    def _load_pending_jobs(self):
        """Lädt ausstehende Jobs aus der Datenbank"""
        try:
            with get_db_session() as db:
                pending_jobs = db.query(Job).filter(
                    Job.status.in_([JobStatus.PENDING, JobStatus.QUEUED])
                ).order_by(Job.priority.desc(), Job.created_at).all()

                for job in pending_jobs:
                    self._queue.put(QueuedJob(
                        priority=-job.priority,  # Negative for max-priority behavior
                        created_at=job.created_at.timestamp(),
                        job_id=job.job_id,
                        job_type=job.job_type,
                        input_data=job.input_data or {},
                    ))

                logger.info(f"{len(pending_jobs)} ausstehende Jobs geladen")
        except Exception as e:
            logger.warning(f"Jobs laden fehlgeschlagen: {e}")

    def register_worker(self, job_type: JobType, worker_func: Callable):
        """
        Registriert Worker für Job-Typ

        Args:
            job_type: Art des Jobs
            worker_func: Funktion die den Job ausführt
                         Signatur: (job_id: str, input_data: dict) -> dict
        """
        self._workers[job_type.value] = worker_func
        logger.info(f"Worker registriert: {job_type.value}")

    def add_callback(self, callback: Callable):
        """Registriert Callback für Job-Events"""
        self._callbacks.append(callback)

    def _notify(self, event: str, job_id: str, data: dict = None):
        """Benachrichtigt alle Callbacks"""
        for callback in self._callbacks:
            try:
                callback(event, job_id, data or {})
            except Exception as e:
                print(f"[ERROR] Callback Fehler: {e}")

    def create_job(
        self,
        job_type: JobType,
        input_data: dict,
        user_email: str = None,
        api_key_id: int = None,
        order_id: str = None,
        priority: int = 0,
    ) -> str:
        """
        Erstellt neuen Job

        Args:
            job_type: Art des Jobs
            input_data: Eingabedaten
            user_email: Optional - User Email
            api_key_id: Optional - API Key ID
            order_id: Optional - Order ID (für Stripe-Zahlungen)
            priority: Priorität (höher = früher)

        Returns:
            job_id: Eindeutige Job-ID
        """
        job_id = f"job_{uuid.uuid4().hex[:16]}"

        # Save to database
        try:
            with get_db_session() as db:
                job = Job(
                    job_id=job_id,
                    order_id=order_id,
                    job_type=job_type,
                    status=JobStatus.QUEUED,
                    priority=priority,
                    user_email=user_email,
                    api_key_id=api_key_id,
                    input_data=input_data,
                    max_retries=Config.MAX_RETRIES,
                )
                db.add(job)

            # Add to queue (after successful DB commit)
            self._queue.put(QueuedJob(
                priority=-priority,
                created_at=time.time(),
                job_id=job_id,
                job_type=job_type,
                input_data=input_data,
            ))

            logger.info(f"Job erstellt: {job_id} ({job_type.value})")
            self._notify('job_created', job_id, {'job_type': job_type.value})

            return job_id

        except Exception as e:
            logger.error(f"Job erstellen fehlgeschlagen: {e}")
            raise

    def get_job(self, job_id: str) -> Optional[dict]:
        """Gibt Job-Informationen zurück"""
        with get_db_session() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                return job.to_dict()
            return None

    def get_jobs(
        self,
        status: JobStatus = None,
        job_type: JobType = None,
        user_email: str = None,
        limit: int = 100,
    ) -> List[dict]:
        """Gibt gefilterte Jobs zurück"""
        with get_db_session() as db:
            query = db.query(Job)

            if status:
                query = query.filter(Job.status == status)
            if job_type:
                query = query.filter(Job.job_type == job_type)
            if user_email:
                query = query.filter(Job.user_email == user_email)

            jobs = query.order_by(Job.created_at.desc()).limit(limit).all()
            return [job.to_dict() for job in jobs]

    def cancel_job(self, job_id: str) -> bool:
        """Bricht Job ab"""
        try:
            with get_db_session() as db:
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if not job:
                    return False

                if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.utcnow()
                    logger.info(f"Job abgebrochen: {job_id}")
                    self._notify('job_cancelled', job_id)
                    return True

                return False
        except Exception as e:
            logger.error(f"Job abbrechen fehlgeschlagen: {e}")
            return False

    def _execute_job(self, queued_job: QueuedJob):
        """Führt einen Job aus"""
        job_id = queued_job.job_id
        job_type = queued_job.job_type.value

        try:
            # Update status to running
            with get_db_session() as db:
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if not job or job.status == JobStatus.CANCELLED:
                    return

                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()

            logger.info(f"Job gestartet: {job_id}")
            self._notify('job_started', job_id)

            # Find worker
            worker = self._workers.get(job_type)
            if not worker:
                raise ValueError(f"Kein Worker für {job_type} registriert")

            # Execute worker
            result = worker(job_id, queued_job.input_data)

            # Success - update in separate transaction
            with get_db_session() as db:
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    job.output_data = result

                    # Update token counts if present
                    if result and 'tokens_input' in result:
                        job.tokens_input = result.get('tokens_input', 0)
                    if result and 'tokens_output' in result:
                        job.tokens_output = result.get('tokens_output', 0)

            logger.info(f"Job abgeschlossen: {job_id}")
            self._notify('job_completed', job_id, result)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Job fehlgeschlagen: {job_id} - {error_msg}")

            # Update job with error in separate transaction
            try:
                with get_db_session() as db:
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        job.retry_count += 1
                        job.error_message = error_msg

                        if job.retry_count < job.max_retries:
                            # Retry
                            job.status = JobStatus.QUEUED
                            retry_count = job.retry_count
                            priority = job.priority
                        else:
                            # Failed
                            job.status = JobStatus.FAILED
                            job.completed_at = datetime.utcnow()
                            retry_count = None
                            priority = None

                # Re-queue outside of transaction if retry needed
                if retry_count is not None:
                    self._queue.put(QueuedJob(
                        priority=-(priority - 1),
                        created_at=time.time(),
                        job_id=job_id,
                        job_type=queued_job.job_type,
                        input_data=queued_job.input_data,
                    ))
                    logger.info(f"Job wird wiederholt: {job_id} (Versuch {retry_count})")
            except Exception as db_error:
                logger.error(f"Fehler beim Aktualisieren des fehlgeschlagenen Jobs: {db_error}")

            self._notify('job_failed', job_id, {'error': error_msg})

        finally:
            # Remove from active jobs
            with self._lock:
                if job_id in self._active_jobs:
                    del self._active_jobs[job_id]

    def _process_loop(self):
        """Haupt-Processing Loop"""
        while self._running:
            try:
                # Check if we can run more jobs
                with self._lock:
                    active_count = len(self._active_jobs)

                if active_count >= self.max_concurrent:
                    time.sleep(0.5)
                    continue

                # Try to get next job (non-blocking)
                try:
                    queued_job = self._queue.get(timeout=1.0)
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error getting job from queue: {e}")
                    continue

                # Check if job is still valid
                with get_db_session() as db:
                    job = db.query(Job).filter(Job.job_id == queued_job.job_id).first()
                    job_valid = job and job.status != JobStatus.CANCELLED

                if not job_valid:
                    continue

                # Start job thread
                thread = threading.Thread(
                    target=self._execute_job,
                    args=(queued_job,),
                    daemon=True
                )
                thread.start()

                with self._lock:
                    self._active_jobs[queued_job.job_id] = {
                        'thread': thread,
                        'started_at': time.time(),
                        'job_type': queued_job.job_type.value,
                    }

            except Exception as e:
                logger.error(f"Process Loop Fehler: {e}")
                time.sleep(1)

    def _check_timeouts(self):
        """Überwacht Jobs auf Timeouts"""
        while self._running:
            try:
                time.sleep(30)  # Check every 30 seconds

                current_time = time.time()
                timed_out_jobs = []

                with self._lock:
                    for job_id, job_info in self._active_jobs.items():
                        elapsed = current_time - job_info['started_at']
                        if elapsed > self.job_timeout:
                            timed_out_jobs.append(job_id)

                # Handle timed out jobs
                for job_id in timed_out_jobs:
                    self._handle_timeout(job_id)

            except Exception as e:
                logger.error(f"Timeout check Fehler: {e}")

    def _handle_timeout(self, job_id: str):
        """Behandelt einen Job-Timeout"""
        logger.warning(f"Job timeout: {job_id}")

        try:
            with get_db_session() as db:
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job and job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Job timed out after {self.job_timeout} seconds"
                    job.completed_at = datetime.utcnow()

            logger.warning(f"Job abgebrochen wegen Timeout: {job_id}")
            self._notify('job_timeout', job_id, {'timeout_seconds': self.job_timeout})

        except Exception as e:
            logger.error(f"Timeout handling Fehler für {job_id}: {e}")
        finally:
            # Remove from active jobs
            with self._lock:
                if job_id in self._active_jobs:
                    del self._active_jobs[job_id]

    def start(self):
        """Startet die Job-Queue"""
        if self._running:
            return

        self._running = True

        # Start main processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        # Start timeout monitoring thread
        self._timeout_thread = threading.Thread(target=self._check_timeouts, daemon=True)
        self._timeout_thread.start()

        logger.info(f"Job Queue gestartet (max {self.max_concurrent} concurrent, timeout {self.job_timeout}s)")

    def stop(self):
        """Stoppt die Job-Queue"""
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._timeout_thread:
            self._timeout_thread.join(timeout=5.0)
            self._timeout_thread = None

        # Shutdown executor
        self._executor.shutdown(wait=False)

        logger.info("Job Queue gestoppt")

    @property
    def queue_size(self) -> int:
        """Anzahl Jobs in der Queue"""
        return self._queue.qsize()

    @property
    def active_job_count(self) -> int:
        """Anzahl aktiver Jobs"""
        with self._lock:
            return len(self._active_jobs)

    def get_stats(self) -> dict:
        """Gibt Queue-Statistiken zurück"""
        with get_db_session() as db:
            total = db.query(Job).count()
            pending = db.query(Job).filter(Job.status == JobStatus.PENDING).count()
            queued = db.query(Job).filter(Job.status == JobStatus.QUEUED).count()
            running = db.query(Job).filter(Job.status == JobStatus.RUNNING).count()
            completed = db.query(Job).filter(Job.status == JobStatus.COMPLETED).count()
            failed = db.query(Job).filter(Job.status == JobStatus.FAILED).count()
            timed_out = db.query(Job).filter(
                Job.error_message.like('%timed out%')
            ).count()

            return {
                'queue_size': self.queue_size,
                'active_jobs': self.active_job_count,
                'max_concurrent': self.max_concurrent,
                'job_timeout_seconds': self.job_timeout,
                'total_jobs': total,
                'by_status': {
                    'pending': pending,
                    'queued': queued,
                    'running': running,
                    'completed': completed,
                    'failed': failed,
                    'timed_out': timed_out,
                },
                'active_job_details': self.get_active_job_details(),
            }

    def get_active_job_details(self) -> List[dict]:
        """Gibt Details zu allen aktiven Jobs zurück"""
        current_time = time.time()
        details = []

        with self._lock:
            for job_id, job_info in self._active_jobs.items():
                elapsed = current_time - job_info['started_at']
                remaining = max(0, self.job_timeout - elapsed)
                details.append({
                    'job_id': job_id,
                    'job_type': job_info['job_type'],
                    'running_seconds': round(elapsed, 1),
                    'timeout_remaining': round(remaining, 1),
                    'percent_timeout': round((elapsed / self.job_timeout) * 100, 1),
                })

        return details


# Singleton Instance
_queue_instance: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Gibt Singleton-Instanz zurück"""
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = JobQueue()
    return _queue_instance
