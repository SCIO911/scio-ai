#!/usr/bin/env python3
"""
SCIO - Vollautomatischer Worker
Verarbeitet Jobs automatisch ohne manuelle Eingriffe
"""

import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from backend.config import Config
from backend.models import SessionLocal
from backend.models.job import Job, JobStatus, JobType
from backend.automation.notifications import get_notification_service


class AutoWorker:
    """
    Vollautomatischer Worker

    Features:
    - Automatische Job-Verarbeitung
    - Automatisches Model-Downloading
    - Automatische Ergebnis-Lieferung
    - Auto-Recovery bei Fehlern
    - Automatische Benachrichtigungen
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processed_orders = set()

    def _check_new_orders(self):
        """Prüft auf neue bezahlte Bestellungen"""
        try:
            for order_dir in Config.ORDERS_DIR.iterdir():
                if not order_dir.is_dir():
                    continue

                order_id = order_dir.name
                if order_id in self._processed_orders:
                    continue

                metadata_file = order_dir / 'metadata.json'
                if not metadata_file.exists():
                    continue

                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Nur bezahlte Bestellungen mit Dataset
                if metadata.get('status') != 'paid':
                    continue

                if not any((order_dir / f'dataset{ext}').exists()
                          for ext in ['.jsonl', '.json', '.csv', '.txt']):
                    continue

                # Job bereits erstellt?
                if metadata.get('job_created'):
                    self._processed_orders.add(order_id)
                    continue

                # Neuen Training-Job erstellen
                self._create_training_job(order_id, metadata, order_dir)
                self._processed_orders.add(order_id)

        except Exception as e:
            print(f"[ERROR] Order-Check Fehler: {e}")

    def _create_training_job(self, order_id: str, metadata: dict, order_dir: Path):
        """Erstellt automatisch Training-Job"""
        from backend.services.job_queue import get_job_queue

        # Find dataset
        dataset_path = None
        for ext in ['.jsonl', '.json', '.csv', '.txt']:
            path = order_dir / f'dataset{ext}'
            if path.exists():
                dataset_path = str(path)
                break

        if not dataset_path:
            return

        model_size = metadata.get('model_size', 'llama-7b')
        email = metadata.get('email')

        print(f"[AUTO] Erstelle automatischen Training-Job für {order_id}")

        try:
            queue = get_job_queue()
            job_id = queue.create_job(
                job_type=JobType.LLM_TRAINING,
                input_data={
                    'model_id': model_size,
                    'dataset_path': dataset_path,
                    'order_id': order_id,
                    'epochs': 3,
                    'use_4bit': True,
                },
                user_email=email,
                order_id=order_id,
                priority=10,
            )

            # Update metadata
            metadata['job_created'] = True
            metadata['job_id'] = job_id
            metadata['job_created_at'] = datetime.now().isoformat()

            with open(order_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Benachrichtigung
            notifier = get_notification_service()
            notifier.notify_new_order(
                order_id,
                email,
                metadata.get('amount_eur', 0),
                model_size
            )

            print(f"[OK] Training-Job erstellt: {job_id}")

        except Exception as e:
            print(f"[ERROR] Job-Erstellung fehlgeschlagen: {e}")

    def _monitor_jobs(self):
        """Überwacht laufende Jobs"""
        try:
            db = SessionLocal()

            # Prüfe auf abgeschlossene Jobs
            completed_jobs = db.query(Job).filter(
                Job.status == JobStatus.COMPLETED,
                Job.output_data.isnot(None)
            ).all()

            for job in completed_jobs:
                # Benachrichtigung nur einmal
                if job.output_data.get('notified'):
                    continue

                # Benachrichtigung senden
                notifier = get_notification_service()
                notifier.notify_job_completed(
                    job.job_id,
                    job.job_type.value,
                    job.duration_seconds,
                    job.user_email
                )

                # Als benachrichtigt markieren
                output = job.output_data or {}
                output['notified'] = True
                output['notified_at'] = datetime.utcnow().isoformat()
                job.output_data = output
                db.commit()

            # Prüfe auf fehlgeschlagene Jobs
            failed_jobs = db.query(Job).filter(
                Job.status == JobStatus.FAILED,
                Job.retry_count >= Job.max_retries
            ).all()

            for job in failed_jobs:
                if job.output_data and job.output_data.get('failure_notified'):
                    continue

                notifier = get_notification_service()
                notifier.notify_job_failed(job.job_id, job.error_message or 'Unknown error')

                output = job.output_data or {}
                output['failure_notified'] = True
                job.output_data = output
                db.commit()

            db.close()

        except Exception as e:
            print(f"[ERROR] Job-Monitor Fehler: {e}")

    def _auto_download_models(self):
        """Lädt häufig verwendete Modelle automatisch herunter"""
        try:
            # Prüfe ob torch verfügbar
            try:
                import torch
                from transformers import AutoTokenizer
            except ImportError:
                return

            # Standard-Modelle für Auto-Download
            auto_download = [
                'mistralai/Mistral-7B-Instruct-v0.2',
            ]

            for model_id in auto_download:
                cache_dir = Config.MODELS_DIR / 'cache'

                # Prüfe ob bereits gecached
                if (cache_dir / model_id.replace('/', '--')).exists():
                    continue

                print(f"[DOWNLOAD] Lade Modell automatisch: {model_id}")

                try:
                    # Nur Tokenizer laden (kleiner, schneller)
                    AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=str(cache_dir),
                        trust_remote_code=True
                    )
                    print(f"[OK] Tokenizer gecached: {model_id}")
                except Exception as e:
                    print(f"[WARN]  Model-Download übersprungen: {e}")

        except Exception as e:
            print(f"[ERROR] Auto-Download Fehler: {e}")

    def _activate_gpu_rentals(self):
        """Aktiviert GPU-Rentals automatisch wenn System idle"""
        try:
            from backend.services.hardware_monitor import get_hardware_monitor
            from backend.services.job_queue import get_job_queue
            from backend.integrations.vastai import get_vastai
            from backend.integrations.runpod import get_runpod

            monitor = get_hardware_monitor()
            queue = get_job_queue()
            status = monitor.get_status()

            # Nur wenn keine aktiven Jobs und GPU nicht ausgelastet
            if queue.active_job_count > 0:
                return

            if not status.gpus:
                return

            avg_gpu_util = sum(g.gpu_utilization for g in status.gpus) / len(status.gpus)

            if avg_gpu_util < 10:  # GPU idle
                # Vast.ai aktivieren
                vastai = get_vastai()
                if vastai._enabled:
                    machines = vastai.get_my_machines()
                    for machine in machines:
                        vastai.set_machine_available(machine.machine_id, True)

        except Exception as e:
            print(f"[ERROR] GPU-Rental Aktivierung Fehler: {e}")

    def _run_loop(self):
        """Haupt-Loop"""
        check_interval = 10  # Sekunden

        while self._running:
            try:
                # Neue Bestellungen prüfen
                self._check_new_orders()

                # Jobs überwachen
                self._monitor_jobs()

                # GPU-Rentals aktivieren wenn idle
                self._activate_gpu_rentals()

            except Exception as e:
                print(f"[ERROR] AutoWorker Loop Fehler: {e}")

            time.sleep(check_interval)

    def start(self):
        """Startet den AutoWorker"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Initial: Models vorladen
        threading.Thread(target=self._auto_download_models, daemon=True).start()

        print("[OK] AutoWorker gestartet - Vollautomatischer Betrieb aktiv")

    def stop(self):
        """Stoppt den AutoWorker"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("[STOP] AutoWorker gestoppt")


# Singleton
_auto_worker: Optional[AutoWorker] = None


def get_auto_worker() -> AutoWorker:
    global _auto_worker
    if _auto_worker is None:
        _auto_worker = AutoWorker()
    return _auto_worker
