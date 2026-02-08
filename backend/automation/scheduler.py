#!/usr/bin/env python3
"""
SCIO - Automatischer Scheduler
Führt regelmäßige Tasks automatisch aus
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable, List
import schedule

from backend.config import Config


class AutoScheduler:
    """
    Automatischer Task-Scheduler

    Führt aus:
    - Stündliche Health-Checks
    - Tägliche Earnings-Reports
    - Monatliche API-Key Resets
    - Automatische Model-Downloads
    - GPU-Rental Preis-Optimierung
    - Cleanup alter Daten
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tasks: List[dict] = []

    def add_task(self, name: str, func: Callable, schedule_str: str):
        """
        Fügt Task hinzu

        schedule_str Examples:
        - "every 1 hour"
        - "every day at 09:00"
        - "every monday at 10:00"
        """
        self._tasks.append({
            'name': name,
            'func': func,
            'schedule': schedule_str,
        })

    def _setup_default_tasks(self):
        """Konfiguriert Standard-Tasks"""
        from backend.automation.notifications import get_notification_service

        # Health Check - alle 5 Minuten
        schedule.every(5).minutes.do(self._health_check)

        # Earnings Report - täglich um 20:00
        schedule.every().day.at("20:00").do(self._daily_earnings_report)

        # API Key Reset - 1. des Monats
        schedule.every().day.at("00:01").do(self._check_monthly_reset)

        # GPU Rental Preis-Optimierung - alle 6 Stunden
        schedule.every(6).hours.do(self._optimize_gpu_prices)

        # Cleanup - täglich um 03:00
        schedule.every().day.at("03:00").do(self._cleanup_old_data)

        # Model Cache Check - alle 12 Stunden
        schedule.every(12).hours.do(self._check_model_cache)

        print("[OK] Scheduler Tasks konfiguriert")

    def _health_check(self):
        """Prüft System-Gesundheit"""
        try:
            from backend.services.hardware_monitor import get_hardware_monitor
            from backend.automation.notifications import get_notification_service

            monitor = get_hardware_monitor()
            status = monitor.get_status()
            notifier = get_notification_service()

            # GPU Temperatur-Check
            for gpu in status.gpus:
                if gpu.temperature > 85:
                    notifier.notify_system_warning(
                        "GPU Temperatur Hoch",
                        f"{gpu.name}: {gpu.temperature}°C"
                    )

                if gpu.vram_usage_percent > 95:
                    notifier.notify_system_warning(
                        "VRAM Fast Voll",
                        f"{gpu.name}: {gpu.vram_usage_percent:.0f}% verwendet"
                    )

            # RAM Check
            if status.ram and status.ram.usage_percent > 90:
                notifier.notify_system_warning(
                    "RAM Fast Voll",
                    f"{status.ram.usage_percent:.0f}% verwendet"
                )

        except Exception as e:
            print(f"[ERROR] Health Check Fehler: {e}")

    def _daily_earnings_report(self):
        """Sendet täglichen Einnahmen-Report"""
        try:
            from backend.models import SessionLocal
            from backend.models.earnings import Earning
            from backend.automation.notifications import get_notification_service

            db = SessionLocal()
            notifier = get_notification_service()

            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=now.weekday())
            month_start = today_start.replace(day=1)

            # Query earnings
            today_earnings = db.query(Earning).filter(
                Earning.created_at >= today_start,
                Earning.earning_type == 'income'
            ).all()

            week_earnings = db.query(Earning).filter(
                Earning.created_at >= week_start,
                Earning.earning_type == 'income'
            ).all()

            month_earnings = db.query(Earning).filter(
                Earning.created_at >= month_start,
                Earning.earning_type == 'income'
            ).all()

            today_total = sum(e.amount_cents for e in today_earnings) / 100
            week_total = sum(e.amount_cents for e in week_earnings) / 100
            month_total = sum(e.amount_cents for e in month_earnings) / 100

            db.close()

            notifier.notify_earnings_daily(today_total, week_total, month_total)

        except Exception as e:
            print(f"[ERROR] Earnings Report Fehler: {e}")

    def _check_monthly_reset(self):
        """Prüft ob monatlicher API-Key Reset nötig"""
        if datetime.now().day != 1:
            return

        try:
            from backend.services.api_keys import get_api_key_service

            api_service = get_api_key_service()
            api_service.reset_monthly_usage()
            print("[OK] Monatlicher API-Key Reset durchgeführt")

        except Exception as e:
            print(f"[ERROR] Monthly Reset Fehler: {e}")

    def _optimize_gpu_prices(self):
        """Optimiert GPU-Rental Preise automatisch"""
        try:
            from backend.integrations.vastai import get_vastai

            vastai = get_vastai()
            if vastai._enabled:
                vastai.auto_price_machines()
                print("[OK] Vast.ai Preise optimiert")

        except Exception as e:
            print(f"[ERROR] GPU Preis-Optimierung Fehler: {e}")

    def _cleanup_old_data(self):
        """Bereinigt alte Daten"""
        try:
            from pathlib import Path
            import shutil

            # Lösche generierte Bilder älter als 7 Tage
            gen_dir = Config.DATA_DIR / 'generated'
            if gen_dir.exists():
                cutoff = datetime.now().timestamp() - (7 * 24 * 60 * 60)
                for item in gen_dir.iterdir():
                    if item.is_dir() and item.stat().st_mtime < cutoff:
                        shutil.rmtree(item)

            # Lösche alte Logs älter als 30 Tage
            logs_dir = Config.LOGS_DIR
            if logs_dir.exists():
                cutoff = datetime.now().timestamp() - (30 * 24 * 60 * 60)
                for item in logs_dir.iterdir():
                    if item.is_file() and item.stat().st_mtime < cutoff:
                        item.unlink()

            print("[OK] Alte Daten bereinigt")

        except Exception as e:
            print(f"[ERROR] Cleanup Fehler: {e}")

    def _check_model_cache(self):
        """Prüft Model-Cache"""
        try:
            from backend.workers.base_worker import model_manager

            # Wenn zu viele Modelle geladen, älteste entladen
            if len(model_manager._loaded_models) > model_manager.max_models:
                model_manager._evict_oldest()
                print("[OK] Model-Cache bereinigt")

        except Exception as e:
            print(f"[ERROR] Model Cache Check Fehler: {e}")

    def _run_loop(self):
        """Scheduler Loop"""
        while self._running:
            schedule.run_pending()
            time.sleep(1)

    def start(self):
        """Startet den Scheduler"""
        if self._running:
            return

        self._setup_default_tasks()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[OK] Auto-Scheduler gestartet")

    def stop(self):
        """Stoppt den Scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("[STOP] Auto-Scheduler gestoppt")


# Singleton
_scheduler: Optional[AutoScheduler] = None


def get_scheduler() -> AutoScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AutoScheduler()
    return _scheduler
