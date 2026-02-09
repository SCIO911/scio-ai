#!/usr/bin/env python3
"""
SCIO - Autonomous Money Maker

Automatisches Geldverdienen durch:
- GPU-Vermietung auf Vast.ai (HOST-ONLY)
- Automatische Preisoptimierung
- Earnings-Tracking & Reporting
- Intelligente Ressourcen-Allokation
"""

import time
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from backend.config import Config
from backend.automation.notifications import get_notification_service


@dataclass
class EarningsReport:
    """Earnings-Bericht"""
    period_start: datetime
    period_end: datetime
    vastai_earnings: float = 0.0
    job_earnings: float = 0.0
    total_earnings: float = 0.0
    gpu_hours_rented: float = 0.0
    jobs_completed: int = 0
    avg_price_per_hour: float = 0.0


@dataclass
class MoneyMakerStats:
    """Statistiken für MoneyMaker"""
    total_earnings_usd: float = 0.0
    today_earnings_usd: float = 0.0
    this_week_earnings_usd: float = 0.0
    this_month_earnings_usd: float = 0.0
    gpu_rental_hours: float = 0.0
    jobs_processed: int = 0
    uptime_hours: float = 0.0
    current_hourly_rate: float = 0.0
    estimated_daily: float = 0.0
    estimated_monthly: float = 0.0
    last_earning_event: Optional[datetime] = None


class MoneyMaker:
    """
    SCIO Autonomous Money Maker

    Verdient automatisch Geld durch GPU-Vermietung.
    Optimiert Preise und maximiert Einnahmen.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = MoneyMakerStats()
        self._start_time: Optional[datetime] = None
        self._earnings_log: List[Dict] = []
        self._price_history: List[Dict] = []

        # Konfiguration
        self._min_price = Config.VASTAI_MIN_PRICE
        self._max_price = Config.VASTAI_MAX_PRICE
        self._target_utilization = 0.7  # 70% Ziel-Auslastung

        # Earnings-Datei
        self._earnings_file = Config.DATA_DIR / 'earnings_history.json'
        self._load_earnings_history()

    def _load_earnings_history(self):
        """Lädt Earnings-Historie"""
        try:
            if self._earnings_file.exists():
                with open(self._earnings_file, 'r') as f:
                    data = json.load(f)
                    self._earnings_log = data.get('earnings', [])
                    self._stats.total_earnings_usd = data.get('total_earnings', 0.0)
                    print(f"[MONEY] Earnings-Historie geladen: ${self._stats.total_earnings_usd:.2f} total")
        except Exception as e:
            print(f"[WARN] Earnings-Historie laden fehlgeschlagen: {e}")

    def _save_earnings_history(self):
        """Speichert Earnings-Historie"""
        try:
            self._earnings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._earnings_file, 'w') as f:
                json.dump({
                    'total_earnings': self._stats.total_earnings_usd,
                    'earnings': self._earnings_log[-1000:],  # Letzte 1000 Events
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"[ERROR] Earnings speichern fehlgeschlagen: {e}")

    def _log_earning(self, source: str, amount: float, details: dict = None):
        """Loggt ein Earning-Event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'amount_usd': amount,
            'details': details or {},
        }
        self._earnings_log.append(event)
        self._stats.total_earnings_usd += amount
        self._stats.last_earning_event = datetime.now()

        # Tägliche/Wöchentliche/Monatliche Stats aktualisieren
        self._update_period_stats()

        # Speichern
        self._save_earnings_history()

        print(f"[MONEY] +${amount:.4f} von {source} (Total: ${self._stats.total_earnings_usd:.2f})")

    def _update_period_stats(self):
        """Aktualisiert Perioden-Statistiken"""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)

        self._stats.today_earnings_usd = sum(
            e['amount_usd'] for e in self._earnings_log
            if datetime.fromisoformat(e['timestamp']) >= today_start
        )

        self._stats.this_week_earnings_usd = sum(
            e['amount_usd'] for e in self._earnings_log
            if datetime.fromisoformat(e['timestamp']) >= week_start
        )

        self._stats.this_month_earnings_usd = sum(
            e['amount_usd'] for e in self._earnings_log
            if datetime.fromisoformat(e['timestamp']) >= month_start
        )

    def _optimize_pricing(self):
        """
        Optimiert Vast.ai-Preise basierend auf Marktbedingungen

        Strategie:
        - Bei hoher Nachfrage: Preis erhöhen
        - Bei niedriger Auslastung: Preis senken
        - Immer im konfigurierten Bereich bleiben
        """
        try:
            from backend.integrations.vastai import get_vastai

            vastai = get_vastai()
            if not vastai._enabled:
                return

            machines = vastai.get_my_machines()
            if not machines:
                return

            for machine in machines:
                # Berechne optimalen Preis
                current_price = machine.current_bid
                reliability = machine.reliability
                rentals = machine.rentals_count

                # Basis-Preis nach GPU
                if 'RTX 5090' in machine.gpu_name:
                    base_price = self._max_price
                elif 'RTX 4090' in machine.gpu_name:
                    base_price = self._max_price * 0.9
                elif 'RTX 4080' in machine.gpu_name or 'RTX 3090' in machine.gpu_name:
                    base_price = (self._min_price + self._max_price) / 2
                else:
                    base_price = self._min_price

                # Reliability-Bonus
                if reliability > 0.98:
                    base_price *= 1.15
                elif reliability > 0.95:
                    base_price *= 1.10
                elif reliability < 0.80:
                    base_price *= 0.85

                # Erfahrungs-Bonus (viele Rentals = vertrauenswürdig)
                if rentals > 100:
                    base_price *= 1.05
                elif rentals > 50:
                    base_price *= 1.02

                # Preis im Bereich halten
                optimal_price = max(self._min_price, min(self._max_price, base_price))

                # Nur ändern wenn signifikant anders (>5%)
                if abs(optimal_price - current_price) / max(current_price, 0.01) > 0.05:
                    vastai.set_machine_price(machine.machine_id, round(optimal_price, 2))
                    self._price_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'machine_id': machine.machine_id,
                        'old_price': current_price,
                        'new_price': optimal_price,
                    })

        except Exception as e:
            print(f"[ERROR] Preis-Optimierung fehlgeschlagen: {e}")

    def _check_and_activate_rentals(self):
        """Prüft und aktiviert GPU-Vermietung"""
        try:
            from backend.services.hardware_monitor import get_hardware_monitor
            from backend.services.job_queue import get_job_queue
            from backend.integrations.vastai import get_vastai

            monitor = get_hardware_monitor()
            queue = get_job_queue()
            vastai = get_vastai()

            if not vastai._enabled:
                return

            status = monitor.get_status()
            active_jobs = queue.active_job_count
            queued_jobs = getattr(queue, 'queued_job_count', queue.queue_size)

            # GPU-Auslastung berechnen
            if status.gpus:
                avg_gpu_util = sum(g.gpu_utilization for g in status.gpus) / len(status.gpus)
            else:
                avg_gpu_util = 0

            # Entscheidung: Vermieten oder nicht?
            should_rent = (
                active_jobs == 0 and
                queued_jobs == 0 and
                avg_gpu_util < 15  # GPU praktisch idle
            )

            machines = vastai.get_my_machines()

            for machine in machines:
                # Verfügbarkeit setzen
                vastai.set_machine_available(machine.machine_id, should_rent)

            if should_rent and machines:
                self._stats.current_hourly_rate = sum(m.current_bid for m in machines)
                self._stats.estimated_daily = self._stats.current_hourly_rate * 24
                self._stats.estimated_monthly = self._stats.estimated_daily * 30

        except Exception as e:
            print(f"[ERROR] Rental-Aktivierung fehlgeschlagen: {e}")

    def _track_vastai_earnings(self):
        """Trackt Vast.ai-Einnahmen"""
        try:
            from backend.integrations.vastai import get_vastai

            vastai = get_vastai()
            if not vastai._enabled:
                return

            earnings = vastai.get_earnings()

            if earnings.get('enabled') and 'total_earned_usd' in earnings:
                new_total = earnings['total_earned_usd']

                # Prüfe auf neue Einnahmen
                previous_total = getattr(self, '_last_vastai_total', 0)

                if new_total > previous_total:
                    earned = new_total - previous_total
                    self._log_earning('vastai', earned, {
                        'active_instances': earnings.get('active_instances', 0),
                        'hourly_rate': earnings.get('current_hourly_usd', 0),
                    })

                self._last_vastai_total = new_total

                # Stats aktualisieren
                self._stats.current_hourly_rate = earnings.get('current_hourly_usd', 0)
                self._stats.estimated_daily = earnings.get('daily_estimate_usd', 0)
                self._stats.estimated_monthly = earnings.get('monthly_estimate_usd', 0)

        except Exception as e:
            print(f"[ERROR] Earnings-Tracking fehlgeschlagen: {e}")

    def _send_daily_report(self):
        """Sendet täglichen Earnings-Report"""
        try:
            now = datetime.now()

            # Nur um Mitternacht senden
            if now.hour != 0 or now.minute > 5:
                return

            # Prüfe ob heute schon gesendet
            last_report = getattr(self, '_last_daily_report', None)
            if last_report and last_report.date() == now.date():
                return

            self._last_daily_report = now

            report = f"""
SCIO Daily Earnings Report
===========================
Datum: {now.strftime('%Y-%m-%d')}

Gestern verdient: ${self._stats.today_earnings_usd:.2f}
Diese Woche: ${self._stats.this_week_earnings_usd:.2f}
Dieser Monat: ${self._stats.this_month_earnings_usd:.2f}
Gesamt: ${self._stats.total_earnings_usd:.2f}

Aktuelle Rate: ${self._stats.current_hourly_rate:.2f}/h
Geschätzt heute: ${self._stats.estimated_daily:.2f}
Geschätzt Monat: ${self._stats.estimated_monthly:.2f}

GPU-Stunden vermietet: {self._stats.gpu_rental_hours:.1f}h
Jobs verarbeitet: {self._stats.jobs_processed}
"""

            notifier = get_notification_service()
            notifier.send_message("SCIO Daily Report", report)

            print("[REPORT] Täglicher Earnings-Report gesendet")

        except Exception as e:
            print(f"[ERROR] Daily Report fehlgeschlagen: {e}")

    def _run_loop(self):
        """Haupt-Loop für automatisches Geldverdienen"""
        check_interval = 30  # 30 Sekunden
        price_check_counter = 0
        earnings_check_counter = 0

        print("[MONEY] MoneyMaker Loop gestartet - Automatisches Geldverdienen aktiv!")

        while self._running:
            try:
                # GPU-Vermietung aktivieren/deaktivieren
                self._check_and_activate_rentals()

                # Alle 5 Minuten: Preise optimieren
                price_check_counter += 1
                if price_check_counter >= 10:  # 10 * 30s = 5min
                    self._optimize_pricing()
                    price_check_counter = 0

                # Jede Minute: Earnings tracken
                earnings_check_counter += 1
                if earnings_check_counter >= 2:  # 2 * 30s = 1min
                    self._track_vastai_earnings()
                    earnings_check_counter = 0

                # Täglichen Report prüfen
                self._send_daily_report()

                # Uptime aktualisieren
                if self._start_time:
                    self._stats.uptime_hours = (datetime.now() - self._start_time).total_seconds() / 3600

            except Exception as e:
                print(f"[ERROR] MoneyMaker Loop Fehler: {e}")

            time.sleep(check_interval)

    def start(self):
        """Startet den MoneyMaker"""
        if self._running:
            return

        self._running = True
        self._start_time = datetime.now()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        print(f"""
================================================================
  SCIO MONEY MAKER - AKTIVIERT
================================================================
  Automatisches Geldverdienen durch GPU-Vermietung

  Features:
  [OK] Vast.ai GPU-Vermietung (HOST-ONLY)
  [OK] Automatische Preisoptimierung
  [OK] Intelligente Ressourcen-Allokation
  [OK] Echtzeit Earnings-Tracking
  [OK] Taegliche Reports

  Bisherige Einnahmen: ${self._stats.total_earnings_usd:>10.2f}
================================================================
""")

    def stop(self):
        """Stoppt den MoneyMaker"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._save_earnings_history()
        print("[STOP] MoneyMaker gestoppt")

    def get_stats(self) -> MoneyMakerStats:
        """Gibt aktuelle Statistiken zurück"""
        return self._stats

    def get_earnings_report(self, days: int = 7) -> EarningsReport:
        """Erstellt Earnings-Report für die letzten X Tage"""
        now = datetime.now()
        start = now - timedelta(days=days)

        period_earnings = [
            e for e in self._earnings_log
            if datetime.fromisoformat(e['timestamp']) >= start
        ]

        vastai_total = sum(
            e['amount_usd'] for e in period_earnings
            if e['source'] == 'vastai'
        )

        job_total = sum(
            e['amount_usd'] for e in period_earnings
            if e['source'] == 'job'
        )

        return EarningsReport(
            period_start=start,
            period_end=now,
            vastai_earnings=vastai_total,
            job_earnings=job_total,
            total_earnings=vastai_total + job_total,
        )


# Singleton
_money_maker: Optional[MoneyMaker] = None


def get_money_maker() -> MoneyMaker:
    """Gibt MoneyMaker-Singleton zurück"""
    global _money_maker
    if _money_maker is None:
        _money_maker = MoneyMaker()
    return _money_maker
