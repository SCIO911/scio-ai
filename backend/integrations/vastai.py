#!/usr/bin/env python3
"""
SCIO - Vast.ai Integration
GPU-Rental Plattform
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass

import requests

from backend.config import Config


@dataclass
class VastMachine:
    """Vast.ai Machine Info"""
    machine_id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float
    cpu_cores: int
    ram_gb: float
    disk_gb: float
    inet_up_mbps: float
    inet_down_mbps: float
    min_bid: float
    current_bid: float
    rentals_count: int
    reliability: float


@dataclass
class VastInstance:
    """Vast.ai Instance (laufende Rental)"""
    instance_id: int
    machine_id: int
    status: str  # running, loading, stopping
    gpu_name: str
    num_gpus: int
    price_per_hour: float
    started_at: datetime
    ssh_host: str
    ssh_port: int
    total_earned: float


class VastAIIntegration:
    """
    Vast.ai Integration

    Features:
    - Maschinen-Registrierung
    - Automatisches Pricing
    - Earnings-Tracking
    - Health-Monitoring
    """

    API_BASE = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.VASTAI_API_KEY
        self._enabled = Config.VASTAI_ENABLED and bool(self.api_key)
        self._machines: Dict[int, VastMachine] = {}
        self._instances: Dict[int, VastInstance] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._total_earnings = 0.0
        self._callbacks: List = []

        if not self._enabled:
            print("[WARN]  Vast.ai Integration deaktiviert")

    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Macht API-Request"""
        url = f"{self.API_BASE}/{endpoint}"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        try:
            if method == 'GET':
                resp = requests.get(url, headers=headers, params=data, timeout=30)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                resp = requests.post(url, headers=headers, json=data, timeout=30)
            elif method == 'PUT':
                headers['Content-Type'] = 'application/json'
                resp = requests.put(url, headers=headers, json=data, timeout=30)
            elif method == 'DELETE':
                resp = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unknown method: {method}")

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            print(f"[ERROR] Vast.ai API Fehler: {e}")
            raise

    def add_callback(self, callback):
        """Registriert Callback für Events"""
        self._callbacks.append(callback)

    def _notify(self, event: str, data: dict = None):
        """Benachrichtigt Callbacks"""
        for callback in self._callbacks:
            try:
                callback(event, data or {})
            except:
                pass

    def get_my_machines(self) -> List[VastMachine]:
        """Gibt eigene Maschinen zurück"""
        if not self._enabled:
            return []

        try:
            result = self._request('GET', 'machines/mine')
            machines = []

            for m in result.get('machines', []):
                machine = VastMachine(
                    machine_id=m['id'],
                    gpu_name=m.get('gpu_name', 'Unknown'),
                    num_gpus=m.get('num_gpus', 1),
                    gpu_ram_gb=m.get('gpu_ram', 0) / 1024,
                    cpu_cores=m.get('cpu_cores', 0),
                    ram_gb=m.get('cpu_ram', 0) / 1024,
                    disk_gb=m.get('disk_space', 0),
                    inet_up_mbps=m.get('inet_up', 0),
                    inet_down_mbps=m.get('inet_down', 0),
                    min_bid=m.get('min_bid', 0),
                    current_bid=m.get('dph_total', 0),
                    rentals_count=m.get('rentals_count', 0),
                    reliability=m.get('reliability', 0),
                )
                machines.append(machine)
                self._machines[machine.machine_id] = machine

            return machines

        except Exception as e:
            print(f"[ERROR] Maschinen abrufen fehlgeschlagen: {e}")
            return []

    def get_my_instances(self) -> List[VastInstance]:
        """Gibt aktive Instanzen (Rentals) zurück"""
        if not self._enabled:
            return []

        try:
            result = self._request('GET', 'instances', {'owner': 'me'})
            instances = []

            for i in result.get('instances', []):
                instance = VastInstance(
                    instance_id=i['id'],
                    machine_id=i.get('machine_id', 0),
                    status=i.get('actual_status', 'unknown'),
                    gpu_name=i.get('gpu_name', 'Unknown'),
                    num_gpus=i.get('num_gpus', 1),
                    price_per_hour=i.get('dph_total', 0),
                    started_at=datetime.fromisoformat(i['start_date'].replace('Z', '+00:00'))
                    if i.get('start_date') else datetime.now(),
                    ssh_host=i.get('ssh_host', ''),
                    ssh_port=i.get('ssh_port', 22),
                    total_earned=i.get('total_earned', 0),
                )
                instances.append(instance)
                self._instances[instance.instance_id] = instance

            return instances

        except Exception as e:
            print(f"[ERROR] Instanzen abrufen fehlgeschlagen: {e}")
            return []

    def set_machine_price(self, machine_id: int, price_per_hour: float) -> bool:
        """Setzt Preis für Maschine"""
        if not self._enabled:
            return False

        try:
            self._request('PUT', f'machines/{machine_id}/', {
                'min_bid': price_per_hour,
            })
            print(f"[MONEY] Preis für Maschine {machine_id} auf ${price_per_hour:.2f}/h gesetzt")
            return True

        except Exception as e:
            print(f"[ERROR] Preis setzen fehlgeschlagen: {e}")
            return False

    def set_machine_available(self, machine_id: int, available: bool) -> bool:
        """Setzt Verfügbarkeit einer Maschine"""
        if not self._enabled:
            return False

        try:
            self._request('PUT', f'machines/{machine_id}/', {
                'listed': available,
            })
            status = "verfügbar" if available else "offline"
            print(f"📌 Maschine {machine_id} ist jetzt {status}")
            return True

        except Exception as e:
            print(f"[ERROR] Verfügbarkeit setzen fehlgeschlagen: {e}")
            return False

    def auto_price_machines(self):
        """Passt Preise automatisch an"""
        if not self._enabled:
            return

        machines = self.get_my_machines()

        for machine in machines:
            # Calculate optimal price based on GPU and specs
            base_price = Config.VASTAI_MIN_PRICE

            # Adjust for GPU
            if 'RTX 5090' in machine.gpu_name or 'RTX 4090' in machine.gpu_name:
                base_price = Config.VASTAI_MAX_PRICE
            elif 'RTX 4080' in machine.gpu_name or 'RTX 3090' in machine.gpu_name:
                base_price = (Config.VASTAI_MIN_PRICE + Config.VASTAI_MAX_PRICE) / 2
            elif machine.gpu_ram_gb >= 24:
                base_price = Config.VASTAI_MAX_PRICE * 0.8

            # Adjust for reliability
            if machine.reliability > 0.95:
                base_price *= 1.1
            elif machine.reliability < 0.8:
                base_price *= 0.9

            # Set price
            self.set_machine_price(machine.machine_id, round(base_price, 2))

    def get_earnings(self) -> dict:
        """Gibt Earnings-Übersicht zurück"""
        if not self._enabled:
            return {'enabled': False}

        try:
            # Get account info
            result = self._request('GET', 'users/current')

            balance = result.get('balance', 0)
            total_earned = result.get('total_earned', 0)

            # Calculate from instances
            instances = self.get_my_instances()
            current_hourly = sum(i.price_per_hour for i in instances if i.status == 'running')
            daily_estimate = current_hourly * 24
            monthly_estimate = daily_estimate * 30

            return {
                'enabled': True,
                'balance_usd': balance,
                'total_earned_usd': total_earned,
                'current_hourly_usd': round(current_hourly, 2),
                'daily_estimate_usd': round(daily_estimate, 2),
                'monthly_estimate_usd': round(monthly_estimate, 2),
                'active_instances': len([i for i in instances if i.status == 'running']),
                'total_machines': len(self._machines),
            }

        except Exception as e:
            print(f"[ERROR] Earnings abrufen fehlgeschlagen: {e}")
            return {'enabled': True, 'error': str(e)}

    def _monitor_loop(self):
        """Monitoring Loop"""
        while self._running:
            try:
                # Refresh instances
                instances = self.get_my_instances()

                # Check for new rentals
                for instance in instances:
                    if instance.status == 'running':
                        self._notify('rental_active', {
                            'instance_id': instance.instance_id,
                            'gpu_name': instance.gpu_name,
                            'price_per_hour': instance.price_per_hour,
                        })

                # Update earnings
                earnings = self.get_earnings()
                self._total_earnings = earnings.get('total_earned_usd', 0)

            except Exception as e:
                print(f"[WARN]  Vast.ai Monitor Fehler: {e}")

            time.sleep(300)  # 5 Minuten

    def start_monitor(self):
        """Startet Background-Monitor"""
        if not self._enabled or self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("[OK] Vast.ai Monitor gestartet")

    def stop_monitor(self):
        """Stoppt Monitor"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        print("[STOP] Vast.ai Monitor gestoppt")

    def get_status(self) -> dict:
        """Gibt Integration-Status zurück"""
        return {
            'enabled': self._enabled,
            'running': self._running,
            'machines': len(self._machines),
            'active_instances': len([i for i in self._instances.values() if i.status == 'running']),
            'total_earnings_usd': self._total_earnings,
        }


# Singleton Instance
_vastai_instance: Optional[VastAIIntegration] = None


def get_vastai() -> VastAIIntegration:
    """Gibt Singleton-Instanz zurück"""
    global _vastai_instance
    if _vastai_instance is None:
        _vastai_instance = VastAIIntegration()
    return _vastai_instance
