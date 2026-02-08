#!/usr/bin/env python3
"""
SCIO - Hardware Monitor Service
Ueberwacht GPU, CPU, RAM in Echtzeit
Verwendet nvidia-smi fuer zuverlaessiges GPU-Monitoring
"""

import time
import threading
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import psutil


@dataclass
class GPUInfo:
    """GPU Informationen"""
    index: int
    name: str
    uuid: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    vram_usage_percent: float
    gpu_utilization: int
    memory_utilization: int
    temperature: int
    power_usage_watts: float
    power_limit_watts: float
    fan_speed: int

    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'name': self.name,
            'uuid': self.uuid,
            'vram': {
                'total_mb': self.vram_total_mb,
                'used_mb': self.vram_used_mb,
                'free_mb': self.vram_free_mb,
                'usage_percent': round(self.vram_usage_percent, 1),
            },
            'utilization': {
                'gpu': self.gpu_utilization,
                'memory': self.memory_utilization,
            },
            'temperature': self.temperature,
            'power': {
                'usage_watts': round(self.power_usage_watts, 1),
                'limit_watts': round(self.power_limit_watts, 1),
            },
            'fan_speed': self.fan_speed,
        }


@dataclass
class CPUInfo:
    """CPU Informationen"""
    usage_percent: float
    usage_per_core: List[float]
    core_count: int
    thread_count: int
    frequency_current_mhz: float
    frequency_max_mhz: float

    def to_dict(self) -> dict:
        return {
            'usage_percent': round(self.usage_percent, 1),
            'usage_per_core': [round(u, 1) for u in self.usage_per_core],
            'core_count': self.core_count,
            'thread_count': self.thread_count,
            'frequency': {
                'current_mhz': round(self.frequency_current_mhz, 0),
                'max_mhz': round(self.frequency_max_mhz, 0),
            },
        }


@dataclass
class RAMInfo:
    """RAM Informationen"""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float

    def to_dict(self) -> dict:
        return {
            'total_gb': round(self.total_gb, 2),
            'used_gb': round(self.used_gb, 2),
            'available_gb': round(self.available_gb, 2),
            'usage_percent': round(self.usage_percent, 1),
        }


@dataclass
class SystemStatus:
    """Gesamter System-Status"""
    timestamp: float
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu: Optional[CPUInfo] = None
    ram: Optional[RAMInfo] = None
    is_busy: bool = False
    active_jobs: int = 0

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'gpus': [gpu.to_dict() for gpu in self.gpus],
            'cpu': self.cpu.to_dict() if self.cpu else None,
            'ram': self.ram.to_dict() if self.ram else None,
            'is_busy': self.is_busy,
            'active_jobs': self.active_jobs,
            'capacity': self.get_capacity(),
        }

    def get_capacity(self) -> dict:
        """Berechnet verfuegbare Kapazitaet"""
        if not self.gpus:
            return {'available': False, 'reason': 'No GPU detected'}

        total_vram_free = sum(gpu.vram_free_mb for gpu in self.gpus)
        max_vram_free = max(gpu.vram_free_mb for gpu in self.gpus)

        can_run_7b = max_vram_free >= 14000
        can_run_13b = max_vram_free >= 26000
        can_run_70b = total_vram_free >= 48000

        return {
            'available': can_run_7b,
            'vram_free_total_gb': round(total_vram_free / 1024, 1),
            'vram_free_max_gpu_gb': round(max_vram_free / 1024, 1),
            'can_run': {
                '7b': can_run_7b,
                '13b': can_run_13b,
                '70b': can_run_70b,
            },
        }


class HardwareMonitor:
    """
    Hardware Monitor Service
    Verwendet nvidia-smi fuer GPU-Monitoring (zuverlaessiger als pynvml)
    """

    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[SystemStatus], None]] = []
        self._last_status: Optional[SystemStatus] = None
        self._nvidia_smi_available = False
        self._active_jobs = 0
        self._gpu_count = 0

        # Check nvidia-smi availability
        self._check_nvidia_smi()

    def _check_nvidia_smi(self):
        """Prueft ob nvidia-smi verfuegbar ist"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self._nvidia_smi_available = True
                self._gpu_count = len(result.stdout.strip().split('\n'))
                print(f"[OK] nvidia-smi verfuegbar - {self._gpu_count} GPU(s) gefunden")
            else:
                print("[WARN] nvidia-smi nicht verfuegbar")
        except Exception as e:
            print(f"[WARN] nvidia-smi Fehler: {e}")

    def add_callback(self, callback: Callable[[SystemStatus], None]):
        """Registriert Callback fuer Status-Updates"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SystemStatus], None]):
        """Entfernt Callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def set_active_jobs(self, count: int):
        """Setzt Anzahl aktiver Jobs"""
        self._active_jobs = count

    def _parse_value(self, text: str, default=0) -> float:
        """Parst numerischen Wert aus Text"""
        if not text or text == 'N/A' or text == '[N/A]':
            return default
        try:
            # Remove units like "MiB", "W", "%", "C"
            clean = text.replace('MiB', '').replace('W', '').replace('%', '').replace('C', '').strip()
            return float(clean)
        except (ValueError, TypeError, AttributeError):
            return default

    def get_gpu_info(self) -> List[GPUInfo]:
        """Liest GPU-Informationen via nvidia-smi"""
        gpus = []

        if not self._nvidia_smi_available:
            return gpus

        try:
            # Query nvidia-smi with XML output for complete data
            result = subprocess.run(
                ['nvidia-smi', '-q', '-x'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return gpus

            root = ET.fromstring(result.stdout)

            for idx, gpu in enumerate(root.findall('gpu')):
                try:
                    name_elem = gpu.find('product_name')
                    uuid_elem = gpu.find('uuid')
                    name = name_elem.text if name_elem is not None and name_elem.text else 'Unknown GPU'
                    uuid = uuid_elem.text if uuid_elem is not None and uuid_elem.text else ''

                    # Memory
                    vram_total = 0
                    vram_used = 0
                    vram_free = 0
                    fb_memory = gpu.find('fb_memory_usage')
                    if fb_memory is not None:
                        t_elem = fb_memory.find('total')
                        u_elem = fb_memory.find('used')
                        f_elem = fb_memory.find('free')
                        if t_elem is not None:
                            vram_total = int(self._parse_value(t_elem.text))
                        if u_elem is not None:
                            vram_used = int(self._parse_value(u_elem.text))
                        if f_elem is not None:
                            vram_free = int(self._parse_value(f_elem.text))

                    # Utilization
                    gpu_util = 0
                    mem_util = 0
                    utilization = gpu.find('utilization')
                    if utilization is not None:
                        gu_elem = utilization.find('gpu_util')
                        if gu_elem is not None:
                            gpu_util = int(self._parse_value(gu_elem.text))
                        mu_elem = utilization.find('memory_util')
                        if mu_elem is not None:
                            mem_util = int(self._parse_value(mu_elem.text))

                    # Temperature
                    temp = 0
                    temperature = gpu.find('temperature')
                    if temperature is not None:
                        temp_elem = temperature.find('gpu_temp')
                        if temp_elem is not None:
                            temp = int(self._parse_value(temp_elem.text))

                    # Power (RTX 5090 uses instant_power_draw instead of power_draw)
                    power = gpu.find('gpu_power_readings')
                    if power is None:
                        power = gpu.find('power_readings')
                    power_draw = 0
                    power_limit = 0
                    if power is not None:
                        # Try instant_power_draw first (RTX 50 series), then power_draw
                        pd_elem = power.find('instant_power_draw')
                        if pd_elem is None:
                            pd_elem = power.find('power_draw')
                        if pd_elem is not None:
                            power_draw = self._parse_value(pd_elem.text)

                        pl_elem = power.find('current_power_limit')
                        if pl_elem is not None:
                            power_limit = self._parse_value(pl_elem.text)
                        if power_limit == 0:
                            pl_elem = power.find('default_power_limit')
                            if pl_elem is not None:
                                power_limit = self._parse_value(pl_elem.text)

                    # Fan (may not exist on all GPUs)
                    fan_speed = 0
                    fan_speed_elem = gpu.find('fan_speed')
                    if fan_speed_elem is not None and fan_speed_elem.text:
                        fan_speed = int(self._parse_value(fan_speed_elem.text))

                    gpus.append(GPUInfo(
                        index=idx,
                        name=name,
                        uuid=uuid,
                        vram_total_mb=vram_total,
                        vram_used_mb=vram_used,
                        vram_free_mb=vram_free,
                        vram_usage_percent=(vram_used / vram_total * 100) if vram_total > 0 else 0,
                        gpu_utilization=gpu_util,
                        memory_utilization=mem_util,
                        temperature=temp,
                        power_usage_watts=power_draw,
                        power_limit_watts=power_limit,
                        fan_speed=fan_speed,
                    ))

                except Exception as e:
                    print(f"[WARN] GPU Parse Error: {e}")
                    continue

        except subprocess.TimeoutExpired:
            print("[WARN] nvidia-smi timeout")
        except Exception as e:
            print(f"[WARN] nvidia-smi error: {e}")

        return gpus

    def get_cpu_info(self) -> CPUInfo:
        """Liest CPU-Informationen"""
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_count = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1

        try:
            cpu_freq = psutil.cpu_freq()
            freq_current = cpu_freq.current if cpu_freq else 0
            freq_max = cpu_freq.max if cpu_freq else 0
        except (AttributeError, RuntimeError, OSError):
            freq_current = 0
            freq_max = 0

        return CPUInfo(
            usage_percent=cpu_percent,
            usage_per_core=cpu_per_core,
            core_count=cpu_count,
            thread_count=cpu_threads,
            frequency_current_mhz=freq_current,
            frequency_max_mhz=freq_max,
        )

    def get_ram_info(self) -> RAMInfo:
        """Liest RAM-Informationen"""
        mem = psutil.virtual_memory()

        return RAMInfo(
            total_gb=mem.total / (1024 ** 3),
            used_gb=mem.used / (1024 ** 3),
            available_gb=mem.available / (1024 ** 3),
            usage_percent=mem.percent,
        )

    def get_status(self) -> SystemStatus:
        """Liest kompletten System-Status"""
        gpus = self.get_gpu_info()
        cpu = self.get_cpu_info()
        ram = self.get_ram_info()

        is_busy = False
        if gpus:
            max_gpu_util = max(gpu.gpu_utilization for gpu in gpus)
            is_busy = max_gpu_util > 50 or self._active_jobs > 0

        status = SystemStatus(
            timestamp=time.time(),
            gpus=gpus,
            cpu=cpu,
            ram=ram,
            is_busy=is_busy,
            active_jobs=self._active_jobs,
        )

        self._last_status = status
        return status

    def _monitor_loop(self):
        """Monitoring Loop"""
        while self._running:
            try:
                status = self.get_status()

                for callback in self._callbacks:
                    try:
                        callback(status)
                    except Exception as e:
                        pass

            except Exception as e:
                pass

            time.sleep(self.update_interval)

    def start(self):
        """Startet den Monitor"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[OK] Hardware Monitor gestartet")

    def stop(self):
        """Stoppt den Monitor"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[STOP] Hardware Monitor gestoppt")

    @property
    def last_status(self) -> Optional[SystemStatus]:
        """Letzter bekannter Status"""
        return self._last_status

    def __del__(self):
        self.stop()


# Singleton Instance
_monitor_instance: Optional[HardwareMonitor] = None


def get_hardware_monitor() -> HardwareMonitor:
    """Gibt Singleton-Instanz zurueck"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = HardwareMonitor()
    return _monitor_instance
