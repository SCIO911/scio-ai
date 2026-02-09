"""
Hardware Monitor
================

Echtzeit-Ueberwachung der Hardware-Auslastung.
"""

import subprocess
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import json


@dataclass
class CPUStats:
    """CPU-Statistiken"""
    usage_percent: float = 0.0
    frequency_mhz: int = 0
    temperature_c: Optional[float] = None
    per_core_usage: List[float] = None

    def __post_init__(self):
        if self.per_core_usage is None:
            self.per_core_usage = []


@dataclass
class RAMStats:
    """RAM-Statistiken"""
    total_gb: float = 0.0
    used_gb: float = 0.0
    available_gb: float = 0.0
    usage_percent: float = 0.0


@dataclass
class GPUStats:
    """GPU-Statistiken"""
    name: str = ""
    usage_percent: float = 0.0
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    memory_percent: float = 0.0
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    fan_speed_percent: Optional[int] = None


@dataclass
class DiskStats:
    """Disk-Statistiken"""
    drive: str = ""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    usage_percent: float = 0.0
    read_speed_mbps: float = 0.0
    write_speed_mbps: float = 0.0


@dataclass
class NetworkStats:
    """Netzwerk-Statistiken"""
    interface: str = ""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    speed_mbps: float = 0.0


@dataclass
class SystemStats:
    """Gesamtsystem-Statistiken"""
    timestamp: float = 0.0
    cpu: CPUStats = None
    ram: RAMStats = None
    gpus: List[GPUStats] = None
    disks: List[DiskStats] = None
    network: List[NetworkStats] = None

    def __post_init__(self):
        if self.cpu is None:
            self.cpu = CPUStats()
        if self.ram is None:
            self.ram = RAMStats()
        if self.gpus is None:
            self.gpus = []
        if self.disks is None:
            self.disks = []
        if self.network is None:
            self.network = []


class HardwareMonitor:
    """Echtzeit Hardware-Monitor"""

    def __init__(self, history_size: int = 60):
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[SystemStats], None]] = []
        self._interval = 1.0

    def _run_powershell(self, command: str) -> str:
        """Fuehrt PowerShell-Befehl aus"""
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _run_command(self, command: str) -> str:
        """Fuehrt Shell-Befehl aus"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def get_cpu_stats(self) -> CPUStats:
        """Holt aktuelle CPU-Statistiken"""
        stats = CPUStats()

        # CPU usage via PowerShell
        ps_cmd = """
        $cpu = Get-CimInstance Win32_Processor
        @{
            Usage = $cpu.LoadPercentage
            Frequency = $cpu.CurrentClockSpeed
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            stats.usage_percent = float(data.get('Usage', 0) or 0)
            stats.frequency_mhz = int(data.get('Frequency', 0) or 0)
        except Exception:
            pass

        return stats

    def get_ram_stats(self) -> RAMStats:
        """Holt aktuelle RAM-Statistiken"""
        stats = RAMStats()

        ps_cmd = """
        $os = Get-CimInstance Win32_OperatingSystem
        @{
            TotalMB = [math]::Round($os.TotalVisibleMemorySize / 1024, 2)
            FreeMB = [math]::Round($os.FreePhysicalMemory / 1024, 2)
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            total_mb = float(data.get('TotalMB', 0))
            free_mb = float(data.get('FreeMB', 0))
            used_mb = total_mb - free_mb

            stats.total_gb = total_mb / 1024
            stats.available_gb = free_mb / 1024
            stats.used_gb = used_mb / 1024
            stats.usage_percent = (used_mb / total_mb * 100) if total_mb > 0 else 0
        except Exception:
            pass

        return stats

    def get_gpu_stats(self) -> List[GPUStats]:
        """Holt aktuelle GPU-Statistiken"""
        gpus = []

        # NVIDIA GPU via nvidia-smi
        output = self._run_command(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,fan.speed --format=csv,noheader,nounits"
        )

        if output and "Error" not in output:
            for line in output.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    try:
                        mem_used = int(parts[2]) if parts[2] != "[N/A]" else 0
                        mem_total = int(parts[3]) if parts[3] != "[N/A]" else 0

                        gpu = GPUStats(
                            name=parts[0],
                            usage_percent=float(parts[1]) if parts[1] != "[N/A]" else 0,
                            memory_used_mb=mem_used,
                            memory_total_mb=mem_total,
                            memory_percent=(mem_used / mem_total * 100) if mem_total > 0 else 0,
                        )

                        if len(parts) > 4 and parts[4] != "[N/A]":
                            gpu.temperature_c = float(parts[4])
                        if len(parts) > 5 and parts[5] != "[N/A]":
                            gpu.power_draw_w = float(parts[5])
                        if len(parts) > 6 and parts[6] != "[N/A]":
                            gpu.fan_speed_percent = int(parts[6])

                        gpus.append(gpu)
                    except (ValueError, IndexError):
                        pass

        return gpus

    def get_disk_stats(self) -> List[DiskStats]:
        """Holt aktuelle Disk-Statistiken"""
        disks = []

        ps_cmd = """
        Get-Volume | Where-Object DriveLetter | ForEach-Object {
            @{
                Drive = $_.DriveLetter
                TotalGB = [math]::Round($_.Size / 1GB, 2)
                FreeGB = [math]::Round($_.SizeRemaining / 1GB, 2)
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]

            for item in data:
                total = float(item.get('TotalGB', 0))
                free = float(item.get('FreeGB', 0))
                used = total - free

                disks.append(DiskStats(
                    drive=f"{item.get('Drive', '?')}:",
                    total_gb=total,
                    used_gb=used,
                    free_gb=free,
                    usage_percent=(used / total * 100) if total > 0 else 0,
                ))
        except Exception:
            pass

        return disks

    def get_network_stats(self) -> List[NetworkStats]:
        """Holt aktuelle Netzwerk-Statistiken"""
        networks = []

        ps_cmd = """
        Get-NetAdapterStatistics | ForEach-Object {
            @{
                Name = $_.Name
                BytesSent = $_.SentBytes
                BytesRecv = $_.ReceivedBytes
                PacketsSent = $_.SentUnicastPackets
                PacketsRecv = $_.ReceivedUnicastPackets
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]

            for item in data:
                networks.append(NetworkStats(
                    interface=item.get('Name', 'Unknown'),
                    bytes_sent=int(item.get('BytesSent', 0)),
                    bytes_recv=int(item.get('BytesRecv', 0)),
                    packets_sent=int(item.get('PacketsSent', 0)),
                    packets_recv=int(item.get('PacketsRecv', 0)),
                ))
        except Exception:
            pass

        return networks

    def get_current_stats(self) -> SystemStats:
        """Holt alle aktuellen Statistiken"""
        return SystemStats(
            timestamp=time.time(),
            cpu=self.get_cpu_stats(),
            ram=self.get_ram_stats(),
            gpus=self.get_gpu_stats(),
            disks=self.get_disk_stats(),
            network=self.get_network_stats(),
        )

    def add_callback(self, callback: Callable[[SystemStats], None]) -> None:
        """Fuegt Callback fuer neue Statistiken hinzu"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SystemStats], None]) -> None:
        """Entfernt Callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self, interval: float = 1.0) -> None:
        """Startet kontinuierliches Monitoring"""
        if self._running:
            return

        self._interval = interval
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stoppt Monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Monitoring-Schleife"""
        while self._running:
            try:
                stats = self.get_current_stats()
                self.history.append(stats)

                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception:
                        pass

            except Exception:
                pass

            time.sleep(self._interval)

    def get_history(self) -> List[SystemStats]:
        """Gibt Monitoring-Historie zurueck"""
        return list(self.history)

    def get_average_stats(self, seconds: int = 60) -> Optional[SystemStats]:
        """Berechnet Durchschnittswerte der letzten n Sekunden"""
        if not self.history:
            return None

        now = time.time()
        relevant = [s for s in self.history if now - s.timestamp <= seconds]

        if not relevant:
            return None

        avg = SystemStats(timestamp=now)
        avg.cpu.usage_percent = sum(s.cpu.usage_percent for s in relevant) / len(relevant)
        avg.ram.usage_percent = sum(s.ram.usage_percent for s in relevant) / len(relevant)
        avg.ram.used_gb = sum(s.ram.used_gb for s in relevant) / len(relevant)

        if relevant[0].gpus:
            avg.gpus = [GPUStats(
                name=relevant[0].gpus[0].name,
                usage_percent=sum(s.gpus[0].usage_percent for s in relevant if s.gpus) / len(relevant),
                memory_used_mb=int(sum(s.gpus[0].memory_used_mb for s in relevant if s.gpus) / len(relevant)),
                memory_total_mb=relevant[0].gpus[0].memory_total_mb,
            )]

        return avg

    def print_status(self) -> str:
        """Gibt formatierten Status-String zurueck"""
        stats = self.get_current_stats()

        lines = [
            "=" * 50,
            "SYSTEM STATUS",
            "=" * 50,
            f"CPU: {stats.cpu.usage_percent:.1f}% @ {stats.cpu.frequency_mhz}MHz",
            f"RAM: {stats.ram.used_gb:.1f}/{stats.ram.total_gb:.1f}GB ({stats.ram.usage_percent:.1f}%)",
        ]

        for gpu in stats.gpus:
            temp_str = f", {gpu.temperature_c}C" if gpu.temperature_c else ""
            power_str = f", {gpu.power_draw_w:.0f}W" if gpu.power_draw_w else ""
            lines.append(
                f"GPU: {gpu.usage_percent:.0f}% | "
                f"VRAM: {gpu.memory_used_mb}/{gpu.memory_total_mb}MB{temp_str}{power_str}"
            )

        for disk in stats.disks:
            lines.append(f"Disk {disk.drive} {disk.used_gb:.0f}/{disk.total_gb:.0f}GB ({disk.usage_percent:.0f}%)")

        lines.append("=" * 50)
        return "\n".join(lines)


# Singleton-Instanz
_monitor_instance: Optional[HardwareMonitor] = None


def get_monitor() -> HardwareMonitor:
    """Holt oder erstellt Monitor-Singleton"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = HardwareMonitor()
    return _monitor_instance
