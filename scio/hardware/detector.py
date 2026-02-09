"""
Hardware-Detektor
=================

Automatische Erkennung aller Hardware-Komponenten.
"""

import subprocess
import platform
import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re


class HardwareDetector:
    """Erkennt und sammelt Hardware-Informationen"""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def detect_all(self) -> Dict[str, Any]:
        """Erkennt alle Hardware-Komponenten"""
        return {
            "system": self.detect_system(),
            "cpu": self.detect_cpu(),
            "ram": self.detect_ram(),
            "gpu": self.detect_gpu(),
            "storage": self.detect_storage(),
            "network": self.detect_network(),
            "display": self.detect_display(),
            "audio": self.detect_audio(),
            "usb": self.detect_usb(),
        }

    def _run_powershell(self, command: str) -> str:
        """Fuehrt PowerShell-Befehl aus"""
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"

    def _run_command(self, command: str) -> str:
        """Fuehrt Shell-Befehl aus"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"

    def detect_system(self) -> Dict[str, Any]:
        """Erkennt System-Informationen"""
        ps_cmd = """
        $cs = Get-CimInstance Win32_ComputerSystem
        $os = Get-CimInstance Win32_OperatingSystem
        $bios = Get-CimInstance Win32_BIOS
        @{
            Hostname = $cs.Name
            Manufacturer = $cs.Manufacturer
            Model = $cs.Model
            SystemType = $cs.SystemType
            OSName = $os.Caption
            OSVersion = $os.Version
            OSBuild = $os.BuildNumber
            BIOSVersion = $bios.SMBIOSBIOSVersion
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            return json.loads(output)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {
                "Hostname": platform.node(),
                "OSName": platform.system(),
                "OSVersion": platform.version(),
            }

    def detect_cpu(self) -> Dict[str, Any]:
        """Erkennt CPU-Informationen"""
        ps_cmd = """
        $cpu = Get-CimInstance Win32_Processor
        @{
            Name = $cpu.Name
            Manufacturer = $cpu.Manufacturer
            Cores = $cpu.NumberOfCores
            Threads = $cpu.NumberOfLogicalProcessors
            MaxClockMHz = $cpu.MaxClockSpeed
            CurrentClockMHz = $cpu.CurrentClockSpeed
            L2CacheKB = $cpu.L2CacheSize
            L3CacheKB = $cpu.L3CacheSize
            Architecture = $cpu.Architecture
            AddressWidth = $cpu.AddressWidth
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            return json.loads(output)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"Cores": os.cpu_count()}

    def detect_ram(self) -> Dict[str, Any]:
        """Erkennt RAM-Informationen"""
        ps_cmd = """
        $mem = Get-CimInstance Win32_PhysicalMemory
        $total = ($mem | Measure-Object Capacity -Sum).Sum
        @{
            TotalGB = [math]::Round($total / 1GB, 2)
            Modules = @($mem | ForEach-Object {
                @{
                    Manufacturer = $_.Manufacturer
                    PartNumber = $_.PartNumber.Trim()
                    CapacityGB = [math]::Round($_.Capacity / 1GB, 2)
                    SpeedMHz = $_.Speed
                    FormFactor = $_.FormFactor
                    DeviceLocator = $_.DeviceLocator
                }
            })
        } | ConvertTo-Json -Depth 3
        """
        try:
            output = self._run_powershell(ps_cmd)
            return json.loads(output)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}

    def detect_gpu(self) -> List[Dict[str, Any]]:
        """Erkennt GPU-Informationen"""
        gpus = []

        # Windows GPU detection
        ps_cmd = """
        Get-CimInstance Win32_VideoController | ForEach-Object {
            @{
                Name = $_.Name
                AdapterRAM = $_.AdapterRAM
                DriverVersion = $_.DriverVersion
                VideoProcessor = $_.VideoProcessor
                Status = $_.Status
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            gpus.extend(data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # NVIDIA-specific
        nvidia_info = self._detect_nvidia_gpu()
        if nvidia_info:
            for i, gpu in enumerate(gpus):
                if "NVIDIA" in gpu.get("Name", ""):
                    gpus[i].update(nvidia_info)

        return gpus

    def _detect_nvidia_gpu(self) -> Optional[Dict[str, Any]]:
        """NVIDIA GPU-spezifische Informationen"""
        try:
            output = self._run_command(
                "nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,"
                "driver_version,temperature.gpu,power.draw,utilization.gpu,"
                "compute_cap --format=csv,noheader,nounits"
            )
            if "Error" not in output and output:
                parts = [p.strip() for p in output.split(",")]
                if len(parts) >= 9:
                    return {
                        "NvidiaName": parts[0],
                        "VRAMTotalMB": int(parts[1]),
                        "VRAMFreeMB": int(parts[2]),
                        "VRAMUsedMB": int(parts[3]),
                        "DriverVersion": parts[4],
                        "TemperatureC": int(parts[5]) if parts[5] != "[N/A]" else None,
                        "PowerDrawW": float(parts[6]) if parts[6] != "[N/A]" else None,
                        "UtilizationPercent": int(parts[7]) if parts[7] != "[N/A]" else None,
                        "ComputeCapability": parts[8],
                        "CUDAAvailable": True,
                    }
        except (ValueError, IndexError, TypeError):
            pass
        return None

    def detect_storage(self) -> List[Dict[str, Any]]:
        """Erkennt Speicher-Informationen"""
        ps_cmd = """
        Get-CimInstance Win32_DiskDrive | ForEach-Object {
            @{
                Model = $_.Model
                SizeGB = [math]::Round($_.Size / 1GB, 2)
                MediaType = $_.MediaType
                InterfaceType = $_.InterfaceType
                SerialNumber = $_.SerialNumber
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return data
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def detect_partitions(self) -> List[Dict[str, Any]]:
        """Erkennt Partitionen und Volumes"""
        ps_cmd = """
        Get-Volume | Where-Object DriveLetter | ForEach-Object {
            @{
                DriveLetter = $_.DriveLetter
                FileSystem = $_.FileSystemType
                SizeGB = [math]::Round($_.Size / 1GB, 2)
                FreeGB = [math]::Round($_.SizeRemaining / 1GB, 2)
                Label = $_.FileSystemLabel
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return data
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def detect_network(self) -> List[Dict[str, Any]]:
        """Erkennt Netzwerk-Adapter"""
        ps_cmd = """
        Get-NetAdapter | Where-Object Status -eq 'Up' | ForEach-Object {
            @{
                Name = $_.Name
                InterfaceDescription = $_.InterfaceDescription
                MacAddress = $_.MacAddress
                LinkSpeedMbps = [math]::Round($_.LinkSpeed.Split()[0])
                Status = $_.Status
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return data
        except (json.JSONDecodeError, ValueError, TypeError, ImportError):
            return []

    def detect_display(self) -> List[Dict[str, Any]]:
        """Erkennt Displays"""
        ps_cmd = """
        Get-CimInstance Win32_DesktopMonitor | ForEach-Object {
            @{
                Name = $_.Name
                ScreenWidth = $_.ScreenWidth
                ScreenHeight = $_.ScreenHeight
                PPI = $_.PixelsPerXLogicalInch
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return data
        except (json.JSONDecodeError, ValueError, TypeError, ImportError):
            return []

    def detect_audio(self) -> List[Dict[str, Any]]:
        """Erkennt Audio-Geraete"""
        ps_cmd = """
        Get-CimInstance Win32_SoundDevice | ForEach-Object {
            @{
                Name = $_.Name
                Manufacturer = $_.Manufacturer
                Status = $_.Status
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return data
        except (json.JSONDecodeError, ValueError, TypeError, ImportError):
            return []

    def detect_usb(self) -> List[Dict[str, Any]]:
        """Erkennt USB-Geraete"""
        ps_cmd = """
        Get-PnpDevice -Class USB | Where-Object Status -eq 'OK' |
        Select-Object -First 20 | ForEach-Object {
            @{
                Name = $_.FriendlyName
                Status = $_.Status
                InstanceId = $_.InstanceId
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return data
        except (json.JSONDecodeError, ValueError, TypeError, ImportError):
            return []

    def detect_drivers(self) -> List[Dict[str, Any]]:
        """Erkennt installierte Treiber"""
        ps_cmd = """
        Get-CimInstance Win32_PnPSignedDriver |
        Where-Object DeviceName | Select-Object -First 50 |
        ForEach-Object {
            @{
                DeviceName = $_.DeviceName
                DriverVersion = $_.DriverVersion
                Manufacturer = $_.Manufacturer
                DriverDate = $_.DriverDate
            }
        } | ConvertTo-Json
        """
        try:
            output = self._run_powershell(ps_cmd)
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return data
        except (json.JSONDecodeError, ValueError, TypeError, ImportError):
            return []

    def get_hardware_summary(self) -> str:
        """Gibt eine lesbare Hardware-Zusammenfassung zurueck"""
        hw = self.detect_all()

        lines = [
            "=" * 60,
            "SCIO HARDWARE SUMMARY",
            "=" * 60,
            "",
            f"System: {hw['system'].get('Manufacturer', 'Unknown')} {hw['system'].get('Model', '')}",
            f"OS: {hw['system'].get('OSName', 'Unknown')} {hw['system'].get('OSVersion', '')}",
            "",
            "CPU:",
            f"  {hw['cpu'].get('Name', 'Unknown')}",
            f"  Cores: {hw['cpu'].get('Cores', '?')} | Threads: {hw['cpu'].get('Threads', '?')}",
            f"  L3 Cache: {hw['cpu'].get('L3CacheKB', 0) // 1024}MB",
            "",
            "RAM:",
            f"  Total: {hw['ram'].get('TotalGB', '?')}GB",
        ]

        if hw['ram'].get('Modules'):
            for mod in hw['ram']['Modules']:
                lines.append(f"  - {mod.get('Manufacturer', '')} {mod.get('CapacityGB', '')}GB @ {mod.get('SpeedMHz', '')}MHz")

        lines.extend(["", "GPU:"])
        for gpu in hw['gpu']:
            name = gpu.get('Name', 'Unknown')
            vram = gpu.get('VRAMTotalMB', gpu.get('AdapterRAM', 0))
            if isinstance(vram, int) and vram > 0:
                if vram > 1000000000:  # bytes
                    vram = f"{vram // (1024**3)}GB"
                elif vram > 1000:  # MB
                    vram = f"{vram // 1024}GB"
                else:
                    vram = f"{vram}MB"
            lines.append(f"  - {name} ({vram})")

        lines.extend(["", "Storage:"])
        for disk in hw['storage']:
            lines.append(f"  - {disk.get('Model', 'Unknown')}: {disk.get('SizeGB', '?')}GB")

        lines.extend(["", "Network:"])
        for net in hw['network']:
            lines.append(f"  - {net.get('InterfaceDescription', net.get('Name', 'Unknown'))}")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def save_to_json(self, filepath: str) -> None:
        """Speichert Hardware-Info als JSON"""
        hw = self.detect_all()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(hw, f, indent=2, default=str)

    def check_cuda_available(self) -> bool:
        """Prueft ob CUDA verfuegbar ist"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Fallback: Check nvidia-smi
            output = self._run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
            return "NVIDIA" in output

    def get_cuda_devices(self) -> List[Dict[str, Any]]:
        """Holt CUDA-Geraete-Informationen"""
        try:
            import torch
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                })
            return devices
        except (json.JSONDecodeError, ValueError, TypeError, ImportError):
            return []
