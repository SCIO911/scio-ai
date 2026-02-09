"""
SCIO Hardware Module
====================

Hardware-Erkennung, Monitoring und Optimierung.
Angepasst an: Schenker XMG NEO (A25)

Hardware-Profil:
- CPU: AMD Ryzen 9 9955HX3D (16C/32T, 128MB L3 3D V-Cache)
- RAM: 96GB DDR5-5600
- GPU: NVIDIA RTX 5090 Laptop (24GB VRAM)
- GPU2: AMD Radeon 610M (Integrated)
- Storage: 16TB NVMe (2x 8TB) + 20GB RAM Disk
- Display: 16" 2560x1600
- Network: Wi-Fi 6E + 2.5GbE
"""

from scio.hardware.detector import HardwareDetector
from scio.hardware.monitor import HardwareMonitor
from scio.hardware.gpu import GPUManager
from scio.hardware.config import HardwareConfig, SYSTEM_PROFILE

__all__ = [
    'HardwareDetector',
    'HardwareMonitor',
    'GPUManager',
    'HardwareConfig',
    'SYSTEM_PROFILE',
]
