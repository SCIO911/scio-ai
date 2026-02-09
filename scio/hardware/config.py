"""
Hardware-Konfiguration
======================

Optimierte Einstellungen basierend auf der erkannten Hardware.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class GPUType(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    NONE = "none"


class StorageType(Enum):
    NVME = "nvme"
    SSD = "ssd"
    HDD = "hdd"
    RAMDISK = "ramdisk"


@dataclass
class CPUConfig:
    """CPU-Konfiguration"""
    name: str = "AMD Ryzen 9 9955HX3D"
    cores: int = 16
    threads: int = 32
    base_clock_mhz: int = 2500
    l2_cache_kb: int = 16384
    l3_cache_kb: int = 131072  # 128MB 3D V-Cache
    architecture: str = "Zen5"

    @property
    def optimal_workers(self) -> int:
        """Optimale Anzahl Worker-Prozesse"""
        return self.threads - 2  # 30 workers, 2 reserved for system

    @property
    def optimal_threads_per_worker(self) -> int:
        """Optimale Threads pro Worker"""
        return 2


@dataclass
class RAMConfig:
    """RAM-Konfiguration"""
    total_gb: float = 96.0
    speed_mhz: int = 5600
    type: str = "DDR5"
    channels: int = 2
    modules: int = 2

    @property
    def available_for_processing_gb(self) -> float:
        """Verfuegbarer RAM fuer Verarbeitung (80% des Totals)"""
        return self.total_gb * 0.8

    @property
    def max_model_size_gb(self) -> float:
        """Maximale Modellgroesse im RAM"""
        return self.total_gb * 0.6


@dataclass
class GPUConfig:
    """GPU-Konfiguration"""
    name: str = ""
    type: GPUType = GPUType.NONE
    vram_gb: float = 0
    driver_version: str = ""
    compute_capability: str = ""
    is_primary: bool = False
    supports_cuda: bool = False
    supports_tensor_cores: bool = False


@dataclass
class StorageConfig:
    """Storage-Konfiguration"""
    name: str = ""
    type: StorageType = StorageType.SSD
    size_tb: float = 0
    path: str = ""
    read_speed_mbps: int = 0
    write_speed_mbps: int = 0


@dataclass
class NetworkConfig:
    """Netzwerk-Konfiguration"""
    name: str = ""
    type: str = ""  # wifi, ethernet, bluetooth
    speed_mbps: int = 0
    mac_address: str = ""


@dataclass
class DisplayConfig:
    """Display-Konfiguration"""
    name: str = ""
    width: int = 0
    height: int = 0
    ppi: int = 0
    refresh_rate: int = 60


@dataclass
class HardwareConfig:
    """Komplette Hardware-Konfiguration"""
    hostname: str = "AXIS"
    system_manufacturer: str = "SchenkerTechnologiesGmbH"
    system_model: str = "XMG NEO (A25)"
    os_name: str = "Windows 11 Pro"
    os_version: str = "10.0.26200"

    cpu: CPUConfig = field(default_factory=CPUConfig)
    ram: RAMConfig = field(default_factory=RAMConfig)
    gpus: List[GPUConfig] = field(default_factory=list)
    storage: List[StorageConfig] = field(default_factory=list)
    network: List[NetworkConfig] = field(default_factory=list)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def get_primary_gpu(self) -> Optional[GPUConfig]:
        """Holt die primaere GPU"""
        for gpu in self.gpus:
            if gpu.is_primary:
                return gpu
        return self.gpus[0] if self.gpus else None

    def get_nvidia_gpu(self) -> Optional[GPUConfig]:
        """Holt die NVIDIA GPU"""
        for gpu in self.gpus:
            if gpu.type == GPUType.NVIDIA:
                return gpu
        return None

    def get_fastest_storage(self) -> Optional[StorageConfig]:
        """Holt den schnellsten Speicher"""
        priority = [StorageType.RAMDISK, StorageType.NVME, StorageType.SSD, StorageType.HDD]
        for storage_type in priority:
            for storage in self.storage:
                if storage.type == storage_type:
                    return storage
        return self.storage[0] if self.storage else None

    def to_dict(self) -> Dict:
        """Konvertiert zu Dictionary"""
        return {
            "hostname": self.hostname,
            "system_manufacturer": self.system_manufacturer,
            "system_model": self.system_model,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "cpu": {
                "name": self.cpu.name,
                "cores": self.cpu.cores,
                "threads": self.cpu.threads,
                "base_clock_mhz": self.cpu.base_clock_mhz,
                "l3_cache_mb": self.cpu.l3_cache_kb // 1024,
            },
            "ram": {
                "total_gb": self.ram.total_gb,
                "speed_mhz": self.ram.speed_mhz,
                "type": self.ram.type,
            },
            "gpus": [
                {
                    "name": gpu.name,
                    "type": gpu.type.value,
                    "vram_gb": gpu.vram_gb,
                    "is_primary": gpu.is_primary,
                }
                for gpu in self.gpus
            ],
            "storage_tb": sum(s.size_tb for s in self.storage),
        }


# Vordefiniertes System-Profil fuer dieses Geraet
SYSTEM_PROFILE = HardwareConfig(
    hostname="AXIS",
    system_manufacturer="SchenkerTechnologiesGmbH",
    system_model="XMG NEO (A25)",
    os_name="Windows 11 Pro",
    os_version="10.0.26200",
    cpu=CPUConfig(
        name="AMD Ryzen 9 9955HX3D",
        cores=16,
        threads=32,
        base_clock_mhz=2500,
        l2_cache_kb=16384,
        l3_cache_kb=131072,
        architecture="Zen5",
    ),
    ram=RAMConfig(
        total_gb=96.0,
        speed_mhz=5600,
        type="DDR5",
        channels=2,
        modules=2,
    ),
    gpus=[
        GPUConfig(
            name="NVIDIA GeForce RTX 5090 Laptop GPU",
            type=GPUType.NVIDIA,
            vram_gb=24.0,
            driver_version="591.74",
            compute_capability="10.0",
            is_primary=True,
            supports_cuda=True,
            supports_tensor_cores=True,
        ),
        GPUConfig(
            name="AMD Radeon 610M",
            type=GPUType.AMD,
            vram_gb=2.0,
            driver_version="32.0.13034.7001",
            is_primary=False,
            supports_cuda=False,
            supports_tensor_cores=False,
        ),
    ],
    storage=[
        StorageConfig(
            name="Romex RAMDISK",
            type=StorageType.RAMDISK,
            size_tb=0.02,
            path="R:",
            read_speed_mbps=50000,
            write_speed_mbps=50000,
        ),
        StorageConfig(
            name="WD_BLACK SN850X 8000GB",
            type=StorageType.NVME,
            size_tb=8.0,
            path="C:",
            read_speed_mbps=7300,
            write_speed_mbps=6600,
        ),
        StorageConfig(
            name="Samsung SSD 9100 PRO 8TB",
            type=StorageType.NVME,
            size_tb=8.0,
            path="D:",
            read_speed_mbps=14500,
            write_speed_mbps=12000,
        ),
    ],
    network=[
        NetworkConfig(
            name="Intel Wi-Fi 6E AX210 160MHz",
            type="wifi",
            speed_mbps=2400,
        ),
        NetworkConfig(
            name="Realtek Gaming 2.5GbE Family Controller",
            type="ethernet",
            speed_mbps=2500,
        ),
        NetworkConfig(
            name="Intel Wireless Bluetooth",
            type="bluetooth",
            speed_mbps=3,
        ),
    ],
    display=DisplayConfig(
        name="NE160QDM-NM9",
        width=2560,
        height=1600,
        ppi=144,
        refresh_rate=240,
    ),
)


# Optimierte Einstellungen basierend auf der Hardware - MAXIMALE LEISTUNG
class OptimalSettings:
    """Optimale Einstellungen fuer SCIO - 100% Hardware-Nutzung"""

    # Parallelisierung - MAXIMALE CPU-NUTZUNG
    MAX_WORKERS = 32  # Alle 32 Threads nutzen
    THREADS_PER_WORKER = 1  # Ein Thread pro Worker fuer maximale Parallelitaet

    # Speicher - MAXIMALE RAM-NUTZUNG
    MAX_MEMORY_GB = 90.0  # 90GB von 96GB nutzen (6GB fuer System)
    CHUNK_SIZE_MB = 1024  # Groessere Chunks fuer 128MB L3 V-Cache
    PREFETCH_BUFFER_GB = 16.0  # Grosser Prefetch-Buffer

    # GPU - MAXIMALE GPU-NUTZUNG
    USE_GPU = True
    GPU_MEMORY_FRACTION = 0.95  # 95% des VRAM nutzen (~23GB von 24GB)
    CUDA_VISIBLE_DEVICES = "0"  # RTX 5090 Blackwell
    ENABLE_TENSOR_CORES = True  # 5th Gen Tensor Cores
    MIXED_PRECISION = True  # BF16 fuer Blackwell-Optimierung
    ENABLE_CUDA_GRAPHS = True  # CUDA Graphs fuer weniger Overhead
    ENABLE_FLASH_ATTENTION = True  # Flash Attention 3
    COMPILE_MODE = "max-autotune"  # torch.compile mit maximaler Optimierung

    # Storage - MAXIMALE I/O-NUTZUNG
    TEMP_DIR = "R:\\"  # RAM Disk (50GB/s)
    CACHE_DIR = "D:\\SCIO\\cache"  # Samsung 9100 PRO (14.5GB/s)
    DATA_DIR = "D:\\SCIO\\data"  # Samsung 9100 PRO
    MODEL_DIR = "C:\\SCIO\\models"  # WD Black SN850X (7.3GB/s)
    ASYNC_IO = True  # Asynchrone I/O-Operationen
    IO_THREADS = 8  # Dedizierte I/O-Threads

    # Netzwerk
    MAX_CONNECTIONS = 500  # Mehr parallele Verbindungen
    TIMEOUT_SECONDS = 30
    KEEPALIVE = True

    # ML-spezifisch - MAXIMALE THROUGHPUT
    BATCH_SIZE = 128  # Groessere Batches fuer 24GB VRAM
    MICRO_BATCH_SIZE = 32  # Fuer Gradient Accumulation
    GRADIENT_ACCUMULATION = 4
    DATALOADER_WORKERS = 16  # Mehr Data Loading Workers
    PIN_MEMORY = True
    NON_BLOCKING = True  # Non-blocking CUDA transfers
    PERSISTENT_WORKERS = True  # Workers am Leben halten

    # Performance Tuning
    CUDNN_BENCHMARK = True  # cuDNN Autotuning
    TF32_ENABLED = True  # TensorFloat-32 fuer Matmul
    DETERMINISTIC = False  # Nicht-deterministisch fuer Speed

    @classmethod
    def get_pytorch_settings(cls) -> Dict:
        """PyTorch-spezifische Einstellungen fuer maximale Performance"""
        return {
            "device": "cuda:0",
            "dtype": "bfloat16",
            "compile": True,
            "compile_mode": cls.COMPILE_MODE,
            "num_workers": cls.DATALOADER_WORKERS,
            "pin_memory": cls.PIN_MEMORY,
            "batch_size": cls.BATCH_SIZE,
            "persistent_workers": cls.PERSISTENT_WORKERS,
            "non_blocking": cls.NON_BLOCKING,
            "cudnn_benchmark": cls.CUDNN_BENCHMARK,
        }

    @classmethod
    def get_tensorflow_settings(cls) -> Dict:
        """TensorFlow-spezifische Einstellungen"""
        return {
            "gpu_memory_growth": True,
            "memory_limit": int(SYSTEM_PROFILE.gpus[0].vram_gb * 1024 * 0.95),
            "mixed_precision": "mixed_bfloat16",
            "xla": True,
        }

    @classmethod
    def configure_for_max_performance(cls):
        """Konfiguriert PyTorch fuer maximale Hardware-Nutzung"""
        try:
            import torch

            # CUDA Settings
            if torch.cuda.is_available():
                # Setze Memory Fraction
                torch.cuda.set_per_process_memory_fraction(cls.GPU_MEMORY_FRACTION)

                # cuDNN Benchmark
                torch.backends.cudnn.benchmark = cls.CUDNN_BENCHMARK

                # TF32 fuer schnellere Matmul
                torch.backends.cuda.matmul.allow_tf32 = cls.TF32_ENABLED
                torch.backends.cudnn.allow_tf32 = cls.TF32_ENABLED

                # BFloat16 als Standard
                torch.set_default_dtype(torch.bfloat16)

                # CUDA Caching Allocator optimieren
                import os
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

                print(f"PyTorch konfiguriert fuer RTX 5090:")
                print(f"  - VRAM: {cls.GPU_MEMORY_FRACTION*100:.0f}% (~{SYSTEM_PROFILE.gpus[0].vram_gb * cls.GPU_MEMORY_FRACTION:.1f}GB)")
                print(f"  - cuDNN Benchmark: {cls.CUDNN_BENCHMARK}")
                print(f"  - TF32: {cls.TF32_ENABLED}")
                print(f"  - BFloat16: Aktiviert")

                return True
        except ImportError:
            pass
        return False

    @classmethod
    def get_system_summary(cls) -> str:
        """Gibt Hardware-Zusammenfassung zurueck"""
        gpu = SYSTEM_PROFILE.get_nvidia_gpu()
        return f"""
================================================================================
SCIO HARDWARE PROFIL - MAXIMALE LEISTUNG
================================================================================
System:   {SYSTEM_PROFILE.system_model} ({SYSTEM_PROFILE.system_manufacturer})
OS:       {SYSTEM_PROFILE.os_name}

CPU:      {SYSTEM_PROFILE.cpu.name}
          {SYSTEM_PROFILE.cpu.cores} Kerne / {SYSTEM_PROFILE.cpu.threads} Threads @ {SYSTEM_PROFILE.cpu.base_clock_mhz} MHz
          L3 Cache: {SYSTEM_PROFILE.cpu.l3_cache_kb // 1024} MB (3D V-Cache)

RAM:      {SYSTEM_PROFILE.ram.total_gb} GB {SYSTEM_PROFILE.ram.type}-{SYSTEM_PROFILE.ram.speed_mhz}
          Nutzbar: {cls.MAX_MEMORY_GB} GB

GPU:      {gpu.name if gpu else 'Keine'}
          VRAM: {gpu.vram_gb if gpu else 0} GB (Nutzung: {cls.GPU_MEMORY_FRACTION*100:.0f}%)
          Compute: {gpu.compute_capability if gpu else 'N/A'}

Storage:  RAM Disk: 20 GB @ 50 GB/s
          NVMe 1: 8 TB @ 14.5 GB/s (Samsung 9100 PRO)
          NVMe 2: 8 TB @ 7.3 GB/s (WD Black SN850X)

Settings: Workers: {cls.MAX_WORKERS} | Batch: {cls.BATCH_SIZE} | DataLoader: {cls.DATALOADER_WORKERS}
================================================================================
"""
