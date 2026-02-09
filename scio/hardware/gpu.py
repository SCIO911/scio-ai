"""
GPU Manager
===========

GPU-Verwaltung und Optimierung fuer CUDA/NVIDIA.
"""

import os
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class CUDADevice:
    """CUDA-Geraet-Informationen"""
    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    compute_capability: str
    driver_version: str
    is_available: bool = True


class GPUManager:
    """Verwaltet GPU-Ressourcen"""

    def __init__(self):
        self._devices: List[CUDADevice] = []
        self._torch_available = False
        self._tensorflow_available = False
        self._refresh_devices()

    def _run_nvidia_smi(self, query: str) -> str:
        """Fuehrt nvidia-smi Query aus"""
        try:
            result = subprocess.run(
                f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _refresh_devices(self) -> None:
        """Aktualisiert Geraete-Liste"""
        self._devices = []

        # Check via nvidia-smi
        output = self._run_nvidia_smi(
            "index,name,memory.total,memory.free,compute_cap,driver_version"
        )

        if output:
            for line in output.split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    try:
                        self._devices.append(CUDADevice(
                            index=int(parts[0]),
                            name=parts[1],
                            total_memory_mb=int(parts[2]),
                            free_memory_mb=int(parts[3]),
                            compute_capability=parts[4],
                            driver_version=parts[5],
                        ))
                    except (ValueError, IndexError):
                        pass

        # Check PyTorch
        try:
            import torch
            self._torch_available = torch.cuda.is_available()
        except ImportError:
            self._torch_available = False

        # Check TensorFlow
        try:
            import tensorflow as tf
            self._tensorflow_available = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            self._tensorflow_available = False

    @property
    def devices(self) -> List[CUDADevice]:
        """Alle CUDA-Geraete"""
        return self._devices

    @property
    def device_count(self) -> int:
        """Anzahl CUDA-Geraete"""
        return len(self._devices)

    @property
    def primary_device(self) -> Optional[CUDADevice]:
        """Primaeres CUDA-Geraet"""
        return self._devices[0] if self._devices else None

    @property
    def is_cuda_available(self) -> bool:
        """Ist CUDA verfuegbar"""
        return len(self._devices) > 0

    @property
    def is_torch_available(self) -> bool:
        """Ist PyTorch mit CUDA verfuegbar"""
        return self._torch_available

    @property
    def is_tensorflow_available(self) -> bool:
        """Ist TensorFlow mit GPU verfuegbar"""
        return self._tensorflow_available

    def get_device(self, index: int = 0) -> Optional[CUDADevice]:
        """Holt Geraet nach Index"""
        if 0 <= index < len(self._devices):
            return self._devices[index]
        return None

    def get_free_memory(self, device_index: int = 0) -> int:
        """Holt freien VRAM in MB"""
        output = self._run_nvidia_smi("memory.free")
        if output:
            for i, line in enumerate(output.split("\n")):
                if i == device_index:
                    try:
                        return int(line.strip())
                    except ValueError:
                        pass
        return 0

    def get_utilization(self, device_index: int = 0) -> float:
        """Holt GPU-Auslastung in Prozent"""
        output = self._run_nvidia_smi("utilization.gpu")
        if output:
            for i, line in enumerate(output.split("\n")):
                if i == device_index:
                    try:
                        return float(line.strip())
                    except ValueError:
                        pass
        return 0.0

    def get_temperature(self, device_index: int = 0) -> Optional[float]:
        """Holt GPU-Temperatur in Celsius"""
        output = self._run_nvidia_smi("temperature.gpu")
        if output:
            for i, line in enumerate(output.split("\n")):
                if i == device_index:
                    try:
                        return float(line.strip())
                    except ValueError:
                        pass
        return None

    def get_power_draw(self, device_index: int = 0) -> Optional[float]:
        """Holt Stromverbrauch in Watt"""
        output = self._run_nvidia_smi("power.draw")
        if output and "[N/A]" not in output:
            for i, line in enumerate(output.split("\n")):
                if i == device_index:
                    try:
                        return float(line.strip())
                    except ValueError:
                        pass
        return None

    def set_cuda_device(self, device_index: int) -> None:
        """Setzt aktives CUDA-Geraet"""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

        if self._torch_available:
            try:
                import torch
                torch.cuda.set_device(device_index)
            except Exception:
                pass

    def configure_pytorch(
        self,
        device_index: int = 0,
        memory_fraction: float = 0.9,
        allow_growth: bool = True,
    ) -> Dict[str, Any]:
        """Konfiguriert PyTorch fuer optimale GPU-Nutzung"""
        config = {}

        try:
            import torch

            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}

            # Set device
            device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(device)

            # Memory settings
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device_index)

            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Get device properties
            props = torch.cuda.get_device_properties(device_index)

            config = {
                "device": str(device),
                "device_name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "cudnn_benchmark": True,
                "tf32_enabled": True,
                "memory_fraction": memory_fraction,
            }

        except ImportError:
            config = {"error": "PyTorch not installed"}
        except Exception as e:
            config = {"error": str(e)}

        return config

    def configure_tensorflow(
        self,
        device_index: int = 0,
        memory_limit_mb: Optional[int] = None,
        allow_growth: bool = True,
    ) -> Dict[str, Any]:
        """Konfiguriert TensorFlow fuer optimale GPU-Nutzung"""
        config = {}

        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return {"error": "No GPU available"}

            if device_index < len(gpus):
                gpu = gpus[device_index]

                if memory_limit_mb:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                elif allow_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Enable mixed precision
                tf.keras.mixed_precision.set_global_policy('mixed_float16')

                config = {
                    "device": gpu.name,
                    "memory_growth": allow_growth,
                    "memory_limit_mb": memory_limit_mb,
                    "mixed_precision": "mixed_float16",
                }

        except ImportError:
            config = {"error": "TensorFlow not installed"}
        except Exception as e:
            config = {"error": str(e)}

        return config

    def get_optimal_batch_size(
        self,
        model_memory_mb: int,
        sample_memory_mb: int,
        device_index: int = 0,
        safety_margin: float = 0.2,
    ) -> int:
        """Berechnet optimale Batch-Groesse basierend auf VRAM"""
        device = self.get_device(device_index)
        if not device:
            return 1

        free_memory = self.get_free_memory(device_index)
        usable_memory = (free_memory - model_memory_mb) * (1 - safety_margin)

        if usable_memory <= 0 or sample_memory_mb <= 0:
            return 1

        batch_size = int(usable_memory / sample_memory_mb)
        return max(1, batch_size)

    @contextmanager
    def memory_context(self, device_index: int = 0):
        """Context Manager fuer GPU-Speicherverwaltung"""
        initial_memory = self.get_free_memory(device_index)

        try:
            yield
        finally:
            # Cleanup
            if self._torch_available:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if self._tensorflow_available:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:
                    pass

            final_memory = self.get_free_memory(device_index)
            # Log memory usage if needed

    def clear_cache(self) -> None:
        """Leert GPU-Cache"""
        if self._torch_available:
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        if self._tensorflow_available:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Holt aktuellen GPU-Status"""
        status = {
            "cuda_available": self.is_cuda_available,
            "torch_available": self._torch_available,
            "tensorflow_available": self._tensorflow_available,
            "device_count": self.device_count,
            "devices": [],
        }

        for device in self._devices:
            free_mem = self.get_free_memory(device.index)
            status["devices"].append({
                "index": device.index,
                "name": device.name,
                "total_memory_mb": device.total_memory_mb,
                "free_memory_mb": free_mem,
                "used_memory_mb": device.total_memory_mb - free_mem,
                "utilization_percent": self.get_utilization(device.index),
                "temperature_c": self.get_temperature(device.index),
                "power_draw_w": self.get_power_draw(device.index),
                "compute_capability": device.compute_capability,
            })

        return status

    def print_status(self) -> str:
        """Gibt formatierten GPU-Status zurueck"""
        status = self.get_status()

        lines = [
            "=" * 50,
            "GPU STATUS",
            "=" * 50,
            f"CUDA Available: {status['cuda_available']}",
            f"PyTorch CUDA: {status['torch_available']}",
            f"TensorFlow GPU: {status['tensorflow_available']}",
            f"Device Count: {status['device_count']}",
            "",
        ]

        for device in status["devices"]:
            lines.extend([
                f"[GPU {device['index']}] {device['name']}",
                f"  Memory: {device['used_memory_mb']}/{device['total_memory_mb']}MB "
                f"({device['used_memory_mb']/device['total_memory_mb']*100:.1f}%)",
                f"  Utilization: {device['utilization_percent']:.1f}%",
            ])

            if device['temperature_c']:
                lines.append(f"  Temperature: {device['temperature_c']}C")
            if device['power_draw_w']:
                lines.append(f"  Power Draw: {device['power_draw_w']:.1f}W")

            lines.append(f"  Compute Capability: {device['compute_capability']}")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)


# Singleton-Instanz
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Holt oder erstellt GPU-Manager-Singleton"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
