#!/usr/bin/env python3
"""
SCIO AI-Workstation - Vollautomatischer Starter
Startet alle Services automatisch und ueberwacht sie
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import sys

# Must be set before ANY imports to suppress torch/triton warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ═══════════════════════════════════════════════════════════════
# GPU & CPU OPTIMIERUNGEN - Maximale Hardware-Nutzung
# ═══════════════════════════════════════════════════════════════

# CUDA Optimierungen für RTX 5090
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Tensor Core Optimierungen (TF32 für bessere Performance)
os.environ['NVIDIA_TF32_OVERRIDE'] = '1'

# CPU Threading Optimierungen (16 Cores / 32 Threads)
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

# Tokenizers Parallelisierung
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# HuggingFace Cache im Projekt-Verzeichnis
os.environ['HF_HOME'] = 'C:/SCIO/data/models/cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/SCIO/data/models/cache'

import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress noisy loggers
for logger_name in ['torch', 'triton', 'absl', 'tensorflow']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import time
import subprocess
from pathlib import Path
from datetime import datetime

# Projekt-Verzeichnis
PROJECT_DIR = Path(__file__).parent
os.chdir(PROJECT_DIR)

# Fuege Projekt zum Path hinzu
sys.path.insert(0, str(PROJECT_DIR))


def print_banner():
    print("""
+======================================================================+
|                                                                      |
|    SSSS   CCCC  III   OOO                                            |
|   S      C       I   O   O                                           |
|    SSS   C       I   O   O                                           |
|       S  C       I   O   O                                           |
|   SSSS    CCCC  III   OOO                                            |
|                                                                      |
|   Service Computer Intelligence Organization                         |
|   AI-WORKSTATION v2.1 - MAXIMUM PERFORMANCE                          |
|   Optimiert fuer RTX 5090 + 32 Threads + 94GB RAM                    |
|                                                                      |
+======================================================================+
    """)


def check_dependencies():
    """Prueft ob alle Dependencies verfuegbar sind"""
    print("[INFO] Pruefe Dependencies...")

    required = [
        'flask', 'flask_cors', 'flask_socketio', 'stripe', 'dotenv',
        'sqlalchemy', 'psutil', 'requests', 'httpx'
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[WARN] Fehlende Pakete: {', '.join(missing)}")
        print("[INFO] Bitte 'pip install -r requirements.txt' ausfuehren")
    else:
        print("[OK] Alle Core-Dependencies OK")

    # Hardware-Info
    import psutil
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"[OK] CPU: {cpu_cores} Cores / {cpu_threads} Threads")
    print(f"[OK] RAM: {ram_gb:.1f} GB")

    # Optional: GPU-Pakete (suppress triton warning during import)
    try:
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()  # Suppress triton warning
        try:
            import torch
        finally:
            sys.stderr = old_stderr
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[OK] GPU: {gpu_name} ({vram_total:.0f} GB VRAM)")

            # Zeige aktive Optimierungen
            if torch.backends.cuda.matmul.allow_tf32:
                print("[OK] TF32 Tensor Cores aktiviert")
            if torch.backends.cudnn.benchmark:
                print("[OK] cuDNN Benchmark aktiviert")
        else:
            print("[WARN] PyTorch ohne CUDA - GPU-Features limitiert")
    except ImportError:
        print("[WARN] PyTorch nicht installiert - GPU-Features deaktiviert")

    # Flash Attention Check
    try:
        import flash_attn
        print("[OK] Flash Attention 2 verfuegbar")
    except ImportError:
        print("[INFO] Flash Attention nicht installiert - SDPA wird verwendet")


def create_directories():
    """Erstellt alle benoetigten Verzeichnisse"""
    dirs = [
        PROJECT_DIR / 'data',
        PROJECT_DIR / 'data' / 'orders',
        PROJECT_DIR / 'data' / 'models',
        PROJECT_DIR / 'data' / 'logs',
        PROJECT_DIR / 'data' / 'generated',
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("[OK] Verzeichnisse erstellt")


def start_server():
    """Startet den Haupt-Server"""
    from backend.app import app, socketio, start_services, print_banner as server_banner
    from backend.config import Config

    server_banner()
    start_services()

    print("\n[OK] PRODUCTION SERVER")
    print(f"[OK] http://localhost:{Config.PORT}")
    print(f"[OK] Admin: http://localhost:{Config.PORT}/admin/")
    print("[OK] System bereit - Vollautomatischer Betrieb\n")

    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=False,
        use_reloader=False,
        log_output=False,
        allow_unsafe_werkzeug=True,
    )


def main():
    print_banner()
    print(f"[DATE] Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DIR] Projekt: {PROJECT_DIR}\n")

    # Setup
    check_dependencies()
    create_directories()

    # Start
    print("\n" + "=" * 70)
    start_server()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOP] Server gestoppt")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
