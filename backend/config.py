#!/usr/bin/env python3
"""
SCIO AI-Workstation - Zentrale Konfiguration

ALLES IN C:\\SCIO - Keine externen Abhaengigkeiten
"""

import os
import secrets
import logging
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: C:\SCIO (alles ist hier enthalten)
BASE_DIR = Path(__file__).parent.parent

# Load .env from C:\SCIO\.env
load_dotenv(BASE_DIR / '.env')


class Config:
    """Zentrale Konfigurationsklasse"""

    # Pfade
    BASE_DIR = BASE_DIR
    DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
    ORDERS_DIR = Path(os.getenv('ORDERS_DIR', DATA_DIR / 'orders'))
    MODELS_DIR = Path(os.getenv('MODELS_DIR', DATA_DIR / 'models'))
    LOGS_DIR = Path(os.getenv('LOGS_DIR', DATA_DIR / 'logs'))

    # Datenbank
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{DATA_DIR}/scio.db')

    # Service
    SERVICE_NAME = os.getenv('SERVICE_NAME', 'SCIO')
    SERVICE_URL = os.getenv('SERVICE_URL', 'http://localhost:5000')
    SERVICE_EMAIL = os.getenv('SERVICE_EMAIL', 'noreply@scio.ai')
    # Generate secure secret key if not provided (warn in production)
    _secret_key_env = os.getenv('SECRET_KEY')
    if not _secret_key_env:
        _secret_key_env = secrets.token_hex(32)
        logging.getLogger(__name__).warning(
            "SECRET_KEY not set in environment! Using generated key. "
            "Set SECRET_KEY in .env for production."
        )
    SECRET_KEY = _secret_key_env

    # Server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'  # Production default

    # Stripe
    STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')
    STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
    STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

    # Pricing (in Cents)
    PRICES = {
        'llama-7b': int(os.getenv('PRICE_LLAMA_7B', 20000)),      # 200€
        'llama-13b': int(os.getenv('PRICE_LLAMA_13B', 40000)),    # 400€
        'llama-70b': int(os.getenv('PRICE_LLAMA_70B', 100000)),   # 1000€
    }

    # API Pricing (pro 1000 Tokens in Cents)
    API_PRICES = {
        'input': int(os.getenv('API_PRICE_INPUT', 1)),    # 0.01€ / 1k tokens
        'output': int(os.getenv('API_PRICE_OUTPUT', 3)),  # 0.03€ / 1k tokens
    }

    # Image Generation Pricing (pro Bild in Cents)
    IMAGE_PRICES = {
        'sd-1.5': int(os.getenv('PRICE_SD15', 5)),        # 0.05€
        'sdxl': int(os.getenv('PRICE_SDXL', 10)),         # 0.10€
        'flux': int(os.getenv('PRICE_FLUX', 15)),         # 0.15€
    }

    # Hardware Limits - Maximale Auslastung
    MAX_VRAM_USAGE = float(os.getenv('MAX_VRAM_USAGE', 0.98))     # 98%
    MAX_RAM_USAGE = float(os.getenv('MAX_RAM_USAGE', 0.95))       # 95%
    MAX_CPU_USAGE = float(os.getenv('MAX_CPU_USAGE', 1.0))        # 100%

    # Job Queue - Optimiert für High-End Hardware
    MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', 12))
    JOB_TIMEOUT_SECONDS = int(os.getenv('JOB_TIMEOUT_SECONDS', 86400))  # 24h
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))

    # Performance Settings
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', 16))
    TORCH_NUM_THREADS = int(os.getenv('TORCH_NUM_THREADS', 16))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
    PREFETCH_FACTOR = int(os.getenv('PREFETCH_FACTOR', 4))

    # GPU Optimierungen
    USE_FLASH_ATTENTION = os.getenv('USE_FLASH_ATTENTION', 'true').lower() == 'true'
    USE_BETTERTRANSFORMER = os.getenv('USE_BETTERTRANSFORMER', 'true').lower() == 'true'
    LOAD_IN_8BIT = os.getenv('LOAD_IN_8BIT', 'false').lower() == 'true'
    LOAD_IN_4BIT = os.getenv('LOAD_IN_4BIT', 'false').lower() == 'true'

    # LLM Settings - Erweitert für 24GB VRAM
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
    MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', 32768))
    MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 8192))
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 8))

    # Available Models - Optimiert für RTX 5090 mit 24GB VRAM
    AVAILABLE_MODELS = {
        'mistral-7b': {
            'name': 'Mistral 7B Instruct',
            'hf_id': 'mistralai/Mistral-7B-Instruct-v0.2',
            'vram_gb': 14,
            'context_length': 32768,
        },
        'mistral-nemo': {
            'name': 'Mistral Nemo 12B',
            'hf_id': 'mistralai/Mistral-Nemo-Instruct-2407',
            'vram_gb': 24,
            'context_length': 128000,
        },
        'llama-7b': {
            'name': 'Llama 2 7B Chat',
            'hf_id': 'meta-llama/Llama-2-7b-chat-hf',
            'vram_gb': 14,
            'context_length': 4096,
        },
        'llama-13b': {
            'name': 'Llama 2 13B Chat',
            'hf_id': 'meta-llama/Llama-2-13b-chat-hf',
            'vram_gb': 26,
            'context_length': 4096,
        },
        'llama3-8b': {
            'name': 'Llama 3 8B Instruct',
            'hf_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'vram_gb': 16,
            'context_length': 8192,
        },
        'qwen-7b': {
            'name': 'Qwen 1.5 7B Chat',
            'hf_id': 'Qwen/Qwen1.5-7B-Chat',
            'vram_gb': 14,
            'context_length': 32768,
        },
        'qwen2-7b': {
            'name': 'Qwen 2 7B Instruct',
            'hf_id': 'Qwen/Qwen2-7B-Instruct',
            'vram_gb': 14,
            'context_length': 131072,
        },
        'gemma2-9b': {
            'name': 'Gemma 2 9B Instruct',
            'hf_id': 'google/gemma-2-9b-it',
            'vram_gb': 18,
            'context_length': 8192,
        },
        'phi3-medium': {
            'name': 'Phi-3 Medium 14B',
            'hf_id': 'microsoft/Phi-3-medium-4k-instruct',
            'vram_gb': 28,
            'context_length': 4096,
        },
    }

    # Vast.ai Integration
    VASTAI_API_KEY = os.getenv('VASTAI_API_KEY')
    VASTAI_ENABLED = os.getenv('VASTAI_ENABLED', 'false').lower() == 'true'
    VASTAI_MIN_PRICE = float(os.getenv('VASTAI_MIN_PRICE', 0.30))
    VASTAI_MAX_PRICE = float(os.getenv('VASTAI_MAX_PRICE', 0.80))

    # RunPod Integration
    RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
    RUNPOD_ENABLED = os.getenv('RUNPOD_ENABLED', 'false').lower() == 'true'

    # Notifications
    DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    @classmethod
    def init_dirs(cls):
        """Erstellt alle benötigten Verzeichnisse"""
        for dir_path in [cls.DATA_DIR, cls.ORDERS_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_info(cls, model_id: str) -> dict:
        """Gibt Model-Info zurück oder Default"""
        return cls.AVAILABLE_MODELS.get(model_id, cls.AVAILABLE_MODELS['mistral-7b'])


# Initialisiere Verzeichnisse
Config.init_dirs()
