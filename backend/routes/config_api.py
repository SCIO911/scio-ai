#!/usr/bin/env python3
"""
SCIO - Configuration Management API
Runtime-Konfiguration ohne Neustart
"""

import threading
from flask import Blueprint, jsonify, request
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

config_api_bp = Blueprint('config_api', __name__, url_prefix='/api/config')


@dataclass
class RuntimeConfig:
    """
    Runtime-Konfiguration für SCIO

    Diese Werte können zur Laufzeit geändert werden ohne Neustart.
    """
    # Job Queue
    job_max_concurrent: int = 12
    job_timeout_seconds: int = 86400
    job_max_retries: int = 3

    # Hardware Thresholds
    hardware_max_vram_usage: float = 0.98
    hardware_max_ram_usage: float = 0.95
    hardware_max_cpu_usage: float = 1.0

    # LLM Settings
    llm_default_model: str = "mistral-7b"
    llm_max_context_length: int = 32768
    llm_max_new_tokens: int = 8192
    llm_default_temperature: float = 0.7

    # Image Generation
    image_default_model: str = "flux-schnell"
    image_default_steps: int = 4
    image_max_resolution: int = 2048

    # Rate Limiting
    rate_requests_per_second: int = 20
    rate_requests_per_minute: int = 200
    rate_burst_size: int = 50

    # Degradation
    degradation_level: str = "normal"
    degradation_auto_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_audit_enabled: bool = True

    # Features
    feature_learning_enabled: bool = True
    feature_decision_enabled: bool = True
    feature_orchestration_enabled: bool = True

    # Last update
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_by: str = ""


class ConfigManager:
    """
    Runtime Configuration Manager

    Ermöglicht Änderung von Konfigurationswerten zur Laufzeit
    und benachrichtigt registrierte Listener über Änderungen.
    """

    def __init__(self):
        self._config = RuntimeConfig()
        self._lock = threading.Lock()
        self._listeners: list = []
        self._history: list = []
        self._max_history = 100

    def get_all(self) -> Dict[str, Any]:
        """Gibt alle Konfigurationswerte zurück"""
        with self._lock:
            return asdict(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        """Gibt einzelnen Konfigurationswert zurück"""
        with self._lock:
            return getattr(self._config, key, default)

    def set(self, key: str, value: Any, updated_by: str = "api") -> bool:
        """
        Setzt einzelnen Konfigurationswert

        Args:
            key: Konfigurationsschlüssel
            value: Neuer Wert
            updated_by: Wer hat die Änderung vorgenommen

        Returns:
            True bei Erfolg
        """
        with self._lock:
            if not hasattr(self._config, key):
                return False

            old_value = getattr(self._config, key)

            # Typ-Validierung
            expected_type = type(old_value)
            try:
                if expected_type == bool and isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes')
                elif expected_type in (int, float):
                    value = expected_type(value)
            except (ValueError, TypeError):
                return False

            # Wert setzen
            setattr(self._config, key, value)
            self._config.updated_at = datetime.now().isoformat()
            self._config.updated_by = updated_by

            # History speichern
            self._history.append({
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "updated_by": updated_by,
                "timestamp": datetime.now().isoformat()
            })
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Listener benachrichtigen
            self._notify_listeners(key, old_value, value)

            return True

    def set_many(self, updates: Dict[str, Any], updated_by: str = "api") -> Dict[str, bool]:
        """Setzt mehrere Konfigurationswerte"""
        results = {}
        for key, value in updates.items():
            results[key] = self.set(key, value, updated_by)
        return results

    def add_listener(self, callback):
        """Registriert Listener für Konfigurationsänderungen"""
        self._listeners.append(callback)

    def remove_listener(self, callback):
        """Entfernt Listener"""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, key: str, old_value: Any, new_value: Any):
        """Benachrichtigt alle Listener"""
        for listener in self._listeners:
            try:
                listener(key, old_value, new_value)
            except Exception:
                pass

    def get_history(self, limit: int = 50) -> list:
        """Gibt Änderungshistorie zurück"""
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def reset_to_defaults(self):
        """Setzt alle Werte auf Default zurück"""
        with self._lock:
            self._config = RuntimeConfig()
            self._config.updated_at = datetime.now().isoformat()
            self._config.updated_by = "reset"


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Gibt globale ConfigManager-Instanz zurück"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# ═══════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@config_api_bp.route('', methods=['GET'])
@config_api_bp.route('/', methods=['GET'])
def get_config():
    """
    Gibt aktuelle Konfiguration zurück

    Returns:
        JSON mit allen Konfigurationswerten
    """
    manager = get_config_manager()
    return jsonify({
        "config": manager.get_all(),
        "editable_keys": _get_editable_keys()
    })


@config_api_bp.route('/<key>', methods=['GET'])
def get_config_value(key: str):
    """Gibt einzelnen Konfigurationswert zurück"""
    manager = get_config_manager()
    value = manager.get(key)

    if value is None:
        return jsonify({"error": f"Key '{key}' not found"}), 404

    return jsonify({
        "key": key,
        "value": value
    })


@config_api_bp.route('', methods=['PUT', 'PATCH'])
@config_api_bp.route('/', methods=['PUT', 'PATCH'])
def update_config():
    """
    Aktualisiert Konfigurationswerte

    Request Body:
        {"key": "value", ...}

    Returns:
        Ergebnis der Updates
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    manager = get_config_manager()
    client_id = request.headers.get('X-API-Key', request.remote_addr)

    results = manager.set_many(data, updated_by=client_id)

    success_count = sum(1 for v in results.values() if v)
    failed_keys = [k for k, v in results.items() if not v]

    response = {
        "updated": success_count,
        "failed": len(failed_keys),
        "results": results
    }

    if failed_keys:
        response["failed_keys"] = failed_keys

    return jsonify(response)


@config_api_bp.route('/<key>', methods=['PUT', 'PATCH'])
def update_single_config(key: str):
    """Aktualisiert einzelnen Konfigurationswert"""
    data = request.get_json()
    if not data or 'value' not in data:
        return jsonify({"error": "No value provided"}), 400

    manager = get_config_manager()
    client_id = request.headers.get('X-API-Key', request.remote_addr)

    success = manager.set(key, data['value'], updated_by=client_id)

    if not success:
        return jsonify({"error": f"Failed to update '{key}'"}), 400

    return jsonify({
        "key": key,
        "value": manager.get(key),
        "updated": True
    })


@config_api_bp.route('/history', methods=['GET'])
def get_config_history():
    """Gibt Änderungshistorie zurück"""
    limit = request.args.get('limit', 50, type=int)
    manager = get_config_manager()

    return jsonify({
        "history": manager.get_history(limit)
    })


@config_api_bp.route('/reset', methods=['POST'])
def reset_config():
    """Setzt Konfiguration auf Defaults zurück"""
    manager = get_config_manager()
    manager.reset_to_defaults()

    return jsonify({
        "status": "reset",
        "config": manager.get_all()
    })


@config_api_bp.route('/schema', methods=['GET'])
def get_config_schema():
    """Gibt Schema der Konfiguration zurück"""
    return jsonify({
        "schema": _get_config_schema()
    })


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _get_editable_keys() -> list:
    """Gibt Liste der editierbaren Schlüssel zurück"""
    config = RuntimeConfig()
    return [
        key for key in asdict(config).keys()
        if not key.startswith('_') and key not in ('updated_at', 'updated_by')
    ]


def _get_config_schema() -> Dict[str, Any]:
    """Gibt Schema für Konfiguration zurück"""
    return {
        "job_max_concurrent": {"type": "integer", "min": 1, "max": 100, "description": "Max concurrent jobs"},
        "job_timeout_seconds": {"type": "integer", "min": 60, "max": 604800, "description": "Job timeout in seconds"},
        "job_max_retries": {"type": "integer", "min": 0, "max": 10, "description": "Max job retries"},

        "hardware_max_vram_usage": {"type": "float", "min": 0.5, "max": 1.0, "description": "Max VRAM usage (0-1)"},
        "hardware_max_ram_usage": {"type": "float", "min": 0.5, "max": 1.0, "description": "Max RAM usage (0-1)"},
        "hardware_max_cpu_usage": {"type": "float", "min": 0.5, "max": 1.0, "description": "Max CPU usage (0-1)"},

        "llm_default_model": {"type": "string", "description": "Default LLM model"},
        "llm_max_context_length": {"type": "integer", "min": 1024, "max": 131072, "description": "Max context length"},
        "llm_max_new_tokens": {"type": "integer", "min": 1, "max": 32768, "description": "Max new tokens"},
        "llm_default_temperature": {"type": "float", "min": 0.0, "max": 2.0, "description": "Default temperature"},

        "image_default_model": {"type": "string", "description": "Default image model"},
        "image_default_steps": {"type": "integer", "min": 1, "max": 100, "description": "Default inference steps"},
        "image_max_resolution": {"type": "integer", "min": 512, "max": 4096, "description": "Max image resolution"},

        "rate_requests_per_second": {"type": "integer", "min": 1, "max": 1000, "description": "Requests per second limit"},
        "rate_requests_per_minute": {"type": "integer", "min": 1, "max": 10000, "description": "Requests per minute limit"},
        "rate_burst_size": {"type": "integer", "min": 1, "max": 1000, "description": "Burst size for rate limiting"},

        "degradation_level": {"type": "string", "enum": ["normal", "reduced", "minimal", "emergency"], "description": "Degradation level"},
        "degradation_auto_enabled": {"type": "boolean", "description": "Enable auto degradation"},

        "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"], "description": "Log level"},
        "log_audit_enabled": {"type": "boolean", "description": "Enable audit logging"},

        "feature_learning_enabled": {"type": "boolean", "description": "Enable learning module"},
        "feature_decision_enabled": {"type": "boolean", "description": "Enable decision engine"},
        "feature_orchestration_enabled": {"type": "boolean", "description": "Enable orchestration"},
    }
