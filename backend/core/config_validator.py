#!/usr/bin/env python3
"""
SCIO Config Validator
Validiert Konfigurationswerte beim Start
"""

import os
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConfigRule:
    """Regel für Konfigurationsvalidierung"""
    name: str
    required: bool = False
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    error_message: str = ""
    sensitive: bool = False  # Nicht in Logs ausgeben


class ConfigValidator:
    """
    Validiert Konfigurationswerte

    Verwendung:
        validator = ConfigValidator()
        validator.add_rule(ConfigRule('DATABASE_URL', required=True))
        validator.add_rule(ConfigRule('PORT', default=5000, validator=lambda x: 1 <= int(x) <= 65535))

        errors = validator.validate(config_dict)
    """

    def __init__(self):
        self.rules: Dict[str, ConfigRule] = {}
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Definiert Standard-Validierungsregeln"""

        # Kritische Konfigurationen
        self.add_rule(ConfigRule(
            'SECRET_KEY',
            required=True,
            validator=lambda x: len(x) >= 32,
            error_message="SECRET_KEY must be at least 32 characters",
            sensitive=True
        ))

        self.add_rule(ConfigRule(
            'PORT',
            default=5000,
            validator=lambda x: 1 <= int(x) <= 65535,
            error_message="PORT must be between 1 and 65535"
        ))

        self.add_rule(ConfigRule(
            'HOST',
            default='0.0.0.0',
            validator=lambda x: x in ['0.0.0.0', '127.0.0.1', 'localhost'] or self._is_valid_ip(x),
            error_message="HOST must be a valid IP address"
        ))

        # Stripe (optional aber wenn gesetzt, muss gültig sein)
        self.add_rule(ConfigRule(
            'STRIPE_SECRET_KEY',
            required=False,
            validator=lambda x: x is None or x.startswith(('sk_test_', 'sk_live_')),
            error_message="STRIPE_SECRET_KEY must start with sk_test_ or sk_live_",
            sensitive=True
        ))

        # Paths
        self.add_rule(ConfigRule(
            'DATA_DIR',
            default='./data',
            validator=lambda x: self._is_valid_path(x),
            error_message="DATA_DIR must be a valid directory path"
        ))

        self.add_rule(ConfigRule(
            'MODELS_DIR',
            default='./models',
            validator=lambda x: self._is_valid_path(x),
            error_message="MODELS_DIR must be a valid directory path"
        ))

        # Numerische Grenzen
        self.add_rule(ConfigRule(
            'MAX_WORKERS',
            default=4,
            validator=lambda x: 1 <= int(x) <= 100,
            error_message="MAX_WORKERS must be between 1 and 100"
        ))

        self.add_rule(ConfigRule(
            'JOB_TIMEOUT',
            default=300,
            validator=lambda x: 10 <= int(x) <= 86400,
            error_message="JOB_TIMEOUT must be between 10 and 86400 seconds"
        ))

    def add_rule(self, rule: ConfigRule):
        """Fügt Validierungsregel hinzu"""
        self.rules[rule.name] = rule

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validiert Konfiguration

        Returns:
            Liste von Fehlermeldungen
        """
        errors = []

        for name, rule in self.rules.items():
            value = config.get(name, os.environ.get(name, rule.default))

            # Required check
            if rule.required and value is None:
                errors.append(f"Missing required config: {name}")
                continue

            if value is None:
                continue

            # Custom validator
            if rule.validator:
                try:
                    if not rule.validator(value):
                        errors.append(rule.error_message or f"Invalid value for {name}")
                except Exception as e:
                    errors.append(f"Validation error for {name}: {e}")

        return errors

    def validate_and_raise(self, config: Dict[str, Any]):
        """Validiert und wirft Exception bei Fehlern"""
        errors = self.validate(config)
        if errors:
            raise ConfigValidationError(
                "Configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    def get_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Gibt Konfiguration mit Defaults zurück"""
        result = dict(config)
        for name, rule in self.rules.items():
            if name not in result and rule.default is not None:
                result[name] = rule.default
        return result

    @staticmethod
    def _is_valid_ip(ip: str) -> bool:
        """Prüft ob gültige IP-Adresse"""
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(part) <= 255 for part in parts)
        except ValueError:
            return False

    @staticmethod
    def _is_valid_path(path: str) -> bool:
        """Prüft ob gültiger Pfad (muss nicht existieren)"""
        try:
            Path(path)
            return True
        except Exception:
            return False


class ConfigValidationError(Exception):
    """Exception für Konfigurationsfehler"""
    pass


# Globale Instanz
_validator: Optional[ConfigValidator] = None


def get_config_validator() -> ConfigValidator:
    """Gibt globale Validator-Instanz zurück"""
    global _validator
    if _validator is None:
        _validator = ConfigValidator()
    return _validator


def validate_config(config: Dict[str, Any] = None) -> List[str]:
    """
    Validiert Konfiguration (nutzt os.environ wenn config=None)

    Returns:
        Liste von Fehlermeldungen
    """
    validator = get_config_validator()
    return validator.validate(config or dict(os.environ))
