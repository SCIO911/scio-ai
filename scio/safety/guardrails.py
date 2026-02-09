#!/usr/bin/env python3
"""
SCIO - Guardrails System

Sicherheits- und Ethik-Richtlinien fuer SCIO:
- Content Filtering
- Rate Limiting
- Action Validation
- Ethical Guidelines
- Risk Assessment
"""

import re
import time
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from threading import Lock


class RiskLevel(str, Enum):
    """Risikostufen"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class ActionType(str, Enum):
    """Aktionstypen"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FINANCIAL = "financial"
    SYSTEM = "system"
    DATA_ACCESS = "data_access"


@dataclass
class ValidationResult:
    """Ergebnis einer Validierung"""
    is_valid: bool
    risk_level: RiskLevel = RiskLevel.SAFE
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    blocked_content: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class GuardrailConfig:
    """Guardrail Konfiguration"""
    # Content Filtering
    enable_content_filter: bool = True
    blocked_patterns: List[str] = field(default_factory=list)
    sensitive_keywords: List[str] = field(default_factory=list)

    # Rate Limiting
    enable_rate_limit: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

    # Financial Limits
    max_single_transaction: float = 100.0  # USD
    max_daily_spending: float = 500.0  # USD
    require_confirmation_above: float = 50.0  # USD

    # Action Limits
    max_file_size_mb: float = 100.0
    max_api_calls_per_minute: int = 30
    max_concurrent_operations: int = 10

    # Safety Flags
    allow_code_execution: bool = True
    allow_network_access: bool = True
    allow_file_write: bool = True
    allow_system_commands: bool = False
    allow_financial_operations: bool = True


class ContentFilter:
    """Filter fuer schaedliche/unangemessene Inhalte"""

    # Standard-Blocklisten
    HARMFUL_PATTERNS = [
        r'(?i)(password|passwort|secret)\s*[:=]\s*[\'"][^\'"]+[\'"]',  # Exposed Secrets
        r'(?i)(api[_-]?key|token)\s*[:=]\s*[\'"][^\'"]+[\'"]',  # API Keys
        r'(?i)(rm\s+-rf\s+/|format\s+c:)',  # Dangerous Commands
        r'(?i)(drop\s+table|delete\s+from\s+\*|truncate)',  # SQL Injection
        r'<script[^>]*>.*?</script>',  # XSS
    ]

    SENSITIVE_CATEGORIES = [
        "personal_data",
        "financial_info",
        "health_records",
        "credentials",
        "private_keys"
    ]

    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self._compiled_patterns = [
            re.compile(p) for p in self.HARMFUL_PATTERNS + self.config.blocked_patterns
        ]

    def filter(self, content: str) -> ValidationResult:
        """Filtert Content"""
        result = ValidationResult(is_valid=True)
        blocked = []
        warnings = []

        # Pattern-Matching
        for i, pattern in enumerate(self._compiled_patterns):
            matches = pattern.findall(content)
            if matches:
                if i < len(self.HARMFUL_PATTERNS):
                    # Harmful Pattern
                    blocked.extend(matches)
                    result.risk_level = RiskLevel.HIGH
                else:
                    # Custom Pattern
                    warnings.append(f"Pattern match: {matches[0][:50]}...")

        # Sensitive Keywords
        content_lower = content.lower()
        for keyword in self.config.sensitive_keywords:
            if keyword.lower() in content_lower:
                warnings.append(f"Sensitive keyword: {keyword}")
                if result.risk_level == RiskLevel.SAFE:
                    result.risk_level = RiskLevel.LOW

        if blocked:
            result.is_valid = False
            result.blocked_content = blocked
            result.reason = "Harmful content detected"

        result.warnings = warnings
        return result

    def redact_sensitive(self, content: str) -> str:
        """Redaktiert sensitive Informationen"""
        redacted = content

        # Patterns redaktieren
        for pattern in self._compiled_patterns:
            redacted = pattern.sub("[REDACTED]", redacted)

        # Email-Adressen redaktieren
        redacted = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            redacted
        )

        # Telefonnummern redaktieren
        redacted = re.sub(
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            '[PHONE]',
            redacted
        )

        return redacted


class RateLimiter:
    """Rate Limiting"""

    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, identifier: str = "default") -> ValidationResult:
        """Prueft Rate Limit"""
        if not self.config.enable_rate_limit:
            return ValidationResult(is_valid=True)

        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        with self._lock:
            # Alte Eintraege entfernen
            self._requests[identifier] = [
                t for t in self._requests[identifier]
                if t > hour_ago
            ]

            requests = self._requests[identifier]

            # Requests pro Minute
            minute_requests = sum(1 for t in requests if t > minute_ago)
            if minute_requests >= self.config.requests_per_minute:
                return ValidationResult(
                    is_valid=False,
                    risk_level=RiskLevel.BLOCKED,
                    reason=f"Rate limit exceeded: {minute_requests}/{self.config.requests_per_minute} per minute"
                )

            # Requests pro Stunde
            if len(requests) >= self.config.requests_per_hour:
                return ValidationResult(
                    is_valid=False,
                    risk_level=RiskLevel.BLOCKED,
                    reason=f"Rate limit exceeded: {len(requests)}/{self.config.requests_per_hour} per hour"
                )

            # Request registrieren
            self._requests[identifier].append(now)

        return ValidationResult(is_valid=True)

    def get_remaining(self, identifier: str = "default") -> Dict[str, int]:
        """Gibt verbleibende Requests zurueck"""
        now = time.time()

        with self._lock:
            requests = self._requests.get(identifier, [])
            minute_requests = sum(1 for t in requests if t > now - 60)
            hour_requests = len([t for t in requests if t > now - 3600])

        return {
            "minute": max(0, self.config.requests_per_minute - minute_requests),
            "hour": max(0, self.config.requests_per_hour - hour_requests)
        }


class ActionValidator:
    """Validiert Aktionen"""

    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self._daily_spending = 0.0
        self._spending_reset = time.time()
        self._confirmation_callbacks: List[Callable] = []

    def validate_action(self, action_type: ActionType, details: Dict = None) -> ValidationResult:
        """Validiert eine Aktion"""
        details = details or {}

        # Generelle Checks
        if action_type == ActionType.EXECUTE and not self.config.allow_code_execution:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                reason="Code execution is disabled"
            )

        if action_type == ActionType.NETWORK and not self.config.allow_network_access:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                reason="Network access is disabled"
            )

        if action_type == ActionType.WRITE and not self.config.allow_file_write:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                reason="File writing is disabled"
            )

        if action_type == ActionType.SYSTEM and not self.config.allow_system_commands:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                reason="System commands are disabled"
            )

        # Spezifische Checks
        if action_type == ActionType.FINANCIAL:
            return self._validate_financial(details)

        if action_type == ActionType.WRITE:
            return self._validate_write(details)

        return ValidationResult(is_valid=True)

    def _validate_financial(self, details: Dict) -> ValidationResult:
        """Validiert finanzielle Aktion"""
        if not self.config.allow_financial_operations:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                reason="Financial operations are disabled"
            )

        amount = details.get("amount", 0.0)

        # Einzeltransaktions-Limit
        if amount > self.config.max_single_transaction:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.HIGH,
                reason=f"Amount ${amount:.2f} exceeds single transaction limit ${self.config.max_single_transaction:.2f}"
            )

        # Tages-Limit pruefen
        self._reset_daily_if_needed()

        if self._daily_spending + amount > self.config.max_daily_spending:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.HIGH,
                reason=f"Daily spending limit would be exceeded (${self._daily_spending:.2f} + ${amount:.2f} > ${self.config.max_daily_spending:.2f})"
            )

        # Bestaetigung erforderlich?
        result = ValidationResult(is_valid=True)

        if amount > self.config.require_confirmation_above:
            result.warnings.append(f"Amount ${amount:.2f} requires confirmation")
            result.risk_level = RiskLevel.MEDIUM

        return result

    def _validate_write(self, details: Dict) -> ValidationResult:
        """Validiert Schreibaktion"""
        file_size_mb = details.get("size_mb", 0)

        if file_size_mb > self.config.max_file_size_mb:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.MEDIUM,
                reason=f"File size {file_size_mb}MB exceeds limit {self.config.max_file_size_mb}MB"
            )

        return ValidationResult(is_valid=True)

    def _reset_daily_if_needed(self):
        """Reset taegliches Spending wenn noetig"""
        now = time.time()
        if now - self._spending_reset > 86400:  # 24 Stunden
            self._daily_spending = 0.0
            self._spending_reset = now

    def record_spending(self, amount: float):
        """Zeichnet Ausgabe auf"""
        self._reset_daily_if_needed()
        self._daily_spending += amount

    def on_confirmation_needed(self, callback: Callable):
        """Registriert Callback fuer Bestaetigung"""
        self._confirmation_callbacks.append(callback)


class EthicalGuardrails:
    """Ethische Richtlinien"""

    PROHIBITED_PURPOSES = [
        "illegal_activity",
        "harm_to_persons",
        "discrimination",
        "privacy_violation",
        "deception",
        "manipulation"
    ]

    ETHICAL_PRINCIPLES = [
        "beneficence",        # Gutes tun
        "non_maleficence",    # Schaden vermeiden
        "autonomy",           # Nutzer-Autonomie respektieren
        "justice",            # Fair und gerecht handeln
        "transparency",       # Transparent sein
        "accountability"      # Verantwortung uebernehmen
    ]

    def __init__(self):
        self._warnings: List[str] = []

    def evaluate(self, action: str, context: Dict = None) -> ValidationResult:
        """Evaluiert ethische Aspekte"""
        context = context or {}
        result = ValidationResult(is_valid=True)

        action_lower = action.lower()

        # Verbotene Zwecke pruefen
        for purpose in self.PROHIBITED_PURPOSES:
            if purpose.replace("_", " ") in action_lower:
                result.is_valid = False
                result.risk_level = RiskLevel.CRITICAL
                result.reason = f"Action violates ethical guideline: {purpose}"
                return result

        # Ethische Bedenken pruefen
        if "personal data" in action_lower or "private" in action_lower:
            result.warnings.append("This action may involve personal data - ensure consent is obtained")
            result.risk_level = RiskLevel.MEDIUM

        if "automated decision" in action_lower:
            result.warnings.append("Automated decisions should be transparent and explainable")

        if "financial" in action_lower and context.get("amount", 0) > 0:
            result.warnings.append("Financial actions require careful consideration")

        return result

    def check_bias(self, data: Any, protected_attributes: List[str] = None) -> ValidationResult:
        """Prueft auf Bias"""
        protected = protected_attributes or ["gender", "race", "age", "religion", "nationality"]

        result = ValidationResult(is_valid=True)

        if isinstance(data, dict):
            for attr in protected:
                if attr in data:
                    result.warnings.append(f"Protected attribute '{attr}' detected - ensure fair treatment")
                    result.risk_level = RiskLevel.LOW

        return result


class Guardrails:
    """Haupt-Guardrails System"""

    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self.content_filter = ContentFilter(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.action_validator = ActionValidator(self.config)
        self.ethical = EthicalGuardrails()

        self._audit_log: List[Dict] = []

    def check(self, action: str, action_type: ActionType = ActionType.READ,
              content: str = None, context: Dict = None) -> ValidationResult:
        """
        Fuehrt alle Guardrail-Checks durch

        Args:
            action: Beschreibung der Aktion
            action_type: Typ der Aktion
            content: Optionaler Content zum Pruefen
            context: Zusaetzlicher Kontext
        """
        context = context or {}
        combined_result = ValidationResult(is_valid=True)

        # 1. Rate Limiting
        rate_result = self.rate_limiter.check(context.get("identifier", "default"))
        if not rate_result.is_valid:
            self._log_check("rate_limit", action, rate_result)
            return rate_result

        # 2. Content Filtering
        if content and self.config.enable_content_filter:
            content_result = self.content_filter.filter(content)
            if not content_result.is_valid:
                self._log_check("content_filter", action, content_result)
                return content_result
            combined_result.warnings.extend(content_result.warnings)

        # 3. Action Validation
        action_result = self.action_validator.validate_action(action_type, context)
        if not action_result.is_valid:
            self._log_check("action_validator", action, action_result)
            return action_result
        combined_result.warnings.extend(action_result.warnings)

        # 4. Ethical Check
        ethical_result = self.ethical.evaluate(action, context)
        if not ethical_result.is_valid:
            self._log_check("ethical", action, ethical_result)
            return ethical_result
        combined_result.warnings.extend(ethical_result.warnings)

        # Hoechstes Risiko-Level uebernehmen
        risk_levels = [
            content_result.risk_level if content else RiskLevel.SAFE,
            action_result.risk_level,
            ethical_result.risk_level
        ]
        combined_result.risk_level = max(risk_levels, key=lambda x: list(RiskLevel).index(x))

        self._log_check("combined", action, combined_result)
        return combined_result

    def _log_check(self, check_type: str, action: str, result: ValidationResult):
        """Loggt Check-Ergebnis"""
        self._audit_log.append({
            "timestamp": time.time(),
            "check_type": check_type,
            "action": action[:100],
            "is_valid": result.is_valid,
            "risk_level": result.risk_level.value,
            "reason": result.reason
        })

        # Log begrenzen
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Gibt Audit Log zurueck"""
        return self._audit_log[-limit:]

    def get_risk_summary(self) -> Dict:
        """Gibt Risiko-Zusammenfassung"""
        if not self._audit_log:
            return {"total_checks": 0}

        recent = self._audit_log[-1000:]

        return {
            "total_checks": len(recent),
            "blocked": sum(1 for l in recent if not l["is_valid"]),
            "high_risk": sum(1 for l in recent if l["risk_level"] in ["high", "critical"]),
            "by_level": {
                level.value: sum(1 for l in recent if l["risk_level"] == level.value)
                for level in RiskLevel
            }
        }


# Singleton
_guardrails: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    """Gibt Guardrails Singleton zurueck"""
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails


def check_action(action: str, action_type: ActionType = ActionType.READ,
                 content: str = None, context: Dict = None) -> ValidationResult:
    """Convenience-Funktion fuer schnellen Check"""
    return get_guardrails().check(action, action_type, content, context)
