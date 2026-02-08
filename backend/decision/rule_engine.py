#!/usr/bin/env python3
"""
SCIO - Rule Engine
Regelbasierte Entscheidungen und Geschäftslogik
"""

import re
import json
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class RulePriority(int, Enum):
    """Priorität von Regeln"""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    FALLBACK = 0


class RuleOperator(str, Enum):
    """Vergleichsoperatoren für Regeln"""
    EQUALS = 'eq'
    NOT_EQUALS = 'neq'
    GREATER = 'gt'
    GREATER_EQ = 'gte'
    LESS = 'lt'
    LESS_EQ = 'lte'
    IN = 'in'
    NOT_IN = 'not_in'
    CONTAINS = 'contains'
    MATCHES = 'matches'  # Regex
    EXISTS = 'exists'
    NOT_EXISTS = 'not_exists'


@dataclass
class Condition:
    """Eine Bedingung in einer Regel"""
    field: str
    operator: RuleOperator
    value: Any

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluiert die Bedingung gegen den Kontext"""
        # Feld aus Kontext extrahieren (unterstützt nested fields wie "user.role")
        field_value = context
        for part in self.field.split('.'):
            if isinstance(field_value, dict):
                field_value = field_value.get(part)
            else:
                field_value = None
                break

        # Operator anwenden
        if self.operator == RuleOperator.EQUALS:
            return field_value == self.value
        elif self.operator == RuleOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == RuleOperator.GREATER:
            return field_value is not None and field_value > self.value
        elif self.operator == RuleOperator.GREATER_EQ:
            return field_value is not None and field_value >= self.value
        elif self.operator == RuleOperator.LESS:
            return field_value is not None and field_value < self.value
        elif self.operator == RuleOperator.LESS_EQ:
            return field_value is not None and field_value <= self.value
        elif self.operator == RuleOperator.IN:
            return field_value in self.value
        elif self.operator == RuleOperator.NOT_IN:
            return field_value not in self.value
        elif self.operator == RuleOperator.CONTAINS:
            return self.value in str(field_value) if field_value else False
        elif self.operator == RuleOperator.MATCHES:
            return bool(re.match(self.value, str(field_value))) if field_value else False
        elif self.operator == RuleOperator.EXISTS:
            return field_value is not None
        elif self.operator == RuleOperator.NOT_EXISTS:
            return field_value is None

        return False


@dataclass
class Rule:
    """
    Eine Geschäftsregel mit Bedingungen und Aktionen
    """
    id: str
    name: str
    description: str
    conditions: List[Condition]
    action: str
    action_params: Dict[str, Any] = field(default_factory=dict)
    priority: RulePriority = RulePriority.NORMAL
    enabled: bool = True
    match_all: bool = True  # True = AND, False = OR
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    hit_count: int = 0
    last_triggered: Optional[datetime] = None

    def matches(self, context: Dict[str, Any]) -> bool:
        """Prüft ob alle/eine Bedingung(en) erfüllt sind"""
        if not self.enabled:
            return False

        if not self.conditions:
            return True  # Keine Bedingungen = immer wahr

        results = [c.evaluate(context) for c in self.conditions]

        if self.match_all:
            return all(results)
        else:
            return any(results)

    def trigger(self) -> Dict[str, Any]:
        """Gibt die Aktion zurück und aktualisiert Tracking"""
        self.hit_count += 1
        self.last_triggered = datetime.now()
        return {
            "action": self.action,
            "params": self.action_params,
            "rule_id": self.id,
            "rule_name": self.name,
        }


@dataclass
class RuleSet:
    """Eine Sammlung von Regeln für einen Bereich"""
    name: str
    rules: List[Rule] = field(default_factory=list)

    def add_rule(self, rule: Rule):
        """Fügt eine Regel hinzu"""
        self.rules.append(rule)
        # Nach Priorität sortieren
        self.rules.sort(key=lambda r: r.priority.value, reverse=True)

    def remove_rule(self, rule_id: str):
        """Entfernt eine Regel"""
        self.rules = [r for r in self.rules if r.id != rule_id]

    def find_matching(self, context: Dict[str, Any], first_only: bool = True) -> List[Rule]:
        """Findet passende Regeln"""
        matching = [r for r in self.rules if r.matches(context)]
        if first_only and matching:
            return [matching[0]]
        return matching


class RuleEngine:
    """
    SCIO Rule Engine

    Verarbeitet Geschäftsregeln für automatisierte Entscheidungen.
    Unterstützt:
    - Bedingungs-basierte Regeln
    - Prioritäten und Reihenfolge
    - AND/OR Logik
    - Verschachtelte Felder
    - Regex-Matching
    - Tracking und Statistiken
    """

    def __init__(self):
        self.rule_sets: Dict[str, RuleSet] = {}
        self.global_rules: List[Rule] = []
        self.action_handlers: Dict[str, Callable] = {}
        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert die Rule Engine mit Standard-Regeln"""
        try:
            # Standard-Regelsets erstellen
            self._create_api_rules()
            self._create_resource_rules()
            self._create_security_rules()

            self._initialized = True
            print("[OK] Rule Engine initialisiert")
            return True
        except Exception as e:
            print(f"[ERROR] Rule Engine Fehler: {e}")
            return False

    def _create_api_rules(self):
        """Erstellt API-bezogene Regeln"""
        api_rules = RuleSet("api")

        # Rate Limiting
        api_rules.add_rule(Rule(
            id="rate_limit_exceeded",
            name="Rate Limit Exceeded",
            description="Blockiert Anfragen bei überschrittenem Rate Limit",
            conditions=[
                Condition("requests_per_minute", RuleOperator.GREATER, 60),
                Condition("is_premium", RuleOperator.EQUALS, False)
            ],
            action="rate_limit",
            action_params={"wait_seconds": 60, "message": "Rate limit exceeded"},
            priority=RulePriority.CRITICAL
        ))

        # Premium Priority
        api_rules.add_rule(Rule(
            id="premium_priority",
            name="Premium User Priority",
            description="Priorisiert Premium-Nutzer",
            conditions=[
                Condition("is_premium", RuleOperator.EQUALS, True)
            ],
            action="prioritize",
            action_params={"boost": 50},
            priority=RulePriority.HIGH
        ))

        # Large Request Handling
        api_rules.add_rule(Rule(
            id="large_request",
            name="Large Request Handler",
            description="Behandelt große Anfragen speziell",
            conditions=[
                Condition("token_count", RuleOperator.GREATER, 10000)
            ],
            action="queue_async",
            action_params={"timeout": 300},
            priority=RulePriority.NORMAL
        ))

        self.rule_sets["api"] = api_rules

    def _create_resource_rules(self):
        """Erstellt Ressourcen-Management Regeln"""
        resource_rules = RuleSet("resource")

        # VRAM Critical
        resource_rules.add_rule(Rule(
            id="vram_critical",
            name="VRAM Critical",
            description="Kritischer VRAM-Stand",
            conditions=[
                Condition("vram_percent", RuleOperator.GREATER, 95)
            ],
            action="emergency_cleanup",
            action_params={"unload_count": 2},
            priority=RulePriority.CRITICAL
        ))

        # Model Preload
        resource_rules.add_rule(Rule(
            id="model_preload",
            name="Model Preload",
            description="Lädt häufig genutzte Modelle vor",
            conditions=[
                Condition("vram_percent", RuleOperator.LESS, 50),
                Condition("queue_size", RuleOperator.LESS, 3)
            ],
            action="preload_model",
            action_params={"model": "default"},
            priority=RulePriority.LOW
        ))

        self.rule_sets["resource"] = resource_rules

    def _create_security_rules(self):
        """Erstellt Sicherheits-Regeln"""
        security_rules = RuleSet("security")

        # Invalid API Key
        security_rules.add_rule(Rule(
            id="invalid_api_key",
            name="Invalid API Key",
            description="Blockiert ungültige API Keys",
            conditions=[
                Condition("api_key_valid", RuleOperator.EQUALS, False)
            ],
            action="reject",
            action_params={"status": 401, "message": "Invalid API key"},
            priority=RulePriority.CRITICAL
        ))

        # Suspicious Pattern
        security_rules.add_rule(Rule(
            id="suspicious_pattern",
            name="Suspicious Pattern",
            description="Erkennt verdächtige Muster",
            conditions=[
                Condition("input", RuleOperator.MATCHES, r".*(DROP\s+TABLE|DELETE\s+FROM|<script>).*")
            ],
            action="block",
            action_params={"reason": "Suspicious input pattern"},
            priority=RulePriority.CRITICAL
        ))

        self.rule_sets["security"] = security_rules

    def evaluate(self,
                 rule_set_name: str,
                 context: Dict[str, Any],
                 execute: bool = False) -> Dict[str, Any]:
        """
        Evaluiert Regeln gegen den Kontext

        Args:
            rule_set_name: Name des Regelsets
            context: Kontext-Daten
            execute: Ob die Aktion ausgeführt werden soll

        Returns:
            Dict mit Ergebnis
        """
        rule_set = self.rule_sets.get(rule_set_name)
        if not rule_set:
            return {
                "matched": False,
                "error": f"RuleSet '{rule_set_name}' not found"
            }

        # Auch globale Regeln prüfen
        all_rules = self.global_rules + rule_set.rules
        all_rules.sort(key=lambda r: r.priority.value, reverse=True)

        for rule in all_rules:
            if rule.matches(context):
                result = rule.trigger()

                # Aktion ausführen wenn gewünscht
                if execute and result["action"] in self.action_handlers:
                    handler = self.action_handlers[result["action"]]
                    try:
                        result["execution_result"] = handler(result["params"], context)
                    except Exception as e:
                        result["execution_error"] = str(e)

                return {
                    "matched": True,
                    "result": result
                }

        return {
            "matched": False,
            "message": "No matching rules"
        }

    def evaluate_all(self,
                     context: Dict[str, Any],
                     rule_sets: List[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluiert alle passenden Regeln

        Args:
            context: Kontext-Daten
            rule_sets: Optional: Nur bestimmte Regelsets

        Returns:
            Liste aller Matches
        """
        results = []

        sets_to_check = rule_sets or list(self.rule_sets.keys())

        for set_name in sets_to_check:
            result = self.evaluate(set_name, context, execute=False)
            if result.get("matched"):
                result["rule_set"] = set_name
                results.append(result)

        return results

    def add_rule_set(self, rule_set: RuleSet):
        """Fügt ein Regelset hinzu"""
        self.rule_sets[rule_set.name] = rule_set

    def add_rule(self, rule_set_name: str, rule: Rule):
        """Fügt eine Regel zu einem Set hinzu"""
        if rule_set_name not in self.rule_sets:
            self.rule_sets[rule_set_name] = RuleSet(rule_set_name)
        self.rule_sets[rule_set_name].add_rule(rule)

    def add_global_rule(self, rule: Rule):
        """Fügt eine globale Regel hinzu (wird immer geprüft)"""
        self.global_rules.append(rule)
        self.global_rules.sort(key=lambda r: r.priority.value, reverse=True)

    def register_action_handler(self, action: str, handler: Callable):
        """Registriert einen Handler für eine Aktion"""
        self.action_handlers[action] = handler

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        stats = {
            "rule_sets": {},
            "global_rules_count": len(self.global_rules),
            "action_handlers_count": len(self.action_handlers),
            "total_rules": len(self.global_rules)
        }

        for name, rule_set in self.rule_sets.items():
            set_stats = {
                "rules_count": len(rule_set.rules),
                "rules": []
            }

            for rule in rule_set.rules:
                set_stats["rules"].append({
                    "id": rule.id,
                    "name": rule.name,
                    "priority": rule.priority.name,
                    "enabled": rule.enabled,
                    "hit_count": rule.hit_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                })

            stats["rule_sets"][name] = set_stats
            stats["total_rules"] += len(rule_set.rules)

        return stats

    def export_rules(self) -> str:
        """Exportiert alle Regeln als JSON"""
        export = {
            "rule_sets": {},
            "global_rules": []
        }

        for name, rule_set in self.rule_sets.items():
            export["rule_sets"][name] = {
                "name": rule_set.name,
                "rules": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "description": r.description,
                        "conditions": [
                            {"field": c.field, "operator": c.operator.value, "value": c.value}
                            for c in r.conditions
                        ],
                        "action": r.action,
                        "action_params": r.action_params,
                        "priority": r.priority.value,
                        "enabled": r.enabled,
                        "match_all": r.match_all
                    }
                    for r in rule_set.rules
                ]
            }

        return json.dumps(export, indent=2, default=str)


# Singleton
_rule_engine: Optional[RuleEngine] = None

def get_rule_engine() -> RuleEngine:
    """Gibt Singleton-Instanz zurück"""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngine()
    return _rule_engine
