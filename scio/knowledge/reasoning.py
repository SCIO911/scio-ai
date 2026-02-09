"""
SCIO Reasoning Engine

Logisches Schlussfolgern und Inferenz ueber Wissen.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from enum import Enum, auto
from datetime import datetime

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class RuleType(str, Enum):
    """Typen von Logik-Regeln"""
    IMPLICATION = "implication"  # A -> B
    EQUIVALENCE = "equivalence"  # A <-> B
    CONJUNCTION = "conjunction"  # A AND B
    DISJUNCTION = "disjunction"  # A OR B
    NEGATION = "negation"  # NOT A
    UNIVERSAL = "universal"  # Fuer alle X
    EXISTENTIAL = "existential"  # Es existiert X
    CUSTOM = "custom"


class InferenceStatus(str, Enum):
    """Status einer Inferenz"""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"


@dataclass
class LogicRule:
    """Eine logische Regel fuer Inferenz"""

    id: str = field(default_factory=lambda: generate_id("rule"))
    name: str = ""
    rule_type: RuleType = RuleType.IMPLICATION
    antecedent: List[str] = field(default_factory=list)  # Voraussetzungen
    consequent: List[str] = field(default_factory=list)  # Schlussfolgerungen
    confidence: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "rule_type": self.rule_type.value,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "confidence": self.confidence,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogicRule':
        return cls(
            id=data.get("id", generate_id("rule")),
            name=data.get("name", ""),
            rule_type=RuleType(data.get("rule_type", "implication")),
            antecedent=data.get("antecedent", []),
            consequent=data.get("consequent", []),
            confidence=data.get("confidence", 1.0),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else now_utc(),
        )

    def matches(self, facts: Set[str]) -> bool:
        """Prueft ob alle Antezedenzien in den Fakten enthalten sind"""
        return all(ant in facts for ant in self.antecedent)

    def apply(self, facts: Set[str]) -> Set[str]:
        """Wendet Regel an und gibt neue Fakten zurueck"""
        if self.matches(facts):
            return set(self.consequent)
        return set()

    def to_text(self) -> str:
        """Konvertiert Regel zu lesbarem Text"""
        ant = " AND ".join(self.antecedent) if self.antecedent else "TRUE"
        con = " AND ".join(self.consequent) if self.consequent else "TRUE"
        return f"IF ({ant}) THEN ({con})"


@dataclass
class InferenceResult:
    """Ergebnis einer Inferenz"""

    conclusion: str
    status: InferenceStatus
    confidence: float
    supporting_rules: List[LogicRule] = field(default_factory=list)
    supporting_facts: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status == InferenceStatus.VALID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusion": self.conclusion,
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_rules": [r.name for r in self.supporting_rules],
            "supporting_facts": self.supporting_facts,
            "reasoning_chain": self.reasoning_chain,
        }

    def explain(self) -> str:
        """Erzeugt Erklaerung der Inferenz"""
        lines = [f"Conclusion: {self.conclusion}"]
        lines.append(f"Status: {self.status.value}")
        lines.append(f"Confidence: {self.confidence:.2f}")

        if self.reasoning_chain:
            lines.append("\nReasoning:")
            for i, step in enumerate(self.reasoning_chain, 1):
                lines.append(f"  {i}. {step}")

        if self.supporting_facts:
            lines.append("\nSupporting Facts:")
            for fact in self.supporting_facts:
                lines.append(f"  - {fact}")

        return "\n".join(lines)


class ReasoningEngine:
    """
    Engine fuer logisches Schlussfolgern.

    Features:
    - Forward Chaining (Vorwaertsverkettung)
    - Backward Chaining (Rueckwaertsverkettung)
    - Konfidenz-Propagation
    - Erklaerung der Schlussfolgerungen
    """

    def __init__(self):
        self._rules: Dict[str, LogicRule] = {}
        self._facts: Set[str] = set()
        self._derived_facts: Dict[str, Tuple[float, List[str]]] = {}  # fact -> (confidence, rule_ids)

    def add_rule(self, rule: LogicRule) -> LogicRule:
        """Fuegt eine Regel hinzu"""
        self._rules[rule.id] = rule
        logger.debug("Rule added", id=rule.id, name=rule.name)
        return rule

    def create_rule(
        self,
        name: str,
        antecedent: List[str],
        consequent: List[str],
        confidence: float = 1.0,
        description: str = "",
    ) -> LogicRule:
        """Erstellt und fuegt eine neue Regel hinzu"""
        rule = LogicRule(
            name=name,
            antecedent=antecedent,
            consequent=consequent,
            confidence=confidence,
            description=description,
        )
        return self.add_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Entfernt eine Regel"""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def add_fact(self, fact: str) -> None:
        """Fuegt ein Faktum hinzu"""
        self._facts.add(fact.lower())

    def add_facts(self, facts: List[str]) -> None:
        """Fuegt mehrere Fakten hinzu"""
        for fact in facts:
            self.add_fact(fact)

    def remove_fact(self, fact: str) -> bool:
        """Entfernt ein Faktum"""
        fact_lower = fact.lower()
        if fact_lower in self._facts:
            self._facts.discard(fact_lower)
            return True
        return False

    def clear_facts(self) -> None:
        """Loescht alle Fakten"""
        self._facts.clear()
        self._derived_facts.clear()

    def get_facts(self) -> Set[str]:
        """Gibt alle Fakten zurueck"""
        return self._facts.copy()

    def forward_chain(self, max_iterations: int = 100) -> List[InferenceResult]:
        """
        Forward Chaining - leitet neue Fakten aus bestehenden ab.

        Wendet Regeln iterativ an, bis keine neuen Fakten mehr abgeleitet werden.
        """
        results = []
        all_facts = self._facts.copy()
        iterations = 0

        while iterations < max_iterations:
            new_facts = set()
            iterations += 1

            for rule in self._rules.values():
                if rule.matches(all_facts):
                    derived = rule.apply(all_facts)
                    for fact in derived:
                        fact_lower = fact.lower()
                        if fact_lower not in all_facts:
                            new_facts.add(fact_lower)

                            # Berechne Konfidenz
                            ant_confidences = [
                                self._derived_facts.get(a, (1.0, []))[0]
                                for a in rule.antecedent
                            ]
                            confidence = rule.confidence * min(ant_confidences) if ant_confidences else rule.confidence

                            self._derived_facts[fact_lower] = (confidence, [rule.id])

                            results.append(InferenceResult(
                                conclusion=fact,
                                status=InferenceStatus.VALID,
                                confidence=confidence,
                                supporting_rules=[rule],
                                supporting_facts=list(rule.antecedent),
                                reasoning_chain=[
                                    f"Given: {', '.join(rule.antecedent)}",
                                    f"Applied rule: {rule.name}",
                                    f"Concluded: {fact}",
                                ],
                            ))

            if not new_facts:
                break

            all_facts.update(new_facts)

        self._facts = all_facts
        logger.debug("Forward chaining complete", iterations=iterations, new_facts=len(results))
        return results

    def backward_chain(self, goal: str) -> InferenceResult:
        """
        Backward Chaining - prueft ob ein Ziel aus den Fakten ableitbar ist.
        """
        goal_lower = goal.lower()

        # Ziel bereits bekannt?
        if goal_lower in self._facts:
            return InferenceResult(
                conclusion=goal,
                status=InferenceStatus.VALID,
                confidence=self._derived_facts.get(goal_lower, (1.0, []))[0],
                supporting_facts=[goal],
                reasoning_chain=[f"'{goal}' is a known fact"],
            )

        # Suche Regeln, die das Ziel ableiten
        for rule in self._rules.values():
            if goal_lower in [c.lower() for c in rule.consequent]:
                # Pruefe ob alle Antezedenzien erfuellbar sind
                all_satisfied = True
                sub_results = []

                for antecedent in rule.antecedent:
                    sub_result = self.backward_chain(antecedent)
                    if not sub_result.is_valid:
                        all_satisfied = False
                        break
                    sub_results.append(sub_result)

                if all_satisfied:
                    # Berechne kombinierte Konfidenz
                    confidence = rule.confidence
                    for sr in sub_results:
                        confidence *= sr.confidence

                    chain = []
                    for sr in sub_results:
                        chain.extend(sr.reasoning_chain)
                    chain.append(f"Applied rule: {rule.name}")
                    chain.append(f"Concluded: {goal}")

                    return InferenceResult(
                        conclusion=goal,
                        status=InferenceStatus.VALID,
                        confidence=confidence,
                        supporting_rules=[rule] + [r for sr in sub_results for r in sr.supporting_rules],
                        supporting_facts=list(rule.antecedent),
                        reasoning_chain=chain,
                    )

        # Ziel nicht ableitbar
        return InferenceResult(
            conclusion=goal,
            status=InferenceStatus.INVALID,
            confidence=0.0,
            reasoning_chain=[f"Cannot derive '{goal}' from known facts and rules"],
        )

    def query(self, question: str) -> InferenceResult:
        """Beantwortet eine Frage durch Inferenz"""
        # Einfache Frage-Parsing
        question_lower = question.lower().strip()

        # "Is X true?" oder "X?"
        if question_lower.startswith("is "):
            goal = question_lower[3:].rstrip("?").strip()
        else:
            goal = question_lower.rstrip("?").strip()

        # Versuche zuerst Forward Chaining
        self.forward_chain()

        # Dann Backward Chaining fuer das Ziel
        return self.backward_chain(goal)

    def explain(self, fact: str) -> str:
        """Erklaert wie ein Fakt abgeleitet wurde"""
        result = self.backward_chain(fact)
        return result.explain()

    def check_consistency(self) -> List[Tuple[str, str]]:
        """Prueft auf Widersprueche in den Fakten"""
        contradictions = []

        # Suche nach einfachen Widerspruechen (A und NOT A)
        for fact in self._facts:
            negated = f"not {fact}"
            if negated in self._facts:
                contradictions.append((fact, negated))

            # Pruefe auch umgekehrt
            if fact.startswith("not "):
                positive = fact[4:]
                if positive in self._facts:
                    contradictions.append((positive, fact))

        return contradictions

    def get_rules(self) -> List[LogicRule]:
        """Gibt alle Regeln zurueck"""
        return list(self._rules.values())

    def stats(self) -> Dict[str, Any]:
        """Gibt Statistiken zurueck"""
        return {
            "rule_count": len(self._rules),
            "fact_count": len(self._facts),
            "derived_fact_count": len(self._derived_facts),
            "rules_by_type": {
                rt.value: sum(1 for r in self._rules.values() if r.rule_type == rt)
                for rt in RuleType
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialisiert die Engine"""
        return {
            "rules": [r.to_dict() for r in self._rules.values()],
            "facts": list(self._facts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningEngine':
        """Deserialisiert eine Engine"""
        engine = cls()
        for rule_data in data.get("rules", []):
            engine.add_rule(LogicRule.from_dict(rule_data))
        for fact in data.get("facts", []):
            engine.add_fact(fact)
        return engine


__all__ = [
    'RuleType',
    'InferenceStatus',
    'LogicRule',
    'InferenceResult',
    'ReasoningEngine',
]
