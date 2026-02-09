"""
SCIO Reasoning Engine

Implementiert formales logisches Reasoning nach wissenschaftlichen Standards:
- Deduktives Reasoning (Syllogismen, Modus Ponens, etc.)
- Induktives Reasoning (Generalisierung aus Beobachtungen)
- Abduktives Reasoning (Inference to Best Explanation)
- Bayesianisches Reasoning (probabilistische Schlussfolgerungen)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import re
import math
from collections import defaultdict

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class ReasoningType(str, Enum):
    """Arten des Reasoning."""
    DEDUCTIVE = "deductive"      # Von Allgemein zu Spezifisch (garantiert)
    INDUCTIVE = "inductive"      # Von Spezifisch zu Allgemein (wahrscheinlich)
    ABDUCTIVE = "abductive"      # Beste Erklärung (hypothetisch)
    ANALOGICAL = "analogical"    # Ähnlichkeitsbasiert
    CAUSAL = "causal"           # Ursache-Wirkung
    BAYESIAN = "bayesian"       # Probabilistisch


class ValidityStatus(str, Enum):
    """Gültigkeit eines Arguments."""
    VALID = "valid"             # Logisch gültig
    INVALID = "invalid"         # Logisch ungültig
    SOUND = "sound"             # Gültig + wahre Prämissen
    UNSOUND = "unsound"         # Gültig aber falsche Prämissen
    UNKNOWN = "unknown"         # Nicht bestimmbar


class FallacyType(str, Enum):
    """Logische Fehlschlüsse."""
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_DILEMMA = "false_dilemma"
    SLIPPERY_SLOPE = "slippery_slope"
    CIRCULAR_REASONING = "circular_reasoning"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    HASTY_GENERALIZATION = "hasty_generalization"
    POST_HOC = "post_hoc"
    EQUIVOCATION = "equivocation"
    NONE = "none"


@dataclass
class Premise:
    """
    Eine Prämisse - eine Aussage die als wahr angenommen wird.

    Attributes:
        content: Der Inhalt der Prämisse
        confidence: Vertrauen in die Wahrheit (0.0 - 1.0)
        source: Quelle der Prämisse
        is_empirical: Basiert auf Beobachtung
        is_axiom: Ist ein Axiom (per Definition wahr)
    """
    id: str = field(default_factory=lambda: generate_id("prem"))
    content: str = ""
    confidence: float = 1.0
    source: Optional[str] = None
    is_empirical: bool = False
    is_axiom: bool = False
    domain: str = "general"

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))

    def __str__(self) -> str:
        conf = f" [{self.confidence:.0%}]" if self.confidence < 1.0 else ""
        return f"{self.content}{conf}"

    def __hash__(self) -> int:
        return hash(self.content)

    def __eq__(self, other) -> bool:
        if isinstance(other, Premise):
            return self.content == other.content
        return False


@dataclass
class Conclusion:
    """
    Eine Schlussfolgerung - das Ergebnis eines Arguments.

    Attributes:
        content: Der Inhalt der Schlussfolgerung
        confidence: Vertrauen basierend auf Prämissen
        reasoning_type: Art des verwendeten Reasoning
        validity: Gültigkeit der Schlussfolgerung
    """
    id: str = field(default_factory=lambda: generate_id("conc"))
    content: str = ""
    confidence: float = 0.0
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    validity: ValidityStatus = ValidityStatus.UNKNOWN
    derived_from: List[str] = field(default_factory=list)  # Premise IDs

    def __str__(self) -> str:
        return f"{self.content} [{self.validity.value}, {self.confidence:.0%}]"


@dataclass
class Argument:
    """
    Ein vollständiges Argument mit Prämissen und Schlussfolgerung.
    """
    id: str = field(default_factory=lambda: generate_id("arg"))
    premises: List[Premise] = field(default_factory=list)
    conclusion: Optional[Conclusion] = None
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    validity: ValidityStatus = ValidityStatus.UNKNOWN
    soundness: ValidityStatus = ValidityStatus.UNKNOWN
    fallacies: List[FallacyType] = field(default_factory=list)
    explanation: str = ""
    timestamp: datetime = field(default_factory=now_utc)

    def add_premise(self, content: str, confidence: float = 1.0, **kwargs) -> Premise:
        """Fügt eine Prämisse hinzu."""
        premise = Premise(content=content, confidence=confidence, **kwargs)
        self.premises.append(premise)
        return premise

    def set_conclusion(self, content: str, **kwargs) -> Conclusion:
        """Setzt die Schlussfolgerung."""
        self.conclusion = Conclusion(
            content=content,
            reasoning_type=self.reasoning_type,
            derived_from=[p.id for p in self.premises],
            **kwargs
        )
        return self.conclusion

    def is_valid(self) -> bool:
        return self.validity in [ValidityStatus.VALID, ValidityStatus.SOUND]

    def is_sound(self) -> bool:
        return self.soundness == ValidityStatus.SOUND

    def __str__(self) -> str:
        lines = ["Argument:"]
        for i, p in enumerate(self.premises, 1):
            lines.append(f"  P{i}: {p}")
        lines.append(f"  ∴ {self.conclusion}" if self.conclusion else "  ∴ ?")
        lines.append(f"  Status: {self.validity.value}")
        if self.fallacies and self.fallacies != [FallacyType.NONE]:
            lines.append(f"  Fallacies: {', '.join(f.value for f in self.fallacies)}")
        return "\n".join(lines)


class LogicalReasoner:
    """
    Formaler logischer Reasoner.

    Implementiert:
    - Propositionale Logik
    - Prädikatenlogik (eingeschränkt)
    - Syllogistische Logik
    - Modus Ponens/Tollens
    """

    def __init__(self):
        # Bekannte logische Regeln
        self._rules: Dict[str, Callable] = {
            "modus_ponens": self._modus_ponens,
            "modus_tollens": self._modus_tollens,
            "hypothetical_syllogism": self._hypothetical_syllogism,
            "disjunctive_syllogism": self._disjunctive_syllogism,
            "conjunction": self._conjunction,
            "simplification": self._simplification,
        }

        # Bekannte Fakten
        self._knowledge_base: Set[str] = set()

        logger.info("LogicalReasoner initialized")

    def add_fact(self, fact: str) -> None:
        """Fügt einen Fakt zur Wissensbasis hinzu."""
        self._knowledge_base.add(fact.strip().lower())

    def query(self, statement: str) -> Tuple[bool, float, str]:
        """
        Prüft ob eine Aussage aus der Wissensbasis folgt.

        Returns:
            (is_true, confidence, explanation)
        """
        statement = statement.strip().lower()

        # Direkte Übereinstimmung
        if statement in self._knowledge_base:
            return True, 1.0, "Direkt in Wissensbasis gefunden."

        # Versuche Ableitung durch Inferenz-Regeln
        derived, confidence, chain = self._try_inference(statement)
        if derived:
            return True, confidence, f"Abgeleitet durch: {' -> '.join(chain)}"

        # Versuche partielle Übereinstimmung
        partial_matches = [f for f in self._knowledge_base if statement in f or f in statement]
        if partial_matches:
            return False, 0.3, f"Teilweise relevant: {partial_matches[0]}"

        return False, 0.0, "Kann nicht aus Wissensbasis abgeleitet werden."

    def _try_inference(self, target: str, depth: int = 3) -> Tuple[bool, float, List[str]]:
        """
        Versucht eine Aussage durch Anwendung von Inferenz-Regeln abzuleiten.
        """
        if depth <= 0:
            return False, 0.0, []

        # Suche nach Implikationen in der Wissensbasis: "wenn X dann Y"
        for fact in self._knowledge_base:
            # Modus Ponens: wenn "A impliziert B" und "A" bekannt, dann "B"
            if " impliziert " in fact or " dann " in fact:
                parts = fact.replace(" impliziert ", "|").replace(" dann ", "|").split("|")
                if len(parts) == 2:
                    antecedent, consequent = parts[0].strip(), parts[1].strip()
                    if consequent == target and antecedent in self._knowledge_base:
                        return True, 0.9, [antecedent, f"-> {consequent}"]

            # Transitivität: wenn "A ist B" und "B ist C", dann "A ist C"
            if " ist " in fact and " ist " in target:
                fact_parts = fact.split(" ist ")
                target_parts = target.split(" ist ")
                if len(fact_parts) == 2 and len(target_parts) == 2:
                    if fact_parts[0] == target_parts[0]:
                        bridge = f"{fact_parts[1]} ist {target_parts[1]}"
                        if bridge in self._knowledge_base:
                            return True, 0.85, [fact, bridge, f"-> {target}"]

        return False, 0.0, []

    def validate_argument(self, argument: Argument) -> Argument:
        """
        Validiert ein Argument auf logische Gültigkeit.
        """
        if not argument.premises or not argument.conclusion:
            argument.validity = ValidityStatus.INVALID
            argument.explanation = "Argument unvollständig."
            return argument

        # Prüfe auf Fehlschlüsse
        argument.fallacies = self._detect_fallacies(argument)

        # Bestimme Gültigkeit basierend auf Reasoning-Typ
        if argument.reasoning_type == ReasoningType.DEDUCTIVE:
            argument.validity, argument.explanation = self._validate_deductive(argument)
        elif argument.reasoning_type == ReasoningType.INDUCTIVE:
            argument.validity, argument.explanation = self._validate_inductive(argument)
        elif argument.reasoning_type == ReasoningType.ABDUCTIVE:
            argument.validity, argument.explanation = self._validate_abductive(argument)

        # Bestimme Solidität
        if argument.validity == ValidityStatus.VALID:
            all_premises_true = all(p.confidence >= 0.9 for p in argument.premises)
            if all_premises_true:
                argument.soundness = ValidityStatus.SOUND
            else:
                argument.soundness = ValidityStatus.UNSOUND

        # Berechne Confidence der Conclusion
        if argument.conclusion:
            argument.conclusion.confidence = self._calculate_conclusion_confidence(argument)
            argument.conclusion.validity = argument.validity

        return argument

    def _validate_deductive(self, argument: Argument) -> Tuple[ValidityStatus, str]:
        """Validiert deduktives Argument."""
        premises = [p.content.lower() for p in argument.premises]
        conclusion = argument.conclusion.content.lower() if argument.conclusion else ""

        # Prüfe auf bekannte gültige Formen

        # Modus Ponens: P → Q, P ⊢ Q
        for p in premises:
            if "→" in p or "wenn" in p or "dann" in p or "implies" in p:
                # Vereinfachte Prüfung
                parts = re.split(r'→|wenn|dann|implies|if|then', p, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    antecedent = parts[0].strip()
                    consequent = parts[-1].strip()

                    # Prüfe ob Antezedent in Prämissen
                    if any(antecedent in other_p for other_p in premises if other_p != p):
                        if consequent in conclusion:
                            return ValidityStatus.VALID, "Modus Ponens angewendet."

        # Syllogismus: Alle A sind B, Alle B sind C ⊢ Alle A sind C
        all_statements = []
        for p in premises:
            match = re.match(r'alle?\s+(\w+)\s+sind\s+(\w+)', p, re.IGNORECASE)
            if match:
                all_statements.append((match.group(1), match.group(2)))

        if len(all_statements) >= 2:
            # Prüfe auf Kettenregel
            for (a, b) in all_statements:
                for (c, d) in all_statements:
                    if b.lower() == c.lower():
                        expected = f"alle {a} sind {d}".lower()
                        if expected in conclusion:
                            return ValidityStatus.VALID, "Syllogismus (Barbara) angewendet."

        # Standardfall: Können nicht validieren
        return ValidityStatus.UNKNOWN, "Konnte logische Gültigkeit nicht automatisch bestimmen."

    def _validate_inductive(self, argument: Argument) -> Tuple[ValidityStatus, str]:
        """Validiert induktives Argument."""
        # Induktive Argumente sind nie deduktiv gültig
        # aber können stark oder schwach sein

        num_premises = len(argument.premises)
        avg_confidence = sum(p.confidence for p in argument.premises) / num_premises if num_premises else 0

        if num_premises >= 5 and avg_confidence >= 0.8:
            return ValidityStatus.VALID, f"Starke induktive Unterstützung ({num_premises} Beobachtungen)."
        elif num_premises >= 3:
            return ValidityStatus.UNKNOWN, f"Moderate induktive Unterstützung ({num_premises} Beobachtungen)."
        else:
            return ValidityStatus.INVALID, "Zu wenige Beobachtungen für verlässliche Induktion."

    def _validate_abductive(self, argument: Argument) -> Tuple[ValidityStatus, str]:
        """Validiert abduktives Argument (Inference to Best Explanation)."""
        # Abduktive Argumente bewerten wir nach Erklärungskraft

        if argument.conclusion and argument.premises:
            # Prüfe ob Conclusion alle Prämissen erklären würde
            explanation_coverage = len(argument.premises)  # Vereinfacht

            if explanation_coverage >= 3:
                return ValidityStatus.VALID, "Gute abduktive Erklärung (hohe Abdeckung)."
            else:
                return ValidityStatus.UNKNOWN, "Abduktive Erklärung benötigt mehr Bestätigung."

        return ValidityStatus.INVALID, "Unvollständige abduktive Argumentation."

    def _detect_fallacies(self, argument: Argument) -> List[FallacyType]:
        """Erkennt logische Fehlschlüsse."""
        fallacies = []

        premises_text = " ".join(p.content.lower() for p in argument.premises)
        conclusion_text = argument.conclusion.content.lower() if argument.conclusion else ""

        # Zirkelschluss: Conclusion in Prämissen
        if conclusion_text and conclusion_text in premises_text:
            fallacies.append(FallacyType.CIRCULAR_REASONING)

        # Ad Hominem: Angriff auf Person statt Argument
        ad_hominem_patterns = ["du bist", "er ist", "sie ist", "dumm", "inkompetent", "lügner"]
        if any(pattern in premises_text for pattern in ad_hominem_patterns):
            if not any(pattern in conclusion_text for pattern in ad_hominem_patterns):
                fallacies.append(FallacyType.AD_HOMINEM)

        # Falsches Dilemma: Nur zwei Optionen präsentiert
        if "entweder" in premises_text and "oder" in premises_text:
            if premises_text.count("oder") == 1:  # Nur zwei Optionen
                fallacies.append(FallacyType.FALSE_DILEMMA)

        # Hastige Generalisierung: Wenige Fälle → Allgemeine Aussage
        if argument.reasoning_type == ReasoningType.INDUCTIVE:
            if len(argument.premises) < 3 and "alle" in conclusion_text:
                fallacies.append(FallacyType.HASTY_GENERALIZATION)

        # Post Hoc: Zeitliche Folge ≠ Kausalität
        temporal_words = ["danach", "nachdem", "anschließend", "folglich"]
        causal_words = ["deshalb", "daher", "verursacht", "weil"]
        if any(t in premises_text for t in temporal_words):
            if any(c in conclusion_text for c in causal_words):
                fallacies.append(FallacyType.POST_HOC)

        if not fallacies:
            fallacies.append(FallacyType.NONE)

        return fallacies

    def _calculate_conclusion_confidence(self, argument: Argument) -> float:
        """Berechnet die Konfidenz der Schlussfolgerung."""
        if not argument.premises:
            return 0.0

        # Basis: Produkt der Prämissen-Konfidenzen
        premise_confidence = 1.0
        for p in argument.premises:
            premise_confidence *= p.confidence

        # Modifikator basierend auf Reasoning-Typ
        type_modifier = {
            ReasoningType.DEDUCTIVE: 1.0,    # Vollständige Übertragung
            ReasoningType.INDUCTIVE: 0.8,    # Reduktion für Unsicherheit
            ReasoningType.ABDUCTIVE: 0.6,    # Weitere Reduktion
            ReasoningType.ANALOGICAL: 0.5,   # Analogien sind unsicherer
            ReasoningType.CAUSAL: 0.7,       # Kausalität ist schwer zu etablieren
            ReasoningType.BAYESIAN: 0.9,     # Mathematisch fundiert
        }.get(argument.reasoning_type, 0.5)

        # Modifikator für Fehlschlüsse
        fallacy_modifier = 1.0
        if argument.fallacies and FallacyType.NONE not in argument.fallacies:
            fallacy_modifier = 0.3  # Starke Reduktion bei Fehlschlüssen

        # Validitäts-Modifikator
        validity_modifier = {
            ValidityStatus.SOUND: 1.0,
            ValidityStatus.VALID: 0.95,
            ValidityStatus.UNKNOWN: 0.5,
            ValidityStatus.INVALID: 0.1,
            ValidityStatus.UNSOUND: 0.2,
        }.get(argument.validity, 0.5)

        final_confidence = premise_confidence * type_modifier * fallacy_modifier * validity_modifier
        return max(0.0, min(1.0, final_confidence))

    # Logische Regeln
    def _modus_ponens(self, p_implies_q: str, p: str) -> Optional[str]:
        """
        Modus Ponens: P → Q, P ⊢ Q

        Wenn 'P impliziert Q' und 'P' wahr ist, dann ist 'Q' wahr.

        Args:
            p_implies_q: Implikation der Form "wenn P dann Q" oder "P → Q"
            p: Die Prämisse P

        Returns:
            Q wenn erfolgreich abgeleitet, sonst None
        """
        # Normalisiere Input
        impl_lower = p_implies_q.lower().strip()
        p_lower = p.lower().strip()

        # Verschiedene Implikations-Patterns
        patterns = [
            (r'wenn\s+(.+?)\s+dann\s+(.+)', impl_lower),
            (r'(.+?)\s+impliziert\s+(.+)', impl_lower),
            (r'(.+?)\s*→\s*(.+)', impl_lower),
            (r'(.+?)\s+implies\s+(.+)', impl_lower),
            (r'if\s+(.+?)\s+then\s+(.+)', impl_lower),
        ]

        for pattern, text in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                antecedent = match.group(1).strip()
                consequent = match.group(2).strip()

                # Prüfe ob P mit Antezedent übereinstimmt
                if self._terms_match(p_lower, antecedent):
                    return consequent

        return None

    def _modus_tollens(self, p_implies_q: str, not_q: str) -> Optional[str]:
        """
        Modus Tollens: P → Q, ¬Q ⊢ ¬P

        Wenn 'P impliziert Q' und 'Q' falsch ist, dann ist 'P' falsch.

        Args:
            p_implies_q: Implikation der Form "wenn P dann Q"
            not_q: Die Negation von Q

        Returns:
            ¬P wenn erfolgreich abgeleitet, sonst None
        """
        impl_lower = p_implies_q.lower().strip()
        not_q_lower = not_q.lower().strip()

        # Extrahiere Q aus ¬Q
        q_value = self._extract_negated(not_q_lower)
        if not q_value:
            return None

        # Patterns für Implikation
        patterns = [
            (r'wenn\s+(.+?)\s+dann\s+(.+)', impl_lower),
            (r'(.+?)\s+impliziert\s+(.+)', impl_lower),
            (r'(.+?)\s*→\s*(.+)', impl_lower),
            (r'if\s+(.+?)\s+then\s+(.+)', impl_lower),
        ]

        for pattern, text in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                antecedent = match.group(1).strip()
                consequent = match.group(2).strip()

                # Prüfe ob Q mit Konsequenz übereinstimmt
                if self._terms_match(q_value, consequent):
                    return f"nicht {antecedent}"

        return None

    def _hypothetical_syllogism(self, p_implies_q: str, q_implies_r: str) -> Optional[str]:
        """
        Hypothetischer Syllogismus: P → Q, Q → R ⊢ P → R

        Wenn 'P impliziert Q' und 'Q impliziert R', dann 'P impliziert R'.

        Args:
            p_implies_q: Erste Implikation "P → Q"
            q_implies_r: Zweite Implikation "Q → R"

        Returns:
            "P → R" wenn erfolgreich abgeleitet, sonst None
        """
        # Extrahiere P, Q aus erster Implikation
        p, q1 = self._parse_implication(p_implies_q)
        if not p or not q1:
            return None

        # Extrahiere Q, R aus zweiter Implikation
        q2, r = self._parse_implication(q_implies_r)
        if not q2 or not r:
            return None

        # Prüfe ob Q1 und Q2 übereinstimmen
        if self._terms_match(q1, q2):
            return f"wenn {p} dann {r}"

        return None

    def _disjunctive_syllogism(self, p_or_q: str, not_p: str) -> Optional[str]:
        """
        Disjunktiver Syllogismus: P ∨ Q, ¬P ⊢ Q

        Wenn 'P oder Q' und 'nicht P', dann 'Q'.

        Args:
            p_or_q: Disjunktion "P oder Q"
            not_p: Negation von P

        Returns:
            Q wenn erfolgreich abgeleitet, sonst None
        """
        disj_lower = p_or_q.lower().strip()
        not_p_lower = not_p.lower().strip()

        # Extrahiere P aus ¬P
        p_value = self._extract_negated(not_p_lower)
        if not p_value:
            return None

        # Parse Disjunktion
        parts = re.split(r'\s+oder\s+|\s+or\s+|\s*∨\s*', disj_lower, maxsplit=1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        # Prüfe welche Seite negiert wurde
        if self._terms_match(p_value, left):
            return right
        elif self._terms_match(p_value, right):
            return left

        return None

    def _conjunction(self, p: str, q: str) -> str:
        """
        Konjunktion: P, Q ⊢ P ∧ Q

        Wenn P und Q beide wahr sind, dann ist 'P und Q' wahr.

        Args:
            p: Erste Aussage
            q: Zweite Aussage

        Returns:
            Konjunktion von P und Q
        """
        p_clean = p.strip()
        q_clean = q.strip()

        # Vermeide doppelte Klammern
        if not p_clean.startswith("("):
            p_clean = f"({p_clean})"
        if not q_clean.startswith("("):
            q_clean = f"({q_clean})"

        return f"{p_clean} und {q_clean}"

    def _simplification(self, p_and_q: str) -> Tuple[str, str]:
        """
        Simplifikation: P ∧ Q ⊢ P, Q

        Aus einer Konjunktion können beide Teile abgeleitet werden.

        Args:
            p_and_q: Konjunktion "P und Q"

        Returns:
            Tuple (P, Q)
        """
        conj_lower = p_and_q.lower().strip()

        # Entferne äußere Klammern falls vorhanden
        if conj_lower.startswith("(") and conj_lower.endswith(")"):
            conj_lower = conj_lower[1:-1]

        # Parse Konjunktion
        parts = re.split(r'\s+und\s+|\s+and\s+|\s*∧\s*', conj_lower, maxsplit=1)
        if len(parts) == 2:
            left = parts[0].strip().strip("()")
            right = parts[1].strip().strip("()")
            return left, right

        return "", ""

    def _parse_implication(self, implication: str) -> Tuple[Optional[str], Optional[str]]:
        """Parst eine Implikation und gibt (Antezedent, Konsequenz) zurück."""
        impl_lower = implication.lower().strip()

        patterns = [
            r'wenn\s+(.+?)\s+dann\s+(.+)',
            r'(.+?)\s+impliziert\s+(.+)',
            r'(.+?)\s*→\s*(.+)',
            r'if\s+(.+?)\s+then\s+(.+)',
        ]

        for pattern in patterns:
            match = re.match(pattern, impl_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()

        return None, None

    def _extract_negated(self, negation: str) -> Optional[str]:
        """Extrahiert den negierten Term aus einer Negation."""
        neg_lower = negation.lower().strip()

        patterns = [
            r'nicht\s+(.+)',
            r'not\s+(.+)',
            r'¬\s*(.+)',
            r'kein\s+(.+)',
            r'keine\s+(.+)',
        ]

        for pattern in patterns:
            match = re.match(pattern, neg_lower)
            if match:
                return match.group(1).strip()

        return None

    def _terms_match(self, term1: str, term2: str) -> bool:
        """Prüft ob zwei Terme semantisch übereinstimmen."""
        t1 = term1.lower().strip()
        t2 = term2.lower().strip()

        # Exakte Übereinstimmung
        if t1 == t2:
            return True

        # Teilstring-Übereinstimmung (für flexibleres Matching)
        if t1 in t2 or t2 in t1:
            return True

        # Wort-Überlappung
        words1 = set(t1.split())
        words2 = set(t2.split())
        overlap = len(words1 & words2)
        total = min(len(words1), len(words2))

        if total > 0 and overlap / total >= 0.7:
            return True

        return False


class BayesianReasoner:
    """
    Bayesianisches Reasoning für probabilistische Schlussfolgerungen.

    P(H|E) = P(E|H) * P(H) / P(E)
    """

    def __init__(self):
        # Prior-Wahrscheinlichkeiten
        self._priors: Dict[str, float] = {}

        # Likelihood: P(E|H)
        self._likelihoods: Dict[Tuple[str, str], float] = {}

        logger.info("BayesianReasoner initialized")

    def set_prior(self, hypothesis: str, probability: float) -> None:
        """Setzt die Prior-Wahrscheinlichkeit einer Hypothese."""
        self._priors[hypothesis] = max(0.0, min(1.0, probability))

    def set_likelihood(self, evidence: str, hypothesis: str, probability: float) -> None:
        """Setzt P(E|H) - Wahrscheinlichkeit der Evidenz gegeben der Hypothese."""
        self._likelihoods[(evidence, hypothesis)] = max(0.0, min(1.0, probability))

    def update(self, hypothesis: str, evidence: str) -> float:
        """
        Aktualisiert die Wahrscheinlichkeit einer Hypothese gegeben neuer Evidenz.

        Returns:
            Posterior-Wahrscheinlichkeit P(H|E)
        """
        prior = self._priors.get(hypothesis, 0.5)
        likelihood = self._likelihoods.get((evidence, hypothesis), 0.5)

        # P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H)
        # Vereinfacht: Annahme P(E|¬H) = 0.5
        p_e_given_not_h = 0.5
        p_e = likelihood * prior + p_e_given_not_h * (1 - prior)

        if p_e == 0:
            return 0.0

        # Bayes' Theorem
        posterior = (likelihood * prior) / p_e

        # Update prior für nächste Iteration
        self._priors[hypothesis] = posterior

        return posterior

    def calculate_posterior(
        self,
        prior: float,
        likelihood: float,
        false_positive_rate: float = 0.1,
    ) -> float:
        """
        Berechnet Posterior aus gegebenen Werten.

        Args:
            prior: P(H) - Prior-Wahrscheinlichkeit
            likelihood: P(E|H) - Sensitivität/True Positive Rate
            false_positive_rate: P(E|¬H) - False Positive Rate

        Returns:
            Posterior P(H|E)
        """
        p_e = likelihood * prior + false_positive_rate * (1 - prior)

        if p_e == 0:
            return 0.0

        return (likelihood * prior) / p_e


class ScientificMethod:
    """
    Implementiert die wissenschaftliche Methode.

    1. Beobachtung
    2. Fragestellung
    3. Hypothesenbildung
    4. Vorhersage
    5. Experiment/Test
    6. Analyse
    7. Schlussfolgerung
    """

    def __init__(self):
        self.logical_reasoner = LogicalReasoner()
        self.bayesian_reasoner = BayesianReasoner()

        # Aktueller wissenschaftlicher Prozess
        self._observations: List[str] = []
        self._question: Optional[str] = None
        self._hypotheses: List[Dict[str, Any]] = []
        self._predictions: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []
        self._conclusion: Optional[str] = None

        logger.info("ScientificMethod initialized")

    def observe(self, observation: str, source: Optional[str] = None) -> None:
        """Schritt 1: Beobachtung aufzeichnen."""
        self._observations.append({
            "content": observation,
            "source": source,
            "timestamp": now_utc().isoformat(),
        })

    def ask(self, question: str) -> None:
        """Schritt 2: Forschungsfrage formulieren."""
        self._question = question

    def hypothesize(
        self,
        hypothesis: str,
        prior_probability: float = 0.5,
        testable: bool = True,
        falsifiable: bool = True,
    ) -> Dict[str, Any]:
        """Schritt 3: Hypothese formulieren."""
        h = {
            "id": generate_id("hyp"),
            "content": hypothesis,
            "prior": prior_probability,
            "testable": testable,
            "falsifiable": falsifiable,
            "status": "proposed",
            "evidence_for": [],
            "evidence_against": [],
        }
        self._hypotheses.append(h)
        self.bayesian_reasoner.set_prior(hypothesis, prior_probability)
        return h

    def predict(self, hypothesis_id: str, prediction: str, if_true: str, if_false: str) -> Dict[str, Any]:
        """Schritt 4: Vorhersage aus Hypothese ableiten."""
        p = {
            "id": generate_id("pred"),
            "hypothesis_id": hypothesis_id,
            "prediction": prediction,
            "if_true": if_true,
            "if_false": if_false,
            "outcome": None,
        }
        self._predictions.append(p)
        return p

    def test(self, prediction_id: str, result: str, supports_hypothesis: bool) -> Dict[str, Any]:
        """Schritt 5: Test durchführen und Ergebnis aufzeichnen."""
        # Finde Vorhersage
        pred = next((p for p in self._predictions if p["id"] == prediction_id), None)
        if not pred:
            raise ValueError(f"Prediction {prediction_id} nicht gefunden")

        pred["outcome"] = result

        # Finde Hypothese
        hyp = next((h for h in self._hypotheses if h["id"] == pred["hypothesis_id"]), None)
        if hyp:
            if supports_hypothesis:
                hyp["evidence_for"].append(result)
            else:
                hyp["evidence_against"].append(result)

            # Bayesianische Aktualisierung
            likelihood = 0.9 if supports_hypothesis else 0.1
            self.bayesian_reasoner.set_likelihood(result, hyp["content"], likelihood)
            new_prob = self.bayesian_reasoner.update(hyp["content"], result)
            hyp["posterior"] = new_prob

        r = {
            "id": generate_id("res"),
            "prediction_id": prediction_id,
            "result": result,
            "supports": supports_hypothesis,
            "timestamp": now_utc().isoformat(),
        }
        self._results.append(r)
        return r

    def analyze(self) -> Dict[str, Any]:
        """Schritt 6: Ergebnisse analysieren."""
        analysis = {
            "observations_count": len(self._observations),
            "question": self._question,
            "hypotheses": [],
            "best_hypothesis": None,
            "confidence": 0.0,
        }

        best_prob = 0.0
        best_hyp = None

        for hyp in self._hypotheses:
            prob = hyp.get("posterior", hyp["prior"])
            evidence_ratio = len(hyp["evidence_for"]) / max(1, len(hyp["evidence_for"]) + len(hyp["evidence_against"]))

            hyp_analysis = {
                "hypothesis": hyp["content"],
                "posterior_probability": prob,
                "evidence_for": len(hyp["evidence_for"]),
                "evidence_against": len(hyp["evidence_against"]),
                "evidence_ratio": evidence_ratio,
                "status": self._determine_hypothesis_status(hyp),
            }
            analysis["hypotheses"].append(hyp_analysis)

            if prob > best_prob:
                best_prob = prob
                best_hyp = hyp

        if best_hyp:
            analysis["best_hypothesis"] = best_hyp["content"]
            analysis["confidence"] = best_prob

        return analysis

    def conclude(self) -> str:
        """Schritt 7: Schlussfolgerung ziehen."""
        analysis = self.analyze()

        if not analysis["best_hypothesis"]:
            self._conclusion = "Keine Hypothese konnte bestätigt werden. Weitere Forschung erforderlich."
        elif analysis["confidence"] >= 0.95:
            self._conclusion = f"Die Hypothese '{analysis['best_hypothesis']}' wird stark unterstützt (Konfidenz: {analysis['confidence']:.1%})."
        elif analysis["confidence"] >= 0.75:
            self._conclusion = f"Die Hypothese '{analysis['best_hypothesis']}' wird moderat unterstützt (Konfidenz: {analysis['confidence']:.1%}). Weitere Tests empfohlen."
        else:
            self._conclusion = f"Keine Hypothese konnte mit ausreichender Konfidenz bestätigt werden. Höchste: {analysis['best_hypothesis']} ({analysis['confidence']:.1%})."

        return self._conclusion

    def _determine_hypothesis_status(self, hypothesis: Dict) -> str:
        """Bestimmt den Status einer Hypothese."""
        prob = hypothesis.get("posterior", hypothesis["prior"])
        evidence_for = len(hypothesis["evidence_for"])
        evidence_against = len(hypothesis["evidence_against"])

        if prob >= 0.95 and evidence_for >= 3:
            return "strongly_supported"
        elif prob >= 0.75 and evidence_for >= 1:
            return "supported"
        elif prob <= 0.25 and evidence_against >= 1:
            return "refuted"
        elif prob <= 0.05 and evidence_against >= 3:
            return "strongly_refuted"
        else:
            return "inconclusive"

    def get_full_report(self) -> str:
        """Erstellt einen vollständigen wissenschaftlichen Bericht."""
        analysis = self.analyze()

        lines = [
            "=" * 60,
            "WISSENSCHAFTLICHER BERICHT",
            "=" * 60,
            "",
            f"Forschungsfrage: {self._question or 'Nicht definiert'}",
            "",
            "BEOBACHTUNGEN:",
            "-" * 40,
        ]

        for obs in self._observations:
            source = f" (Quelle: {obs['source']})" if obs.get('source') else ""
            lines.append(f"  - {obs['content']}{source}")

        lines.extend([
            "",
            "HYPOTHESEN:",
            "-" * 40,
        ])

        for hyp in analysis["hypotheses"]:
            lines.append(f"  H: {hyp['hypothesis']}")
            lines.append(f"     Posterior: {hyp['posterior_probability']:.1%}")
            lines.append(f"     Evidenz: +{hyp['evidence_for']} / -{hyp['evidence_against']}")
            lines.append(f"     Status: {hyp['status']}")
            lines.append("")

        lines.extend([
            "SCHLUSSFOLGERUNG:",
            "-" * 40,
            f"  {self._conclusion or self.conclude()}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


class ReasoningEngine:
    """
    Hauptschnittstelle für logisches Reasoning.

    Kombiniert alle Reasoning-Systeme zu einem kohärenten Interface.
    """

    def __init__(self):
        self.logical = LogicalReasoner()
        self.bayesian = BayesianReasoner()
        self.scientific = ScientificMethod()

        logger.info("ReasoningEngine initialized")

    def reason(
        self,
        premises: List[str],
        conclusion: str,
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
    ) -> Argument:
        """
        Konstruiert und validiert ein Argument.
        """
        arg = Argument(reasoning_type=reasoning_type)

        for premise in premises:
            arg.add_premise(premise)

        arg.set_conclusion(conclusion)

        return self.logical.validate_argument(arg)

    def evaluate_claim(
        self,
        claim: str,
        evidence: List[str],
        prior: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Bewertet eine Behauptung basierend auf Evidenz.
        """
        # Setze Prior
        self.bayesian.set_prior(claim, prior)

        # Update mit jeder Evidenz
        for ev in evidence:
            # Vereinfachte Likelihood-Schätzung
            # In Realität würde dies durch ein LLM oder Wissensbase bestimmt
            likelihood = 0.7  # Annahme: Evidenz ist moderat unterstützend
            self.bayesian.set_likelihood(ev, claim, likelihood)
            self.bayesian.update(claim, ev)

        posterior = self.bayesian._priors.get(claim, prior)

        return {
            "claim": claim,
            "prior": prior,
            "posterior": posterior,
            "evidence_count": len(evidence),
            "verdict": self._get_verdict(posterior),
        }

    def _get_verdict(self, probability: float) -> str:
        """Gibt ein Urteil basierend auf Wahrscheinlichkeit."""
        if probability >= 0.95:
            return "Sehr wahrscheinlich wahr"
        elif probability >= 0.75:
            return "Wahrscheinlich wahr"
        elif probability >= 0.5:
            return "Möglicherweise wahr"
        elif probability >= 0.25:
            return "Wahrscheinlich falsch"
        else:
            return "Sehr wahrscheinlich falsch"

    def chain_of_thought(self, question: str, context: List[str]) -> List[str]:
        """
        Generiert eine Gedankenkette (Chain of Thought) für eine Frage.
        """
        thoughts = [
            f"Frage: {question}",
            "",
            "Analyse:",
        ]

        # Analysiere Kontext
        for i, ctx in enumerate(context, 1):
            thoughts.append(f"  Fakt {i}: {ctx}")

        thoughts.extend([
            "",
            "Schlussfolgerung:",
            "  [Hier würde das LLM die Schlussfolgerung generieren]",
        ])

        return thoughts
