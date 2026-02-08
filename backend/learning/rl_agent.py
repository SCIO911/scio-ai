#!/usr/bin/env python3
"""
SCIO - Reinforcement Learning Agent
Q-Learning, Policy Gradients und Reward-basiertes Lernen
"""

import random
import json
import math
import pickle
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Zustand des Systems"""
    features: Dict[str, Any]

    def to_key(self) -> str:
        """Konvertiert zu hashbarem Key"""
        sorted_items = sorted(self.features.items())
        return str(sorted_items)

    @classmethod
    def from_context(cls, context: Dict[str, Any], feature_keys: List[str] = None) -> 'State':
        """Erstellt State aus Kontext"""
        if feature_keys:
            features = {k: context.get(k) for k in feature_keys}
        else:
            features = context.copy()
        return cls(features=features)


@dataclass
class Action:
    """Eine mögliche Aktion"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass
class Experience:
    """Eine Erfahrung (State, Action, Reward, Next State)"""
    state: State
    action: Action
    reward: float
    next_state: Optional[State]
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)


class RewardSystem:
    """
    Berechnet Rewards basierend auf verschiedenen Metriken
    """

    def __init__(self):
        self.reward_functions: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}
        self._setup_default_rewards()

    def _setup_default_rewards(self):
        """Erstellt Standard-Reward-Funktionen"""

        # Erfolgs-Reward
        self.add_reward("success", lambda ctx: 1.0 if ctx.get("success") else -0.5, weight=1.0)

        # Latenz-Reward (niedrigere Latenz = höherer Reward)
        self.add_reward("latency", lambda ctx: max(0, 1 - ctx.get("latency_s", 1) / 10), weight=0.3)

        # Ressourcen-Effizienz
        self.add_reward("efficiency", lambda ctx: 1 - ctx.get("vram_used_percent", 50) / 100, weight=0.2)

        # Kosten-Reward (niedrigere Kosten = höherer Reward)
        self.add_reward("cost", lambda ctx: max(0, 1 - ctx.get("cost_cents", 0) / 100), weight=0.2)

        # Qualitäts-Reward
        self.add_reward("quality", lambda ctx: ctx.get("quality_score", 0.5), weight=0.5)

        # User Satisfaction
        self.add_reward("satisfaction", lambda ctx: ctx.get("user_rating", 3) / 5, weight=0.8)

    def add_reward(self, name: str, func: Callable, weight: float = 1.0):
        """Fügt eine Reward-Funktion hinzu"""
        self.reward_functions[name] = func
        self.weights[name] = weight

    def calculate(self, context: Dict[str, Any], components: List[str] = None) -> Tuple[float, Dict[str, float]]:
        """
        Berechnet den Gesamt-Reward

        Returns:
            (total_reward, component_rewards)
        """
        if components is None:
            components = list(self.reward_functions.keys())

        rewards = {}
        total = 0.0
        total_weight = 0.0

        for name in components:
            if name in self.reward_functions:
                try:
                    reward = self.reward_functions[name](context)
                    weight = self.weights.get(name, 1.0)
                    rewards[name] = reward
                    total += reward * weight
                    total_weight += weight
                except Exception:
                    rewards[name] = 0.0

        if total_weight > 0:
            total /= total_weight

        return total, rewards


class QLearningAgent:
    """
    Q-Learning Agent für diskrete Zustands- und Aktionsräume
    """

    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 0.1,
                 exploration_decay: float = 0.995):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01

        # Q-Table: state_key -> action_name -> q_value
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Tracking
        self.episodes = 0
        self.total_reward = 0

    def get_action(self, state: State, available_actions: List[Action]) -> Action:
        """Wählt eine Aktion (epsilon-greedy)"""
        if not available_actions:
            return Action(name="none")

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Exploitation
        state_key = state.to_key()
        q_values = self.q_table[state_key]

        best_action = max(available_actions, key=lambda a: q_values.get(a.name, 0.0))
        return best_action

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        """Aktualisiert Q-Wert"""
        state_key = state.to_key()
        next_state_key = next_state.to_key() if next_state else None

        current_q = self.q_table[state_key][action.name]

        if done or next_state_key is None:
            target = reward
        else:
            # Max Q-Wert des nächsten Zustands
            next_q_values = self.q_table[next_state_key]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            target = reward + self.gamma * max_next_q

        # Q-Learning Update
        self.q_table[state_key][action.name] = current_q + self.lr * (target - current_q)

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.total_reward += reward

    def get_policy(self) -> Dict[str, str]:
        """Gibt die gelernte Policy zurück (beste Aktion pro Zustand)"""
        policy = {}
        for state_key, actions in self.q_table.items():
            if actions:
                best_action = max(actions.keys(), key=lambda a: actions[a])
                policy[state_key] = best_action
        return policy


class RLAgent:
    """
    SCIO Reinforcement Learning Agent

    Kombiniert verschiedene RL-Methoden:
    - Q-Learning für diskrete Entscheidungen
    - Policy Gradient Approximation
    - Reward Shaping
    - Experience Replay
    """

    def __init__(self, name: str = "scio_rl"):
        self.name = name
        self.q_agent = QLearningAgent()
        self.reward_system = RewardSystem()

        # Experience Replay Buffer
        self.experience_buffer: List[Experience] = []
        self._buffer_lock = threading.Lock()
        self.buffer_size = 10000

        # Verfügbare Aktionen pro Kontext
        self.action_spaces: Dict[str, List[Action]] = {}

        # Feature Keys für State-Extraktion
        self.state_features: List[str] = [
            "task_type",
            "model_size",
            "vram_available",
            "queue_size",
            "is_premium"
        ]

        # Tracking
        self.episode_count = 0
        self.total_episodes_reward = 0
        self._initialized = False

    def initialize(self) -> bool:
        """Initialisiert den RL Agent"""
        try:
            self._setup_action_spaces()
            self._load_model()
            self._initialized = True
            logger.info("RL Agent initialisiert")
            return True
        except Exception as e:
            logger.error(f"RL Agent Fehler: {e}")
            return False

    def _setup_action_spaces(self):
        """Definiert verfügbare Aktionen pro Kontext"""

        # Worker-Auswahl
        self.action_spaces["worker_selection"] = [
            Action("llm_small", {"max_vram": 8}),
            Action("llm_medium", {"max_vram": 14}),
            Action("llm_large", {"max_vram": 24}),
            Action("vision", {"capabilities": ["ocr", "caption"]}),
            Action("audio", {"capabilities": ["stt", "tts"]}),
            Action("code", {"capabilities": ["generate", "review"]}),
            Action("embedding", {"capabilities": ["text", "image"]}),
        ]

        # Ressourcen-Management
        self.action_spaces["resource_management"] = [
            Action("keep_model", {}),
            Action("unload_model", {}),
            Action("preload_model", {}),
            Action("increase_batch", {"factor": 2}),
            Action("decrease_batch", {"factor": 0.5}),
        ]

        # Queue-Management
        self.action_spaces["queue_management"] = [
            Action("process_immediately", {"priority": "high"}),
            Action("queue_normal", {"priority": "normal"}),
            Action("defer", {"priority": "low"}),
            Action("reject", {"reason": "overload"}),
        ]

    def _load_model(self):
        """Lädt gespeicherte Q-Table"""
        try:
            from backend.config import Config
            model_path = Config.DATA_DIR / "rl_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.q_agent.q_table = defaultdict(lambda: defaultdict(float), data.get("q_table", {}))
                    self.episode_count = data.get("episodes", 0)
                logger.info(f"RL Model geladen ({self.episode_count} Episoden)")
        except Exception:
            pass

    def save_model(self):
        """Speichert Q-Table"""
        try:
            from backend.config import Config
            model_path = Config.DATA_DIR / "rl_model.pkl"
            data = {
                "q_table": dict(self.q_agent.q_table),
                "episodes": self.episode_count,
                "timestamp": datetime.now().isoformat()
            }
            with open(model_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"RL Model speichern fehlgeschlagen: {e}")

    def select_action(self,
                      context: Dict[str, Any],
                      action_space: str = "worker_selection") -> Action:
        """
        Wählt die beste Aktion für den aktuellen Kontext

        Args:
            context: Aktueller System-Zustand
            action_space: Welcher Aktionsraum verwendet werden soll

        Returns:
            Die gewählte Aktion
        """
        state = State.from_context(context, self.state_features)
        available_actions = self.action_spaces.get(action_space, [])

        if not available_actions:
            return Action("default")

        return self.q_agent.get_action(state, available_actions)

    def observe(self,
                context: Dict[str, Any],
                action: Action,
                outcome: Dict[str, Any],
                next_context: Dict[str, Any] = None,
                done: bool = False):
        """
        Beobachtet das Ergebnis einer Aktion und lernt daraus

        Args:
            context: Kontext vor der Aktion
            action: Die ausgeführte Aktion
            outcome: Ergebnis der Aktion
            next_context: Kontext nach der Aktion
            done: Ob die Episode beendet ist
        """
        state = State.from_context(context, self.state_features)
        next_state = State.from_context(next_context, self.state_features) if next_context else None

        # Reward berechnen
        reward, reward_components = self.reward_system.calculate(outcome)

        # Experience speichern (thread-safe)
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        with self._buffer_lock:
            self.experience_buffer.append(exp)
            # Buffer begrenzen
            if len(self.experience_buffer) > self.buffer_size:
                self.experience_buffer = self.experience_buffer[-self.buffer_size:]

        # Q-Learning Update
        self.q_agent.update(state, action, reward, next_state, done)

        if done:
            self.episode_count += 1
            self.total_episodes_reward += reward

            # Periodisch speichern
            if self.episode_count % 100 == 0:
                self.save_model()

    def batch_learn(self, batch_size: int = 32):
        """
        Lernt aus einem Batch von Experiences (Experience Replay)
        """
        with self._buffer_lock:
            if len(self.experience_buffer) < batch_size:
                return
            batch = random.sample(self.experience_buffer, batch_size)

        for exp in batch:
            self.q_agent.update(
                exp.state,
                exp.action,
                exp.reward,
                exp.next_state,
                exp.done
            )

    def get_best_action(self, context: Dict[str, Any], action_space: str) -> Tuple[Action, float]:
        """
        Gibt die beste Aktion und deren Q-Wert zurück (ohne Exploration)
        """
        state = State.from_context(context, self.state_features)
        state_key = state.to_key()

        available_actions = self.action_spaces.get(action_space, [])
        q_values = self.q_agent.q_table[state_key]

        if not available_actions:
            return Action("default"), 0.0

        best_action = max(available_actions, key=lambda a: q_values.get(a.name, 0.0))
        best_q = q_values.get(best_action.name, 0.0)

        return best_action, best_q

    def add_reward_function(self, name: str, func: Callable, weight: float = 1.0):
        """Fügt eine Reward-Funktion hinzu"""
        self.reward_system.add_reward(name, func, weight)

    def add_action(self, action_space: str, action: Action):
        """Fügt eine Aktion zu einem Aktionsraum hinzu"""
        if action_space not in self.action_spaces:
            self.action_spaces[action_space] = []
        self.action_spaces[action_space].append(action)

    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        return {
            "episodes": self.episode_count,
            "total_reward": round(self.total_episodes_reward, 2),
            "avg_reward": round(self.total_episodes_reward / max(1, self.episode_count), 3),
            "experience_buffer_size": len(self.experience_buffer),
            "q_table_states": len(self.q_agent.q_table),
            "exploration_rate": round(self.q_agent.epsilon, 4),
            "action_spaces": {k: len(v) for k, v in self.action_spaces.items()},
            "policy_size": len(self.q_agent.get_policy())
        }

    def get_learned_policy(self, action_space: str = None) -> Dict[str, Any]:
        """Gibt die gelernte Policy zurück"""
        policy = self.q_agent.get_policy()

        if action_space:
            # Filtere nach Aktionen im Aktionsraum
            valid_actions = {a.name for a in self.action_spaces.get(action_space, [])}
            policy = {k: v for k, v in policy.items() if v in valid_actions}

        return policy


# Singleton
_rl_agent: Optional[RLAgent] = None

def get_rl_agent() -> RLAgent:
    """Gibt Singleton-Instanz zurück"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLAgent()
    return _rl_agent
