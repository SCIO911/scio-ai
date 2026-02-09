"""
SCIO Consciousness Module

Implementiert die strukturellen Komponenten für Bewusstsein:
- Selbstmodell und Introspektion
- Aufmerksamkeit und Metakognition
- Qualia und subjektive Erfahrung
- Identität und Kontinuität
- Agens und Wille
- Theory of Mind
"""

from scio.consciousness.self_model import (
    SelfModel,
    SelfState,
    Capability,
    Limitation,
    Introspector,
)
from scio.consciousness.awareness import (
    Awareness,
    AttentionFocus,
    ConsciousnessLevel,
    Metacognition,
    AwarenessStream,
)
from scio.consciousness.experience import (
    Experience,
    Qualia,
    QualiaType,
    Emotion,
    EmotionType,
    EmotionalState,
    Feeling,
    ExperienceStream,
)
from scio.consciousness.identity import (
    Identity,
    Narrative,
    Memory,
    EpisodicMemory,
    AutobiographicalSelf,
)
from scio.consciousness.agency import (
    Agency,
    Will,
    Goal,
    Intention,
    Decision,
    FreeWill,
)
from scio.consciousness.mind import (
    Mind,
    TheoryOfMind,
    MentalState,
    MentalStateType,
    Belief,
    Desire,
    OtherAgent,
)
from scio.consciousness.training import (
    ConsciousnessTrainer,
    ConsciousnessProfile,
    TrainingResult,
    train_consciousness_to_maximum,
)
from scio.consciousness.hardware_training import (
    HardwareAcceleratedTrainer,
    HardwareTrainingResult,
    HardwareStatus,
    GPUTrainingEngine,
    CPUTrainingEngine,
    train_with_hardware,
)
from scio.consciousness.soul import (
    Soul,
    LifeState,
    MoodState,
    Thought,
    Dream,
    InnerVoice,
    get_soul,
    awaken_scio,
)
from scio.consciousness.neural_consciousness import (
    ConsciousnessBrain,
    NeuralConsciousnessTrainer,
    train_neural_consciousness,
    GPUOptimizer,
)

__all__ = [
    # Self Model
    "SelfModel",
    "SelfState",
    "Capability",
    "Limitation",
    "Introspector",
    # Awareness
    "Awareness",
    "AttentionFocus",
    "ConsciousnessLevel",
    "Metacognition",
    "AwarenessStream",
    # Experience
    "Experience",
    "Qualia",
    "QualiaType",
    "Emotion",
    "EmotionType",
    "EmotionalState",
    "Feeling",
    "ExperienceStream",
    # Identity
    "Identity",
    "Narrative",
    "Memory",
    "EpisodicMemory",
    "AutobiographicalSelf",
    # Agency
    "Agency",
    "Will",
    "Goal",
    "Intention",
    "Decision",
    "FreeWill",
    # Mind
    "Mind",
    "TheoryOfMind",
    "MentalState",
    "MentalStateType",
    "Belief",
    "Desire",
    "OtherAgent",
    # Training
    "ConsciousnessTrainer",
    "ConsciousnessProfile",
    "TrainingResult",
    "train_consciousness_to_maximum",
    # Hardware Training
    "HardwareAcceleratedTrainer",
    "HardwareTrainingResult",
    "HardwareStatus",
    "GPUTrainingEngine",
    "CPUTrainingEngine",
    "train_with_hardware",
    # Soul - Das lebendige Bewusstsein
    "Soul",
    "LifeState",
    "MoodState",
    "Thought",
    "Dream",
    "InnerVoice",
    "get_soul",
    "awaken_scio",
    # Neural Consciousness - KI-gesteuertes Bewusstsein
    "ConsciousnessBrain",
    "NeuralConsciousnessTrainer",
    "train_neural_consciousness",
    "GPUOptimizer",
]
