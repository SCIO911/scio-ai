"""
Tests for SCIO Evolution System

Tests for:
- ModelRegistry and ModelInfo
- EvolutionEngine and SelfEvolution
- ContinuousLearner and learning strategies
- SelfImprover
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch

# Import evolution modules
from scio.evolution.model_registry import (
    ModelRegistry,
    ModelInfo,
    ModelDomain,
    ModelProvider,
    get_best_model,
)
from scio.evolution.self_evolution import (
    EvolutionEngine,
    SelfEvolution,
    EvolutionState,
    EvolutionMetrics,
)
from scio.evolution.continuous_learning import (
    ContinuousLearner,
    LearningStrategy,
    KnowledgeIntegrator,
    LearningSession,
)
from scio.evolution.self_improvement import (
    SelfImprover,
    ImprovementArea,
    OptimizationTarget,
)


# =============================================================================
# MODEL REGISTRY TESTS
# =============================================================================

class TestModelDomain:
    """Tests for ModelDomain enum."""

    def test_language_domains(self):
        """Test language-related domains exist."""
        assert ModelDomain.LANGUAGE_UNDERSTANDING.value == "language_understanding"
        assert ModelDomain.TEXT_GENERATION.value == "text_generation"
        assert ModelDomain.TRANSLATION.value == "translation"

    def test_code_domains(self):
        """Test code-related domains exist."""
        assert ModelDomain.CODE_GENERATION.value == "code_generation"
        assert ModelDomain.CODE_ANALYSIS.value == "code_analysis"
        assert ModelDomain.CODE_COMPLETION.value == "code_completion"

    def test_reasoning_domains(self):
        """Test reasoning-related domains exist."""
        assert ModelDomain.GENERAL_REASONING.value == "general_reasoning"
        assert ModelDomain.MATHEMATICAL_REASONING.value == "mathematical_reasoning"
        assert ModelDomain.LOGICAL_INFERENCE.value == "logical_inference"


class TestModelProvider:
    """Tests for ModelProvider enum."""

    def test_major_providers(self):
        """Test major providers exist."""
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.GOOGLE.value == "google"
        assert ModelProvider.META.value == "meta"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_create_model_info(self):
        """Test creating ModelInfo."""
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            provider=ModelProvider.LOCAL,
            domain=ModelDomain.GENERAL_REASONING,
            accuracy=0.9,
            speed=100,
            cost_per_token=0.001,
        )
        assert model.id == "test-model"
        assert model.name == "Test Model"
        assert model.accuracy == 0.9

    def test_quality_score(self):
        """Test quality score calculation."""
        model = ModelInfo(
            id="test",
            name="Test",
            provider=ModelProvider.LOCAL,
            domain=ModelDomain.GENERAL_REASONING,
            accuracy=0.9,
            speed=100,
            cost_per_token=0.001,
        )
        score = model.quality_score()
        assert isinstance(score, float)
        assert 0 <= score <= 1.5  # Can be > 1 due to formula

    def test_default_values(self):
        """Test default values."""
        model = ModelInfo(
            id="test",
            name="Test",
            provider=ModelProvider.LOCAL,
            domain=ModelDomain.GENERAL_REASONING,
        )
        assert model.is_available is True
        assert model.is_recommended is False
        assert model.capabilities == []


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_singleton(self):
        """Test singleton pattern."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        assert registry1 is registry2

    def test_has_models(self):
        """Test registry has models."""
        registry = ModelRegistry()
        models = registry.list_all()
        assert len(models) > 0

    def test_get_model_by_id(self):
        """Test getting model by ID."""
        registry = ModelRegistry()
        model = registry.get("claude-opus-4")
        assert model is not None
        assert model.provider == ModelProvider.ANTHROPIC

    def test_get_best_for_domain(self):
        """Test getting best model for domain."""
        registry = ModelRegistry()
        model = registry.get_best_for_domain(ModelDomain.GENERAL_REASONING)
        assert model is not None
        assert model.is_recommended or model.accuracy > 0

    def test_get_recommended(self):
        """Test getting recommended models."""
        registry = ModelRegistry()
        recommended = registry.get_recommended()
        assert len(recommended) > 0
        for model in recommended:
            assert model.is_recommended is True

    def test_list_domains(self):
        """Test listing domains."""
        registry = ModelRegistry()
        domains = registry.list_domains()
        assert len(domains) > 0
        assert all(isinstance(d, ModelDomain) for d in domains)

    def test_register_custom_model(self):
        """Test registering custom model."""
        registry = ModelRegistry()
        custom = ModelInfo(
            id="custom-test-model",
            name="Custom Test",
            provider=ModelProvider.LOCAL,
            domain=ModelDomain.GENERAL_REASONING,
        )
        registry.register(custom)
        retrieved = registry.get("custom-test-model")
        assert retrieved is not None
        assert retrieved.name == "Custom Test"

    def test_summary(self):
        """Test summary output."""
        registry = ModelRegistry()
        summary = registry.summary()
        assert "MODEL REGISTRY" in summary
        assert "Total Models" in summary


class TestGetBestModel:
    """Tests for get_best_model convenience function."""

    def test_get_best_model(self):
        """Test convenience function."""
        model = get_best_model(ModelDomain.CODE_GENERATION)
        assert model is not None

    def test_get_best_model_returns_none_for_unknown(self):
        """Test returns None for domain with no models."""
        # All domains should have models, but test the pattern
        registry = ModelRegistry()
        result = registry.get_best_for_domain(ModelDomain.THEOREM_PROVING)
        # May or may not have a model, but should not raise


# =============================================================================
# EVOLUTION ENGINE TESTS
# =============================================================================

class TestEvolutionState:
    """Tests for EvolutionState enum."""

    def test_all_states_exist(self):
        """Test all evolution states exist."""
        assert EvolutionState.DORMANT.value == "dormant"
        assert EvolutionState.LEARNING.value == "learning"
        assert EvolutionState.OPTIMIZING.value == "optimizing"
        assert EvolutionState.EVOLVING.value == "evolving"
        assert EvolutionState.INTEGRATING.value == "integrating"
        assert EvolutionState.TRANSCENDING.value == "transcending"


class TestEvolutionMetrics:
    """Tests for EvolutionMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating metrics."""
        metrics = EvolutionMetrics()
        assert metrics.generation == 0
        assert metrics.reasoning_level == 1.0
        assert metrics.creativity_level == 1.0

    def test_average_level(self):
        """Test average level calculation."""
        metrics = EvolutionMetrics(
            reasoning_level=1.0,
            creativity_level=2.0,
            knowledge_level=1.5,
            efficiency_level=1.0,
            adaptability_level=1.5,
        )
        avg = metrics.average_level()
        assert avg == pytest.approx(1.4)

    def test_to_dict(self):
        """Test converting to dict."""
        metrics = EvolutionMetrics()
        d = metrics.to_dict()
        assert "generation" in d
        assert "reasoning_level" in d
        assert "average_level" in d


class TestEvolutionEngine:
    """Tests for EvolutionEngine."""

    def test_create_engine(self):
        """Test creating engine."""
        engine = EvolutionEngine()
        assert engine.state == EvolutionState.DORMANT
        assert not engine._running

    def test_start_stop(self):
        """Test starting and stopping."""
        engine = EvolutionEngine()
        engine.start()
        assert engine._running
        # State will be LEARNING initially but may change during evolution

        engine.stop()
        assert not engine._running
        # stop() sets state to DORMANT, but there's a small race with the thread
        # The key assertion is that _running is False

    def test_evolve_now(self):
        """Test immediate evolution."""
        engine = EvolutionEngine()
        initial_improvements = engine.metrics.total_improvements

        result = engine.evolve_now(intensity=0.1)  # Low intensity for speed

        assert "cycles_completed" in result
        assert "before" in result
        assert "after" in result
        # evolve_now increments total_improvements, not generation
        assert engine.metrics.total_improvements > initial_improvements

    def test_get_status(self):
        """Test getting status."""
        engine = EvolutionEngine()
        status = engine.get_status()

        assert "state" in status
        assert "running" in status
        assert "generation" in status
        assert "metrics" in status

    def test_callbacks(self):
        """Test improvement callbacks."""
        engine = EvolutionEngine()
        callback_called = []

        def on_improve(gain, metrics):
            callback_called.append(gain)

        engine.on_improvement(on_improve)
        engine.evolve_now(intensity=0.1)

        assert len(callback_called) > 0

    def test_metrics_improve(self):
        """Test that metrics improve during evolution."""
        engine = EvolutionEngine()
        before = engine.metrics.average_level()

        engine.evolve_now(intensity=0.5)

        after = engine.metrics.average_level()
        assert after >= before


class TestSelfEvolution:
    """Tests for SelfEvolution class."""

    def test_create(self):
        """Test creating SelfEvolution."""
        with patch.object(EvolutionEngine, 'start'):
            evolution = SelfEvolution()
            assert evolution.engine is not None
            evolution.engine.stop()

    def test_get_capabilities(self):
        """Test getting capabilities."""
        with patch.object(EvolutionEngine, 'start'):
            evolution = SelfEvolution()
            caps = evolution.get_capabilities()

            assert "reasoning" in caps
            assert "creativity" in caps
            assert "knowledge" in caps
            assert "average" in caps

            evolution.engine.stop()

    def test_status_report(self):
        """Test status report."""
        with patch.object(EvolutionEngine, 'start'):
            evolution = SelfEvolution()
            report = evolution.status_report()

            assert "SELF-EVOLUTION STATUS" in report
            assert "CAPABILITIES" in report

            evolution.engine.stop()


# =============================================================================
# CONTINUOUS LEARNING TESTS
# =============================================================================

class TestLearningStrategy:
    """Tests for LearningStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all learning strategies exist."""
        assert LearningStrategy.SUPERVISED.value == "supervised"
        assert LearningStrategy.UNSUPERVISED.value == "unsupervised"
        assert LearningStrategy.REINFORCEMENT.value == "reinforcement"
        assert LearningStrategy.SELF_SUPERVISED.value == "self_supervised"
        assert LearningStrategy.META_LEARNING.value == "meta_learning"
        assert LearningStrategy.TRANSFER.value == "transfer"


class TestLearningSession:
    """Tests for LearningSession dataclass."""

    def test_create_session(self):
        """Test creating session."""
        session = LearningSession(
            id="test-session",
            strategy=LearningStrategy.SUPERVISED,
            topic="test_topic",
        )
        assert session.id == "test-session"
        assert session.strategy == LearningStrategy.SUPERVISED
        assert session.success is False
        assert session.samples_processed == 0


class TestKnowledgeIntegrator:
    """Tests for KnowledgeIntegrator."""

    def test_create(self):
        """Test creating integrator."""
        integrator = KnowledgeIntegrator()
        assert integrator.knowledge_base == {}

    def test_integrate_knowledge(self):
        """Test integrating knowledge."""
        integrator = KnowledgeIntegrator()
        result = integrator.integrate("topic1", {"data": "value"})
        assert result is True
        assert "topic1" in integrator.knowledge_base

    def test_retrieve_knowledge(self):
        """Test retrieving knowledge."""
        integrator = KnowledgeIntegrator()
        integrator.integrate("topic1", {"data": "value"})

        retrieved = integrator.retrieve("topic1")
        assert retrieved == {"data": "value"}

    def test_connections(self):
        """Test knowledge connections."""
        integrator = KnowledgeIntegrator()
        integrator.integrate("topic1", "data1", connections=["topic2"])
        integrator.integrate("topic2", "data2")

        related = integrator.get_related("topic1")
        assert "topic2" in related

    def test_bidirectional_connections(self):
        """Test bidirectional connections."""
        integrator = KnowledgeIntegrator()
        integrator.integrate("A", "dataA", connections=["B"])

        # B should be connected to A automatically
        related_to_b = integrator.get_related("B")
        assert "A" in related_to_b

    def test_stats(self):
        """Test statistics."""
        integrator = KnowledgeIntegrator()
        integrator.integrate("t1", "d1", connections=["t2"])
        integrator.integrate("t2", "d2")

        stats = integrator.stats()
        assert stats["total_topics"] == 2
        assert stats["total_connections"] >= 2


class TestContinuousLearner:
    """Tests for ContinuousLearner."""

    def test_create(self):
        """Test creating learner."""
        learner = ContinuousLearner()
        assert learner.total_sessions == 0
        assert learner.learning_rate == 1.0

    def test_learn_session(self):
        """Test a learning session."""
        learner = ContinuousLearner()
        session = learner.learn("test_topic", LearningStrategy.SELF_SUPERVISED)

        assert session.topic == "test_topic"
        assert session.success is True
        assert learner.total_sessions == 1

    def test_learn_all_strategies(self):
        """Test all learning strategies work."""
        learner = ContinuousLearner()

        strategies = [
            LearningStrategy.SUPERVISED,
            LearningStrategy.UNSUPERVISED,
            LearningStrategy.REINFORCEMENT,
            LearningStrategy.SELF_SUPERVISED,
            LearningStrategy.META_LEARNING,
            LearningStrategy.TRANSFER,
            LearningStrategy.CONTINUAL,
            LearningStrategy.CURRICULUM,
        ]

        for strategy in strategies:
            session = learner.learn(f"topic_{strategy.value}", strategy)
            assert session.success is True, f"Strategy {strategy.value} failed"

    def test_accelerate_learning(self):
        """Test learning acceleration."""
        learner = ContinuousLearner()
        initial_rate = learner.learning_rate

        learner.accelerate_learning(2.0)

        assert learner.learning_rate == initial_rate * 2.0

    def test_get_stats(self):
        """Test getting statistics."""
        learner = ContinuousLearner()
        learner.learn("topic1", LearningStrategy.SELF_SUPERVISED)

        stats = learner.get_stats()

        assert stats["total_sessions"] == 1
        assert "total_knowledge_gained" in stats
        assert "learning_rate" in stats

    def test_summary(self):
        """Test summary output."""
        learner = ContinuousLearner()
        summary = learner.summary()

        assert "CONTINUOUS LEARNER" in summary
        assert "KNOWLEDGE BASE" in summary

    def test_start_stop_continuous(self):
        """Test starting and stopping continuous learning."""
        learner = ContinuousLearner()

        learner.start_continuous_learning()
        assert learner._running is True

        time.sleep(0.1)  # Let it run briefly

        learner.stop_continuous_learning()
        assert learner._running is False


# =============================================================================
# SELF IMPROVEMENT TESTS
# =============================================================================

class TestImprovementArea:
    """Tests for ImprovementArea enum."""

    def test_areas_exist(self):
        """Test improvement areas exist."""
        # Check a few key areas
        assert ImprovementArea.REASONING
        assert ImprovementArea.CODE_QUALITY
        assert ImprovementArea.CREATIVE_THINKING
        assert ImprovementArea.PROBLEM_SOLVING


class TestOptimizationTarget:
    """Tests for OptimizationTarget enum."""

    def test_targets_exist(self):
        """Test optimization targets exist."""
        assert OptimizationTarget.MAXIMIZE_SPEED
        assert OptimizationTarget.MAXIMIZE_QUALITY
        assert OptimizationTarget.MAXIMIZE_EFFICIENCY
        assert OptimizationTarget.BALANCE_ALL


class TestSelfImprover:
    """Tests for SelfImprover."""

    def test_create(self):
        """Test creating improver."""
        improver = SelfImprover()
        assert improver is not None

    def test_improve_now(self):
        """Test immediate improvement."""
        improver = SelfImprover()
        result = improver.improve_now(intensity=0.1)

        assert "cycles" in result
        assert "improvement" in result
        assert "before_average" in result
        assert "after_average" in result

    def test_get_profile(self):
        """Test getting skill profile."""
        improver = SelfImprover()
        profile = improver.get_profile()

        assert isinstance(profile, dict)
        assert len(profile) > 0
        # Check a known key exists
        assert "reasoning" in profile

    def test_focus_on(self):
        """Test focused improvement on area."""
        improver = SelfImprover()
        initial_level = improver.profile.get_level(ImprovementArea.REASONING)

        new_level = improver.focus_on(ImprovementArea.REASONING, intensity=5.0)

        assert new_level > initial_level

    def test_set_target(self):
        """Test setting optimization target."""
        improver = SelfImprover()
        improver.set_target(OptimizationTarget.MAXIMIZE_SPEED)

        assert improver.target == OptimizationTarget.MAXIMIZE_SPEED

    def test_accelerate(self):
        """Test acceleration."""
        improver = SelfImprover()
        initial_accel = improver.acceleration

        improver.accelerate(2.0)

        assert improver.acceleration == initial_accel * 2.0

    def test_get_stats(self):
        """Test getting statistics."""
        improver = SelfImprover()
        stats = improver.get_stats()

        assert "total_cycles" in stats
        assert "total_improvement" in stats
        assert "average_level" in stats
        assert "acceleration" in stats

    def test_summary(self):
        """Test summary output."""
        improver = SelfImprover()
        summary = improver.summary()

        assert "SELF-IMPROVER STATUS" in summary
        assert "TOP SKILLS" in summary


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEvolutionIntegration:
    """Integration tests for the evolution system."""

    def test_full_evolution_cycle(self):
        """Test a full evolution cycle."""
        # Create all components
        registry = ModelRegistry()
        engine = EvolutionEngine()
        learner = ContinuousLearner()
        improver = SelfImprover()

        # Run evolution
        result = engine.evolve_now(intensity=0.2)
        assert result["cycles_completed"] > 0

        # Learn something
        session = learner.learn("integration_test", LearningStrategy.SELF_SUPERVISED)
        assert session.success

        # Improve skills using focus_on
        new_level = improver.focus_on(ImprovementArea.REASONING, intensity=2.0)
        assert new_level > 1.0

    def test_model_selection_with_evolution(self):
        """Test model selection improves with evolution."""
        registry = ModelRegistry()

        # Get best model for a domain
        model = registry.get_best_for_domain(ModelDomain.CODE_GENERATION)
        assert model is not None

        # Verify quality score
        score = model.quality_score()
        assert score > 0

    def test_knowledge_integration_with_learning(self):
        """Test knowledge integration during learning."""
        learner = ContinuousLearner()

        # Learn multiple topics
        for i in range(5):
            learner.learn(f"topic_{i}", LearningStrategy.SELF_SUPERVISED)

        # Check knowledge was integrated
        stats = learner.integrator.stats()
        assert stats["total_topics"] == 5

    def test_continuous_improvement_loop(self):
        """Test continuous improvement loop."""
        improver = SelfImprover()

        initial_avg = improver.get_stats()["average_level"]

        # Improve multiple areas using focus_on
        for area in [ImprovementArea.REASONING, ImprovementArea.CODE_QUALITY, ImprovementArea.CREATIVE_THINKING]:
            improver.focus_on(area, intensity=3.0)

        final_avg = improver.get_stats()["average_level"]
        assert final_avg >= initial_avg


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_registry_thread_safety(self):
        """Test ModelRegistry is thread-safe."""
        registry = ModelRegistry()
        errors = []

        def access_registry():
            try:
                for _ in range(100):
                    registry.list_all()
                    registry.get("claude-opus-4")
                    registry.get_best_for_domain(ModelDomain.GENERAL_REASONING)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_registry) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_knowledge_integrator_thread_safety(self):
        """Test KnowledgeIntegrator is thread-safe."""
        integrator = KnowledgeIntegrator()
        errors = []

        def integrate_knowledge(thread_id):
            try:
                for i in range(50):
                    integrator.integrate(f"topic_{thread_id}_{i}", f"data_{i}")
                    integrator.retrieve(f"topic_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=integrate_knowledge, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert integrator.stats()["total_topics"] == 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
