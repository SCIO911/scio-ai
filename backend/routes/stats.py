#!/usr/bin/env python3
"""
SCIO - Dashboard Stats API
Aggregierte Statistiken für Frontend-Dashboard
"""

import time
from datetime import datetime, timedelta
from flask import Blueprint, jsonify
from typing import Dict, Any, List

stats_bp = Blueprint('stats', __name__, url_prefix='/api/stats')

# Start time für Uptime-Berechnung
_start_time = time.time()


@stats_bp.route('', methods=['GET'])
@stats_bp.route('/', methods=['GET'])
def get_dashboard_stats():
    """
    Dashboard-Statistiken (Aggregiert)

    Gibt einen Überblick über alle SCIO-Systeme:
    - System: Uptime, Version
    - Queue: Jobs pending, running, completed
    - Hardware: GPU, CPU, RAM
    - API: Requests, Latency, Errors
    - AI Modules: Decision, Learning, Orchestrator
    - Health: Status aller Komponenten

    Returns:
        JSON mit aggregierten Dashboard-Daten
    """
    stats = {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - _start_time),
        "version": "2.0.0",
    }

    # ─── Queue Stats ──────────────────────────────────────────────────
    stats["queue"] = _get_queue_stats()

    # ─── Hardware Stats ───────────────────────────────────────────────
    stats["hardware"] = _get_hardware_stats()

    # ─── API Stats ────────────────────────────────────────────────────
    stats["api"] = _get_api_stats()

    # ─── AI Modules Stats ─────────────────────────────────────────────
    stats["ai_modules"] = _get_ai_module_stats()

    # ─── Health Status ────────────────────────────────────────────────
    stats["health"] = _get_health_status()

    # ─── Money Maker Stats ──────────────────────────────────────────────
    stats["earnings"] = _get_earnings_stats()

    return jsonify(stats)


@stats_bp.route('/queue', methods=['GET'])
def get_queue_stats():
    """Job Queue Statistiken"""
    return jsonify(_get_queue_stats())


@stats_bp.route('/hardware', methods=['GET'])
def get_hardware_stats():
    """Hardware Statistiken"""
    return jsonify(_get_hardware_stats())


@stats_bp.route('/api', methods=['GET'])
def get_api_stats_endpoint():
    """API Statistiken"""
    return jsonify(_get_api_stats())


@stats_bp.route('/ai', methods=['GET'])
def get_ai_stats():
    """AI Module Statistiken"""
    return jsonify(_get_ai_module_stats())


@stats_bp.route('/health', methods=['GET'])
def get_health_stats():
    """Health Status aller Module"""
    return jsonify(_get_health_status())


@stats_bp.route('/earnings', methods=['GET'])
def get_earnings_stats_endpoint():
    """
    Earnings Statistiken (MoneyMaker)

    Zeigt automatische Einnahmen durch GPU-Vermietung.
    """
    return jsonify(_get_earnings_stats())


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _get_queue_stats() -> Dict[str, Any]:
    """Sammelt Job Queue Statistiken"""
    try:
        from backend.services.job_queue import get_job_queue
        queue = get_job_queue()

        stats = queue.get_stats()
        return {
            "pending": stats.get("pending", 0),
            "queued": stats.get("queued", 0),
            "running": queue.active_job_count,
            "completed": stats.get("completed", 0),
            "failed": stats.get("failed", 0),
            "total_processed": stats.get("total", 0),
            "is_running": queue.is_running,
            "max_concurrent": queue.max_concurrent,
        }
    except Exception as e:
        return {"error": str(e), "available": False}


def _get_hardware_stats() -> Dict[str, Any]:
    """Sammelt Hardware Statistiken"""
    try:
        from backend.services.hardware_monitor import get_hardware_monitor
        monitor = get_hardware_monitor()
        status = monitor.get_status()

        gpus = []
        for gpu in status.gpus:
            gpus.append({
                "index": gpu.index,
                "name": gpu.name,
                "vram_used_gb": round(gpu.vram_used_mb / 1024, 2),
                "vram_total_gb": round(gpu.vram_total_mb / 1024, 2),
                "vram_percent": round(gpu.vram_usage_percent, 1),
                "utilization_percent": gpu.gpu_utilization,
                "temperature_c": gpu.temperature,
            })

        cpu_percent = status.cpu.usage_percent if status.cpu else 0
        ram_used = status.ram.used_gb if status.ram else 0
        ram_total = status.ram.total_gb if status.ram else 1

        return {
            "gpus": gpus,
            "gpu_count": len(gpus),
            "cpu_usage_percent": round(cpu_percent, 1),
            "ram_used_gb": round(ram_used, 2),
            "ram_total_gb": round(ram_total, 2),
            "ram_percent": round(ram_used / max(ram_total, 1) * 100, 1),
            "is_busy": status.is_busy,
        }
    except Exception as e:
        return {"error": str(e), "available": False}


def _get_api_stats() -> Dict[str, Any]:
    """Sammelt API Statistiken"""
    try:
        from backend.monitoring.prometheus_exporter import get_metrics
        metrics = get_metrics()

        # Request Counts
        total_requests = sum(metrics.requests_total.values.values())
        error_requests = sum(metrics.request_errors.values.values())

        # Request Durations (average from histogram)
        total_duration = sum(metrics.request_duration.sums.values())
        total_count = sum(metrics.request_duration.counts.values())
        avg_latency_ms = (total_duration / max(total_count, 1)) * 1000

        return {
            "requests_total": int(total_requests),
            "errors_total": int(error_requests),
            "error_rate_percent": round(error_requests / max(total_requests, 1) * 100, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "rate_limit_hits": int(sum(metrics.rate_limit_hits.values.values())),
        }
    except Exception as e:
        return {"error": str(e), "available": False}


def _get_ai_module_stats() -> Dict[str, Any]:
    """Sammelt AI Module Statistiken"""
    stats = {}

    # Decision Engine
    try:
        from backend.decision import get_decision_engine
        engine = get_decision_engine()
        engine_stats = engine.get_statistics()
        stats["decision_engine"] = {
            "total_decisions": engine_stats.get("total_decisions", 0),
            "avg_confidence": round(engine_stats.get("avg_confidence", 0), 3),
            "trees_count": engine_stats.get("trees", 0),
        }
    except Exception:
        stats["decision_engine"] = {"available": False}

    # Rule Engine
    try:
        from backend.decision import get_rule_engine
        engine = get_rule_engine()
        engine_stats = engine.get_statistics()
        stats["rule_engine"] = {
            "total_evaluations": engine_stats.get("total_evaluations", 0),
            "rules_count": engine_stats.get("rules", 0),
        }
    except Exception:
        stats["rule_engine"] = {"available": False}

    # RL Agent
    try:
        from backend.learning import get_rl_agent
        agent = get_rl_agent()
        agent_stats = agent.get_statistics()
        stats["rl_agent"] = {
            "total_actions": agent_stats.get("total_actions", 0),
            "avg_reward": round(agent_stats.get("avg_reward", 0), 3),
            "exploration_rate": round(agent_stats.get("epsilon", 0), 3),
        }
    except Exception:
        stats["rl_agent"] = {"available": False}

    # Continuous Learner
    try:
        from backend.learning import get_continuous_learner
        learner = get_continuous_learner()
        learner_stats = learner.get_statistics()
        stats["continuous_learner"] = {
            "patterns_learned": learner_stats.get("patterns", 0),
            "observations": learner_stats.get("observations", 0),
        }
    except Exception:
        stats["continuous_learner"] = {"available": False}

    # Knowledge Graph
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()
        kg_stats = kg.get_statistics()
        stats["knowledge_graph"] = {
            "entities": kg_stats.get("entities", 0),
            "relations": kg_stats.get("relations", 0),
        }
    except Exception:
        stats["knowledge_graph"] = {"available": False}

    # Orchestrator
    try:
        from backend.orchestration import get_orchestrator
        orch = get_orchestrator()
        orch_stats = orch.get_statistics()
        stats["orchestrator"] = {
            "modules": orch_stats.get("registered_modules", 0),
            "workflows_executed": orch_stats.get("workflows_executed", 0),
        }
    except Exception:
        stats["orchestrator"] = {"available": False}

    # Capability Engine
    try:
        from backend.capabilities import get_capability_engine
        caps = get_capability_engine()
        stats["capability_engine"] = {
            "tools_registered": caps.registry._tool_count,
            "chains_registered": len(getattr(caps, '_chains', {})),
        }
    except Exception:
        stats["capability_engine"] = {"available": False}

    return stats


def _get_health_status() -> Dict[str, Any]:
    """Sammelt Health Status aller Module"""
    try:
        from backend.core.reliability import get_health_checker
        health = get_health_checker()
        results = health.check_all()

        status = {}
        all_healthy = True

        for name, check_result in results.items():
            status[name] = {
                "healthy": check_result.healthy,
                "latency_ms": round(check_result.latency_ms, 2),
                "message": check_result.message,
            }
            if not check_result.healthy:
                all_healthy = False

        return {
            "overall": "healthy" if all_healthy else "degraded",
            "checks": status,
        }
    except Exception as e:
        return {"error": str(e), "overall": "unknown"}


def _get_earnings_stats() -> Dict[str, Any]:
    """Sammelt Earnings Statistiken vom MoneyMaker"""
    try:
        from backend.automation.money_maker import get_money_maker
        money_maker = get_money_maker()
        stats = money_maker.get_stats()

        return {
            "total_earnings_usd": round(stats.total_earnings_usd, 2),
            "today_usd": round(stats.today_earnings_usd, 2),
            "this_week_usd": round(stats.this_week_earnings_usd, 2),
            "this_month_usd": round(stats.this_month_earnings_usd, 2),
            "current_hourly_rate": round(stats.current_hourly_rate, 2),
            "estimated_daily_usd": round(stats.estimated_daily, 2),
            "estimated_monthly_usd": round(stats.estimated_monthly, 2),
            "uptime_hours": round(stats.uptime_hours, 1),
            "gpu_rental_hours": round(stats.gpu_rental_hours, 1),
            "jobs_processed": stats.jobs_processed,
            "last_earning": stats.last_earning_event.isoformat() if stats.last_earning_event else None,
            "status": "active",
        }
    except Exception as e:
        return {"error": str(e), "status": "unavailable"}


# ═══════════════════════════════════════════════════════════════════════════
# REAL-TIME STATS (für WebSocket)
# ═══════════════════════════════════════════════════════════════════════════

def get_realtime_stats() -> Dict[str, Any]:
    """
    Kompakte Real-Time Stats für WebSocket-Updates

    Gibt nur die wichtigsten Metriken zurück für schnelle Updates.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "queue": {
            "running": _safe_get(lambda: get_job_queue().active_job_count, 0),
            "queued": _safe_get(lambda: get_job_queue().queue_size, 0),
        },
        "hardware": {
            "gpu_util": _safe_get(
                lambda: get_hardware_monitor().get_status().gpus[0].utilization if get_hardware_monitor().get_status().gpus else 0,
                0
            ),
            "cpu": _safe_get(lambda: get_hardware_monitor().get_status().cpu_percent, 0),
        },
        "health": _safe_get(
            lambda: "healthy" if get_health_checker().is_healthy() else "degraded",
            "unknown"
        ),
    }


def _safe_get(func, default):
    """Sichere Ausführung mit Default-Wert"""
    try:
        return func()
    except Exception:
        return default


# Lazy imports for _safe_get
def get_job_queue():
    from backend.services.job_queue import get_job_queue as _get
    return _get()


def get_hardware_monitor():
    from backend.services.hardware_monitor import get_hardware_monitor as _get
    return _get()


def get_health_checker():
    from backend.core.reliability import get_health_checker as _get
    return _get()
