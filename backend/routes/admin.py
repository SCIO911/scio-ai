#!/usr/bin/env python3
"""
SCIO - Admin Routes
Dashboard und Management-Endpoints
"""

import os
from datetime import datetime, timedelta
from functools import wraps

from flask import request, jsonify, send_from_directory

from . import admin_bp
from backend.config import Config
from backend.services.hardware_monitor import get_hardware_monitor
from backend.services.job_queue import get_job_queue
from backend.services.api_keys import get_api_key_service
from backend.integrations.vastai import get_vastai
from backend.integrations.runpod import get_runpod
from backend.models import SessionLocal
from backend.models.earnings import Earning, EarningSource


def require_admin(f):
    """Decorator für Admin-Authentifizierung"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for admin token in header or query
        admin_token = request.headers.get('X-Admin-Token') or request.args.get('token')

        expected_token = os.getenv('ADMIN_TOKEN', 'scio-admin-2024')

        if admin_token != expected_token:
            return jsonify({'error': 'Unauthorized'}), 401

        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════

@admin_bp.route('/')
def admin_dashboard():
    """Serve Admin Dashboard"""
    from pathlib import Path
    frontend_dir = Path(__file__).parent.parent.parent / 'frontend'
    return send_from_directory(str(frontend_dir), 'admin.html')


@admin_bp.route('/api/overview')
@require_admin
def overview():
    """Dashboard Übersicht"""
    # Hardware Status
    monitor = get_hardware_monitor()
    hw_status = monitor.get_status()

    # Job Queue Stats
    queue = get_job_queue()
    queue_stats = queue.get_stats()

    # Earnings (last 30 days)
    db = SessionLocal()
    try:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        earnings = db.query(Earning).filter(
            Earning.created_at >= thirty_days_ago
        ).all()

        total_earnings = sum(e.amount_cents for e in earnings if e.earning_type and e.earning_type.value == 'income')
        total_expenses = sum(e.amount_cents for e in earnings if e.earning_type and e.earning_type.value == 'expense')

        by_source = {}
        for e in earnings:
            if e.earning_type and e.earning_type.value == 'income' and e.source:
                source = e.source.value
                by_source[source] = by_source.get(source, 0) + e.amount_cents

    finally:
        db.close()

    # Platform Integrations
    vastai = get_vastai()
    runpod = get_runpod()

    return jsonify({
        'hardware': {
            'gpus': [gpu.to_dict() for gpu in hw_status.gpus],
            'cpu': hw_status.cpu.to_dict() if hw_status.cpu else None,
            'ram': hw_status.ram.to_dict() if hw_status.ram else None,
            'is_busy': hw_status.is_busy,
            'capacity': hw_status.get_capacity(),
        },
        'jobs': queue_stats,
        'earnings_30d': {
            'total_eur': total_earnings / 100,
            'expenses_eur': total_expenses / 100,
            'net_eur': (total_earnings - total_expenses) / 100,
            'by_source': {k: v / 100 for k, v in by_source.items()},
        },
        'integrations': {
            'vastai': vastai.get_status(),
            'runpod': runpod.get_status(),
        },
        'timestamp': datetime.utcnow().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════
# HARDWARE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@admin_bp.route('/api/hardware')
@require_admin
def hardware_status():
    """Detaillierter Hardware-Status"""
    monitor = get_hardware_monitor()
    return jsonify(monitor.get_status().to_dict())


@admin_bp.route('/api/hardware/history')
@require_admin
def hardware_history():
    """Hardware-Nutzungsverlauf - Gibt historische Hardware-Metriken zurück"""
    from backend.services.hardware_history import get_hardware_history

    # Query-Parameter
    hours = request.args.get('hours', 24, type=int)
    resolution = request.args.get('resolution', 'minute', type=str)  # minute, hour, day

    history_service = get_hardware_history()
    data = history_service.get_history(hours=hours, resolution=resolution)

    return jsonify({
        'hours': hours,
        'resolution': resolution,
        'data_points': len(data),
        'data': data,
    })


# ═══════════════════════════════════════════════════════════════
# JOB MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@admin_bp.route('/api/jobs')
@require_admin
def list_all_jobs():
    """Alle Jobs"""
    queue = get_job_queue()
    limit = request.args.get('limit', 100, type=int)
    status_filter = request.args.get('status')

    from backend.models.job import JobStatus

    status = None
    if status_filter:
        try:
            status = JobStatus(status_filter)
        except (ValueError, KeyError):
            pass  # Invalid status filter, use None (no filter)

    jobs = queue.get_jobs(status=status, limit=limit)

    return jsonify({
        'jobs': jobs,
        'total': len(jobs),
        'stats': queue.get_stats(),
    })


@admin_bp.route('/api/jobs/<job_id>', methods=['DELETE'])
@require_admin
def admin_cancel_job(job_id: str):
    """Job abbrechen"""
    queue = get_job_queue()
    success = queue.cancel_job(job_id)

    return jsonify({'success': success})


# ═══════════════════════════════════════════════════════════════
# API KEY MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@admin_bp.route('/api/keys', methods=['GET'])
@require_admin
def list_api_keys():
    """Alle API Keys"""
    db = SessionLocal()
    try:
        from backend.models.api_key import APIKey
        keys = db.query(APIKey).order_by(APIKey.created_at.desc()).all()
        return jsonify({
            'keys': [key.to_dict() for key in keys],
            'total': len(keys),
        })
    finally:
        db.close()


@admin_bp.route('/api/keys', methods=['POST'])
@require_admin
def create_api_key():
    """Neuen API Key erstellen"""
    data = request.json
    api_service = get_api_key_service()

    try:
        full_key, key_info = api_service.create_key(
            user_email=data.get('email'),
            name=data.get('name'),
            description=data.get('description'),
            rate_limit_rpm=data.get('rate_limit_rpm', 60),
            rate_limit_tpm=data.get('rate_limit_tpm', 100000),
            monthly_token_limit=data.get('monthly_token_limit', 1000000),
            credits_cents=data.get('credits_cents', 0),
            allowed_models=data.get('allowed_models'),
            allowed_features=data.get('allowed_features', ['inference']),
            expires_days=data.get('expires_days'),
            is_admin=data.get('is_admin', False),
        )

        return jsonify({
            'key': full_key,
            'info': key_info,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@admin_bp.route('/api/keys/<key_prefix>/credits', methods=['POST'])
@require_admin
def add_key_credits(key_prefix: str):
    """Credits zu Key hinzufügen"""
    data = request.json
    amount_cents = data.get('amount_cents', 0)

    # Find key by prefix
    db = SessionLocal()
    try:
        from backend.models.api_key import APIKey
        key = db.query(APIKey).filter(APIKey.key_prefix.like(f"{key_prefix}%")).first()

        if not key:
            return jsonify({'error': 'Key not found'}), 404

        key.credits_cents += amount_cents
        db.commit()

        return jsonify({
            'success': True,
            'new_balance_cents': key.credits_cents,
        })

    finally:
        db.close()


@admin_bp.route('/api/keys/<key_prefix>/deactivate', methods=['POST'])
@require_admin
def deactivate_key(key_prefix: str):
    """Key deaktivieren"""
    db = SessionLocal()
    try:
        from backend.models.api_key import APIKey
        key = db.query(APIKey).filter(APIKey.key_prefix.like(f"{key_prefix}%")).first()

        if not key:
            return jsonify({'error': 'Key not found'}), 404

        key.is_active = False
        db.commit()

        return jsonify({'success': True})

    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
# EARNINGS
# ═══════════════════════════════════════════════════════════════

@admin_bp.route('/api/earnings')
@require_admin
def list_earnings():
    """Einnahmen-Liste"""
    db = SessionLocal()
    try:
        days = request.args.get('days', 30, type=int)
        since = datetime.utcnow() - timedelta(days=days)

        earnings = db.query(Earning).filter(
            Earning.created_at >= since
        ).order_by(Earning.created_at.desc()).all()

        return jsonify({
            'earnings': [e.to_dict() for e in earnings],
            'total': len(earnings),
        })

    finally:
        db.close()


@admin_bp.route('/api/earnings/summary')
@require_admin
def earnings_summary():
    """Einnahmen-Zusammenfassung"""
    db = SessionLocal()
    try:
        # Today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_earnings = db.query(Earning).filter(
            Earning.created_at >= today_start,
            Earning.earning_type == 'income',
        ).all()
        today_total = sum(e.amount_cents for e in today_earnings)

        # This week
        week_start = today_start - timedelta(days=today_start.weekday())
        week_earnings = db.query(Earning).filter(
            Earning.created_at >= week_start,
            Earning.earning_type == 'income',
        ).all()
        week_total = sum(e.amount_cents for e in week_earnings)

        # This month
        month_start = today_start.replace(day=1)
        month_earnings = db.query(Earning).filter(
            Earning.created_at >= month_start,
            Earning.earning_type == 'income',
        ).all()
        month_total = sum(e.amount_cents for e in month_earnings)

        # All time
        all_earnings = db.query(Earning).filter(
            Earning.earning_type == 'income',
        ).all()
        all_total = sum(e.amount_cents for e in all_earnings)

        return jsonify({
            'today_eur': today_total / 100,
            'week_eur': week_total / 100,
            'month_eur': month_total / 100,
            'all_time_eur': all_total / 100,
        })

    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
# PLATFORM INTEGRATIONS
# ═══════════════════════════════════════════════════════════════

@admin_bp.route('/api/vastai')
@require_admin
def vastai_status():
    """Vast.ai Status"""
    vastai = get_vastai()
    return jsonify({
        'status': vastai.get_status(),
        'machines': [vars(m) for m in vastai.get_my_machines()],
        'instances': [vars(i) for i in vastai.get_my_instances()],
        'earnings': vastai.get_earnings(),
    })


@admin_bp.route('/api/vastai/price', methods=['POST'])
@require_admin
def set_vastai_price():
    """Vast.ai Preis setzen"""
    data = request.json
    machine_id = data.get('machine_id')
    price = data.get('price_per_hour')

    vastai = get_vastai()
    success = vastai.set_machine_price(machine_id, price)

    return jsonify({'success': success})


@admin_bp.route('/api/runpod')
@require_admin
def runpod_status():
    """RunPod Status"""
    runpod = get_runpod()
    return jsonify({
        'status': runpod.get_status(),
        'account': runpod.get_myself(),
        'pods': [vars(p) for p in runpod.get_pods()],
        'endpoints': [vars(e) for e in runpod.get_endpoints()],
        'spending': runpod.get_spending(),
    })


# ═══════════════════════════════════════════════════════════════
# SYSTEM MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@admin_bp.route('/api/config')
@require_admin
def get_config():
    """Aktuelle Konfiguration"""
    return jsonify({
        'service_name': Config.SERVICE_NAME,
        'service_url': Config.SERVICE_URL,
        'max_concurrent_jobs': Config.MAX_CONCURRENT_JOBS,
        'available_models': Config.AVAILABLE_MODELS,
        'prices': {
            'training': Config.PRICES,
            'api': Config.API_PRICES,
            'images': Config.IMAGE_PRICES,
        },
        'integrations': {
            'vastai_enabled': Config.VASTAI_ENABLED,
            'runpod_enabled': Config.RUNPOD_ENABLED,
        },
    })


@admin_bp.route('/api/workers')
@require_admin
def workers_status():
    """Worker-Status"""
    from backend.workers.llm_inference import get_inference_worker
    from backend.workers.llm_training import get_training_worker
    from backend.workers.image_gen import get_image_worker

    return jsonify({
        'inference': get_inference_worker().get_status(),
        'training': get_training_worker().get_status(),
        'image_gen': get_image_worker().get_status(),
    })
