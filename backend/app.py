#!/usr/bin/env python3
"""
SCIO AI-Workstation
Vollautomatische AI-Service-Plattform

Features:
- LLM Inference (OpenAI-kompatibel)
- LLM Fine-Tuning (LoRA/QLoRA)
- Image Generation (SDXL, SD)
- Hardware Monitoring
- Job Queue
- API Key Management
- Vast.ai & RunPod Integration
"""

import os
import sys
import json
import time
import hashlib
import threading
import logging
from pathlib import Path
from datetime import datetime

# Suppress noisy loggers
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)

from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import stripe
from dotenv import load_dotenv
import uuid
import signal
import atexit

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import Config
from backend.models import init_db
from backend.routes import api_bp, admin_bp, webhooks_bp, tools_bp
from backend.routes.autonomy import autonomy_bp
from backend.routes.ai_modules import ai_bp
from backend.routes.capabilities import caps_bp
from backend.routes.orchestration import orch_bp
from backend.routes.stats import stats_bp
from backend.routes.openapi import openapi_bp
from backend.routes.config_api import config_api_bp

# Load .env
load_dotenv(Config.BASE_DIR / '.env')

# ===================================================================
# FLASK APP SETUP
# ===================================================================

app = Flask(__name__, static_folder='../frontend')
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Sichere CORS-Konfiguration (nicht mehr wildcard *)
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
]
# Produktions-Domain hinzufügen wenn konfiguriert
if hasattr(Config, 'PRODUCTION_DOMAIN') and Config.PRODUCTION_DOMAIN:
    CORS_ORIGINS.append(f"https://{Config.PRODUCTION_DOMAIN}")

CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# WebSocket Support (mit sicherer CORS-Konfiguration)
socketio = SocketIO(
    app,
    cors_allowed_origins=CORS_ORIGINS,
    async_mode='threading'
)

# Rate Limiting
from backend.core.security import RateLimiter, RateLimitConfig, RateLimitExceeded

rate_limiter = RateLimiter(RateLimitConfig(
    requests_per_second=20,
    requests_per_minute=200,
    requests_per_hour=5000,
    burst_size=50,
    block_duration_seconds=60
))

@app.before_request
def check_rate_limit():
    """Rate Limiting für alle API-Requests"""
    if request.path.startswith('/api/'):
        # Client-ID aus IP oder API-Key
        client_id = request.headers.get('X-API-Key', request.remote_addr)
        if not rate_limiter.is_allowed(client_id):
            remaining = rate_limiter.get_remaining(client_id)
            return jsonify({
                'error': 'Rate limit exceeded',
                'retry_after': 60,
                'remaining': remaining
            }), 429

@app.after_request
def add_security_headers(response):
    """Sicherheits-Header für alle Responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


# ===================================================================
# REQUEST TRACING & AUDIT LOGGING
# ===================================================================

# Audit Logger
audit_logger = logging.getLogger('scio.audit')
audit_logger.setLevel(logging.INFO)

# Request Tracing
@app.before_request
def start_request_tracing():
    """Startet Request-Tracing mit eindeutiger Request-ID"""
    g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4())[:8])
    g.request_start = time.time()

@app.after_request
def complete_request_tracing(response):
    """Komplettiert Request-Tracing, Audit-Logging und Prometheus-Metriken"""
    # Request-ID in Response-Header
    response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')

    # Request-Dauer berechnen
    duration_ms = 0
    duration_s = 0
    if hasattr(g, 'request_start'):
        duration_s = time.time() - g.request_start
        duration_ms = duration_s * 1000
    response.headers['X-Response-Time'] = f"{duration_ms:.2f}ms"

    # Prometheus Metriken
    try:
        from backend.monitoring.prometheus_exporter import get_metrics
        metrics = get_metrics()

        # Request Counter
        endpoint = request.path.split('?')[0][:50]  # Limit length
        metrics.requests_total.inc(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code)
        )

        # Request Duration Histogram
        if duration_s > 0:
            metrics.request_duration.observe(
                duration_s,
                method=request.method,
                endpoint=endpoint
            )

        # Error Counter
        if response.status_code >= 400:
            error_type = "client_error" if response.status_code < 500 else "server_error"
            metrics.request_errors.inc(
                method=request.method,
                endpoint=endpoint,
                error_type=error_type
            )

        # Rate Limit Counter
        if response.status_code == 429:
            metrics.rate_limit_hits.inc()
    except Exception:
        pass  # Metrics sollten nie Request blockieren

    # Audit-Log für API-Requests
    if request.path.startswith('/api/'):
        audit_logger.info(
            f"[{g.request_id}] {request.method} {request.path} "
            f"-> {response.status_code} ({duration_ms:.1f}ms) "
            f"client={request.remote_addr}"
        )

    return response


# ===================================================================
# HEALTH PROBES (Kubernetes-kompatibel)
# ===================================================================

# Globale Shutdown-Flag
_shutdown_requested = False
_ready = False


@app.route('/healthz')
@app.route('/health/live')
def liveness_probe():
    """
    Kubernetes Liveness Probe

    Prüft ob der Prozess noch lebt.
    Bei Failure wird der Container neu gestartet.
    """
    if _shutdown_requested:
        return jsonify({'status': 'shutting_down'}), 503

    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/readyz')
@app.route('/health/ready')
def readiness_probe():
    """
    Kubernetes Readiness Probe

    Prüft ob der Service bereit ist, Traffic zu empfangen.
    Bei Failure wird kein Traffic mehr geroutet.
    """
    if _shutdown_requested:
        return jsonify({'status': 'shutting_down', 'ready': False}), 503

    if not _ready:
        return jsonify({'status': 'starting', 'ready': False}), 503

    # Prüfe kritische Abhängigkeiten
    checks = {}
    all_healthy = True

    # Health Checker aus Core nutzen
    try:
        from backend.core.reliability import get_health_checker
        health = get_health_checker()
        results = health.check_all()

        for name, status in results.items():
            checks[name] = {
                'healthy': status.healthy,
                'latency_ms': status.latency_ms,
                'message': status.message
            }
            if not status.healthy:
                all_healthy = False
    except Exception as e:
        checks['health_checker'] = {'healthy': False, 'message': str(e)}
        all_healthy = False

    if not all_healthy:
        return jsonify({
            'status': 'degraded',
            'ready': False,
            'checks': checks
        }), 503

    return jsonify({
        'status': 'ready',
        'ready': True,
        'checks': checks,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/health/startup')
def startup_probe():
    """
    Kubernetes Startup Probe

    Prüft ob der Service noch am Starten ist.
    Gibt Kubernetes Zeit für langsame Startups.
    """
    if _ready:
        return jsonify({'status': 'started', 'ready': True}), 200

    return jsonify({
        'status': 'starting',
        'ready': False,
        'timestamp': datetime.now().isoformat()
    }), 503


# ===================================================================
# PROMETHEUS METRICS ENDPOINT
# ===================================================================

@app.route('/metrics')
@app.route('/api/metrics')
def prometheus_metrics():
    """
    Prometheus Metrics Endpoint

    Exportiert alle SCIO-Metriken im Prometheus-Text-Format:
    - Job Queue: scio_jobs_queued, scio_jobs_active, scio_jobs_total
    - Hardware: scio_gpu_*, scio_cpu_*, scio_ram_*
    - API: scio_requests_total, scio_request_duration_seconds
    - Decision: scio_decisions_total, scio_decision_confidence
    - Learning: scio_learning_*, scio_rl_rewards
    - Orchestrator: scio_modules_*, scio_events_*, scio_workflows_*

    Usage:
        curl http://localhost:5000/metrics

    Prometheus scrape config:
        - job_name: 'scio'
          static_configs:
            - targets: ['localhost:5000']
    """
    from flask import Response
    from backend.monitoring.prometheus_exporter import get_metrics

    metrics = get_metrics()
    output = metrics.export()

    return Response(output, mimetype='text/plain; version=0.0.4; charset=utf-8')


# ===================================================================
# GRACEFUL SHUTDOWN
# ===================================================================

def graceful_shutdown(signum=None, frame=None):
    """
    Graceful Shutdown Handler

    Beendet alle Services sauber:
    1. Stoppt neue Request-Annahme
    2. Wartet auf laufende Requests
    3. Stoppt Background-Services
    4. Speichert State
    """
    global _shutdown_requested, _ready
    _shutdown_requested = True
    _ready = False

    print("\n[SHUTDOWN] Graceful Shutdown gestartet...")
    shutdown_logger = logging.getLogger('scio.shutdown')

    try:
        # 1. Event Bus stoppen
        try:
            from backend.orchestration.event_bus import get_event_bus
            event_bus = get_event_bus()
            event_bus.stop()
            shutdown_logger.info("Event Bus gestoppt")
        except Exception as e:
            shutdown_logger.warning(f"Event Bus Stop-Fehler: {e}")

        # 2. Job Queue stoppen (laufende Jobs abwarten)
        try:
            from backend.services.job_queue import get_job_queue
            queue = get_job_queue()
            queue.stop(wait=True, timeout=30)
            shutdown_logger.info("Job Queue gestoppt")
        except Exception as e:
            shutdown_logger.warning(f"Job Queue Stop-Fehler: {e}")

        # 3. Multi-Agent System stoppen
        try:
            from backend.agents import get_multi_agent_system
            mas = get_multi_agent_system()
            mas.stop()
            shutdown_logger.info("Multi-Agent System gestoppt")
        except Exception as e:
            shutdown_logger.warning(f"Multi-Agent Stop-Fehler: {e}")

        # 4. Hardware Monitor stoppen
        try:
            from backend.services.hardware_monitor import get_hardware_monitor
            monitor = get_hardware_monitor()
            monitor.stop()
            shutdown_logger.info("Hardware Monitor gestoppt")
        except Exception as e:
            shutdown_logger.warning(f"Hardware Monitor Stop-Fehler: {e}")

        # 5. RL Agent State speichern
        try:
            from backend.learning import get_rl_agent
            agent = get_rl_agent()
            agent.save_state()
            shutdown_logger.info("RL Agent State gespeichert")
        except Exception as e:
            shutdown_logger.warning(f"RL Agent Save-Fehler: {e}")

        print("[SHUTDOWN] Graceful Shutdown abgeschlossen")

    except Exception as e:
        print(f"[SHUTDOWN] Fehler beim Shutdown: {e}")

    # Bei Signal-Handler: Prozess beenden
    if signum is not None:
        sys.exit(0)


# Signal-Handler registrieren
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)
atexit.register(graceful_shutdown)


# Stripe
stripe.api_key = Config.STRIPE_SECRET_KEY
if not stripe.api_key:
    print("[WARN]  WARNING: STRIPE_SECRET_KEY nicht gesetzt!")

# Register Blueprints
app.register_blueprint(api_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(webhooks_bp)
app.register_blueprint(tools_bp)  # Alle AI-Tools
app.register_blueprint(autonomy_bp, url_prefix='/api/autonomy')  # Selbst-Programmierung
app.register_blueprint(ai_bp)  # AI Modules (Decision, Learning, Planning, Knowledge, Agents, Monitoring)
app.register_blueprint(caps_bp)  # Capabilities (100.000+ Tools)
app.register_blueprint(orch_bp)  # Orchestration (Event Bus, Workflows, Coordination)
app.register_blueprint(stats_bp)  # Dashboard Stats API
app.register_blueprint(openapi_bp)  # OpenAPI/Swagger Documentation
app.register_blueprint(config_api_bp)  # Runtime Configuration API


# ===================================================================
# STATIC ROUTES
# ===================================================================

@app.route('/')
def index():
    """Customer Portal"""
    return send_from_directory('../frontend', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend folder"""
    return send_from_directory('../frontend', filename)


@app.route('/docs')
def docs():
    """API Documentation"""
    return send_from_directory('../frontend', 'docs.html')


@app.route('/health')
def health():
    """Health Check"""
    from backend.services.hardware_monitor import get_hardware_monitor
    from backend.services.job_queue import get_job_queue

    monitor = get_hardware_monitor()
    queue = get_job_queue()

    status = monitor.get_status()

    return jsonify({
        'status': 'healthy',
        'service': 'SCIO AI-Workstation',
        'version': '2.0.0',
        'stripe_mode': 'LIVE' if stripe.api_key and 'live' in stripe.api_key else 'TEST',
        'gpu_count': len(status.gpus),
        'gpu_available': len(status.gpus) > 0,
        'is_busy': status.is_busy,
        'active_jobs': queue.active_job_count,
        'queued_jobs': queue.queue_size,
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/config')
def public_config():
    """Public Configuration"""
    return jsonify({
        'stripe_key': Config.STRIPE_PUBLISHABLE_KEY,
        'prices': {
            k: {
                'amount': v,
                'display': f'{v/100:.2f}€'
            } for k, v in Config.PRICES.items()
        },
        'service': Config.SERVICE_NAME,
    })


# ===================================================================
# LEGACY PAYMENT ROUTES (für bestehendes Frontend)
# ===================================================================

@app.route('/api/create-payment-intent', methods=['POST'])
def create_payment():
    """Create Stripe Payment Intent (Legacy)"""
    try:
        data = request.json
        model_size = data.get('model_size')
        email = data.get('email')

        if not model_size or model_size not in Config.PRICES:
            return jsonify({'error': 'Ungültige Model-Größe'}), 400

        if not email or '@' not in email:
            return jsonify({'error': 'Ungültige Email'}), 400

        order_id = hashlib.sha256(
            f"{email}{time.time()}{model_size}".encode()
        ).hexdigest()[:16]

        amount = Config.PRICES[model_size]

        print(f"[INFO] Creating Payment Intent: {order_id} ({amount/100:.2f}€)")

        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='eur',
            automatic_payment_methods={'enabled': True},
            metadata={
                'order_id': order_id,
                'model_size': model_size,
                'email': email,
                'service': 'SCIO'
            },
            receipt_email=email,
            description=f'SCIO AI Training - {model_size.upper()}'
        )

        # Save Order
        order_dir = Config.ORDERS_DIR / order_id
        order_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'order_id': order_id,
            'email': email,
            'model_size': model_size,
            'amount_eur': amount / 100,
            'currency': 'eur',
            'status': 'pending_payment',
            'payment_intent_id': intent.id,
            'created_at': datetime.now().isoformat(),
            'service': Config.SERVICE_NAME
        }

        with open(order_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[OK] Payment Intent created: {order_id}")

        return jsonify({
            'clientSecret': intent.client_secret,
            'order_id': order_id
        })

    except stripe.error.StripeError as e:
        print(f"[ERROR] Stripe Error: {e}")
        return jsonify({'error': f'Stripe-Fehler: {str(e)}'}), 500
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload customer dataset (Legacy)"""
    try:
        order_id = request.form.get('order_id')
        file = request.files.get('file')

        if not order_id or not file:
            return jsonify({'error': 'Fehlende Daten'}), 400

        order_dir = Config.ORDERS_DIR / order_id

        if not order_dir.exists():
            return jsonify({'error': 'Order nicht gefunden'}), 404

        ext = os.path.splitext(file.filename)[1].lower()
        allowed = ['.csv', '.json', '.jsonl', '.txt']

        if ext not in allowed:
            return jsonify({'error': f'Ungültiges Format. Erlaubt: {", ".join(allowed)}'}), 400

        filepath = order_dir / f'dataset{ext}'
        file.save(filepath)

        # Update metadata
        metadata_file = order_dir / 'metadata.json'
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        metadata['dataset_filename'] = file.filename
        metadata['dataset_size_bytes'] = os.path.getsize(filepath)
        metadata['dataset_uploaded_at'] = datetime.now().isoformat()

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[FILE] Dataset uploaded: {order_id} ({file.filename})")

        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"[ERROR] Upload Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/webhook/stripe', methods=['POST'])
def stripe_webhook_legacy():
    """Legacy Stripe Webhook (redirect to new endpoint)"""
    from backend.routes.webhooks import stripe_webhook
    return stripe_webhook()


@app.route('/api/orders')
def list_orders():
    """List all orders (Legacy)"""
    orders = []

    for order_dir in Config.ORDERS_DIR.iterdir():
        if order_dir.is_dir():
            metadata_file = order_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    orders.append({
                        'order_id': metadata['order_id'],
                        'email': metadata['email'],
                        'status': metadata['status'],
                        'amount_eur': metadata['amount_eur'],
                        'created_at': metadata['created_at']
                    })

    orders.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'orders': orders, 'total': len(orders)})


# ===================================================================
# WEBSOCKET HANDLERS
# ===================================================================

@socketio.on('connect')
def handle_connect():
    """WebSocket Connect"""
    print(f"[SOCKET] Client connected: {request.sid}")
    emit('connected', {'status': 'ok'})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket Disconnect"""
    print(f"[SOCKET] Client disconnected: {request.sid}")


@socketio.on('subscribe_hardware')
def handle_subscribe_hardware():
    """Subscribe to Hardware Updates"""
    from backend.services.hardware_monitor import get_hardware_monitor

    def send_update(status):
        socketio.emit('hardware_update', status.to_dict(), room=request.sid)

    monitor = get_hardware_monitor()
    monitor.add_callback(send_update)

    # Send initial status
    status = monitor.get_status()
    emit('hardware_update', status.to_dict())


@socketio.on('subscribe_jobs')
def handle_subscribe_jobs():
    """Subscribe to Job Updates"""
    from backend.services.job_queue import get_job_queue

    def send_update(event, job_id, data):
        socketio.emit('job_update', {
            'event': event,
            'job_id': job_id,
            'data': data,
        }, room=request.sid)

    queue = get_job_queue()
    queue.add_callback(send_update)

    # Send current stats
    emit('jobs_stats', queue.get_stats())


@socketio.on('subscribe_stats')
def handle_subscribe_stats():
    """Subscribe to Real-Time Stats Updates"""
    from backend.routes.stats import get_realtime_stats

    # Send initial stats
    emit('stats_update', get_realtime_stats())


@socketio.on('subscribe_system')
def handle_subscribe_system():
    """
    Subscribe to System Events

    Events:
    - module_status: Status-Änderungen von Modulen
    - error_occurred: System-Fehler
    - warning: Warnungen (Degradation, etc.)
    """
    from backend.orchestration.event_bus import get_event_bus, EventType

    event_bus = get_event_bus()

    def on_system_event(event):
        socketio.emit('system_event', {
            'type': event.type.value,
            'source': event.source,
            'data': event.data,
            'timestamp': event.timestamp.isoformat()
        }, room=request.sid)

    # Subscribe to system events
    event_bus.subscribe(EventType.SYSTEM_ERROR, on_system_event)
    event_bus.subscribe(EventType.SYSTEM_HEALTH, on_system_event)

    emit('subscribed', {'channel': 'system'})


@socketio.on('subscribe_decisions')
def handle_subscribe_decisions():
    """
    Subscribe to Decision Engine Events

    Events:
    - decision_made: Eine Entscheidung wurde getroffen
    - decision_stats: Statistik-Updates (periodisch)
    """
    from backend.orchestration.event_bus import get_event_bus, EventType

    event_bus = get_event_bus()

    def on_decision_event(event):
        socketio.emit('decision_event', {
            'type': event.type.value,
            'data': event.data,
            'timestamp': event.timestamp.isoformat()
        }, room=request.sid)

    event_bus.subscribe(EventType.DECISION_MADE, on_decision_event)
    event_bus.subscribe(EventType.DECISION_FEEDBACK, on_decision_event)

    # Send current stats
    try:
        from backend.decision import get_decision_engine
        engine = get_decision_engine()
        emit('decision_stats', engine.get_statistics())
    except Exception:
        pass

    emit('subscribed', {'channel': 'decisions'})


@socketio.on('subscribe_learning')
def handle_subscribe_learning():
    """
    Subscribe to Learning Module Events

    Events:
    - learning_observation: Neue Beobachtung
    - learning_update: Modell-Update
    - reward_calculated: Reward berechnet
    """
    from backend.orchestration.event_bus import get_event_bus, EventType

    event_bus = get_event_bus()

    def on_learning_event(event):
        socketio.emit('learning_event', {
            'type': event.type.value,
            'data': event.data,
            'timestamp': event.timestamp.isoformat()
        }, room=request.sid)

    event_bus.subscribe(EventType.LEARNING_OBSERVATION, on_learning_event)
    event_bus.subscribe(EventType.LEARNING_UPDATE, on_learning_event)
    event_bus.subscribe(EventType.REWARD_CALCULATED, on_learning_event)

    # Send current stats
    try:
        from backend.learning import get_rl_agent
        agent = get_rl_agent()
        emit('learning_stats', agent.get_statistics())
    except Exception:
        pass

    emit('subscribed', {'channel': 'learning'})


@socketio.on('subscribe_workflows')
def handle_subscribe_workflows():
    """
    Subscribe to Workflow Events

    Events:
    - plan_created: Workflow erstellt
    - plan_step_started: Step gestartet
    - plan_step_completed: Step abgeschlossen
    - plan_completed: Workflow abgeschlossen
    - plan_failed: Workflow fehlgeschlagen
    """
    from backend.orchestration.event_bus import get_event_bus, EventType

    event_bus = get_event_bus()

    def on_workflow_event(event):
        socketio.emit('workflow_event', {
            'type': event.type.value,
            'data': event.data,
            'timestamp': event.timestamp.isoformat()
        }, room=request.sid)

    event_bus.subscribe(EventType.PLAN_CREATED, on_workflow_event)
    event_bus.subscribe(EventType.PLAN_STEP_STARTED, on_workflow_event)
    event_bus.subscribe(EventType.PLAN_STEP_COMPLETED, on_workflow_event)
    event_bus.subscribe(EventType.PLAN_COMPLETED, on_workflow_event)
    event_bus.subscribe(EventType.PLAN_FAILED, on_workflow_event)

    emit('subscribed', {'channel': 'workflows'})


@socketio.on('subscribe_all')
def handle_subscribe_all():
    """Subscribe to all event channels"""
    handle_subscribe_hardware()
    handle_subscribe_jobs()
    handle_subscribe_stats()
    handle_subscribe_system()
    handle_subscribe_decisions()
    handle_subscribe_learning()
    handle_subscribe_workflows()

    emit('subscribed', {'channel': 'all'})


# ===================================================================
# STARTUP - Modular Service Initialization
# ===================================================================

import logging

# Configure module logger
startup_logger = logging.getLogger('scio.startup')


def _init_database():
    """Initialisiert die Datenbank"""
    try:
        init_db()
        startup_logger.info("Datenbank initialisiert")
        return True
    except Exception as e:
        startup_logger.warning(f"Datenbank-Fehler: {e}")
        return False


def _init_hardware_monitor():
    """Startet den Hardware Monitor"""
    try:
        from backend.services.hardware_monitor import get_hardware_monitor
        monitor = get_hardware_monitor()
        monitor.start()
        return True
    except Exception as e:
        startup_logger.warning(f"Hardware Monitor Fehler: {e}")
        return False


def _register_worker(queue, worker_getter, job_types, worker_name):
    """Hilfsfunktion zum Registrieren eines Workers"""
    try:
        worker = worker_getter()
        if worker.initialize():
            for job_type in job_types:
                queue.register_worker(job_type, worker)
            return True
    except Exception as e:
        startup_logger.warning(f"{worker_name} nicht verfügbar: {e}")
    return False


def _init_job_queue():
    """Initialisiert die Job Queue und registriert alle Worker"""
    try:
        from backend.services.job_queue import get_job_queue
        from backend.models.job import JobType

        queue = get_job_queue()

        # Worker-Definitionen: (getter_func, job_types, name)
        worker_configs = [
            # Core Workers
            ('backend.workers.llm_inference', 'get_inference_worker',
             [JobType.LLM_INFERENCE], 'LLM Inference Worker'),
            ('backend.workers.llm_training', 'get_training_worker',
             [JobType.LLM_TRAINING], 'LLM Training Worker'),
            ('backend.workers.image_gen', 'get_image_worker',
             [JobType.IMAGE_GENERATION], 'Image Generation Worker'),

            # Audio Worker
            ('backend.workers.audio_worker', 'get_audio_worker',
             [JobType.SPEECH_TO_TEXT, JobType.TEXT_TO_SPEECH, JobType.MUSIC_GENERATION],
             'Audio Worker'),

            # Video Worker
            ('backend.workers.video_worker', 'get_video_worker',
             [JobType.VIDEO_GENERATION, JobType.IMAGE_TO_VIDEO], 'Video Worker'),

            # Vision Worker
            ('backend.workers.vision_worker', 'get_vision_worker',
             [JobType.IMAGE_CAPTION, JobType.VISUAL_QA, JobType.OCR, JobType.OBJECT_DETECTION],
             'Vision Worker'),

            # Code Worker
            ('backend.workers.code_worker', 'get_code_worker',
             [JobType.CODE_GENERATION, JobType.CODE_COMPLETION, JobType.CODE_REVIEW, JobType.CODE_FIX],
             'Code Worker'),

            # Embedding Worker
            ('backend.workers.embedding_worker', 'get_embedding_worker',
             [JobType.TEXT_EMBEDDING, JobType.IMAGE_EMBEDDING, JobType.SIMILARITY_SEARCH],
             'Embedding Worker'),

            # Upscale Worker
            ('backend.workers.upscale_worker', 'get_upscale_worker',
             [JobType.IMAGE_UPSCALE, JobType.FACE_RESTORE], 'Upscale Worker'),

            # 3D Worker
            ('backend.workers.threed_worker', 'get_threed_worker',
             [JobType.TEXT_TO_3D, JobType.IMAGE_TO_3D], '3D Worker'),

            # Document Worker
            ('backend.workers.document_worker', 'get_document_worker',
             [JobType.DOCUMENT_PARSE, JobType.PDF_EXTRACT, JobType.TEXT_CHUNK],
             'Document Worker'),
        ]

        # Registriere alle Worker dynamisch
        for module_path, getter_name, job_types, worker_name in worker_configs:
            try:
                module = __import__(module_path, fromlist=[getter_name])
                getter_func = getattr(module, getter_name)
                _register_worker(queue, getter_func, job_types, worker_name)
            except Exception as e:
                startup_logger.warning(f"{worker_name} nicht verfügbar: {e}")

        queue.start()
        return True
    except Exception as e:
        startup_logger.warning(f"Job Queue Fehler: {e}")
        return False


def _init_platform_integrations():
    """Initialisiert Platform-Integrationen (Vast.ai, RunPod)"""
    # Vast.ai
    try:
        from backend.integrations.vastai import get_vastai
        vastai = get_vastai()
        if vastai._enabled:
            vastai.start_monitor()
    except Exception as e:
        startup_logger.warning(f"Vast.ai Integration Fehler: {e}")

    # RunPod
    try:
        from backend.integrations.runpod import get_runpod
        runpod = get_runpod()
        if runpod._enabled:
            runpod.start_monitor()
    except Exception as e:
        startup_logger.warning(f"RunPod Integration Fehler: {e}")


def _init_automation_services():
    """Startet Automatisierungs-Services"""
    # Scheduler
    try:
        from backend.automation.scheduler import get_scheduler
        scheduler = get_scheduler()
        scheduler.start()
    except Exception as e:
        startup_logger.warning(f"Scheduler Fehler: {e}")

    # Auto Worker
    try:
        from backend.automation.auto_worker import get_auto_worker
        auto_worker = get_auto_worker()
        auto_worker.start()
    except Exception as e:
        startup_logger.warning(f"AutoWorker Fehler: {e}")

    # Money Maker - Automatisches Geldverdienen
    try:
        from backend.automation.money_maker import get_money_maker
        money_maker = get_money_maker()
        money_maker.start()
    except Exception as e:
        startup_logger.warning(f"MoneyMaker Fehler: {e}")

    # Startup Notification
    try:
        from backend.automation.notifications import get_notification_service
        notifier = get_notification_service()
        notifier.notify_startup()
    except Exception as e:
        startup_logger.warning(f"Startup-Benachrichtigung Fehler: {e}")


def _init_autonomy_engine():
    """Initialisiert die Autonomy Engine für Selbst-Programmierung"""
    try:
        from backend.autonomy import get_autonomy_engine
        autonomy = get_autonomy_engine()
        if autonomy.initialize():
            startup_logger.info("Autonomy Engine initialisiert - Selbst-Programmierung aktiv")
            return True
        else:
            startup_logger.warning("Autonomy Engine konnte nicht initialisiert werden")
            return False
    except Exception as e:
        startup_logger.warning(f"Autonomy Engine Fehler: {e}")
        return False


def _init_ai_modules():
    """Initialisiert erweiterte AI-Module"""
    # Decision Engine
    try:
        from backend.decision import get_decision_engine, get_rule_engine
        decision_engine = get_decision_engine()
        if decision_engine.initialize():
            startup_logger.info("Decision Engine initialisiert")
        rule_engine = get_rule_engine()
        if rule_engine.initialize():
            startup_logger.info("Rule Engine initialisiert")
    except Exception as e:
        startup_logger.warning(f"Decision Module Fehler: {e}")

    # Learning Module
    try:
        from backend.learning import get_rl_agent, get_continuous_learner
        rl_agent = get_rl_agent()
        if rl_agent.initialize():
            startup_logger.info("RL Agent initialisiert")
        learner = get_continuous_learner()
        if learner.initialize():
            startup_logger.info("Continuous Learner initialisiert")
    except Exception as e:
        startup_logger.warning(f"Learning Module Fehler: {e}")

    # Planning Module
    try:
        from backend.planning import get_planner
        planner = get_planner()
        if planner.initialize():
            startup_logger.info("Planner initialisiert (A*, MCTS)")
    except Exception as e:
        startup_logger.warning(f"Planning Module Fehler: {e}")

    # Knowledge Graph
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()
        if kg.initialize():
            startup_logger.info("Knowledge Graph initialisiert")
    except Exception as e:
        startup_logger.warning(f"Knowledge Graph Fehler: {e}")

    # Multi-Agent System
    try:
        from backend.agents import get_multi_agent_system
        mas = get_multi_agent_system()
        if mas.initialize():
            mas.start()
            startup_logger.info("Multi-Agent System initialisiert")
    except Exception as e:
        startup_logger.warning(f"Multi-Agent System Fehler: {e}")

    # Monitoring
    try:
        from backend.monitoring import get_drift_detector, get_performance_tracker
        drift = get_drift_detector()
        if drift.initialize():
            startup_logger.info("Drift Detector initialisiert")
        perf = get_performance_tracker()
        if perf.initialize():
            startup_logger.info("Performance Tracker initialisiert")
    except Exception as e:
        startup_logger.warning(f"Monitoring Module Fehler: {e}")

    # Capability Engine - 100.000+ Tools
    try:
        from backend.capabilities import get_capability_engine
        caps = get_capability_engine()
        if caps.initialize():
            startup_logger.info(f"Capability Engine initialisiert ({caps.registry._tool_count} Tools)")
    except Exception as e:
        startup_logger.warning(f"Capability Engine Fehler: {e}")


def _init_orchestration():
    """Initialisiert den SCIO Orchestrator für Modul-Koordination"""
    try:
        from backend.orchestration import get_orchestrator
        orchestrator = get_orchestrator()
        if orchestrator.initialize():
            startup_logger.info("SCIO Orchestrator initialisiert - Module verbunden")
            return True
        else:
            startup_logger.warning("Orchestrator konnte nicht initialisiert werden")
            return False
    except Exception as e:
        startup_logger.warning(f"Orchestrator Fehler: {e}")
        return False


def _register_health_checks():
    """Registriert Health Checks für alle kritischen Services"""
    from backend.core.reliability import get_health_checker

    health = get_health_checker()

    # Database Health Check
    @health.register("database", timeout_seconds=5.0)
    def check_database():
        try:
            from backend.models import get_session
            from sqlalchemy import text
            session = get_session()
            try:
                session.execute(text("SELECT 1"))
                return True
            finally:
                session.close()
        except Exception:
            return False

    # Job Queue Health Check
    @health.register("job_queue", timeout_seconds=3.0)
    def check_job_queue():
        try:
            from backend.services.job_queue import get_job_queue
            queue = get_job_queue()
            return queue.is_running
        except Exception:
            return False

    # Event Bus Health Check
    @health.register("event_bus", timeout_seconds=2.0)
    def check_event_bus():
        try:
            from backend.orchestration.event_bus import get_event_bus
            bus = get_event_bus()
            return bus._running
        except Exception:
            return False

    # Hardware Monitor Health Check
    @health.register("hardware_monitor", timeout_seconds=3.0)
    def check_hardware():
        try:
            from backend.services.hardware_monitor import get_hardware_monitor
            monitor = get_hardware_monitor()
            status = monitor.get_status()
            return status is not None
        except Exception:
            return False


def start_services():
    """
    Startet alle Background-Services - VOLLAUTOMATISCH

    Diese Funktion orchestriert den Start aller SCIO-Services
    in der richtigen Reihenfolge mit Fehlerbehandlung.
    """
    global _ready
    startup_logger.info("Starte Services (Vollautomatischer Modus)...")

    # 0. Health Checks registrieren
    _register_health_checks()

    # 1. Datenbank initialisieren (kritisch)
    _init_database()

    # 2. Hardware Monitor starten
    _init_hardware_monitor()

    # 3. Job Queue mit allen Workern initialisieren
    _init_job_queue()

    # 4. Platform-Integrationen starten
    _init_platform_integrations()

    # 5. Automatisierungs-Services starten
    _init_automation_services()

    # 6. Autonomy Engine initialisieren
    _init_autonomy_engine()

    # 7. Erweiterte AI-Module initialisieren
    _init_ai_modules()

    # 8. SCIO Orchestrator initialisieren - Verbindet alle Module
    _init_orchestration()

    # 9. Service als bereit markieren
    _ready = True

    startup_logger.info("Alle Services gestartet - VOLLAUTOMATISCHER BETRIEB AKTIV")


def print_banner():
    """Zeigt Startup-Banner"""
    print("\n" + "=" * 70)
    print("[INFO] SCIO AI-WORKSTATION v2.0 - VOLLAUTOMATISCH")
    print("=" * 70)
    print(f"[STRIPE] Stripe: {'LIVE [LIVE]' if stripe.api_key and 'live' in stripe.api_key else 'TEST [TEST]'}")
    print(f"[NET] Vast.ai: {'Enabled [OK]' if Config.VASTAI_ENABLED else 'Disabled [-]'}")
    print(f"[NET] RunPod: {'Enabled [OK]' if Config.RUNPOD_ENABLED else 'Disabled [-]'}")
    print(f"[AUTO] Auto-Mode: AKTIV")
    print("=" * 70)
    print("\n[LAUNCH] AUTOMATISCHE FEATURES:")
    print("   [OK] Automatische Job-Verarbeitung")
    print("   [OK] Automatische Benachrichtigungen")
    print("   [OK] Automatische GPU-Rental wenn idle")
    print("   [OK] Automatische Health-Checks")
    print("   [OK] Automatische Earnings-Reports")
    print("   [OK] Auto-Recovery bei Fehlern")

    print("\n[MONEY] MONEY MAKER - AUTOMATISCHES GELDVERDIENEN:")
    print("   [OK] Vast.ai GPU-Vermietung (HOST-ONLY)")
    print("   [OK] Automatische Preisoptimierung")
    print("   [OK] Echtzeit Earnings-Tracking")
    print("   [OK] Intelligente Ressourcen-Allokation")
    print("   [OK] Taegliche Earnings-Reports")
    print("=" * 70)
    print("\n[BRAIN] AUTONOMIE-SYSTEM:")
    print("   [OK] Self-Awareness - Codebase-Analyse")
    print("   [OK] Capability-Analyzer - Faehigkeiten erkennen")
    print("   [OK] Code-Generator - Selbst-Programmierung")
    print("   [OK] Self-Tester - Code validieren")
    print("   [OK] Evolution-Planner - Weiterentwicklung")
    print("   [OK] Memory - Langzeit-Gedaechtnis")
    print("   [OK] Task-Verifier - 100% Erfuellung garantiert")
    print("=" * 70)
    print("\n[AI] ADVANCED AI MODULES:")
    print("   [OK] Decision Engine - Entscheidungsbaeume & Heuristiken")
    print("   [OK] Rule Engine - Regelbasierte Logik")
    print("   [OK] RL Agent - Q-Learning & Reinforcement")
    print("   [OK] Continuous Learner - Feedback & Patterns")
    print("   [OK] Planner - A*, MCTS, Hierarchical Planning")
    print("   [OK] Knowledge Graph - Entitaeten & Inferenz")
    print("   [OK] Multi-Agent System - Kollaboration")
    print("   [OK] Drift Detector - Anomalie-Erkennung")
    print("   [OK] Performance Tracker - SLA Monitoring")
    print("=" * 70)
    print("\n[TOOLS] CAPABILITY ENGINE (100.000+ Tools):")
    print("   [OK] NLP - Text, Translation, Sentiment, NER, QA")
    print("   [OK] Vision - Classification, Detection, Segmentation, OCR")
    print("   [OK] Audio - STT, TTS, Music, Voice Analysis")
    print("   [OK] Video - Generation, Editing, Tracking, Effects")
    print("   [OK] Generative - Image, 3D, Code, Documents")
    print("   [OK] Code - 20+ Languages, Analysis, Git, Build")
    print("   [OK] Data - Processing, Analytics, Statistics")
    print("   [OK] Documents - PDF, Office, Conversion")
    print("   [OK] Web - Scraping, APIs, SEO, Automation")
    print("   [OK] Security - Encryption, Auth, Scanning")
    print("   [OK] Cloud - AWS, GCP, Azure, Kubernetes, Docker")
    print("   [OK] ML/AI - Training, Deployment, Monitoring")
    print("   [OK] Business - Finance, CRM, HR, Calendar")
    print("   [OK] Science - Math, Statistics, Simulation")
    print("   [OK] Visualization - Charts, Maps, Dashboards")
    print("=" * 70)
    print("\n[ORCH] ORCHESTRATION SYSTEM:")
    print("   [OK] Event Bus - Pub/Sub Cross-Module Kommunikation")
    print("   [OK] Workflow Engine - Multi-Step Orchestrierung")
    print("   [OK] Orchestrator - Zentrale Modul-Koordination")
    print("   [OK] Auto-Feedback - RL lernt aus Job-Ergebnissen")
    print("   [OK] Knowledge Updates - Entscheidungen -> Graph")
    print("   [OK] Drift Alerts - Automatische Benachrichtigung")
    print("   [OK] Health Monitoring - Alle Module ueberwacht")
    print("=" * 70)
    print("\n[PROD] PRODUCTION FEATURES:")
    print("   [OK] Health Probes - /healthz, /readyz (Kubernetes)")
    print("   [OK] Request Tracing - X-Request-ID Header")
    print("   [OK] Audit Logging - API Request/Response Log")
    print("   [OK] Graceful Shutdown - SIGTERM/SIGINT Handler")
    print("   [OK] Rate Limiting - Token Bucket + Multi-Window")
    print("   [OK] Security Headers - XSS, CSRF, Clickjacking")
    print("   [OK] Circuit Breakers - Fault Tolerance")
    print("   [OK] Retry Logic - Exponential Backoff")
    print("=" * 70)
    print("\n[JOB] ENDPOINTS:")
    print("   [NET] http://localhost:5000                     Kunden-Portal")
    print("   [SETUP] http://localhost:5000/admin               Admin Dashboard")
    print("   [DOCS] http://localhost:5000/docs                API Dokumentation")
    print("   [HEALTH] http://localhost:5000/health              Health Check")
    print("   [PROBE] http://localhost:5000/healthz              Liveness Probe")
    print("   [PROBE] http://localhost:5000/readyz               Readiness Probe")
    print("   [BRAIN] http://localhost:5000/api/autonomy         Autonomie-API")
    print("   [AI] http://localhost:5000/api/ai                AI Modules API")
    print("   [CAPS] http://localhost:5000/api/capabilities      100.000+ Tools")
    print("   [ORCH] http://localhost:5000/api/orchestration     Orchestration API")
    print("   [STATS] http://localhost:5000/api/stats            Dashboard Stats")
    print("   [PROM] http://localhost:5000/metrics               Prometheus Metrics")
    print("   [DOCS] http://localhost:5000/api/docs              Swagger UI")
    print("   [CONF] http://localhost:5000/api/config            Runtime Config")
    print("=" * 70)
    print(f"\n[NET] Server: http://localhost:{Config.PORT}")
    print("[AUTO] System laeuft vollautomatisch - keine manuellen Eingriffe noetig!")
    print(f"\n[WARN]  Strg+C zum Beenden\n")


# ===================================================================
# MAIN
# ===================================================================

if __name__ == '__main__':
    print_banner()
    start_services()

    print("\n[OK] PRODUCTION SERVER")
    print(f"[OK] http://{Config.HOST}:{Config.PORT}")
    print(f"[OK] Admin: http://localhost:{Config.PORT}/admin/")
    print("[OK] System bereit - Vollautomatischer Betrieb\n")

    # Run with SocketIO (threading mode for torch compatibility)
    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=False,
        use_reloader=False,
        log_output=False,
        allow_unsafe_werkzeug=True,
    )
