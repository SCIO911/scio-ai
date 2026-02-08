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

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import stripe
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import Config
from backend.models import init_db
from backend.routes import api_bp, admin_bp, webhooks_bp, tools_bp
from backend.routes.autonomy import autonomy_bp
from backend.routes.ai_modules import ai_bp
from backend.routes.capabilities import caps_bp
from backend.routes.orchestration import orch_bp

# Load .env
load_dotenv(Config.BASE_DIR / '.env')

# ===================================================================
# FLASK APP SETUP
# ===================================================================

app = Flask(__name__, static_folder='../frontend')
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app)

# WebSocket Support (threading mode - compatible with torch)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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


# ===================================================================
# STATIC ROUTES
# ===================================================================

@app.route('/')
def index():
    """Customer Portal"""
    return send_from_directory('../frontend', 'index.html')


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


def start_services():
    """
    Startet alle Background-Services - VOLLAUTOMATISCH

    Diese Funktion orchestriert den Start aller SCIO-Services
    in der richtigen Reihenfolge mit Fehlerbehandlung.
    """
    startup_logger.info("Starte Services (Vollautomatischer Modus)...")

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
    print("\n[JOB] ENDPOINTS:")
    print("   [NET] http://localhost:5000                     Kunden-Portal")
    print("   [SETUP] http://localhost:5000/admin               Admin Dashboard")
    print("   [DOCS] http://localhost:5000/docs                API Dokumentation")
    print("   [HEALTH] http://localhost:5000/health              Health Check")
    print("   [BRAIN] http://localhost:5000/api/autonomy         Autonomie-API")
    print("   [AI] http://localhost:5000/api/ai                AI Modules API")
    print("   [CAPS] http://localhost:5000/api/capabilities      100.000+ Tools")
    print("   [ORCH] http://localhost:5000/api/orchestration     Orchestration API")
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
