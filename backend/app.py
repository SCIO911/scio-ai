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
# STARTUP
# ===================================================================

def start_services():
    """Startet alle Background-Services - VOLLAUTOMATISCH"""
    print("[SETUP] Starte Services (Vollautomatischer Modus)...")

    # Initialize Database
    try:
        init_db()
        print("[OK] Datenbank initialisiert")
    except Exception as e:
        print(f"[WARN]  Datenbank-Fehler: {e}")

    # Start Hardware Monitor
    try:
        from backend.services.hardware_monitor import get_hardware_monitor
        monitor = get_hardware_monitor()
        monitor.start()
    except Exception as e:
        print(f"[WARN]  Hardware Monitor Fehler: {e}")

    # Start Job Queue
    try:
        from backend.services.job_queue import get_job_queue
        from backend.models.job import JobType

        queue = get_job_queue()

        # Register Workers
        try:
            from backend.workers.llm_inference import get_inference_worker
            worker = get_inference_worker()
            if worker.initialize():
                queue.register_worker(JobType.LLM_INFERENCE, worker)
        except Exception as e:
            print(f"[WARN]  LLM Inference Worker nicht verfügbar: {e}")

        try:
            from backend.workers.llm_training import get_training_worker
            worker = get_training_worker()
            if worker.initialize():
                queue.register_worker(JobType.LLM_TRAINING, worker)
        except Exception as e:
            print(f"[WARN]  LLM Training Worker nicht verfügbar: {e}")

        try:
            from backend.workers.image_gen import get_image_worker
            worker = get_image_worker()
            if worker.initialize():
                queue.register_worker(JobType.IMAGE_GENERATION, worker)
        except Exception as e:
            print(f"[WARN]  Image Generation Worker nicht verfügbar: {e}")

        # ═══════════════════════════════════════════════════════════════
        # NEUE WORKER - Alle AI-Tools für SCIO
        # ═══════════════════════════════════════════════════════════════

        # Audio Worker (STT, TTS, Music)
        try:
            from backend.workers.audio_worker import get_audio_worker
            worker = get_audio_worker()
            if worker.initialize():
                queue.register_worker(JobType.SPEECH_TO_TEXT, worker)
                queue.register_worker(JobType.TEXT_TO_SPEECH, worker)
                queue.register_worker(JobType.MUSIC_GENERATION, worker)
        except Exception as e:
            print(f"[WARN]  Audio Worker nicht verfügbar: {e}")

        # Video Worker
        try:
            from backend.workers.video_worker import get_video_worker
            worker = get_video_worker()
            if worker.initialize():
                queue.register_worker(JobType.VIDEO_GENERATION, worker)
                queue.register_worker(JobType.IMAGE_TO_VIDEO, worker)
        except Exception as e:
            print(f"[WARN]  Video Worker nicht verfügbar: {e}")

        # Vision Worker (OCR, Captioning, Detection)
        try:
            from backend.workers.vision_worker import get_vision_worker
            worker = get_vision_worker()
            if worker.initialize():
                queue.register_worker(JobType.IMAGE_CAPTION, worker)
                queue.register_worker(JobType.VISUAL_QA, worker)
                queue.register_worker(JobType.OCR, worker)
                queue.register_worker(JobType.OBJECT_DETECTION, worker)
        except Exception as e:
            print(f"[WARN]  Vision Worker nicht verfügbar: {e}")

        # Code Worker
        try:
            from backend.workers.code_worker import get_code_worker
            worker = get_code_worker()
            if worker.initialize():
                queue.register_worker(JobType.CODE_GENERATION, worker)
                queue.register_worker(JobType.CODE_COMPLETION, worker)
                queue.register_worker(JobType.CODE_REVIEW, worker)
                queue.register_worker(JobType.CODE_FIX, worker)
        except Exception as e:
            print(f"[WARN]  Code Worker nicht verfügbar: {e}")

        # Embedding Worker (RAG)
        try:
            from backend.workers.embedding_worker import get_embedding_worker
            worker = get_embedding_worker()
            if worker.initialize():
                queue.register_worker(JobType.TEXT_EMBEDDING, worker)
                queue.register_worker(JobType.IMAGE_EMBEDDING, worker)
                queue.register_worker(JobType.SIMILARITY_SEARCH, worker)
        except Exception as e:
            print(f"[WARN]  Embedding Worker nicht verfügbar: {e}")

        # Upscale Worker
        try:
            from backend.workers.upscale_worker import get_upscale_worker
            worker = get_upscale_worker()
            if worker.initialize():
                queue.register_worker(JobType.IMAGE_UPSCALE, worker)
                queue.register_worker(JobType.FACE_RESTORE, worker)
        except Exception as e:
            print(f"[WARN]  Upscale Worker nicht verfügbar: {e}")

        # 3D Worker
        try:
            from backend.workers.threed_worker import get_threed_worker
            worker = get_threed_worker()
            if worker.initialize():
                queue.register_worker(JobType.TEXT_TO_3D, worker)
                queue.register_worker(JobType.IMAGE_TO_3D, worker)
        except Exception as e:
            print(f"[WARN]  3D Worker nicht verfügbar: {e}")

        # Document Worker
        try:
            from backend.workers.document_worker import get_document_worker
            worker = get_document_worker()
            if worker.initialize():
                queue.register_worker(JobType.DOCUMENT_PARSE, worker)
                queue.register_worker(JobType.PDF_EXTRACT, worker)
                queue.register_worker(JobType.TEXT_CHUNK, worker)
        except Exception as e:
            print(f"[WARN]  Document Worker nicht verfügbar: {e}")

        queue.start()
    except Exception as e:
        print(f"[WARN]  Job Queue Fehler: {e}")

    # Start Platform Integrations
    try:
        from backend.integrations.vastai import get_vastai
        vastai = get_vastai()
        if vastai._enabled:
            vastai.start_monitor()
    except Exception as e:
        print(f"[WARN]  Vast.ai Integration Fehler: {e}")

    try:
        from backend.integrations.runpod import get_runpod
        runpod = get_runpod()
        if runpod._enabled:
            runpod.start_monitor()
    except Exception as e:
        print(f"[WARN]  RunPod Integration Fehler: {e}")

    # Start Automation Services
    try:
        from backend.automation.scheduler import get_scheduler
        scheduler = get_scheduler()
        scheduler.start()
    except Exception as e:
        print(f"[WARN]  Scheduler Fehler: {e}")

    try:
        from backend.automation.auto_worker import get_auto_worker
        auto_worker = get_auto_worker()
        auto_worker.start()
    except Exception as e:
        print(f"[WARN]  AutoWorker Fehler: {e}")

    # Startup Notification
    try:
        from backend.automation.notifications import get_notification_service
        notifier = get_notification_service()
        notifier.notify_startup()
    except Exception as e:
        print(f"[WARN]  Startup-Benachrichtigung Fehler: {e}")

    # ═══════════════════════════════════════════════════════════════
    # AUTONOMY ENGINE - Selbst-Programmierung
    # ═══════════════════════════════════════════════════════════════
    try:
        from backend.autonomy import get_autonomy_engine
        autonomy = get_autonomy_engine()
        if autonomy.initialize():
            print("[OK] Autonomy Engine initialisiert - Selbst-Programmierung aktiv")
        else:
            print("[WARN] Autonomy Engine konnte nicht initialisiert werden")
    except Exception as e:
        print(f"[WARN]  Autonomy Engine Fehler: {e}")

    # ═══════════════════════════════════════════════════════════════
    # ADVANCED AI MODULES - Intelligente Entscheidungsfindung
    # ═══════════════════════════════════════════════════════════════

    # Decision Engine - Entscheidungsbäume und Heuristiken
    try:
        from backend.decision import get_decision_engine, get_rule_engine
        decision_engine = get_decision_engine()
        if decision_engine.initialize():
            print("[OK] Decision Engine initialisiert")
        rule_engine = get_rule_engine()
        if rule_engine.initialize():
            print("[OK] Rule Engine initialisiert")
    except Exception as e:
        print(f"[WARN]  Decision Module Fehler: {e}")

    # Learning Module - RL und Continuous Learning
    try:
        from backend.learning import get_rl_agent, get_continuous_learner
        rl_agent = get_rl_agent()
        if rl_agent.initialize():
            print("[OK] RL Agent initialisiert")
        learner = get_continuous_learner()
        if learner.initialize():
            print("[OK] Continuous Learner initialisiert")
    except Exception as e:
        print(f"[WARN]  Learning Module Fehler: {e}")

    # Planning Module - A*, MCTS, Hierarchical Planning
    try:
        from backend.planning import get_planner
        planner = get_planner()
        if planner.initialize():
            print("[OK] Planner initialisiert (A*, MCTS)")
    except Exception as e:
        print(f"[WARN]  Planning Module Fehler: {e}")

    # Knowledge Graph - Entitäten, Relationen, Inferenz
    try:
        from backend.knowledge import get_knowledge_graph
        kg = get_knowledge_graph()
        if kg.initialize():
            print("[OK] Knowledge Graph initialisiert")
    except Exception as e:
        print(f"[WARN]  Knowledge Graph Fehler: {e}")

    # Multi-Agent System - Kollaboration
    try:
        from backend.agents import get_multi_agent_system
        mas = get_multi_agent_system()
        if mas.initialize():
            mas.start()
            print("[OK] Multi-Agent System initialisiert")
    except Exception as e:
        print(f"[WARN]  Multi-Agent System Fehler: {e}")

    # Monitoring - Drift Detection, Performance Tracking
    try:
        from backend.monitoring import get_drift_detector, get_performance_tracker
        drift = get_drift_detector()
        if drift.initialize():
            print("[OK] Drift Detector initialisiert")
        perf = get_performance_tracker()
        if perf.initialize():
            print("[OK] Performance Tracker initialisiert")
    except Exception as e:
        print(f"[WARN]  Monitoring Module Fehler: {e}")

    print("[OK] Alle Services gestartet - VOLLAUTOMATISCHER BETRIEB AKTIV")


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
    print("\n[JOB] ENDPOINTS:")
    print("   [NET] http://localhost:5000              Kunden-Portal")
    print("   [SETUP] http://localhost:5000/admin        Admin Dashboard")
    print("   [DOCS] http://localhost:5000/docs         API Dokumentation")
    print("   [HEALTH] http://localhost:5000/health       Health Check")
    print("   [BRAIN] http://localhost:5000/api/autonomy  Autonomie-API")
    print("   [AI] http://localhost:5000/api/ai         AI Modules API")
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
