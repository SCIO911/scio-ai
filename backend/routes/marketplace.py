#!/usr/bin/env python3
"""
SCIO - Marketplace API Routes
Öffentliche API für Job-Einreichung und Abrechnung
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import stripe

from backend.config import Config
from backend.services.job_marketplace import get_marketplace
from backend.services.job_queue import get_job_queue, JobType

marketplace_bp = Blueprint('marketplace', __name__)

# Stripe initialisieren
stripe.api_key = Config.STRIPE_SECRET_KEY


@marketplace_bp.route('/submit', methods=['POST'])
def submit_job():
    """
    Öffentlicher Endpoint für Job-Einreichung

    Body:
    {
        "job_type": "image_generation",
        "params": {...},
        "customer_email": "kunde@example.com",
        "payment_method_id": "pm_xxx"  # Optional für Stripe
    }
    """
    try:
        data = request.get_json()

        job_type = data.get('job_type')
        params = data.get('params', {})
        customer_email = data.get('customer_email')

        if not job_type or not customer_email:
            return jsonify({
                'error': 'job_type and customer_email required'
            }), 400

        marketplace = get_marketplace()

        # Job erstellen
        job = marketplace.direct_api.submit_job(
            job_type=job_type,
            params=params,
            customer_id=customer_email
        )

        return jsonify({
            'success': True,
            'job_id': job.job_id,
            'price_usd': job.price_usd,
            'status': job.status,
            'message': 'Job eingereicht. Zahlung erforderlich für Verarbeitung.'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@marketplace_bp.route('/pricing', methods=['GET'])
def get_pricing():
    """Gibt aktuelle Preisliste zurück"""
    marketplace = get_marketplace()

    return jsonify({
        'currency': 'USD',
        'pricing': {
            'llm_inference': {
                'price': 0.002,
                'unit': 'per 1K tokens',
                'description': 'LLM Text Generation (Mistral, LLaMA, etc.)'
            },
            'image_generation': {
                'price': 0.02,
                'unit': 'per image',
                'description': 'Image Generation (FLUX, SDXL, SD3.5)'
            },
            'video_generation': {
                'price': 0.50,
                'unit': 'per second',
                'description': 'Video Generation (CogVideoX, AnimateDiff)'
            },
            'audio_transcription': {
                'price': 0.006,
                'unit': 'per minute',
                'description': 'Speech-to-Text (Whisper Large V3)'
            },
            'audio_tts': {
                'price': 0.015,
                'unit': 'per 1K characters',
                'description': 'Text-to-Speech (XTTS, F5-TTS, Bark)'
            },
            'music_generation': {
                'price': 0.10,
                'unit': 'per 10 seconds',
                'description': 'Music Generation (MusicGen)'
            },
            'object_detection': {
                'price': 0.005,
                'unit': 'per image',
                'description': 'Object Detection (YOLOv10)'
            },
            'ocr': {
                'price': 0.01,
                'unit': 'per page',
                'description': 'OCR Text Extraction (EasyOCR, Surya)'
            },
            'background_removal': {
                'price': 0.01,
                'unit': 'per image',
                'description': 'Background Removal (rembg)'
            },
            'upscaling': {
                'price': 0.03,
                'unit': 'per image',
                'description': 'Image Upscaling 4x (Real-ESRGAN)'
            },
            'face_restoration': {
                'price': 0.02,
                'unit': 'per image',
                'description': 'Face Enhancement (GFPGAN)'
            },
            'embedding': {
                'price': 0.0001,
                'unit': 'per 1K tokens',
                'description': 'Text/Image Embeddings'
            },
            'training_hour': {
                'price': 2.00,
                'unit': 'per GPU hour',
                'description': 'Custom Model Training'
            },
        },
        'hardware': {
            'gpu': 'NVIDIA RTX 5090 (24GB VRAM)',
            'cpu': 'AMD Ryzen 9 9955HX3D (32 Threads)',
            'ram': '94GB DDR5',
        },
        'payment_methods': ['stripe', 'crypto'],
    })


@marketplace_bp.route('/status', methods=['GET'])
def marketplace_status():
    """Gibt Marketplace-Status zurück"""
    marketplace = get_marketplace()
    return jsonify(marketplace.get_status())


@marketplace_bp.route('/earnings', methods=['GET'])
def marketplace_earnings():
    """Gibt Earnings-Report zurück (Admin only)"""
    auth_header = request.headers.get('Authorization', '')

    if f"Bearer {Config.ADMIN_TOKEN}" != auth_header:
        return jsonify({'error': 'Unauthorized'}), 401

    marketplace = get_marketplace()
    return jsonify(marketplace.get_earnings_report())


@marketplace_bp.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Gibt Job-Status zurück"""
    marketplace = get_marketplace()

    # Suche in pending jobs
    for job in marketplace.direct_api._pending_jobs:
        if job.job_id == job_id:
            return jsonify({
                'job_id': job.job_id,
                'status': job.status,
                'job_type': job.job_type,
                'price_usd': job.price_usd,
                'created_at': job.created_at.isoformat(),
            })

    # Suche in completed jobs
    for job in marketplace.direct_api._completed_jobs:
        if job.job_id == job_id:
            return jsonify({
                'job_id': job.job_id,
                'status': job.status,
                'job_type': job.job_type,
                'price_usd': job.price_usd,
                'created_at': job.created_at.isoformat(),
            })

    return jsonify({'error': 'Job not found'}), 404


@marketplace_bp.route('/create-payment', methods=['POST'])
def create_payment():
    """
    Erstellt Stripe PaymentIntent für Job

    Body:
    {
        "job_id": "abc123",
        "customer_email": "kunde@example.com"
    }
    """
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        customer_email = data.get('customer_email')

        if not job_id:
            return jsonify({'error': 'job_id required'}), 400

        marketplace = get_marketplace()

        # Finde Job
        job = None
        for j in marketplace.direct_api._pending_jobs:
            if j.job_id == job_id:
                job = j
                break

        if not job:
            return jsonify({'error': 'Job not found'}), 404

        # Erstelle PaymentIntent
        amount_cents = int(job.price_usd * 100)

        # Minimum $0.50 für Stripe
        if amount_cents < 50:
            amount_cents = 50

        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency='usd',
            metadata={
                'job_id': job_id,
                'job_type': job.job_type,
            },
            receipt_email=customer_email,
            automatic_payment_methods={
                'enabled': True,
                'allow_redirects': 'always'
            },
        )

        return jsonify({
            'client_secret': intent.client_secret,
            'amount_usd': amount_cents / 100,
            'job_id': job_id,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@marketplace_bp.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """
    Stripe Webhook für Zahlungsbestätigungen
    Startet Job nach erfolgreicher Zahlung
    """
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, Config.STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    if event['type'] == 'payment_intent.succeeded':
        intent = event['data']['object']
        job_id = intent['metadata'].get('job_id')

        if job_id:
            # Job zur Verarbeitung freigeben
            marketplace = get_marketplace()

            for job in marketplace.direct_api._pending_jobs:
                if job.job_id == job_id:
                    job.status = 'paid'
                    print(f"[MONEY] Zahlung erhalten für Job {job_id}: ${intent['amount'] / 100:.2f}")

                    # Job in Queue einreihen
                    queue = get_job_queue()
                    queue.enqueue(
                        job_type=_map_job_type(job.job_type),
                        data=job.requirements,
                        priority=5,
                    )
                    break

    return jsonify({'received': True})


def _map_job_type(marketplace_type: str) -> JobType:
    """Mappt Marketplace Job-Typ auf interne JobType"""
    mapping = {
        'llm_inference': JobType.LLM_INFERENCE,
        'image_generation': JobType.IMAGE_GENERATION,
        'video_generation': JobType.VIDEO_GENERATION,
        'audio_transcription': JobType.AUDIO_STT,
        'audio_tts': JobType.AUDIO_TTS,
        'music_generation': JobType.AUDIO_MUSIC,
        'embedding': JobType.TEXT_EMBEDDING,
        'training_hour': JobType.TRAINING,
    }
    return mapping.get(marketplace_type, JobType.CUSTOM)


# API Docs
@marketplace_bp.route('/docs', methods=['GET'])
def api_docs():
    """API Dokumentation"""
    return jsonify({
        'name': 'SCIO AI Marketplace API',
        'version': '1.0.0',
        'base_url': f'{Config.SERVICE_URL}/api/marketplace',
        'endpoints': {
            'GET /pricing': 'Aktuelle Preisliste abrufen',
            'POST /submit': 'Job einreichen',
            'GET /job/<job_id>': 'Job-Status abrufen',
            'POST /create-payment': 'Stripe PaymentIntent erstellen',
            'POST /webhook/stripe': 'Stripe Webhook (intern)',
            'GET /status': 'Marketplace-Status',
        },
        'authentication': 'API Key im Header: Authorization: Bearer YOUR_API_KEY',
        'example': {
            'submit_job': {
                'url': 'POST /api/marketplace/submit',
                'body': {
                    'job_type': 'image_generation',
                    'params': {
                        'prompt': 'A futuristic city at night',
                        'width': 1024,
                        'height': 1024,
                    },
                    'customer_email': 'kunde@example.com'
                }
            }
        }
    })
