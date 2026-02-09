#!/usr/bin/env python3
"""
SCIO - Public Paid API Routes

Öffentliche API-Endpoints für bezahlte Services.
Alle Endpoints erfordern einen gültigen API-Key.
"""

from flask import Blueprint, jsonify, request, g
from backend.services.paid_api import (
    get_paid_api_service,
    require_api_key,
    PRICING,
)

public_api_bp = Blueprint('public_api', __name__, url_prefix='/api/v1/public')


# ============================================================
# API KEY MANAGEMENT
# ============================================================

@public_api_bp.route('/keys', methods=['POST'])
def create_api_key():
    """
    Erstellt neuen API Key

    Body:
        email: str
        name: str

    Returns:
        API Key Details
    """
    data = request.get_json() or {}
    email = data.get('email')
    name = data.get('name', 'API User')

    if not email:
        return jsonify({'error': 'Email required'}), 400

    service = get_paid_api_service()
    api_key = service.create_api_key(email, name, initial_balance=1.0)  # $1 Startguthaben

    return jsonify({
        'api_key': api_key.key,
        'email': api_key.email,
        'balance': api_key.balance,
        'message': 'API Key created. You have $1.00 free credit to start!',
    })


@public_api_bp.route('/keys/balance', methods=['GET'])
def get_balance():
    """Gibt Guthaben für API Key zurück"""
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return jsonify({'error': 'Missing API key'}), 401

    api_key = auth[7:]
    service = get_paid_api_service()
    stats = service.get_usage_stats(api_key)

    if not stats:
        return jsonify({'error': 'Invalid API key'}), 401

    return jsonify(stats)


@public_api_bp.route('/keys/topup', methods=['POST'])
def topup_balance():
    """
    Guthaben aufladen via Stripe

    Body:
        amount: float (USD)
    """
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return jsonify({'error': 'Missing API key'}), 401

    api_key = auth[7:]
    data = request.get_json() or {}
    amount = float(data.get('amount', 10))

    if amount < 5:
        return jsonify({'error': 'Minimum top-up is $5'}), 400

    # Stripe Checkout Session erstellen
    try:
        import stripe
        from backend.config import Config

        stripe.api_key = Config.STRIPE_SECRET_KEY

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'SCIO API Credits',
                        'description': f'${amount:.2f} API credits for SCIO services',
                    },
                    'unit_amount': int(amount * 100),
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f'{request.host_url}api/v1/public/keys/topup/success?session_id={{CHECKOUT_SESSION_ID}}&key={api_key}',
            cancel_url=f'{request.host_url}api/v1/public/keys/topup/cancel',
            metadata={
                'api_key': api_key,
                'amount': str(amount),
            },
        )

        return jsonify({
            'checkout_url': session.url,
            'session_id': session.id,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@public_api_bp.route('/keys/topup/success', methods=['GET'])
def topup_success():
    """Callback nach erfolgreicher Zahlung"""
    session_id = request.args.get('session_id')
    api_key = request.args.get('key')

    try:
        import stripe
        from backend.config import Config

        stripe.api_key = Config.STRIPE_SECRET_KEY
        session = stripe.checkout.Session.retrieve(session_id)

        if session.payment_status == 'paid':
            amount = float(session.metadata.get('amount', 0))
            service = get_paid_api_service()
            service.add_balance(api_key, amount)

            return jsonify({
                'success': True,
                'amount_added': amount,
                'message': f'${amount:.2f} added to your balance!',
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Payment not completed'}), 400


# ============================================================
# PRICING INFO
# ============================================================

@public_api_bp.route('/pricing', methods=['GET'])
def get_pricing():
    """Gibt Preisliste zurück"""
    return jsonify({
        'currency': 'USD',
        'services': PRICING,
    })


# ============================================================
# LLM SERVICES
# ============================================================

@public_api_bp.route('/chat/completions', methods=['POST'])
@require_api_key('chat_completion', lambda r: r.json.get('max_tokens', 500))
def chat_completions():
    """
    Chat Completion API (OpenAI-kompatibel)

    Body:
        model: str (optional)
        messages: list
        max_tokens: int (optional)
        temperature: float (optional)
    """
    data = request.get_json() or {}
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 500)
    temperature = data.get('temperature', 0.7)

    if not messages:
        return jsonify({'error': 'Messages required'}), 400

    try:
        from backend.workers.llm_inference import get_llm_worker

        worker = get_llm_worker()
        result = worker.process(
            job_id=f"api_{g.api_key.user_id}",
            input_data={
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
            }
        )

        return jsonify({
            'id': f"chatcmpl-{result.get('request_id', 'unknown')}",
            'object': 'chat.completion',
            'model': result.get('model', 'scio-llm'),
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': result.get('response', ''),
                },
                'finish_reason': 'stop',
            }],
            'usage': {
                'prompt_tokens': result.get('prompt_tokens', 0),
                'completion_tokens': result.get('completion_tokens', 0),
                'total_tokens': result.get('total_tokens', 0),
            },
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@public_api_bp.route('/embeddings', methods=['POST'])
@require_api_key('embedding', lambda r: len(r.json.get('input', '')))
def create_embeddings():
    """Text Embedding API"""
    data = request.get_json() or {}
    input_text = data.get('input', '')

    if not input_text:
        return jsonify({'error': 'Input required'}), 400

    try:
        from backend.workers.embedding_worker import get_embedding_worker

        worker = get_embedding_worker()
        result = worker.process(
            job_id=f"api_{g.api_key.user_id}",
            input_data={'texts': [input_text] if isinstance(input_text, str) else input_text}
        )

        return jsonify({
            'object': 'list',
            'data': [
                {'object': 'embedding', 'index': i, 'embedding': emb}
                for i, emb in enumerate(result.get('embeddings', []))
            ],
            'model': 'scio-embedding',
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# IMAGE SERVICES
# ============================================================

@public_api_bp.route('/images/generate', methods=['POST'])
@require_api_key('image_generation')
def generate_image():
    """
    Image Generation API

    Body:
        prompt: str
        negative_prompt: str (optional)
        width: int (optional)
        height: int (optional)
        steps: int (optional)
    """
    data = request.get_json() or {}
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400

    try:
        from backend.workers.image_gen import get_image_worker

        worker = get_image_worker()
        result = worker.process(
            job_id=f"api_{g.api_key.user_id}",
            input_data={
                'prompt': prompt,
                'negative_prompt': data.get('negative_prompt', ''),
                'width': data.get('width', 1024),
                'height': data.get('height', 1024),
                'steps': data.get('steps', 25),
            }
        )

        return jsonify({
            'created': int(time.time()) if 'time' in dir() else 0,
            'data': [{
                'url': result.get('output_path', ''),
                'b64_json': result.get('base64', ''),
            }],
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# AUDIO SERVICES
# ============================================================

@public_api_bp.route('/audio/transcriptions', methods=['POST'])
@require_api_key('speech_to_text')
def transcribe_audio():
    """Speech-to-Text API"""
    if 'file' not in request.files:
        return jsonify({'error': 'Audio file required'}), 400

    try:
        from backend.workers.audio_worker import get_audio_worker
        import tempfile

        audio_file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)

            worker = get_audio_worker()
            result = worker.process(
                job_id=f"api_{g.api_key.user_id}",
                input_data={
                    'task': 'transcribe',
                    'audio_path': tmp.name,
                    'language': request.form.get('language', 'auto'),
                }
            )

        return jsonify({
            'text': result.get('text', ''),
            'language': result.get('language', 'unknown'),
            'duration': result.get('duration', 0),
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@public_api_bp.route('/audio/speech', methods=['POST'])
@require_api_key('text_to_speech', lambda r: len(r.json.get('input', '')))
def create_speech():
    """Text-to-Speech API"""
    data = request.get_json() or {}
    text = data.get('input', '')

    if not text:
        return jsonify({'error': 'Input text required'}), 400

    try:
        from backend.workers.audio_worker import get_audio_worker

        worker = get_audio_worker()
        result = worker.process(
            job_id=f"api_{g.api_key.user_id}",
            input_data={
                'task': 'tts',
                'text': text,
                'voice': data.get('voice', 'default'),
                'language': data.get('language', 'en'),
            }
        )

        return jsonify({
            'audio_url': result.get('output_path', ''),
            'audio_base64': result.get('base64', ''),
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# CODE SERVICES
# ============================================================

@public_api_bp.route('/code/generate', methods=['POST'])
@require_api_key('code_generation')
def generate_code():
    """Code Generation API"""
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    language = data.get('language', 'python')

    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400

    try:
        from backend.workers.code_worker import get_code_worker

        worker = get_code_worker()
        result = worker.process(
            job_id=f"api_{g.api_key.user_id}",
            input_data={
                'task': 'generate',
                'prompt': prompt,
                'language': language,
            }
        )

        return jsonify({
            'code': result.get('code', ''),
            'language': language,
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@public_api_bp.route('/code/review', methods=['POST'])
@require_api_key('code_review')
def review_code():
    """Code Review API"""
    data = request.get_json() or {}
    code = data.get('code', '')

    if not code:
        return jsonify({'error': 'Code required'}), 400

    try:
        from backend.workers.code_worker import get_code_worker

        worker = get_code_worker()
        result = worker.analyze_security(code)

        return jsonify({
            'review': result,
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# VISION SERVICES
# ============================================================

@public_api_bp.route('/vision/analyze', methods=['POST'])
@require_api_key('image_analysis')
def analyze_image():
    """Image Analysis API"""
    data = request.get_json() or {}
    image_url = data.get('image_url', '')
    image_base64 = data.get('image_base64', '')

    if not image_url and not image_base64:
        return jsonify({'error': 'Image URL or base64 required'}), 400

    try:
        from backend.workers.vision_worker import get_vision_worker

        worker = get_vision_worker()
        result = worker.process(
            job_id=f"api_{g.api_key.user_id}",
            input_data={
                'task': 'analyze',
                'image_url': image_url,
                'image_base64': image_base64,
                'prompt': data.get('prompt', 'Describe this image in detail.'),
            }
        )

        return jsonify({
            'description': result.get('description', ''),
            'objects': result.get('objects', []),
            'cost': g.api_cost,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# HEALTH & STATUS
# ============================================================

@public_api_bp.route('/health', methods=['GET'])
def health():
    """Public API Health Check"""
    service = get_paid_api_service()
    return jsonify({
        'status': 'healthy',
        'total_api_keys': len(service._api_keys),
        'total_revenue': service.get_total_revenue(),
    })


# Import time für timestamps
import time
