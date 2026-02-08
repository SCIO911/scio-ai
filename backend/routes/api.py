#!/usr/bin/env python3
"""
SCIO - REST API Routes
OpenAI-kompatible Inference API + Custom Endpoints
"""

import time
import uuid
import json
from datetime import datetime
from functools import wraps

from flask import request, jsonify, Response, stream_with_context

from . import api_bp
from backend.config import Config
from backend.services.api_keys import get_api_key_service
from backend.services.job_queue import get_job_queue
from backend.services.hardware_monitor import get_hardware_monitor
from backend.models.job import JobType
from backend.workers.llm_inference import get_inference_worker
from backend.workers.image_gen import get_image_worker


def require_api_key(f):
    """Decorator für API-Key Validierung"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing API key'}), 401

        api_key = auth_header[7:]  # Remove 'Bearer '
        api_service = get_api_key_service()

        valid, key_info, error = api_service.validate_key(api_key)
        if not valid:
            return jsonify({'error': error}), 401

        # Check rate limits
        allowed, limits = api_service.check_rate_limits(api_key)
        if not allowed:
            return jsonify({'error': limits.get('error', 'Rate limit exceeded')}), 429

        # Attach key info to request
        request.api_key = api_key
        request.key_info = key_info

        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════
# OPENAI-KOMPATIBLE CHAT COMPLETIONS API
# ═══════════════════════════════════════════════════════════════

@api_bp.route('/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    """
    OpenAI-kompatible Chat Completions API

    Request:
        {
            "model": "mistral-7b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 1000,
            "temperature": 0.7,
            "stream": false
        }

    Response:
        {
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mistral-7b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "..."},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60
            }
        }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        model = data.get('model', 'mistral-7b')
        messages = data.get('messages', [])
        max_tokens = int(data.get('max_tokens', Config.MAX_NEW_TOKENS))
        temperature = float(data.get('temperature', 0.7))
        top_p = float(data.get('top_p', 0.9))
        stream = bool(data.get('stream', False))

        if not messages:
            return jsonify({'error': 'messages required'}), 400

        # Check if model is allowed for this key
        key_info = request.key_info
        allowed_models = key_info.get('allowed_models')
        if allowed_models and model not in allowed_models:
            return jsonify({'error': f'Model {model} not allowed for this API key'}), 403

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            # Streaming response
            def generate():
                worker = get_inference_worker()
                if not worker.initialize():
                    yield f"data: {json.dumps({'error': 'Worker not available'})}\n\n"
                    return

                try:
                    for chunk in worker.generate_stream(
                        messages=messages,
                        model_id=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    ):
                        response_chunk = {
                            'id': completion_id,
                            'object': 'chat.completion.chunk',
                            'created': created,
                            'model': model,
                            'choices': [{
                                'index': 0,
                                'delta': {'content': chunk},
                                'finish_reason': None,
                            }],
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"

                    # Final chunk
                    final_chunk = {
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model,
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop',
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                },
            )

        else:
            # Non-streaming response
            worker = get_inference_worker()
            if not worker.initialize():
                return jsonify({'error': 'Inference worker not available'}), 503

            result = worker.process(completion_id, {
                'model': model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
            })

            # Record usage
            api_service = get_api_key_service()
            api_service.record_usage(
                request.api_key,
                result.get('tokens_input', 0),
                result.get('tokens_output', 0),
            )

            response = {
                'id': completion_id,
                'object': 'chat.completion',
                'created': created,
                'model': model,
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': result.get('response', ''),
                    },
                    'finish_reason': result.get('finish_reason', 'stop'),
                }],
                'usage': {
                    'prompt_tokens': result.get('tokens_input', 0),
                    'completion_tokens': result.get('tokens_output', 0),
                    'total_tokens': result.get('tokens_input', 0) + result.get('tokens_output', 0),
                },
            }

            return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Chat Completion Error: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/models', methods=['GET'])
@require_api_key
def list_models():
    """Liste verfügbarer Modelle"""
    models = []
    for model_id, info in Config.AVAILABLE_MODELS.items():
        models.append({
            'id': model_id,
            'object': 'model',
            'created': 1700000000,
            'owned_by': 'scio',
            'name': info['name'],
            'context_length': info['context_length'],
        })

    return jsonify({
        'object': 'list',
        'data': models,
    })


# ═══════════════════════════════════════════════════════════════
# IMAGE GENERATION API
# ═══════════════════════════════════════════════════════════════

@api_bp.route('/images/generations', methods=['POST'])
@require_api_key
def generate_images():
    """
    Bild-Generierung API

    Request:
        {
            "model": "sdxl",
            "prompt": "A beautiful sunset",
            "n": 1,
            "size": "1024x1024"
        }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        model = data.get('model', 'sdxl')
        prompt = data.get('prompt', '')
        n = min(int(data.get('n', 1)), 4)
        size = str(data.get('size', '1024x1024'))
        negative_prompt = data.get('negative_prompt', '')

        if not prompt:
            return jsonify({'error': 'prompt required'}), 400

        # Parse size
        try:
            width, height = map(int, size.split('x'))
        except (ValueError, AttributeError):
            width, height = 1024, 1024

        worker = get_image_worker()
        if not worker.initialize():
            return jsonify({'error': 'Image worker not available'}), 503

        result = worker.process(f"img_{uuid.uuid4().hex[:16]}", {
            'model': model,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'num_images': n,
            'width': width,
            'height': height,
        })

        # Record usage (simplified)
        api_service = get_api_key_service()
        cost = Config.IMAGE_PRICES.get(model, 10) * n
        api_service.record_usage(request.api_key, 0, 0, cost)

        images_data = []
        for path in result.get('image_paths', []):
            images_data.append({
                'url': f"/api/v1/images/{path.split('/')[-1]}",
                'revised_prompt': prompt,
            })

        return jsonify({
            'created': int(time.time()),
            'data': images_data,
        })

    except Exception as e:
        print(f"[ERROR] Image Generation Error: {e}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# JOB MANAGEMENT API
# ═══════════════════════════════════════════════════════════════

@api_bp.route('/jobs', methods=['POST'])
@require_api_key
def create_job():
    """Erstellt neuen Job"""
    try:
        data = request.json
        job_type = data.get('job_type')
        input_data = data.get('input', {})
        priority = data.get('priority', 0)

        if not job_type:
            return jsonify({'error': 'job_type required'}), 400

        # Map job type
        type_map = {
            'llm_training': JobType.LLM_TRAINING,
            'llm_inference': JobType.LLM_INFERENCE,
            'image_generation': JobType.IMAGE_GENERATION,
            'batch_inference': JobType.BATCH_INFERENCE,
        }

        if job_type not in type_map:
            return jsonify({'error': f'Invalid job_type: {job_type}'}), 400

        # Check permissions
        key_info = request.key_info
        allowed_features = key_info.get('allowed_features', [])
        if allowed_features and job_type.replace('_', '') not in str(allowed_features):
            return jsonify({'error': 'Job type not allowed for this API key'}), 403

        queue = get_job_queue()
        job_id = queue.create_job(
            job_type=type_map[job_type],
            input_data=input_data,
            api_key_id=key_info.get('id'),
            user_email=key_info.get('user_email'),
            priority=priority,
        )

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'created_at': datetime.utcnow().isoformat(),
        })

    except Exception as e:
        print(f"[ERROR] Create Job Error: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/jobs/<job_id>', methods=['GET'])
@require_api_key
def get_job(job_id: str):
    """Gibt Job-Status zurück"""
    queue = get_job_queue()
    job = queue.get_job(job_id)

    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(job)


@api_bp.route('/jobs', methods=['GET'])
@require_api_key
def list_jobs():
    """Liste Jobs für aktuellen User"""
    key_info = request.key_info
    queue = get_job_queue()

    jobs = queue.get_jobs(
        user_email=key_info.get('user_email'),
        limit=request.args.get('limit', 50, type=int),
    )

    return jsonify({
        'jobs': jobs,
        'total': len(jobs),
    })


@api_bp.route('/jobs/<job_id>', methods=['DELETE'])
@require_api_key
def cancel_job(job_id: str):
    """Bricht Job ab"""
    queue = get_job_queue()
    success = queue.cancel_job(job_id)

    if success:
        return jsonify({'status': 'cancelled'})
    else:
        return jsonify({'error': 'Could not cancel job'}), 400


# ═══════════════════════════════════════════════════════════════
# USAGE & ACCOUNT API
# ═══════════════════════════════════════════════════════════════

@api_bp.route('/usage', methods=['GET'])
@require_api_key
def get_usage():
    """Gibt Usage-Statistiken zurück"""
    key_info = request.key_info

    return jsonify({
        'tokens_used_month': key_info.get('monthly_tokens_used', 0),
        'tokens_limit_month': key_info.get('monthly_token_limit', 0),
        'tokens_remaining': key_info.get('monthly_token_limit', 0) - key_info.get('monthly_tokens_used', 0),
        'total_requests': key_info.get('total_requests', 0),
        'total_spent_eur': key_info.get('total_spent_eur', 0),
        'credits_eur': key_info.get('credits_eur', 0),
    })


@api_bp.route('/status', methods=['GET'])
def system_status():
    """System-Status (öffentlich)"""
    monitor = get_hardware_monitor()
    status = monitor.get_status()

    return jsonify({
        'status': 'online',
        'gpu_available': len(status.gpus) > 0,
        'is_busy': status.is_busy,
        'capacity': status.get_capacity(),
        'timestamp': datetime.utcnow().isoformat(),
    })
