#!/usr/bin/env python3
"""
SCIO - Tools API Routes
Alle AI-Tool Endpunkte
"""

from flask import Blueprint, request, jsonify, send_file
from functools import wraps
import os

from backend.config import Config
from backend.services.job_queue import get_job_queue
from backend.models.job import JobType

tools_bp = Blueprint('tools', __name__, url_prefix='/api/v1/tools')


def require_api_key(f):
    """API Key Decorator - Validates API key against database"""
    @wraps(f)
    def decorated(*args, **kwargs):
        from datetime import datetime
        from backend.models import SessionLocal, APIKey

        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not api_key:
            api_key = request.headers.get('X-API-Key', '')

        if not api_key:
            return jsonify({'error': 'Missing API key'}), 401

        # Validate API key against database
        db = SessionLocal()
        try:
            key_hash = APIKey.hash_key(api_key)
            db_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()

            if not db_key:
                return jsonify({'error': 'Invalid API key'}), 401

            if not db_key.is_active:
                return jsonify({'error': 'API key is deactivated'}), 401

            if db_key.expires_at and datetime.utcnow() > db_key.expires_at:
                return jsonify({'error': 'API key has expired'}), 401

            # Check monthly token limit
            if db_key.monthly_tokens_used >= db_key.monthly_token_limit:
                return jsonify({'error': 'Monthly token limit exceeded'}), 429

            # Update last used timestamp
            db_key.last_used_at = datetime.utcnow()
            db_key.total_requests += 1
            db.commit()

            # Store API key info in request context for usage tracking
            request.api_key_id = db_key.id
            request.api_key_user = db_key.user_email
            request.api_key_is_admin = db_key.is_admin

        except Exception as e:
            db.rollback()
            return jsonify({'error': f'Authentication error: {str(e)}'}), 500
        finally:
            db.close()

        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════
# AUDIO ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/audio/transcribe', methods=['POST'])
@require_api_key
def transcribe_audio():
    """Speech-to-Text (Whisper)"""
    try:
        from backend.workers.audio_worker import get_audio_worker

        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['file']
        language = request.form.get('language')
        model = request.form.get('model', 'whisper-large')

        # Save temp file
        temp_path = f"/tmp/audio_{os.urandom(8).hex()}.wav"
        audio_file.save(temp_path)

        worker = get_audio_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.transcribe(temp_path, language=language)

        os.remove(temp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/audio/tts', methods=['POST'])
@require_api_key
def text_to_speech():
    """Text-to-Speech"""
    try:
        from backend.workers.audio_worker import get_audio_worker

        data = request.json
        text = data.get('text')
        voice = data.get('voice')
        language = data.get('language', 'en')
        model = data.get('model', 'xtts-v2')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        worker = get_audio_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.text_to_speech(text, voice=voice, language=language, model=model)

        if data.get('return_audio', False):
            return send_file(result['output_path'], mimetype='audio/wav')

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/audio/music', methods=['POST'])
@require_api_key
def generate_music():
    """Music Generation"""
    try:
        from backend.workers.audio_worker import get_audio_worker

        data = request.json
        prompt = data.get('prompt')
        duration = data.get('duration', 30)
        model = data.get('model', 'medium')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        worker = get_audio_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.generate_music(prompt, duration=duration, model_size=model)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# VIDEO ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/video/generate', methods=['POST'])
@require_api_key
def generate_video():
    """Text-to-Video / Image-to-Video"""
    try:
        from backend.workers.video_worker import get_video_worker

        data = request.json
        prompt = data.get('prompt')
        image = data.get('image')
        model = data.get('model', 'cogvideox-2b')
        num_frames = data.get('num_frames', 49)
        fps = data.get('fps', 8)

        worker = get_video_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.generate_video(
            prompt=prompt,
            image=image,
            model=model,
            num_frames=num_frames,
            fps=fps,
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# VISION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/vision/caption', methods=['POST'])
@require_api_key
def caption_image():
    """Image Captioning"""
    try:
        from backend.workers.vision_worker import get_vision_worker

        if 'file' in request.files:
            image_file = request.files['file']
            temp_path = f"/tmp/img_{os.urandom(8).hex()}.png"
            image_file.save(temp_path)
            image = temp_path
        else:
            data = request.json
            image = data.get('image')

        model = request.form.get('model') or request.json.get('model', 'blip-large')

        worker = get_vision_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.caption_image(image, model=model)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/vision/vqa', methods=['POST'])
@require_api_key
def visual_qa():
    """Visual Question Answering"""
    try:
        from backend.workers.vision_worker import get_vision_worker

        data = request.json
        image = data.get('image')
        question = data.get('question')
        model = data.get('model', 'llava-1.5-7b')

        if not image or not question:
            return jsonify({'error': 'Image and question required'}), 400

        worker = get_vision_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.visual_qa(image, question, model=model)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/vision/ocr', methods=['POST'])
@require_api_key
def ocr():
    """OCR - Extract text from image"""
    try:
        from backend.workers.vision_worker import get_vision_worker

        if 'file' in request.files:
            image_file = request.files['file']
            temp_path = f"/tmp/ocr_{os.urandom(8).hex()}.png"
            image_file.save(temp_path)
            image = temp_path
        else:
            data = request.json
            image = data.get('image')

        languages = request.form.getlist('languages') or request.json.get('languages', ['en', 'de'])

        worker = get_vision_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.ocr(image, languages=languages)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/vision/detect', methods=['POST'])
@require_api_key
def detect_objects():
    """Object Detection (YOLO)"""
    try:
        from backend.workers.vision_worker import get_vision_worker

        if 'file' in request.files:
            image_file = request.files['file']
            temp_path = f"/tmp/detect_{os.urandom(8).hex()}.png"
            image_file.save(temp_path)
            image = temp_path
        else:
            data = request.json
            image = data.get('image')

        model = request.form.get('model') or request.json.get('model', 'yolov8m')
        confidence = float(request.form.get('confidence', 0.25))

        worker = get_vision_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.detect_objects(image, model=model, confidence=confidence)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# CODE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/code/generate', methods=['POST'])
@require_api_key
def generate_code():
    """Code Generation"""
    try:
        from backend.workers.code_worker import get_code_worker

        data = request.json
        prompt = data.get('prompt')
        language = data.get('language')
        model = data.get('model', 'deepseek-coder-6.7b')
        max_tokens = data.get('max_tokens', 2048)

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        worker = get_code_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.generate_code(prompt, model=model, language=language, max_tokens=max_tokens)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/code/complete', methods=['POST'])
@require_api_key
def complete_code():
    """Code Completion"""
    try:
        from backend.workers.code_worker import get_code_worker

        data = request.json
        code = data.get('code')
        cursor_position = data.get('cursor_position')
        model = data.get('model', 'deepseek-coder-6.7b')

        if not code:
            return jsonify({'error': 'No code provided'}), 400

        worker = get_code_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.complete_code(code, cursor_position, model=model)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/code/explain', methods=['POST'])
@require_api_key
def explain_code():
    """Code Explanation"""
    try:
        from backend.workers.code_worker import get_code_worker

        data = request.json
        code = data.get('code')
        language = data.get('language')
        model = data.get('model', 'deepseek-coder-6.7b')

        if not code:
            return jsonify({'error': 'No code provided'}), 400

        worker = get_code_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.explain_code(code, model=model, language=language)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/code/review', methods=['POST'])
@require_api_key
def review_code():
    """Code Review"""
    try:
        from backend.workers.code_worker import get_code_worker

        data = request.json
        code = data.get('code')
        language = data.get('language')
        model = data.get('model', 'deepseek-coder-6.7b')

        if not code:
            return jsonify({'error': 'No code provided'}), 400

        worker = get_code_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.review_code(code, model=model, language=language)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/code/fix', methods=['POST'])
@require_api_key
def fix_code():
    """Code Bug Fix"""
    try:
        from backend.workers.code_worker import get_code_worker

        data = request.json
        code = data.get('code')
        error = data.get('error')
        language = data.get('language')
        model = data.get('model', 'deepseek-coder-6.7b')

        if not code:
            return jsonify({'error': 'No code provided'}), 400

        worker = get_code_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.fix_code(code, error, model=model, language=language)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# EMBEDDING ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/embeddings/text', methods=['POST'])
@require_api_key
def embed_text():
    """Text Embeddings"""
    try:
        from backend.workers.embedding_worker import get_embedding_worker

        data = request.json
        texts = data.get('texts') or data.get('text')
        model = data.get('model', 'bge-large-en-v1.5')

        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        worker = get_embedding_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.embed_text(texts, model=model)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/embeddings/image', methods=['POST'])
@require_api_key
def embed_image():
    """Image Embeddings (CLIP)"""
    try:
        from backend.workers.embedding_worker import get_embedding_worker

        if 'file' in request.files:
            image_file = request.files['file']
            temp_path = f"/tmp/embed_{os.urandom(8).hex()}.png"
            image_file.save(temp_path)
            images = [temp_path]
        else:
            data = request.json
            images = data.get('images') or [data.get('image')]

        model = request.form.get('model') or request.json.get('model', 'clip-vit-large')

        worker = get_embedding_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.embed_image(images, model=model)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# UPSCALE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/upscale', methods=['POST'])
@require_api_key
def upscale_image():
    """Image Upscaling"""
    try:
        from backend.workers.upscale_worker import get_upscale_worker

        if 'file' in request.files:
            image_file = request.files['file']
            temp_path = f"/tmp/upscale_{os.urandom(8).hex()}.png"
            image_file.save(temp_path)
            image = temp_path
        else:
            data = request.json
            image = data.get('image')

        model = request.form.get('model') or request.json.get('model', 'realesrgan-x4')
        scale = int(request.form.get('scale', 4))

        worker = get_upscale_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.upscale_image(image, model=model, scale=scale)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/face-restore', methods=['POST'])
@require_api_key
def restore_face():
    """Face Restoration"""
    try:
        from backend.workers.upscale_worker import get_upscale_worker

        if 'file' in request.files:
            image_file = request.files['file']
            temp_path = f"/tmp/face_{os.urandom(8).hex()}.png"
            image_file.save(temp_path)
            image = temp_path
        else:
            data = request.json
            image = data.get('image')

        model = request.form.get('model') or request.json.get('model', 'gfpgan')

        worker = get_upscale_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.restore_face(image, model=model)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# 3D ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/3d/text', methods=['POST'])
@require_api_key
def text_to_3d():
    """Text-to-3D"""
    try:
        from backend.workers.threed_worker import get_threed_worker

        data = request.json
        prompt = data.get('prompt')
        model = data.get('model', 'shap-e')
        output_format = data.get('format', 'glb')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        worker = get_threed_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.text_to_3d(prompt, model=model, output_format=output_format)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/3d/image', methods=['POST'])
@require_api_key
def image_to_3d():
    """Image-to-3D"""
    try:
        from backend.workers.threed_worker import get_threed_worker

        if 'file' in request.files:
            image_file = request.files['file']
            temp_path = f"/tmp/3d_{os.urandom(8).hex()}.png"
            image_file.save(temp_path)
            image = temp_path
        else:
            data = request.json
            image = data.get('image')

        model = request.form.get('model') or request.json.get('model', 'triposr')
        output_format = request.form.get('format') or request.json.get('format', 'glb')

        worker = get_threed_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.image_to_3d(image, model=model, output_format=output_format)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# DOCUMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/documents/extract', methods=['POST'])
@require_api_key
def extract_document():
    """Extract text from document"""
    try:
        from backend.workers.document_worker import get_document_worker

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        doc_file = request.files['file']
        temp_path = f"/tmp/doc_{os.urandom(8).hex()}{os.path.splitext(doc_file.filename)[1]}"
        doc_file.save(temp_path)

        worker = get_document_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.extract_text_from_pdf(temp_path)
        os.remove(temp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/documents/parse', methods=['POST'])
@require_api_key
def parse_document():
    """Parse document with advanced extraction"""
    try:
        from backend.workers.document_worker import get_document_worker

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        doc_file = request.files['file']
        temp_path = f"/tmp/doc_{os.urandom(8).hex()}{os.path.splitext(doc_file.filename)[1]}"
        doc_file.save(temp_path)

        extract_images = request.form.get('extract_images', 'true').lower() == 'true'
        extract_tables = request.form.get('extract_tables', 'true').lower() == 'true'

        worker = get_document_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.parse_document(temp_path, extract_images, extract_tables)
        os.remove(temp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@tools_bp.route('/documents/chunk', methods=['POST'])
@require_api_key
def chunk_text():
    """Chunk text for RAG"""
    try:
        from backend.workers.document_worker import get_document_worker

        data = request.json
        text = data.get('text')
        chunk_size = data.get('chunk_size', 1000)
        chunk_overlap = data.get('chunk_overlap', 200)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        worker = get_document_worker()
        if not worker.status.value == 'ready':
            worker.initialize()

        result = worker.chunk_text(text, chunk_size, chunk_overlap)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# AVAILABLE TOOLS ENDPOINT
# ═══════════════════════════════════════════════════════════════

@tools_bp.route('/available', methods=['GET'])
def available_tools():
    """List all available tools"""
    tools = {
        'audio': {
            'transcribe': '/api/v1/tools/audio/transcribe',
            'tts': '/api/v1/tools/audio/tts',
            'music': '/api/v1/tools/audio/music',
        },
        'video': {
            'generate': '/api/v1/tools/video/generate',
        },
        'vision': {
            'caption': '/api/v1/tools/vision/caption',
            'vqa': '/api/v1/tools/vision/vqa',
            'ocr': '/api/v1/tools/vision/ocr',
            'detect': '/api/v1/tools/vision/detect',
        },
        'code': {
            'generate': '/api/v1/tools/code/generate',
            'complete': '/api/v1/tools/code/complete',
            'explain': '/api/v1/tools/code/explain',
            'review': '/api/v1/tools/code/review',
            'fix': '/api/v1/tools/code/fix',
        },
        'embeddings': {
            'text': '/api/v1/tools/embeddings/text',
            'image': '/api/v1/tools/embeddings/image',
        },
        'upscale': {
            'image': '/api/v1/tools/upscale',
            'face': '/api/v1/tools/face-restore',
        },
        '3d': {
            'text': '/api/v1/tools/3d/text',
            'image': '/api/v1/tools/3d/image',
        },
        'documents': {
            'extract': '/api/v1/tools/documents/extract',
            'parse': '/api/v1/tools/documents/parse',
            'chunk': '/api/v1/tools/documents/chunk',
        },
    }
    return jsonify(tools)
