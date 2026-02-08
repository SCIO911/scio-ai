#!/usr/bin/env python3
"""
SCIO - Vision Worker
Image Captioning, OCR, Object Detection, Visual QA
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from io import BytesIO

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    TORCH_AVAILABLE = False

# PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Vision Models
LLAVA_AVAILABLE = False
try:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    LLAVA_AVAILABLE = True
except ImportError:
    pass

BLIP_AVAILABLE = False
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    pass

FLORENCE_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM
    # AutoProcessor already imported with LLaVA
    FLORENCE_AVAILABLE = True
except ImportError:
    pass

# OCR
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

# Object Detection
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


# Available Vision Models
VISION_MODELS = {
    # Vision-Language Models
    'llava-1.5-7b': {
        'name': 'LLaVA 1.5 7B',
        'hf_id': 'llava-hf/llava-1.5-7b-hf',
        'type': 'vlm',
        'vram_gb': 14,
    },
    'llava-1.5-13b': {
        'name': 'LLaVA 1.5 13B',
        'hf_id': 'llava-hf/llava-1.5-13b-hf',
        'type': 'vlm',
        'vram_gb': 26,
    },
    'llava-1.6-vicuna-7b': {
        'name': 'LLaVA 1.6 Vicuna 7B',
        'hf_id': 'llava-hf/llava-v1.6-vicuna-7b-hf',
        'type': 'vlm',
        'vram_gb': 14,
    },

    # Image Captioning
    'blip-base': {
        'name': 'BLIP Base',
        'hf_id': 'Salesforce/blip-image-captioning-base',
        'type': 'caption',
        'vram_gb': 2,
    },
    'blip-large': {
        'name': 'BLIP Large',
        'hf_id': 'Salesforce/blip-image-captioning-large',
        'type': 'caption',
        'vram_gb': 4,
    },
    'blip2-opt-2.7b': {
        'name': 'BLIP-2 OPT 2.7B',
        'hf_id': 'Salesforce/blip2-opt-2.7b',
        'type': 'vlm',
        'vram_gb': 8,
    },

    # Florence
    'florence-2-base': {
        'name': 'Florence-2 Base',
        'hf_id': 'microsoft/Florence-2-base',
        'type': 'florence',
        'vram_gb': 4,
    },
    'florence-2-large': {
        'name': 'Florence-2 Large',
        'hf_id': 'microsoft/Florence-2-large',
        'type': 'florence',
        'vram_gb': 8,
    },

    # Object Detection
    'yolov8n': {'name': 'YOLOv8 Nano', 'type': 'detection', 'vram_gb': 1},
    'yolov8s': {'name': 'YOLOv8 Small', 'type': 'detection', 'vram_gb': 2},
    'yolov8m': {'name': 'YOLOv8 Medium', 'type': 'detection', 'vram_gb': 4},
    'yolov8l': {'name': 'YOLOv8 Large', 'type': 'detection', 'vram_gb': 8},
    'yolov8x': {'name': 'YOLOv8 XLarge', 'type': 'detection', 'vram_gb': 12},
}


class VisionWorker(BaseWorker):
    """
    Vision Worker - Handles all vision AI tasks

    Features:
    - Image Captioning (BLIP, BLIP-2)
    - Visual Question Answering (LLaVA)
    - OCR (EasyOCR, PaddleOCR)
    - Object Detection (YOLOv8)
    - Image Analysis (Florence-2)
    """

    def __init__(self):
        super().__init__("Vision Processing")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._ocr = None
        self._yolo = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        available_features = []

        if LLAVA_AVAILABLE:
            available_features.append("LLaVA")
        if BLIP_AVAILABLE:
            available_features.append("BLIP")
        if FLORENCE_AVAILABLE:
            available_features.append("Florence")
        if EASYOCR_AVAILABLE or PADDLEOCR_AVAILABLE:
            available_features.append("OCR")
        if YOLO_AVAILABLE:
            available_features.append("YOLO")

        if not available_features:
            self._error_message = "No vision libraries available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] Vision Worker bereit (Device: {self._device}, Features: {', '.join(available_features)})")
        return True

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Load image from various sources.

        Args:
            image_input: PIL Image, bytes, base64 string, or file path

        Returns:
            PIL Image object

        Raises:
            ValueError: If image cannot be loaded or is invalid
        """
        if not image_input:
            raise ValueError("Image input is required")

        try:
            if isinstance(image_input, Image.Image):
                return image_input
            elif isinstance(image_input, bytes):
                return Image.open(BytesIO(image_input))
            elif isinstance(image_input, str):
                if image_input.startswith("data:image"):
                    # Base64 encoded
                    try:
                        base64_data = image_input.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        return Image.open(BytesIO(image_bytes))
                    except Exception as e:
                        raise ValueError(f"Invalid base64 image data: {e}")
                else:
                    # File path
                    if not Path(image_input).exists():
                        raise ValueError(f"Image file not found: {image_input}")
                    return Image.open(image_input)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    def _load_vlm(self, model_id: str):
        """Load Vision-Language Model"""
        model_info = VISION_MODELS.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")

        hf_id = model_info['hf_id']

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            if "llava" in model_id.lower():
                processor = AutoProcessor.from_pretrained(hf_id)
                model = LlavaForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            elif "blip2" in model_id.lower():
                processor = Blip2Processor.from_pretrained(hf_id)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    device_map="auto",
                )
            elif "blip" in model_id.lower():
                processor = BlipProcessor.from_pretrained(hf_id)
                model = BlipForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                ).to(self._device)
            elif "florence" in model_id.lower():
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(self._device)
            else:
                raise ValueError(f"Unknown VLM type: {model_id}")

            return {"model": model, "processor": processor}

        result = model_manager.get_model(hf_id, loader)
        self._model = result["model"]
        self._processor = result["processor"]
        self._current_model_id = model_id
        print(f"[OK] {model_info['name']} geladen")

    def caption_image(
        self,
        image: Union[str, bytes, Image.Image],
        model: str = "blip-large",
    ) -> dict:
        """Generate caption for an image"""
        start_time = time.time()

        if self._current_model_id != model:
            self._load_vlm(model)

        img = self._load_image(image).convert("RGB")

        inputs = self._processor(images=img, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=100)

        caption = self._processor.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "caption": caption,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def visual_qa(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        model: str = "llava-1.5-7b",
        max_tokens: int = 512,
    ) -> dict:
        """
        Answer questions about an image.

        Args:
            image: Image input (file path, bytes, or PIL Image)
            question: Question to ask about the image (required, max 2000 chars)
            model: VLM model ID
            max_tokens: Maximum response tokens (1-2048)

        Returns:
            dict with answer, question, model, gpu_seconds

        Raises:
            ValueError: If question is empty or parameters invalid
        """
        start_time = time.time()

        # Validate question
        if not question or not question.strip():
            raise ValueError("Question is required for visual QA")

        question = question.strip()
        if len(question) > 2000:
            raise ValueError("Question too long (max 2000 characters)")

        # Validate max_tokens
        max_tokens = max(1, min(2048, max_tokens))

        # Validate model
        if model not in VISION_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(VISION_MODELS.keys())}")

        if self._current_model_id != model:
            self._load_vlm(model)

        img = self._load_image(image).convert("RGB")

        if "llava" in model.lower():
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self._processor(text=prompt, images=img, return_tensors="pt").to(self._device)
        else:
            inputs = self._processor(images=img, text=question, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_tokens)

        answer = self._processor.decode(generated_ids[0], skip_special_tokens=True)

        # Clean up answer
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()

        return {
            "answer": answer,
            "question": question,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def ocr(
        self,
        image: Union[str, bytes, Image.Image],
        languages: List[str] = None,
    ) -> dict:
        """Extract text from image using OCR"""
        start_time = time.time()

        if languages is None:
            languages = ["en", "de"]

        # Load OCR engine if needed
        if self._ocr is None:
            if EASYOCR_AVAILABLE:
                self._ocr = easyocr.Reader(languages, gpu=self._device == "cuda")
            elif PADDLEOCR_AVAILABLE:
                self._ocr = PaddleOCR(use_angle_cls=True, lang='en')
            else:
                raise ValueError("No OCR engine available")

        img = self._load_image(image)

        # Run OCR
        if EASYOCR_AVAILABLE and isinstance(self._ocr, easyocr.Reader):
            results = self._ocr.readtext(img)
            text_blocks = [{"text": r[1], "confidence": r[2], "bbox": r[0]} for r in results]
            full_text = " ".join([r[1] for r in results])
        else:
            result = self._ocr.ocr(img, cls=True)
            text_blocks = []
            for line in result[0]:
                text_blocks.append({
                    "text": line[1][0],
                    "confidence": line[1][1],
                    "bbox": line[0],
                })
            full_text = " ".join([line[1][0] for line in result[0]])

        return {
            "text": full_text,
            "blocks": text_blocks,
            "engine": "easyocr" if EASYOCR_AVAILABLE else "paddleocr",
            "gpu_seconds": time.time() - start_time,
        }

    def detect_objects(
        self,
        image: Union[str, bytes, Image.Image],
        model: str = "yolov8m",
        confidence: float = 0.25,
    ) -> dict:
        """Detect objects in image"""
        start_time = time.time()

        if not YOLO_AVAILABLE:
            raise ValueError("YOLOv8 not installed")

        # Load YOLO model
        if self._yolo is None or self._current_model_id != model:
            model_name = model.replace("yolo", "yolo") + ".pt"
            self._yolo = YOLO(model_name)
            self._current_model_id = model

        img = self._load_image(image)

        results = self._yolo(img, conf=confidence)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                })

        return {
            "detections": detections,
            "count": len(detections),
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process a vision job"""
        task_type = input_data.get("task", "caption")
        image = input_data.get("image") or input_data.get("file")

        self.notify_progress(job_id, 0.1, f"Starting {task_type}")

        if task_type == "caption":
            model = input_data.get("model", "blip-large")
            result = self.caption_image(image, model=model)

        elif task_type == "vqa" or task_type == "visual_qa":
            question = input_data.get("question", "Describe this image in detail.")
            model = input_data.get("model", "llava-1.5-7b")
            result = self.visual_qa(image, question, model=model)

        elif task_type == "ocr":
            languages = input_data.get("languages", ["en", "de"])
            result = self.ocr(image, languages=languages)

        elif task_type == "detect" or task_type == "detection":
            model = input_data.get("model", "yolov8m")
            confidence = input_data.get("confidence", 0.25)
            result = self.detect_objects(image, model=model, confidence=confidence)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._model = None
        self._processor = None
        self._ocr = None
        self._yolo = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Vision Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        return VISION_MODELS


# Singleton Instance
_vision_worker: Optional[VisionWorker] = None


def get_vision_worker() -> VisionWorker:
    """Get singleton instance"""
    global _vision_worker
    if _vision_worker is None:
        _vision_worker = VisionWorker()
    return _vision_worker
