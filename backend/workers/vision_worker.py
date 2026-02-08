#!/usr/bin/env python3
"""
SCIO - Vision Worker
Die BESTEN Vision-Modelle 2024/2025
Optimiert für RTX 5090 mit 24GB VRAM

Unterstützte Modelle:
- Qwen2-VL - Bestes Open Source VLM
- InternVL2 - Sehr fähiges VLM
- LLaVA 1.6 / LLaVA-NeXT
- Florence-2 - Microsoft's universelles Vision-Modell
- PaliGemma - Google's effizientes VLM
- Phi-3.5-Vision - Microsoft's kompaktes VLM
- Molmo - Allen AI

Features:
- Visual Question Answering
- Image Captioning
- OCR (Surya, EasyOCR, PaddleOCR, GOT-OCR)
- Object Detection (YOLOv10, YOLO-World, RT-DETR)
- Image Segmentation (SAM 2)
- Document Understanding
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
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Transformers für VLMs
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    from transformers import LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# Qwen-VL specifics
QWEN_VL_AVAILABLE = False
try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN_VL_AVAILABLE = True
except ImportError:
    pass

# InternVL
INTERNVL_AVAILABLE = False
try:
    from transformers import AutoModel
    INTERNVL_AVAILABLE = True
except ImportError:
    pass

# OCR Engines
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

SURYA_AVAILABLE = False
try:
    from surya.ocr import run_ocr
    from surya.model.detection import segformer
    from surya.model.recognition import model as recognition_model
    SURYA_AVAILABLE = True
except ImportError:
    pass

# Object Detection
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

# SAM 2 for Segmentation
SAM_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# VERFÜGBARE MODELLE - DIE BESTEN 2024/2025
# ═══════════════════════════════════════════════════════════════

VISION_MODELS = {
    # ═══════════════════════════════════════════════════════════
    # QWEN2-VL - Bestes Open Source VLM
    # ═══════════════════════════════════════════════════════════
    'qwen2-vl-7b': {
        'name': 'Qwen2-VL 7B',
        'hf_id': 'Qwen/Qwen2-VL-7B-Instruct',
        'type': 'vlm',
        'vram_gb': 16,
        'description': 'Bestes Open Source VLM, hervorragend für alles',
        'recommended': True,
    },
    'qwen2-vl-2b': {
        'name': 'Qwen2-VL 2B',
        'hf_id': 'Qwen/Qwen2-VL-2B-Instruct',
        'type': 'vlm',
        'vram_gb': 6,
        'description': 'Schnell und effizient',
    },

    # ═══════════════════════════════════════════════════════════
    # INTERNVL2 - Sehr fähig
    # ═══════════════════════════════════════════════════════════
    'internvl2-8b': {
        'name': 'InternVL2 8B',
        'hf_id': 'OpenGVLab/InternVL2-8B',
        'type': 'vlm',
        'vram_gb': 18,
        'description': 'Chinesisches Top-VLM',
        'recommended': True,
    },
    'internvl2-4b': {
        'name': 'InternVL2 4B',
        'hf_id': 'OpenGVLab/InternVL2-4B',
        'type': 'vlm',
        'vram_gb': 10,
        'description': 'Kompakter InternVL',
    },

    # ═══════════════════════════════════════════════════════════
    # LLAVA - Bewährt
    # ═══════════════════════════════════════════════════════════
    'llava-next-8b': {
        'name': 'LLaVA-NeXT 8B',
        'hf_id': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'type': 'vlm',
        'vram_gb': 14,
        'description': 'Neueste LLaVA Version',
    },
    'llava-1.5-7b': {
        'name': 'LLaVA 1.5 7B',
        'hf_id': 'llava-hf/llava-1.5-7b-hf',
        'type': 'vlm',
        'vram_gb': 14,
    },

    # ═══════════════════════════════════════════════════════════
    # FLORENCE-2 - Microsoft's Universal Vision
    # ═══════════════════════════════════════════════════════════
    'florence-2-large': {
        'name': 'Florence-2 Large',
        'hf_id': 'microsoft/Florence-2-large',
        'type': 'florence',
        'vram_gb': 8,
        'description': 'Universal: Caption, Detection, OCR, Segmentation',
        'recommended': True,
    },
    'florence-2-base': {
        'name': 'Florence-2 Base',
        'hf_id': 'microsoft/Florence-2-base',
        'type': 'florence',
        'vram_gb': 4,
        'description': 'Schneller Florence',
    },

    # ═══════════════════════════════════════════════════════════
    # PALIGEMMA - Google
    # ═══════════════════════════════════════════════════════════
    'paligemma-3b': {
        'name': 'PaliGemma 3B',
        'hf_id': 'google/paligemma-3b-mix-448',
        'type': 'vlm',
        'vram_gb': 8,
        'description': 'Google, effizient',
    },

    # ═══════════════════════════════════════════════════════════
    # PHI-3.5-VISION - Microsoft
    # ═══════════════════════════════════════════════════════════
    'phi-3.5-vision': {
        'name': 'Phi-3.5 Vision',
        'hf_id': 'microsoft/Phi-3.5-vision-instruct',
        'type': 'vlm',
        'vram_gb': 10,
        'description': 'Microsoft, sehr effizient',
    },

    # ═══════════════════════════════════════════════════════════
    # MOLMO - Allen AI
    # ═══════════════════════════════════════════════════════════
    'molmo-7b': {
        'name': 'Molmo 7B',
        'hf_id': 'allenai/Molmo-7B-D-0924',
        'type': 'vlm',
        'vram_gb': 16,
        'description': 'Allen AI, Apache 2.0',
    },

    # ═══════════════════════════════════════════════════════════
    # IMAGE CAPTIONING
    # ═══════════════════════════════════════════════════════════
    'blip2-opt-2.7b': {
        'name': 'BLIP-2 OPT 2.7B',
        'hf_id': 'Salesforce/blip2-opt-2.7b',
        'type': 'caption',
        'vram_gb': 8,
    },
    'blip-large': {
        'name': 'BLIP Large',
        'hf_id': 'Salesforce/blip-image-captioning-large',
        'type': 'caption',
        'vram_gb': 4,
    },

    # ═══════════════════════════════════════════════════════════
    # OBJECT DETECTION
    # ═══════════════════════════════════════════════════════════
    'yolov10x': {
        'name': 'YOLOv10 XLarge',
        'type': 'detection',
        'model_file': 'yolov10x.pt',
        'vram_gb': 12,
        'description': 'Neueste YOLO, beste Genauigkeit',
        'recommended': True,
    },
    'yolov10l': {
        'name': 'YOLOv10 Large',
        'type': 'detection',
        'model_file': 'yolov10l.pt',
        'vram_gb': 8,
    },
    'yolov10m': {
        'name': 'YOLOv10 Medium',
        'type': 'detection',
        'model_file': 'yolov10m.pt',
        'vram_gb': 4,
    },
    'yolov8x': {
        'name': 'YOLOv8 XLarge',
        'type': 'detection',
        'model_file': 'yolov8x.pt',
        'vram_gb': 12,
    },
    'yolo-world-l': {
        'name': 'YOLO-World Large',
        'type': 'detection',
        'model_file': 'yolov8l-worldv2.pt',
        'vram_gb': 10,
        'description': 'Open-vocabulary Detection',
    },

    # ═══════════════════════════════════════════════════════════
    # OCR MODELLE
    # ═══════════════════════════════════════════════════════════
    'surya': {
        'name': 'Surya OCR',
        'type': 'ocr',
        'description': 'Bestes multilinguales OCR',
        'recommended': True,
    },
    'easyocr': {
        'name': 'EasyOCR',
        'type': 'ocr',
        'description': 'Schnell und zuverlässig',
    },
    'paddleocr': {
        'name': 'PaddleOCR',
        'type': 'ocr',
        'description': 'Chinesisch optimiert',
    },

    # ═══════════════════════════════════════════════════════════
    # SEGMENTATION
    # ═══════════════════════════════════════════════════════════
    'sam2-large': {
        'name': 'SAM 2 Large',
        'type': 'segmentation',
        'model_file': 'sam2_hiera_large.pt',
        'vram_gb': 10,
        'description': 'Segment Anything Model 2',
        'recommended': True,
    },
    'sam2-base': {
        'name': 'SAM 2 Base',
        'type': 'segmentation',
        'model_file': 'sam2_hiera_base_plus.pt',
        'vram_gb': 6,
    },
}


class VisionWorker(BaseWorker):
    """
    Vision Worker - Die BESTEN Vision-Modelle

    Features:
    - Visual Question Answering (Qwen2-VL, InternVL2, LLaVA)
    - Image Captioning (BLIP-2, Florence-2)
    - OCR (Surya, EasyOCR, PaddleOCR)
    - Object Detection (YOLOv10, YOLO-World)
    - Image Segmentation (SAM 2)
    - Document Understanding (Florence-2)
    """

    def __init__(self):
        super().__init__("Vision Processing")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._ocr = None
        self._yolo = None
        self._sam = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        available = []

        if TRANSFORMERS_AVAILABLE:
            available.append("VLMs")
        if QWEN_VL_AVAILABLE:
            available.append("Qwen2-VL")
        if EASYOCR_AVAILABLE or PADDLEOCR_AVAILABLE or SURYA_AVAILABLE:
            available.append("OCR")
        if YOLO_AVAILABLE:
            available.append("YOLO")
        if SAM_AVAILABLE:
            available.append("SAM2")

        if not available:
            self._error_message = "Keine Vision-Libraries verfügbar"
            self.status = WorkerStatus.ERROR
            return False

        print(f"[OK] Vision Worker bereit")
        print(f"    Device: {self._device}")
        print(f"    Features: {', '.join(available)}")
        print(f"    Empfohlen: qwen2-vl-7b, florence-2-large, yolov10x")

        self.status = WorkerStatus.READY
        return True

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Load image from various sources"""
        if not image_input:
            raise ValueError("Bild erforderlich")

        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            if image_input.startswith("data:image"):
                base64_data = image_input.split(",")[1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_bytes)).convert("RGB")
            else:
                if not Path(image_input).exists():
                    raise ValueError(f"Bild nicht gefunden: {image_input}")
                return Image.open(image_input).convert("RGB")
        else:
            raise ValueError(f"Ungültiger Bildtyp: {type(image_input)}")

    def _load_vlm(self, model_id: str):
        """Load Vision-Language Model"""
        model_info = VISION_MODELS.get(model_id)
        if not model_info:
            raise ValueError(f"Unbekanntes Modell: {model_id}")

        hf_id = model_info.get('hf_id', '')

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            model = None
            processor = None

            # Qwen2-VL
            if 'qwen2-vl' in model_id.lower():
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                processor = AutoProcessor.from_pretrained(hf_id)
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    device_map="auto",
                )

            # InternVL2
            elif 'internvl' in model_id.lower():
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                    device_map="auto",
                )

            # LLaVA / LLaVA-NeXT
            elif 'llava' in model_id.lower():
                if 'next' in model_id.lower() or '1.6' in model_id:
                    processor = LlavaNextProcessor.from_pretrained(hf_id)
                    model = LlavaNextForConditionalGeneration.from_pretrained(
                        hf_id,
                        torch_dtype=self._dtype,
                        device_map="auto",
                    )
                else:
                    processor = AutoProcessor.from_pretrained(hf_id)
                    model = LlavaForConditionalGeneration.from_pretrained(
                        hf_id,
                        torch_dtype=self._dtype,
                        device_map="auto",
                    )

            # Florence-2
            elif 'florence' in model_id.lower():
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                ).to(self._device)

            # PaliGemma
            elif 'paligemma' in model_id.lower():
                processor = PaliGemmaProcessor.from_pretrained(hf_id)
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                ).to(self._device)

            # Phi-3.5 Vision
            elif 'phi' in model_id.lower():
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                    device_map="auto",
                )

            # Molmo
            elif 'molmo' in model_id.lower():
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                    device_map="auto",
                )

            # BLIP-2
            elif 'blip2' in model_id.lower():
                processor = Blip2Processor.from_pretrained(hf_id)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    device_map="auto",
                )

            # BLIP
            elif 'blip' in model_id.lower():
                processor = BlipProcessor.from_pretrained(hf_id)
                model = BlipForConditionalGeneration.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                ).to(self._device)

            else:
                raise ValueError(f"Unbekannter VLM-Typ: {model_id}")

            print(f"[OK] {model_info['name']} geladen")
            return {"model": model, "processor": processor}

        result = model_manager.get_model(hf_id, loader)
        self._model = result["model"]
        self._processor = result["processor"]
        self._current_model_id = model_id

    def visual_qa(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        model: str = "qwen2-vl-7b",
        max_tokens: int = 512,
    ) -> dict:
        """Answer questions about an image"""
        start_time = time.time()

        if not question or not question.strip():
            raise ValueError("Frage erforderlich")
        question = question.strip()[:2000]
        max_tokens = max(1, min(4096, max_tokens))

        if self._current_model_id != model:
            self._load_vlm(model)

        img = self._load_image(image)

        # Model-spezifische Verarbeitung
        if 'qwen2-vl' in model.lower():
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question}
                ]}
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        elif 'llava' in model.lower():
            if 'next' in model.lower() or '1.6' in model:
                prompt = f"<image>\nUSER: {question}\nASSISTANT:"
            else:
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self._processor(text=prompt, images=img, return_tensors="pt").to(self._device)

        elif 'florence' in model.lower():
            prompt = f"<OD>{question}"
            inputs = self._processor(text=prompt, images=img, return_tensors="pt").to(self._device)

        else:
            inputs = self._processor(images=img, text=question, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        # Decode
        if hasattr(self._processor, 'batch_decode'):
            answer = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            answer = self._processor.decode(generated_ids[0], skip_special_tokens=True)

        # Clean up
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()
        if question in answer:
            answer = answer.replace(question, "").strip()

        return {
            "answer": answer,
            "question": question,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def caption_image(
        self,
        image: Union[str, bytes, Image.Image],
        model: str = "blip-large",
    ) -> dict:
        """Generate caption for an image"""
        return self.visual_qa(image, "Describe this image in detail.", model=model, max_tokens=200)

    def ocr(
        self,
        image: Union[str, bytes, Image.Image],
        engine: str = "auto",
        languages: List[str] = None,
    ) -> dict:
        """Extract text from image using OCR"""
        start_time = time.time()

        if languages is None:
            languages = ["en", "de"]

        img = self._load_image(image)

        # Auto-select best available engine
        if engine == "auto":
            if SURYA_AVAILABLE:
                engine = "surya"
            elif EASYOCR_AVAILABLE:
                engine = "easyocr"
            elif PADDLEOCR_AVAILABLE:
                engine = "paddleocr"
            else:
                raise ValueError("Keine OCR Engine verfügbar")

        text_blocks = []
        full_text = ""

        if engine == "surya" and SURYA_AVAILABLE:
            det_processor, det_model = segformer.load_processor(), segformer.load_model()
            rec_model, rec_processor = recognition_model.load_model(), recognition_model.load_processor()
            results = run_ocr([img], [languages], det_model, det_processor, rec_model, rec_processor)
            for page in results:
                for line in page.text_lines:
                    text_blocks.append({
                        "text": line.text,
                        "confidence": line.confidence,
                        "bbox": line.bbox,
                    })
            full_text = " ".join([b["text"] for b in text_blocks])

        elif engine == "easyocr" and EASYOCR_AVAILABLE:
            if self._ocr is None or not isinstance(self._ocr, easyocr.Reader):
                self._ocr = easyocr.Reader(languages, gpu=self._device == "cuda")
            results = self._ocr.readtext(np.array(img))
            for r in results:
                text_blocks.append({"text": r[1], "confidence": r[2], "bbox": r[0]})
            full_text = " ".join([r[1] for r in results])

        elif engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            if self._ocr is None or not isinstance(self._ocr, PaddleOCR):
                self._ocr = PaddleOCR(use_angle_cls=True, lang='en')
            result = self._ocr.ocr(np.array(img), cls=True)
            if result and result[0]:
                for line in result[0]:
                    text_blocks.append({
                        "text": line[1][0],
                        "confidence": line[1][1],
                        "bbox": line[0],
                    })
                full_text = " ".join([line[1][0] for line in result[0]])

        else:
            raise ValueError(f"OCR Engine nicht verfügbar: {engine}")

        return {
            "text": full_text,
            "blocks": text_blocks,
            "engine": engine,
            "gpu_seconds": time.time() - start_time,
        }

    def detect_objects(
        self,
        image: Union[str, bytes, Image.Image],
        model: str = "yolov10x",
        confidence: float = 0.25,
        classes: List[str] = None,
    ) -> dict:
        """Detect objects in image"""
        start_time = time.time()

        if not YOLO_AVAILABLE:
            raise ValueError("YOLO nicht installiert")

        model_info = VISION_MODELS.get(model)
        if not model_info or model_info.get('type') != 'detection':
            raise ValueError(f"Unbekanntes Detection-Modell: {model}")

        model_file = model_info.get('model_file', f"{model}.pt")

        if self._yolo is None or self._current_model_id != model:
            self._yolo = YOLO(model_file)
            self._current_model_id = model

        img = self._load_image(image)

        results = self._yolo(img, conf=confidence, classes=classes)

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

    def segment(
        self,
        image: Union[str, bytes, Image.Image],
        points: List[List[int]] = None,
        boxes: List[List[int]] = None,
        model: str = "sam2-large",
    ) -> dict:
        """Segment image using SAM 2"""
        start_time = time.time()

        if not SAM_AVAILABLE:
            raise ValueError("SAM 2 nicht installiert")

        model_info = VISION_MODELS.get(model)
        if not model_info:
            raise ValueError(f"Unbekanntes Segmentation-Modell: {model}")

        img = self._load_image(image)
        img_array = np.array(img)

        if self._sam is None or self._current_model_id != model:
            model_file = model_info.get('model_file')
            sam2_model = build_sam2("sam2_hiera_l.yaml", model_file)
            self._sam = SAM2ImagePredictor(sam2_model)
            self._current_model_id = model

        self._sam.set_image(img_array)

        if points:
            point_coords = np.array(points)
            point_labels = np.ones(len(points))
            masks, scores, _ = self._sam.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        elif boxes:
            box_array = np.array(boxes)
            masks, scores, _ = self._sam.predict(
                box=box_array,
                multimask_output=True,
            )
        else:
            masks, scores, _ = self._sam.predict(multimask_output=True)

        return {
            "masks": masks.tolist(),
            "scores": scores.tolist(),
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process a vision job"""
        task_type = input_data.get("task", "vqa")
        image = input_data.get("image") or input_data.get("file")

        self.notify_progress(job_id, 0.1, f"Starte {task_type}")

        if task_type == "vqa" or task_type == "visual_qa":
            question = input_data.get("question", "Describe this image in detail.")
            model = input_data.get("model", "qwen2-vl-7b")
            max_tokens = input_data.get("max_tokens", 512)
            result = self.visual_qa(image, question, model=model, max_tokens=max_tokens)

        elif task_type == "caption":
            model = input_data.get("model", "blip-large")
            result = self.caption_image(image, model=model)

        elif task_type == "ocr":
            engine = input_data.get("engine", "auto")
            languages = input_data.get("languages", ["en", "de"])
            result = self.ocr(image, engine=engine, languages=languages)

        elif task_type == "detect" or task_type == "detection":
            model = input_data.get("model", "yolov10x")
            confidence = input_data.get("confidence", 0.25)
            result = self.detect_objects(image, model=model, confidence=confidence)

        elif task_type == "segment" or task_type == "segmentation":
            model = input_data.get("model", "sam2-large")
            points = input_data.get("points")
            boxes = input_data.get("boxes")
            result = self.segment(image, points=points, boxes=boxes, model=model)

        else:
            raise ValueError(f"Unbekannter Task-Typ: {task_type}")

        self.notify_progress(job_id, 1.0, "Fertig")
        return result

    def cleanup(self):
        """Release resources"""
        self._model = None
        self._processor = None
        self._ocr = None
        self._yolo = None
        self._sam = None
        self._current_model_id = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Vision Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        return VISION_MODELS

    def get_recommended_models(self) -> List[str]:
        """Return recommended models"""
        return [k for k, v in VISION_MODELS.items() if v.get('recommended', False)]


# Singleton
_vision_worker: Optional[VisionWorker] = None


def get_vision_worker() -> VisionWorker:
    """Get singleton instance"""
    global _vision_worker
    if _vision_worker is None:
        _vision_worker = VisionWorker()
    return _vision_worker
