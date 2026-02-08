#!/usr/bin/env python3
"""
SCIO - Image Generation Worker
SDXL, Stable Diffusion, Flux
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, List

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# Try to import image generation libraries
try:
    import torch
    TORCH_AVAILABLE = True
    # CUDA Optimierungen
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DPMSolverMultistepScheduler,
        AutoencoderKL,
    )
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    Image = None  # Placeholder for type hints
    print("[WARN]  Diffusers nicht installiert")

# SD3 Pipeline (optional)
SD3_AVAILABLE = False
try:
    from diffusers import StableDiffusion3Pipeline
    SD3_AVAILABLE = True
except ImportError:
    pass


# Available models - Optimiert für RTX 5090 mit 24GB VRAM
IMAGE_MODELS = {
    'sd-1.5': {
        'name': 'Stable Diffusion 1.5',
        'hf_id': 'runwayml/stable-diffusion-v1-5',
        'pipeline': 'StableDiffusionPipeline',
        'vram_gb': 6,
    },
    'sd-2.1': {
        'name': 'Stable Diffusion 2.1',
        'hf_id': 'stabilityai/stable-diffusion-2-1',
        'pipeline': 'StableDiffusionPipeline',
        'vram_gb': 8,
    },
    'sdxl': {
        'name': 'SDXL 1.0',
        'hf_id': 'stabilityai/stable-diffusion-xl-base-1.0',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 12,
    },
    'sdxl-turbo': {
        'name': 'SDXL Turbo',
        'hf_id': 'stabilityai/sdxl-turbo',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 12,
    },
    'sdxl-lightning': {
        'name': 'SDXL Lightning',
        'hf_id': 'ByteDance/SDXL-Lightning',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 12,
    },
    'sd3-medium': {
        'name': 'Stable Diffusion 3 Medium',
        'hf_id': 'stabilityai/stable-diffusion-3-medium-diffusers',
        'pipeline': 'StableDiffusion3Pipeline',
        'vram_gb': 18,
    },
    'playground-v2.5': {
        'name': 'Playground v2.5',
        'hf_id': 'playgroundai/playground-v2.5-1024px-aesthetic',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 16,
    },
}


class ImageGenerationWorker(BaseWorker):
    """
    Image Generation Worker

    Features:
    - Multiple Models (SD 1.5, SD 2.1, SDXL)
    - Batch Generation
    - Custom Dimensions
    - Seed Control
    """

    def __init__(self):
        super().__init__("Image Generation")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._current_model_id: Optional[str] = None
        self._pipeline = None

    def initialize(self) -> bool:
        """Initialisiert den Worker"""
        if not DIFFUSERS_AVAILABLE:
            self._error_message = "Diffusers library not available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] Image Generation Worker bereit (Device: {self._device})")
        return True

    def _load_model(self, model_id: str):
        """Lädt ein Modell - Optimiert für RTX 5090 mit 24GB VRAM"""
        if model_id not in IMAGE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        model_info = IMAGE_MODELS[model_id]
        hf_id = model_info['hf_id']

        def loader():
            print(f"[LOAD] Lade {model_info['name']} mit GPU-Optimierungen...")

            # Optimale dtype für RTX 5090
            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            # Select pipeline class based on model type
            pipeline_type = model_info['pipeline']

            if pipeline_type == 'StableDiffusion3Pipeline' and SD3_AVAILABLE:
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            elif pipeline_type == 'StableDiffusionXLPipeline':
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16" if self._device == "cuda" else None,
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )

            # Optimize für RTX 5090 - Keine Attention Slicing nötig bei 24GB VRAM
            pipe.to(self._device)

            if self._device == "cuda":
                # xformers für schnellere Attention
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("[OK] xformers aktiviert")
                except (ImportError, ModuleNotFoundError):
                    pass  # xformers not available

                # VAE Optimierungen
                try:
                    pipe.vae.enable_tiling()  # Für hochauflösende Bilder
                    pipe.vae.enable_slicing()  # Batch-Verarbeitung
                except AttributeError:
                    pass  # VAE doesn't support tiling/slicing

                # Model Offload NICHT aktivieren - wir haben genug VRAM
                # pipe.enable_model_cpu_offload()  # DEAKTIVIERT

            # Use DPM++ 2M Karras scheduler für beste Qualität/Geschwindigkeit
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )

            # Compile für schnellere Ausführung
            if hasattr(torch, 'compile') and self._device == "cuda":
                try:
                    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
                    print("[OK] torch.compile für UNet aktiviert")
                except Exception:
                    pass

            return pipe

        self._pipeline = model_manager.get_model(hf_id, loader)
        self._current_model_id = model_id

        print(f"[OK] Modell geladen: {model_info['name']} (VRAM optimiert)")

    def process(self, job_id: str, input_data: dict) -> dict:
        """
        Generiert Bilder

        Args:
            job_id: Job identifier
            input_data: Dictionary with:
                - model: Model-ID (default: sdxl)
                - prompt: Text-Prompt (required, max 2000 chars)
                - negative_prompt: Negative Prompt
                - num_images: Anzahl Bilder (1-10)
                - width: Breite in Pixeln (64-2048, must be divisible by 8)
                - height: Höhe in Pixeln (64-2048, must be divisible by 8)
                - steps: Inference Steps (10-150)
                - guidance_scale: CFG Scale (1.0-30.0)
                - seed: Optional Seed
                - output_dir: Optional Output-Verzeichnis

        Returns:
            dict with output_dir, image_paths, model, prompt, etc.

        Raises:
            ValueError: If parameters are invalid
        """
        start_time = time.time()

        # Extract parameters
        model_id = input_data.get('model', 'sdxl')
        prompt = input_data.get('prompt', '')
        negative_prompt = input_data.get('negative_prompt', '')
        num_images = input_data.get('num_images', 1)
        width = input_data.get('width', 1024)
        height = input_data.get('height', 1024)
        steps = input_data.get('steps', 25)
        guidance_scale = input_data.get('guidance_scale', 7.5)
        seed = input_data.get('seed')

        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required")

        prompt = prompt.strip()
        if len(prompt) > 2000:
            raise ValueError("Prompt too long (max 2000 characters)")

        # Validate model
        if model_id not in IMAGE_MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(IMAGE_MODELS.keys())}")

        # Validate and clamp numerical parameters
        num_images = max(1, min(10, num_images))
        width = max(64, min(2048, width))
        height = max(64, min(2048, height))
        steps = max(10, min(150, steps))
        guidance_scale = max(1.0, min(30.0, guidance_scale))

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Adjust dimensions for non-SDXL models
        if model_id in ['sd-1.5', 'sd-2.1']:
            width = min(width, 768)
            height = min(height, 768)

        # Load model if needed
        if self._current_model_id != model_id or self._pipeline is None:
            self._load_model(model_id)

        self.notify_progress(job_id, 0.1, "Model loaded")

        # Output directory
        output_dir = input_data.get('output_dir') or str(
            Config.DATA_DIR / 'generated' / job_id
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self._device).manual_seed(seed)

        self.notify_progress(job_id, 0.2, "Generating images")

        # Generate images
        images: List[Image.Image] = []
        image_paths: List[str] = []

        for i in range(num_images):
            progress = 0.2 + (i / num_images) * 0.7
            self.notify_progress(job_id, progress, f"Generating image {i+1}/{num_images}")

            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            image = result.images[0]
            images.append(image)

            # Save image
            filename = f"image_{i+1:03d}_{uuid.uuid4().hex[:8]}.png"
            filepath = Path(output_dir) / filename
            image.save(filepath)
            image_paths.append(str(filepath))

            # New seed for next image
            if i < num_images - 1:
                seed += 1
                generator = torch.Generator(device=self._device).manual_seed(seed)

        self.notify_progress(job_id, 1.0, "Complete")

        end_time = time.time()

        return {
            'output_dir': output_dir,
            'image_paths': image_paths,
            'num_images': num_images,
            'model': model_id,
            'prompt': prompt,
            'seed': seed - num_images + 1,  # First seed used
            'width': width,
            'height': height,
            'steps': steps,
            'gpu_seconds': end_time - start_time,
        }

    def generate_single(
        self,
        prompt: str,
        model_id: str = 'sdxl',
        negative_prompt: str = '',
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,
    ):
        """
        Generiert ein einzelnes Bild (für API-Nutzung)

        Returns:
            PIL Image
        """
        # Adjust dimensions for non-SDXL models
        if model_id in ['sd-1.5', 'sd-2.1']:
            width = min(width, 768)
            height = min(height, 768)

        # Load model if needed
        if self._current_model_id != model_id or self._pipeline is None:
            self._load_model(model_id)

        # Generator
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        # Generate
        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]

    def cleanup(self):
        """Gibt Ressourcen frei"""
        self._pipeline = None
        self._current_model_id = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 Image Generation Worker bereinigt")

    def get_available_models(self) -> dict:
        """Gibt verfügbare Modelle zurück"""
        return IMAGE_MODELS


# Singleton Instance
_image_worker: Optional[ImageGenerationWorker] = None


def get_image_worker() -> ImageGenerationWorker:
    """Gibt Singleton-Instanz zurück"""
    global _image_worker
    if _image_worker is None:
        _image_worker = ImageGenerationWorker()
    return _image_worker
