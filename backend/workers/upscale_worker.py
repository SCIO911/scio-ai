#!/usr/bin/env python3
"""
SCIO - Upscale Worker
Image & Video Upscaling, Face Restoration
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

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

# Real-ESRGAN
REALESRGAN_AVAILABLE = False
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    pass

# GFPGAN (Face Restoration)
GFPGAN_AVAILABLE = False
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    pass

# CodeFormer (Face Restoration)
CODEFORMER_AVAILABLE = False
try:
    from codeformer import CodeFormer
    CODEFORMER_AVAILABLE = True
except ImportError:
    pass

# Diffusers Upscaler
DIFFUSERS_UPSCALE_AVAILABLE = False
try:
    from diffusers import StableDiffusionUpscalePipeline
    DIFFUSERS_UPSCALE_AVAILABLE = True
except ImportError:
    pass


# Model download URLs
MODEL_URLS = {
    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    'GFPGANv1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
}


# Available Upscale Models
UPSCALE_MODELS = {
    'realesrgan-x4': {
        'name': 'Real-ESRGAN x4',
        'scale': 4,
        'type': 'esrgan',
        'vram_gb': 2,
    },
    'realesrgan-x2': {
        'name': 'Real-ESRGAN x2',
        'scale': 2,
        'type': 'esrgan',
        'vram_gb': 2,
    },
    'realesrgan-anime': {
        'name': 'Real-ESRGAN Anime',
        'scale': 4,
        'type': 'esrgan',
        'vram_gb': 2,
    },
    'gfpgan': {
        'name': 'GFPGAN (Face)',
        'scale': 2,
        'type': 'face',
        'vram_gb': 4,
    },
    'codeformer': {
        'name': 'CodeFormer (Face)',
        'scale': 2,
        'type': 'face',
        'vram_gb': 4,
    },
    'sd-upscale-x4': {
        'name': 'SD Upscale x4',
        'hf_id': 'stabilityai/stable-diffusion-x4-upscaler',
        'scale': 4,
        'type': 'diffusion',
        'vram_gb': 12,
    },
}


class UpscaleWorker(BaseWorker):
    """
    Upscale Worker - Handles all upscaling tasks

    Features:
    - Image Upscaling (Real-ESRGAN)
    - Face Restoration (GFPGAN, CodeFormer)
    - AI Upscaling (Stable Diffusion)
    - Video Frame Upscaling
    - Batch Processing
    """

    def __init__(self):
        super().__init__("Upscaling")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._esrgan = None
        self._gfpgan = None
        self._codeformer = None
        self._sd_upscaler = None
        self._current_model_id = None
        self._weights_dir = Config.DATA_DIR / "models" / "upscale"

    def initialize(self) -> bool:
        """Initialize the worker"""
        available_features = []

        if REALESRGAN_AVAILABLE:
            available_features.append("Real-ESRGAN")
        if GFPGAN_AVAILABLE:
            available_features.append("GFPGAN")
        if CODEFORMER_AVAILABLE:
            available_features.append("CodeFormer")
        if DIFFUSERS_UPSCALE_AVAILABLE:
            available_features.append("SD-Upscale")

        if not available_features and not PIL_AVAILABLE:
            self._error_message = "No upscaling libraries available"
            self.status = WorkerStatus.ERROR
            return False

        # Ensure weights directory exists
        self._weights_dir.mkdir(parents=True, exist_ok=True)

        self.status = WorkerStatus.READY
        print(f"[OK] Upscale Worker bereit (Device: {self._device}, Features: {', '.join(available_features)})")
        return True

    def _download_model(self, model_name: str) -> Path:
        """
        Download model weights if not present.

        Args:
            model_name: Name of the model (key in MODEL_URLS)

        Returns:
            Path to the model weights file

        Raises:
            RuntimeError: If download fails
        """
        model_path = self._weights_dir / f"{model_name}.pth"

        if model_path.exists():
            return model_path

        if model_name not in MODEL_URLS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")

        url = MODEL_URLS[model_name]
        print(f"[DOWNLOAD] Downloading {model_name}...")

        try:
            # Download with progress
            urllib.request.urlretrieve(url, model_path)
            print(f"[OK] Downloaded {model_name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return model_path
        except Exception as e:
            # Cleanup partial download
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download {model_name}: {e}")

    def _load_esrgan(self, model_type: str = "x4"):
        """Load Real-ESRGAN model with automatic weight download"""
        def loader():
            print(f"[LOAD] Lade Real-ESRGAN {model_type}...")

            if model_type == "anime":
                model_name = "RealESRGAN_x4plus_anime_6B"
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                scale = 4
            elif model_type == "x2":
                model_name = "RealESRGAN_x2plus"
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                scale = 2
            else:
                model_name = "RealESRGAN_x4plus"
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                scale = 4

            # Download if needed
            model_path = self._download_model(model_name)

            upsampler = RealESRGANer(
                scale=scale,
                model_path=str(model_path),
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=self._device == "cuda",
                device=self._device,
            )
            return upsampler

        try:
            self._esrgan = model_manager.get_model(f"esrgan_{model_type}", loader)
            print(f"[OK] Real-ESRGAN {model_type} geladen")
        except Exception as e:
            raise RuntimeError(f"Failed to load Real-ESRGAN {model_type}: {e}")

    def _load_gfpgan(self):
        """Load GFPGAN for face restoration with automatic weight download"""
        def loader():
            print("[LOAD] Lade GFPGAN...")

            # Download if needed
            model_path = self._download_model("GFPGANv1.4")

            return GFPGANer(
                model_path=str(model_path),
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                device=self._device,
            )

        try:
            self._gfpgan = model_manager.get_model("gfpgan", loader)
            print("[OK] GFPGAN geladen")
        except Exception as e:
            raise RuntimeError(f"Failed to load GFPGAN: {e}")

    def _load_sd_upscaler(self):
        """Load Stable Diffusion Upscaler"""
        def loader():
            print("[LOAD] Lade SD Upscaler...")
            pipe = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            pipe.to(self._device)

            if self._device == "cuda":
                pipe.enable_attention_slicing()

            return pipe

        self._sd_upscaler = model_manager.get_model("sd_upscaler", loader)
        print("[OK] SD Upscaler geladen")

    def upscale_image(
        self,
        image: Union[str, Image.Image],
        model: str = "realesrgan-x4",
        scale: int = None,
        output_path: str = None,
    ) -> dict:
        """
        Upscale an image.

        Args:
            image: Image path or PIL Image object
            model: Upscale model (realesrgan-x4, realesrgan-x2, realesrgan-anime, gfpgan, codeformer, sd-upscale-x4)
            scale: Optional custom scale factor (1-8)
            output_path: Optional output file path

        Returns:
            dict with output_path, input_size, output_size, scale, model, gpu_seconds

        Raises:
            ValueError: If image is invalid or model unavailable
        """
        start_time = time.time()

        # Validate model
        if model not in UPSCALE_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(UPSCALE_MODELS.keys())}")

        # Validate scale
        if scale is not None:
            scale = max(1, min(8, scale))

        # Load and validate image
        if isinstance(image, str):
            if not Path(image).exists():
                raise ValueError(f"Image file not found: {image}")
            try:
                img = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to open image: {e}")
            input_path = image
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
            input_path = None
        else:
            raise ValueError("Image must be a file path or PIL Image")

        # Validate image size
        if img.width < 16 or img.height < 16:
            raise ValueError("Image too small (minimum 16x16 pixels)")
        if img.width > 8192 or img.height > 8192:
            raise ValueError("Image too large (maximum 8192x8192 pixels)")

        img_np = np.array(img)

        # Determine output path
        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"upscale_{uuid.uuid4().hex[:8]}.png")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        model_info = UPSCALE_MODELS.get(model, UPSCALE_MODELS['realesrgan-x4'])

        if model_info['type'] == 'esrgan':
            if "anime" in model:
                model_type = "anime"
            elif "x2" in model:
                model_type = "x2"
            else:
                model_type = "x4"

            if self._esrgan is None:
                self._load_esrgan(model_type)

            output, _ = self._esrgan.enhance(img_np, outscale=scale or model_info['scale'])
            output_img = Image.fromarray(output)

        elif model_info['type'] == 'face':
            if self._gfpgan is None and GFPGAN_AVAILABLE:
                self._load_gfpgan()

            if self._gfpgan:
                _, _, output = self._gfpgan.enhance(
                    img_np,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                output_img = Image.fromarray(output)
            else:
                raise ValueError("GFPGAN not available")

        elif model_info['type'] == 'diffusion':
            if self._sd_upscaler is None:
                self._load_sd_upscaler()

            # SD Upscaler needs a prompt
            prompt = "high quality, detailed, sharp"
            output_img = self._sd_upscaler(
                prompt=prompt,
                image=img,
                num_inference_steps=20,
            ).images[0]

        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")

        # Save output
        output_img.save(output_path, quality=95)

        return {
            "output_path": output_path,
            "input_size": f"{img.width}x{img.height}",
            "output_size": f"{output_img.width}x{output_img.height}",
            "scale": output_img.width / img.width,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def upscale_batch(
        self,
        images: List[Union[str, Image.Image]],
        model: str = "realesrgan-x4",
        output_dir: str = None,
    ) -> dict:
        """Upscale multiple images"""
        start_time = time.time()

        if output_dir is None:
            output_dir = str(Config.DATA_DIR / "generated" / f"batch_{uuid.uuid4().hex[:8]}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        for i, img in enumerate(images):
            output_path = str(Path(output_dir) / f"upscaled_{i:04d}.png")
            result = self.upscale_image(img, model=model, output_path=output_path)
            results.append(result)

        return {
            "output_dir": output_dir,
            "count": len(results),
            "results": results,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def restore_face(
        self,
        image: Union[str, Image.Image],
        model: str = "gfpgan",
        output_path: str = None,
    ) -> dict:
        """Restore faces in an image"""
        return self.upscale_image(image, model=model, output_path=output_path)

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process an upscaling job"""
        task_type = input_data.get("task", "upscale")

        self.notify_progress(job_id, 0.1, f"Starting {task_type}")

        if task_type == "upscale":
            image = input_data.get("image") or input_data.get("file")
            model = input_data.get("model", "realesrgan-x4")
            scale = input_data.get("scale")
            result = self.upscale_image(image, model=model, scale=scale)

        elif task_type == "batch":
            images = input_data.get("images")
            model = input_data.get("model", "realesrgan-x4")
            result = self.upscale_batch(images, model=model)

        elif task_type == "face" or task_type == "restore_face":
            image = input_data.get("image") or input_data.get("file")
            model = input_data.get("model", "gfpgan")
            result = self.restore_face(image, model=model)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._esrgan = None
        self._gfpgan = None
        self._codeformer = None
        self._sd_upscaler = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Upscale Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        available = {}
        for model_id, info in UPSCALE_MODELS.items():
            if info['type'] == 'esrgan' and REALESRGAN_AVAILABLE:
                available[model_id] = info
            elif info['type'] == 'face' and (GFPGAN_AVAILABLE or CODEFORMER_AVAILABLE):
                available[model_id] = info
            elif info['type'] == 'diffusion' and DIFFUSERS_UPSCALE_AVAILABLE:
                available[model_id] = info
        return available


# Singleton Instance
_upscale_worker: Optional[UpscaleWorker] = None


def get_upscale_worker() -> UpscaleWorker:
    """Get singleton instance"""
    global _upscale_worker
    if _upscale_worker is None:
        _upscale_worker = UpscaleWorker()
    return _upscale_worker
