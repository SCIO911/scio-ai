#!/usr/bin/env python3
"""
SCIO - Video Worker
Text-to-Video, Image-to-Video, Video Processing
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

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

# Video Generation Libraries
COGVIDEO_AVAILABLE = False
try:
    from diffusers import CogVideoXPipeline
    COGVIDEO_AVAILABLE = True
except ImportError:
    pass

ANIMATEDIFF_AVAILABLE = False
try:
    from diffusers import AnimateDiffPipeline, MotionAdapter
    ANIMATEDIFF_AVAILABLE = True
except ImportError:
    pass

SVD_AVAILABLE = False
try:
    from diffusers import StableVideoDiffusionPipeline
    SVD_AVAILABLE = True
except ImportError:
    pass

# Video I/O
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Available Video Models
VIDEO_MODELS = {
    'cogvideox-2b': {
        'name': 'CogVideoX 2B',
        'hf_id': 'THUDM/CogVideoX-2b',
        'type': 'text2video',
        'vram_gb': 16,
    },
    'cogvideox-5b': {
        'name': 'CogVideoX 5B',
        'hf_id': 'THUDM/CogVideoX-5b',
        'type': 'text2video',
        'vram_gb': 24,
    },
    'animatediff': {
        'name': 'AnimateDiff',
        'hf_id': 'guoyww/animatediff-motion-adapter-v1-5-2',
        'type': 'text2video',
        'vram_gb': 12,
    },
    'svd': {
        'name': 'Stable Video Diffusion',
        'hf_id': 'stabilityai/stable-video-diffusion-img2vid-xt',
        'type': 'img2video',
        'vram_gb': 18,
    },
    'svd-xt': {
        'name': 'SVD-XT (25 frames)',
        'hf_id': 'stabilityai/stable-video-diffusion-img2vid-xt-1-1',
        'type': 'img2video',
        'vram_gb': 18,
    },
}


class VideoWorker(BaseWorker):
    """
    Video Worker - Handles all video AI tasks

    Features:
    - Text-to-Video (CogVideoX)
    - Image-to-Video (SVD)
    - AnimateDiff
    - Video Processing
    """

    def __init__(self):
        super().__init__("Video Generation")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._pipeline = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        available_features = []

        if COGVIDEO_AVAILABLE:
            available_features.append("CogVideoX")
        if ANIMATEDIFF_AVAILABLE:
            available_features.append("AnimateDiff")
        if SVD_AVAILABLE:
            available_features.append("SVD")

        if not available_features:
            self._error_message = "No video libraries available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] Video Worker bereit (Device: {self._device}, Features: {', '.join(available_features)})")
        return True

    def _load_model(self, model_id: str):
        """Load a video model"""
        if model_id not in VIDEO_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        model_info = VIDEO_MODELS[model_id]
        hf_id = model_info['hf_id']

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            if "cogvideo" in model_id.lower():
                pipe = CogVideoXPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                )
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_tiling()

            elif model_id == "animatediff":
                adapter = MotionAdapter.from_pretrained(hf_id, torch_dtype=dtype)
                pipe = AnimateDiffPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    motion_adapter=adapter,
                    torch_dtype=dtype,
                )
                pipe.to(self._device)

            elif "svd" in model_id.lower():
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    variant="fp16",
                )
                pipe.to(self._device)

            else:
                raise ValueError(f"Unknown model type: {model_id}")

            return pipe

        self._pipeline = model_manager.get_model(hf_id, loader)
        self._current_model_id = model_id
        print(f"[OK] {model_info['name']} geladen")

    def generate_video(
        self,
        prompt: str = None,
        image: str = None,
        model: str = "cogvideox-2b",
        num_frames: int = 49,
        fps: int = 8,
        width: int = 720,
        height: int = 480,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: int = None,
        output_path: str = None,
    ) -> dict:
        """
        Generate video from text or image.

        Args:
            prompt: Text description for text-to-video models
            image: Image path for image-to-video models
            model: Model ID (cogvideox-2b, cogvideox-5b, animatediff, svd, svd-xt)
            num_frames: Number of frames (8-120)
            fps: Frames per second (1-60)
            width: Video width (64-1920)
            height: Video height (64-1080)
            num_inference_steps: Denoising steps (10-150)
            guidance_scale: Guidance scale (1.0-30.0)
            seed: Random seed for reproducibility
            output_path: Optional output file path

        Returns:
            dict with output_path, prompt, model, etc.

        Raises:
            ValueError: If parameters are invalid
        """
        start_time = time.time()

        # Validate model
        if model not in VIDEO_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(VIDEO_MODELS.keys())}")

        model_info = VIDEO_MODELS[model]

        # Validate prompt/image based on model type
        if model_info['type'] == 'text2video':
            if not prompt or not prompt.strip():
                raise ValueError("Prompt required for text-to-video models")
            prompt = prompt.strip()
            if len(prompt) > 2000:
                raise ValueError("Prompt too long (max 2000 characters)")
        elif model_info['type'] == 'img2video':
            if not image:
                raise ValueError("Image required for image-to-video models")
            if isinstance(image, str) and not Path(image).exists():
                raise ValueError(f"Image file not found: {image}")

        # Validate and clamp parameters
        num_frames = max(8, min(120, num_frames))
        fps = max(1, min(60, fps))
        width = max(64, min(1920, width))
        height = max(64, min(1080, height))
        num_inference_steps = max(10, min(150, num_inference_steps))
        guidance_scale = max(1.0, min(30.0, guidance_scale))

        # Ensure dimensions are multiples of 8 (required by most models)
        width = (width // 8) * 8
        height = (height // 8) * 8

        if self._current_model_id != model or self._pipeline is None:
            self._load_model(model)

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"video_{uuid.uuid4().hex[:8]}.mp4")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        if model_info['type'] == 'text2video':
            # Text-to-Video
            if "cogvideo" in model.lower():
                video_frames = self._pipeline(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).frames[0]
            else:
                # AnimateDiff
                output = self._pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                video_frames = output.frames[0]

        elif model_info['type'] == 'img2video':
            # Image-to-Video
            if image is None:
                raise ValueError("Image required for img2video models")

            input_image = Image.open(image).convert("RGB")
            input_image = input_image.resize((width, height))

            video_frames = self._pipeline(
                input_image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).frames[0]

        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")

        # Save video
        if IMAGEIO_AVAILABLE:
            imageio.mimsave(output_path, video_frames, fps=fps)
        else:
            # Fallback: save as GIF
            output_path = output_path.replace('.mp4', '.gif')
            video_frames[0].save(
                output_path,
                save_all=True,
                append_images=video_frames[1:],
                duration=int(1000/fps),
                loop=0,
            )

        return {
            "output_path": output_path,
            "prompt": prompt,
            "model": model,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "seed": seed,
            "gpu_seconds": time.time() - start_time,
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process a video job"""
        self.notify_progress(job_id, 0.1, "Starting video generation")

        result = self.generate_video(
            prompt=input_data.get("prompt"),
            image=input_data.get("image"),
            model=input_data.get("model", "cogvideox-2b"),
            num_frames=input_data.get("num_frames", 49),
            fps=input_data.get("fps", 8),
            width=input_data.get("width", 720),
            height=input_data.get("height", 480),
            num_inference_steps=input_data.get("steps", 50),
            guidance_scale=input_data.get("guidance_scale", 6.0),
            seed=input_data.get("seed"),
        )

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._pipeline = None
        self._current_model_id = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Video Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        available = {}
        for model_id, info in VIDEO_MODELS.items():
            if "cogvideo" in model_id and COGVIDEO_AVAILABLE:
                available[model_id] = info
            elif model_id == "animatediff" and ANIMATEDIFF_AVAILABLE:
                available[model_id] = info
            elif "svd" in model_id and SVD_AVAILABLE:
                available[model_id] = info
        return available


# Singleton Instance
_video_worker: Optional[VideoWorker] = None


def get_video_worker() -> VideoWorker:
    """Get singleton instance"""
    global _video_worker
    if _video_worker is None:
        _video_worker = VideoWorker()
    return _video_worker
