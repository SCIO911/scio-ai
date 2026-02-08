#!/usr/bin/env python3
"""
SCIO - 3D Worker
Text-to-3D, Image-to-3D, 3D Model Generation
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
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
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 3D Libraries
SHAP_E_AVAILABLE = False
try:
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh
    SHAP_E_AVAILABLE = True
except ImportError:
    pass

POINT_E_AVAILABLE = False
try:
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config as point_e_diffusion
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    POINT_E_AVAILABLE = True
except ImportError:
    pass

TRIPOSR_AVAILABLE = False
try:
    from tsr.system import TSR
    TRIPOSR_AVAILABLE = True
except ImportError:
    pass

# Trimesh for 3D mesh operations
TRIMESH_AVAILABLE = False
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    pass


# Available 3D Models
THREED_MODELS = {
    'shap-e': {
        'name': 'Shap-E (OpenAI)',
        'type': 'text2mesh',
        'vram_gb': 8,
    },
    'shap-e-img': {
        'name': 'Shap-E Image-to-3D',
        'type': 'img2mesh',
        'vram_gb': 8,
    },
    'point-e': {
        'name': 'Point-E (OpenAI)',
        'type': 'text2pointcloud',
        'vram_gb': 6,
    },
    'triposr': {
        'name': 'TripoSR',
        'type': 'img2mesh',
        'vram_gb': 8,
    },
}


class ThreeDWorker(BaseWorker):
    """
    3D Worker - Handles all 3D generation tasks

    Features:
    - Text-to-3D (Shap-E)
    - Image-to-3D (Shap-E, TripoSR)
    - Point Cloud Generation (Point-E)
    - Mesh Export (OBJ, GLB, PLY)
    """

    def __init__(self):
        super().__init__("3D Generation")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._shap_e_model = None
        self._shap_e_diffusion = None
        self._point_e_model = None
        self._triposr = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        available_features = []

        if SHAP_E_AVAILABLE:
            available_features.append("Shap-E")
        if POINT_E_AVAILABLE:
            available_features.append("Point-E")
        if TRIPOSR_AVAILABLE:
            available_features.append("TripoSR")

        if not available_features:
            self._error_message = "No 3D generation libraries available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] 3D Worker bereit (Device: {self._device}, Features: {', '.join(available_features)})")
        return True

    def _load_shap_e(self, use_image: bool = False):
        """Load Shap-E model"""
        def loader():
            print("[LOAD] Lade Shap-E...")
            if use_image:
                xm = load_model('transmitter', device=self._device)
                model = load_model('image300M', device=self._device)
            else:
                xm = load_model('transmitter', device=self._device)
                model = load_model('text300M', device=self._device)

            diffusion = diffusion_from_config(load_config('diffusion'))
            return {"xm": xm, "model": model, "diffusion": diffusion}

        model_key = "shap_e_img" if use_image else "shap_e_text"
        result = model_manager.get_model(model_key, loader)
        self._shap_e_model = result["model"]
        self._shap_e_xm = result["xm"]
        self._shap_e_diffusion = result["diffusion"]
        print(f"[OK] Shap-E {'(image)' if use_image else '(text)'} geladen")

    def _load_triposr(self):
        """Load TripoSR model"""
        def loader():
            print("[LOAD] Lade TripoSR...")
            model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.to(self._device)
            return model

        self._triposr = model_manager.get_model("triposr", loader)
        print("[OK] TripoSR geladen")

    def text_to_3d(
        self,
        prompt: str,
        model: str = "shap-e",
        num_steps: int = 64,
        guidance_scale: float = 15.0,
        output_format: str = "glb",
        output_path: str = None,
    ) -> dict:
        """
        Generate 3D model from text prompt.

        Args:
            prompt: Text description of the 3D model
            model: Model to use (shap-e)
            num_steps: Number of diffusion steps (16-128)
            guidance_scale: Guidance scale (1.0-30.0)
            output_format: Output format (glb, obj, ply)
            output_path: Optional output file path

        Returns:
            dict with output_path, prompt, format, model, gpu_seconds

        Raises:
            ValueError: If prompt is empty or invalid parameters
        """
        start_time = time.time()

        # Input validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        prompt = prompt.strip()
        if len(prompt) > 1000:
            raise ValueError("Prompt too long (max 1000 characters)")

        if output_format not in ["glb", "obj", "ply"]:
            raise ValueError(f"Invalid output format: {output_format}. Supported: glb, obj, ply")

        num_steps = max(16, min(128, num_steps))  # Clamp to valid range
        guidance_scale = max(1.0, min(30.0, guidance_scale))

        if not SHAP_E_AVAILABLE:
            raise ValueError("Shap-E library not installed. Install with: pip install shap-e")

        try:
            if self._shap_e_model is None:
                self._load_shap_e(use_image=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load Shap-E model: {e}")

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"3d_{uuid.uuid4().hex[:8]}.{output_format}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate latents
        batch_size = 1
        latents = sample_latents(
            batch_size=batch_size,
            model=self._shap_e_model,
            diffusion=self._shap_e_diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        # Decode to mesh
        try:
            for i, latent in enumerate(latents):
                t = decode_latent_mesh(self._shap_e_xm, latent).tri_mesh()

                if output_format == "glb":
                    with open(output_path, 'wb') as f:
                        t.write_glb(f)
                elif output_format == "obj":
                    with open(output_path, 'w') as f:
                        t.write_obj(f)
                elif output_format == "ply":
                    with open(output_path, 'wb') as f:
                        t.write_ply(f)

            # Verify output file was created
            if not Path(output_path).exists():
                raise RuntimeError("Failed to create output file")

            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                raise RuntimeError("Output file is empty")

        except Exception as e:
            # Cleanup failed output
            if Path(output_path).exists():
                Path(output_path).unlink()
            raise RuntimeError(f"Mesh generation failed: {e}")

        return {
            "output_path": output_path,
            "prompt": prompt,
            "format": output_format,
            "model": model,
            "file_size_bytes": file_size,
            "gpu_seconds": time.time() - start_time,
        }

    def image_to_3d(
        self,
        image: Union[str, Image.Image],
        model: str = "triposr",
        output_format: str = "glb",
        output_path: str = None,
    ) -> dict:
        """
        Generate 3D model from image.

        Args:
            image: Image path or PIL Image object
            model: Model to use (triposr, shap-e-img)
            output_format: Output format (glb, obj, ply)
            output_path: Optional output file path

        Returns:
            dict with output_path, format, model, gpu_seconds

        Raises:
            ValueError: If image is invalid or model unavailable
        """
        start_time = time.time()

        # Validate output format
        if output_format not in ["glb", "obj", "ply"]:
            raise ValueError(f"Invalid output format: {output_format}. Supported: glb, obj, ply")

        # Validate model
        if model not in ["triposr", "shap-e-img"]:
            raise ValueError(f"Invalid model: {model}. Supported: triposr, shap-e-img")

        # Load and validate image
        try:
            if isinstance(image, str):
                if not Path(image).exists():
                    raise ValueError(f"Image file not found: {image}")
                img = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                raise ValueError("Image must be a file path or PIL Image")

            # Validate image size
            if img.size[0] < 64 or img.size[1] < 64:
                raise ValueError("Image too small (minimum 64x64 pixels)")
            if img.size[0] > 4096 or img.size[1] > 4096:
                # Resize large images
                img.thumbnail((4096, 4096), Image.Resampling.LANCZOS)

        except Exception as e:
            if "not found" in str(e) or "too small" in str(e):
                raise
            raise ValueError(f"Failed to load image: {e}")

        if output_path is None:
            output_path = str(Config.DATA_DIR / "generated" / f"3d_{uuid.uuid4().hex[:8]}.{output_format}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if model == "triposr" and TRIPOSR_AVAILABLE:
            if self._triposr is None:
                self._load_triposr()

            with torch.inference_mode():
                scene_codes = self._triposr([img], device=self._device)

            mesh = self._triposr.extract_mesh(scene_codes[0])[0]

            if TRIMESH_AVAILABLE:
                mesh.export(output_path)
            else:
                # Fallback: save as OBJ
                if output_format != "obj":
                    output_path = output_path.replace(f".{output_format}", ".obj")
                mesh.export(output_path)

        elif model == "shap-e-img" and SHAP_E_AVAILABLE:
            if self._shap_e_model is None:
                self._load_shap_e(use_image=True)

            latents = sample_latents(
                batch_size=1,
                model=self._shap_e_model,
                diffusion=self._shap_e_diffusion,
                guidance_scale=3.0,
                model_kwargs=dict(images=[img]),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=64,
            )

            for latent in latents:
                t = decode_latent_mesh(self._shap_e_xm, latent).tri_mesh()
                with open(output_path, 'wb') as f:
                    if output_format == "glb":
                        t.write_glb(f)
                    else:
                        t.write_ply(f)

        else:
            raise ValueError(f"Model {model} not available")

        return {
            "output_path": output_path,
            "format": output_format,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process a 3D generation job"""
        task_type = input_data.get("task", "text_to_3d")

        self.notify_progress(job_id, 0.1, f"Starting {task_type}")

        if task_type == "text_to_3d" or task_type == "text2mesh":
            prompt = input_data.get("prompt")
            model = input_data.get("model", "shap-e")
            output_format = input_data.get("format", "glb")
            result = self.text_to_3d(prompt, model=model, output_format=output_format)

        elif task_type == "image_to_3d" or task_type == "img2mesh":
            image = input_data.get("image") or input_data.get("file")
            model = input_data.get("model", "triposr")
            output_format = input_data.get("format", "glb")
            result = self.image_to_3d(image, model=model, output_format=output_format)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._shap_e_model = None
        self._shap_e_diffusion = None
        self._point_e_model = None
        self._triposr = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] 3D Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        available = {}
        for model_id, info in THREED_MODELS.items():
            if "shap-e" in model_id and SHAP_E_AVAILABLE:
                available[model_id] = info
            elif "point-e" in model_id and POINT_E_AVAILABLE:
                available[model_id] = info
            elif "triposr" in model_id and TRIPOSR_AVAILABLE:
                available[model_id] = info
        return available


# Singleton Instance
_threed_worker: Optional[ThreeDWorker] = None


def get_threed_worker() -> ThreeDWorker:
    """Get singleton instance"""
    global _threed_worker
    if _threed_worker is None:
        _threed_worker = ThreeDWorker()
    return _threed_worker
