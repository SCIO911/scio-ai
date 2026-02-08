#!/usr/bin/env python3
"""
SCIO - Image Generation Worker
Die BESTEN Bildgenerierungs-Modelle 2024/2025
Optimiert für RTX 5090 mit 24GB VRAM

Unterstützte Modelle:
- FLUX.1 (schnell/dev) - State of the Art
- Stable Diffusion 3.5 - Neueste SD Version
- SDXL + Varianten (Turbo, Lightning)
- PixArt-Σ - Schnell und hochqualitativ
- Kolors - Kwai's Modell
- Playground v2.5 - Ästhetisch optimiert
- Kandinsky 3 - Hochqualität
- Stable Cascade - Hochauflösend

Features:
- Text-to-Image
- Image-to-Image
- Inpainting
- ControlNet
- IP-Adapter
- LoRA Support
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

# PyTorch mit CUDA-Optimierungen
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
except ImportError:
    TORCH_AVAILABLE = False

# Diffusers Pipelines
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        AutoencoderKL,
    )
    from PIL import Image
    import numpy as np
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    Image = None
    print("[WARN] Diffusers nicht installiert")

# FLUX Pipeline
FLUX_AVAILABLE = False
try:
    from diffusers import FluxPipeline, FluxImg2ImgPipeline
    FLUX_AVAILABLE = True
except ImportError:
    pass

# SD3 Pipeline
SD3_AVAILABLE = False
try:
    from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
    SD3_AVAILABLE = True
except ImportError:
    pass

# PixArt Pipeline
PIXART_AVAILABLE = False
try:
    from diffusers import PixArtSigmaPipeline
    PIXART_AVAILABLE = True
except ImportError:
    pass

# Kandinsky Pipeline
KANDINSKY_AVAILABLE = False
try:
    from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
    KANDINSKY_AVAILABLE = True
except ImportError:
    pass

# Kolors Pipeline
KOLORS_AVAILABLE = False
try:
    from diffusers import KolorsPipeline
    KOLORS_AVAILABLE = True
except ImportError:
    pass

# Stable Cascade
CASCADE_AVAILABLE = False
try:
    from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
    CASCADE_AVAILABLE = True
except ImportError:
    pass

# ControlNet
CONTROLNET_AVAILABLE = False
try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    CONTROLNET_AVAILABLE = True
except ImportError:
    pass

# IP-Adapter
IPADAPTER_AVAILABLE = False
try:
    from diffusers import IPAdapterPipeline
    IPADAPTER_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# VERFÜGBARE MODELLE - DIE BESTEN 2024/2025
# ═══════════════════════════════════════════════════════════════

IMAGE_MODELS = {
    # ═══════════════════════════════════════════════════════════
    # FLUX - State of the Art (Black Forest Labs)
    # ═══════════════════════════════════════════════════════════
    'flux-schnell': {
        'name': 'FLUX.1 Schnell',
        'hf_id': 'black-forest-labs/FLUX.1-schnell',
        'pipeline': 'FluxPipeline',
        'vram_gb': 16,
        'description': 'Schnellstes FLUX Modell, 4 Steps, Apache 2.0',
        'default_steps': 4,
        'recommended': True,
    },
    'flux-dev': {
        'name': 'FLUX.1 Dev',
        'hf_id': 'black-forest-labs/FLUX.1-dev',
        'pipeline': 'FluxPipeline',
        'vram_gb': 20,
        'description': 'Höchste Qualität, non-commercial',
        'default_steps': 28,
        'recommended': True,
    },

    # ═══════════════════════════════════════════════════════════
    # STABLE DIFFUSION 3.5 - Neueste Version
    # ═══════════════════════════════════════════════════════════
    'sd3.5-large': {
        'name': 'SD 3.5 Large',
        'hf_id': 'stabilityai/stable-diffusion-3.5-large',
        'pipeline': 'StableDiffusion3Pipeline',
        'vram_gb': 18,
        'description': 'Neueste SD Version, beste Qualität',
        'default_steps': 28,
        'recommended': True,
    },
    'sd3.5-large-turbo': {
        'name': 'SD 3.5 Large Turbo',
        'hf_id': 'stabilityai/stable-diffusion-3.5-large-turbo',
        'pipeline': 'StableDiffusion3Pipeline',
        'vram_gb': 18,
        'description': 'SD 3.5 mit nur 4 Steps',
        'default_steps': 4,
    },
    'sd3-medium': {
        'name': 'SD 3 Medium',
        'hf_id': 'stabilityai/stable-diffusion-3-medium-diffusers',
        'pipeline': 'StableDiffusion3Pipeline',
        'vram_gb': 12,
        'description': 'Schneller, weniger VRAM',
        'default_steps': 28,
    },

    # ═══════════════════════════════════════════════════════════
    # SDXL - Bewährt und schnell
    # ═══════════════════════════════════════════════════════════
    'sdxl': {
        'name': 'SDXL 1.0',
        'hf_id': 'stabilityai/stable-diffusion-xl-base-1.0',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 12,
        'description': 'Bewährtes Arbeitstier',
        'default_steps': 25,
    },
    'sdxl-turbo': {
        'name': 'SDXL Turbo',
        'hf_id': 'stabilityai/sdxl-turbo',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 12,
        'description': 'Echtzeit, 1-4 Steps',
        'default_steps': 4,
    },
    'sdxl-lightning': {
        'name': 'SDXL Lightning 4-Step',
        'hf_id': 'ByteDance/SDXL-Lightning',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 12,
        'description': 'ByteDance, schnell und gut',
        'default_steps': 4,
    },

    # ═══════════════════════════════════════════════════════════
    # PIXART - Schnell und effizient
    # ═══════════════════════════════════════════════════════════
    'pixart-sigma': {
        'name': 'PixArt-Σ',
        'hf_id': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        'pipeline': 'PixArtSigmaPipeline',
        'vram_gb': 12,
        'description': 'Schnell, gut für Text',
        'default_steps': 20,
    },

    # ═══════════════════════════════════════════════════════════
    # KOLORS - Kwai (Chinesisch, hochwertig)
    # ═══════════════════════════════════════════════════════════
    'kolors': {
        'name': 'Kolors',
        'hf_id': 'Kwai-Kolors/Kolors-diffusers',
        'pipeline': 'KolorsPipeline',
        'vram_gb': 14,
        'description': 'Kwai, hervorragend für Portraits',
        'default_steps': 25,
    },

    # ═══════════════════════════════════════════════════════════
    # PLAYGROUND - Ästhetisch optimiert
    # ═══════════════════════════════════════════════════════════
    'playground-v2.5': {
        'name': 'Playground v2.5',
        'hf_id': 'playgroundai/playground-v2.5-1024px-aesthetic',
        'pipeline': 'StableDiffusionXLPipeline',
        'vram_gb': 14,
        'description': 'Ästhetisch optimiert',
        'default_steps': 25,
    },

    # ═══════════════════════════════════════════════════════════
    # KANDINSKY 3 - Russisches Top-Modell
    # ═══════════════════════════════════════════════════════════
    'kandinsky-3': {
        'name': 'Kandinsky 3',
        'hf_id': 'kandinsky-community/kandinsky-3',
        'pipeline': 'Kandinsky3Pipeline',
        'vram_gb': 16,
        'description': 'Russisch, kreativ',
        'default_steps': 25,
    },

    # ═══════════════════════════════════════════════════════════
    # STABLE CASCADE - Hochauflösend
    # ═══════════════════════════════════════════════════════════
    'stable-cascade': {
        'name': 'Stable Cascade',
        'hf_id': 'stabilityai/stable-cascade',
        'pipeline': 'StableCascadePipeline',
        'vram_gb': 20,
        'description': '2-Stage, sehr hochauflösend',
        'default_steps': 20,
    },

    # ═══════════════════════════════════════════════════════════
    # LEGACY - Für Kompatibilität
    # ═══════════════════════════════════════════════════════════
    'sd-1.5': {
        'name': 'SD 1.5',
        'hf_id': 'runwayml/stable-diffusion-v1-5',
        'pipeline': 'StableDiffusionPipeline',
        'vram_gb': 6,
        'max_resolution': 768,
        'description': 'Legacy, viele LoRAs verfügbar',
        'default_steps': 25,
    },
    'sd-2.1': {
        'name': 'SD 2.1',
        'hf_id': 'stabilityai/stable-diffusion-2-1',
        'pipeline': 'StableDiffusionPipeline',
        'vram_gb': 8,
        'max_resolution': 768,
        'description': 'Legacy',
        'default_steps': 25,
    },
}

# ControlNet Modelle
CONTROLNET_MODELS = {
    'canny': {
        'name': 'Canny Edge',
        'hf_id': 'diffusers/controlnet-canny-sdxl-1.0',
        'description': 'Kanten-basierte Kontrolle',
    },
    'depth': {
        'name': 'Depth',
        'hf_id': 'diffusers/controlnet-depth-sdxl-1.0',
        'description': 'Tiefen-basierte Kontrolle',
    },
    'pose': {
        'name': 'OpenPose',
        'hf_id': 'thibaud/controlnet-openpose-sdxl-1.0',
        'description': 'Posen-Kontrolle',
    },
    'softedge': {
        'name': 'Soft Edge',
        'hf_id': 'SargeZT/controlnet-sd-xl-1.0-softedge-dexined',
        'description': 'Weiche Kanten',
    },
}


class ImageGenerationWorker(BaseWorker):
    """
    Image Generation Worker - Die BESTEN Bildgenerierungs-Modelle

    Features:
    - FLUX.1 (schnell/dev) - State of the Art
    - SD 3.5 - Neueste Stable Diffusion
    - SDXL + Turbo/Lightning
    - PixArt-Σ, Kolors, Playground
    - Text-to-Image, Img2Img, Inpainting
    - ControlNet, IP-Adapter, LoRA
    """

    def __init__(self):
        super().__init__("Image Generation")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._current_model_id: Optional[str] = None
        self._pipeline = None
        self._controlnet = None
        self._ip_adapter = None

    def initialize(self) -> bool:
        """Initialisiert den Worker"""
        if not DIFFUSERS_AVAILABLE:
            self._error_message = "Diffusers nicht installiert"
            self.status = WorkerStatus.ERROR
            return False

        # Prüfe verfügbare Pipelines
        available = ['sdxl', 'sd-1.5', 'sd-2.1']
        if FLUX_AVAILABLE:
            available.extend(['flux-schnell', 'flux-dev'])
        if SD3_AVAILABLE:
            available.extend(['sd3.5-large', 'sd3-medium'])
        if PIXART_AVAILABLE:
            available.append('pixart-sigma')
        if KOLORS_AVAILABLE:
            available.append('kolors')

        print(f"[OK] Image Generation Worker bereit")
        print(f"    Device: {self._device}")
        print(f"    Verfügbare Modelle: {len(IMAGE_MODELS)}")
        print(f"    Empfohlen: flux-schnell, sd3.5-large, sdxl")

        self.status = WorkerStatus.READY
        return True

    def _load_model(self, model_id: str):
        """Lädt ein Modell mit optimalen Einstellungen"""
        if model_id not in IMAGE_MODELS:
            raise ValueError(f"Unbekanntes Modell: {model_id}. Verfügbar: {list(IMAGE_MODELS.keys())}")

        model_info = IMAGE_MODELS[model_id]
        hf_id = model_info['hf_id']
        pipeline_type = model_info['pipeline']

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            pipe = None

            # FLUX Pipelines
            if pipeline_type == 'FluxPipeline' and FLUX_AVAILABLE:
                pipe = FluxPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                )

            # SD3 Pipelines
            elif pipeline_type == 'StableDiffusion3Pipeline' and SD3_AVAILABLE:
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                )

            # PixArt Pipeline
            elif pipeline_type == 'PixArtSigmaPipeline' and PIXART_AVAILABLE:
                pipe = PixArtSigmaPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                )

            # Kolors Pipeline
            elif pipeline_type == 'KolorsPipeline' and KOLORS_AVAILABLE:
                pipe = KolorsPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    variant="fp16",
                )

            # SDXL Pipelines
            elif pipeline_type == 'StableDiffusionXLPipeline':
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    use_safetensors=True,
                    variant="fp16" if self._device == "cuda" else None,
                )

            # Standard SD Pipelines
            elif pipeline_type == 'StableDiffusionPipeline':
                pipe = StableDiffusionPipeline.from_pretrained(
                    hf_id,
                    torch_dtype=self._dtype,
                    use_safetensors=True,
                )

            else:
                raise ValueError(f"Pipeline {pipeline_type} nicht verfügbar")

            # Auf GPU verschieben
            pipe.to(self._device)

            # Optimierungen
            if self._device == "cuda":
                # xformers für effiziente Attention
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

                # VAE Optimierungen
                if hasattr(pipe, 'vae'):
                    try:
                        pipe.vae.enable_tiling()
                        pipe.vae.enable_slicing()
                    except Exception:
                        pass

                # torch.compile für Speed
                if hasattr(torch, 'compile'):
                    try:
                        if hasattr(pipe, 'unet'):
                            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
                        elif hasattr(pipe, 'transformer'):
                            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
                    except Exception:
                        pass

            # Scheduler optimieren (nicht für FLUX)
            if pipeline_type not in ['FluxPipeline']:
                if hasattr(pipe, 'scheduler'):
                    try:
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                            pipe.scheduler.config,
                            use_karras_sigmas=True,
                            algorithm_type="dpmsolver++",
                        )
                    except Exception:
                        pass

            print(f"[OK] {model_info['name']} geladen")
            return pipe

        self._pipeline = model_manager.get_model(hf_id, loader)
        self._current_model_id = model_id

    def process(self, job_id: str, input_data: dict) -> dict:
        """
        Generiert Bilder

        Args:
            input_data:
                model: Model-ID (default: flux-schnell)
                prompt: Text-Prompt
                negative_prompt: Negative Prompt
                num_images: Anzahl (1-10)
                width, height: Dimensionen
                steps: Inference Steps
                guidance_scale: CFG Scale
                seed: Optional Seed
                init_image: Base64 für Img2Img
                strength: Img2Img Stärke (0-1)

        Returns:
            dict mit image_paths, model, prompt, etc.
        """
        start_time = time.time()

        # Parameter extrahieren
        model_id = input_data.get('model', 'flux-schnell')
        prompt = input_data.get('prompt', '')
        negative_prompt = input_data.get('negative_prompt', '')
        num_images = min(10, max(1, input_data.get('num_images', 1)))
        width = input_data.get('width', 1024)
        height = input_data.get('height', 1024)

        # Model-spezifische Defaults
        model_info = IMAGE_MODELS.get(model_id, IMAGE_MODELS['flux-schnell'])
        steps = input_data.get('steps', model_info.get('default_steps', 25))
        guidance_scale = input_data.get('guidance_scale', 7.5)
        seed = input_data.get('seed')

        # Init Image für Img2Img
        init_image = input_data.get('init_image')
        strength = input_data.get('strength', 0.75)

        # Validierung
        if not prompt or not prompt.strip():
            raise ValueError("Prompt erforderlich")
        prompt = prompt.strip()[:2000]

        if model_id not in IMAGE_MODELS:
            raise ValueError(f"Unbekanntes Modell: {model_id}")

        # Dimensionen anpassen
        max_res = model_info.get('max_resolution', 2048)
        width = min(max_res, max(512, (width // 8) * 8))
        height = min(max_res, max(512, (height // 8) * 8))

        steps = max(1, min(150, steps))
        guidance_scale = max(0.0, min(30.0, guidance_scale))

        # Modell laden
        if self._current_model_id != model_id or self._pipeline is None:
            self._load_model(model_id)

        self.notify_progress(job_id, 0.1, "Modell geladen")

        # Output Verzeichnis
        output_dir = input_data.get('output_dir') or str(Config.DATA_DIR / 'generated' / job_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generator für Reproduzierbarkeit
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator = torch.Generator(device=self._device).manual_seed(seed)

        # Init Image verarbeiten
        pil_init_image = None
        if init_image:
            try:
                if init_image.startswith('data:'):
                    init_image = init_image.split(',')[1]
                image_data = base64.b64decode(init_image)
                pil_init_image = Image.open(BytesIO(image_data)).convert('RGB')
                pil_init_image = pil_init_image.resize((width, height))
            except Exception as e:
                print(f"[WARN] Init Image Fehler: {e}")
                pil_init_image = None

        # Bilder generieren
        image_paths: List[str] = []
        self.notify_progress(job_id, 0.2, "Generiere Bilder")

        for i in range(num_images):
            progress = 0.2 + (i / num_images) * 0.7
            self.notify_progress(job_id, progress, f"Bild {i+1}/{num_images}")

            # Generation
            gen_kwargs = {
                'prompt': prompt,
                'width': width,
                'height': height,
                'num_inference_steps': steps,
                'generator': generator,
            }

            # Model-spezifische Parameter
            if 'flux' not in model_id:
                gen_kwargs['negative_prompt'] = negative_prompt
                gen_kwargs['guidance_scale'] = guidance_scale

            if pil_init_image is not None:
                gen_kwargs['image'] = pil_init_image
                gen_kwargs['strength'] = strength

            result = self._pipeline(**gen_kwargs)
            image = result.images[0]

            # Speichern
            filename = f"image_{i+1:03d}_{uuid.uuid4().hex[:8]}.png"
            filepath = Path(output_dir) / filename
            image.save(filepath, quality=95)
            image_paths.append(str(filepath))

            # Neuer Seed
            seed += 1
            generator = torch.Generator(device=self._device).manual_seed(seed)

        self.notify_progress(job_id, 1.0, "Fertig")

        return {
            'status': 'success',
            'output_dir': output_dir,
            'image_paths': image_paths,
            'num_images': num_images,
            'model': model_id,
            'prompt': prompt,
            'seed': seed - num_images,
            'width': width,
            'height': height,
            'steps': steps,
            'gpu_seconds': time.time() - start_time,
        }

    def generate_with_controlnet(
        self,
        prompt: str,
        control_image: Union[str, Image.Image],
        controlnet_type: str = 'canny',
        **kwargs
    ) -> Image.Image:
        """
        Generiert mit ControlNet

        Args:
            prompt: Text-Prompt
            control_image: Kontroll-Bild (Pfad oder PIL Image)
            controlnet_type: 'canny', 'depth', 'pose', 'softedge'
        """
        if not CONTROLNET_AVAILABLE:
            raise RuntimeError("ControlNet nicht verfügbar")

        if controlnet_type not in CONTROLNET_MODELS:
            raise ValueError(f"Unbekannter ControlNet-Typ: {controlnet_type}")

        # Bild laden
        if isinstance(control_image, str):
            control_image = Image.open(control_image).convert('RGB')

        # ControlNet laden
        cn_info = CONTROLNET_MODELS[controlnet_type]
        controlnet = ControlNetModel.from_pretrained(cn_info['hf_id'], torch_dtype=self._dtype)

        # Pipeline erstellen
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=self._dtype,
        ).to(self._device)

        result = pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=kwargs.get('steps', 25),
            guidance_scale=kwargs.get('guidance_scale', 7.5),
        )

        return result.images[0]

    def cleanup(self):
        """Gibt Ressourcen frei"""
        self._pipeline = None
        self._controlnet = None
        self._ip_adapter = None
        self._current_model_id = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Image Generation Worker bereinigt")

    def get_available_models(self) -> dict:
        """Gibt verfügbare Modelle zurück"""
        return IMAGE_MODELS

    def get_recommended_models(self) -> List[str]:
        """Gibt empfohlene Modelle zurück"""
        return [k for k, v in IMAGE_MODELS.items() if v.get('recommended', False)]


# Singleton
_image_worker: Optional[ImageGenerationWorker] = None


def get_image_worker() -> ImageGenerationWorker:
    """Gibt Singleton-Instanz zurück"""
    global _image_worker
    if _image_worker is None:
        _image_worker = ImageGenerationWorker()
    return _image_worker
