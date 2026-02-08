#!/usr/bin/env python3
"""
SCIO - Embedding Worker
Text & Image Embeddings für RAG und Semantic Search
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
except ImportError:
    TORCH_AVAILABLE = False

# Sentence Transformers
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# CLIP for Image Embeddings
CLIP_AVAILABLE = False
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    pass

# PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Vector Database
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass

import numpy as np


# Available Embedding Models
EMBEDDING_MODELS = {
    # Text Embeddings
    'all-MiniLM-L6-v2': {
        'name': 'MiniLM L6',
        'type': 'text',
        'dimensions': 384,
        'vram_gb': 0.5,
    },
    'all-mpnet-base-v2': {
        'name': 'MPNet Base',
        'type': 'text',
        'dimensions': 768,
        'vram_gb': 0.5,
    },
    'bge-large-en-v1.5': {
        'name': 'BGE Large EN',
        'hf_id': 'BAAI/bge-large-en-v1.5',
        'type': 'text',
        'dimensions': 1024,
        'vram_gb': 1,
    },
    'bge-m3': {
        'name': 'BGE M3 (Multilingual)',
        'hf_id': 'BAAI/bge-m3',
        'type': 'text',
        'dimensions': 1024,
        'vram_gb': 2,
    },
    'e5-large-v2': {
        'name': 'E5 Large',
        'hf_id': 'intfloat/e5-large-v2',
        'type': 'text',
        'dimensions': 1024,
        'vram_gb': 1,
    },
    'e5-mistral-7b': {
        'name': 'E5 Mistral 7B',
        'hf_id': 'intfloat/e5-mistral-7b-instruct',
        'type': 'text',
        'dimensions': 4096,
        'vram_gb': 14,
    },
    'gte-large': {
        'name': 'GTE Large',
        'hf_id': 'thenlper/gte-large',
        'type': 'text',
        'dimensions': 1024,
        'vram_gb': 1,
    },
    'nomic-embed-text': {
        'name': 'Nomic Embed Text',
        'hf_id': 'nomic-ai/nomic-embed-text-v1.5',
        'type': 'text',
        'dimensions': 768,
        'vram_gb': 0.5,
    },

    # Image Embeddings
    'clip-vit-base': {
        'name': 'CLIP ViT Base',
        'hf_id': 'openai/clip-vit-base-patch32',
        'type': 'image',
        'dimensions': 512,
        'vram_gb': 1,
    },
    'clip-vit-large': {
        'name': 'CLIP ViT Large',
        'hf_id': 'openai/clip-vit-large-patch14',
        'type': 'image',
        'dimensions': 768,
        'vram_gb': 2,
    },
    'siglip-base': {
        'name': 'SigLIP Base',
        'hf_id': 'google/siglip-base-patch16-224',
        'type': 'image',
        'dimensions': 768,
        'vram_gb': 1,
    },
}


class EmbeddingWorker(BaseWorker):
    """
    Embedding Worker - Handles all embedding tasks

    Features:
    - Text Embeddings
    - Image Embeddings (CLIP)
    - Batch Processing
    - Similarity Search
    - Vector Database Integration (FAISS)
    """

    def __init__(self):
        super().__init__("Embeddings")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._text_model = None
        self._clip_model = None
        self._clip_processor = None
        self._current_model_id = None
        self._vector_stores: Dict[str, Any] = {}

    def initialize(self) -> bool:
        """Initialize the worker"""
        available_features = []

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            available_features.append("Text")
        if CLIP_AVAILABLE:
            available_features.append("CLIP")
        if FAISS_AVAILABLE:
            available_features.append("FAISS")

        if not available_features:
            self._error_message = "No embedding libraries available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] Embedding Worker bereit (Device: {self._device}, Features: {', '.join(available_features)})")
        return True

    def _load_text_model(self, model_id: str):
        """Load text embedding model"""
        model_info = EMBEDDING_MODELS.get(model_id)
        if not model_info or model_info['type'] != 'text':
            model_id = 'bge-large-en-v1.5'
            model_info = EMBEDDING_MODELS[model_id]

        hf_id = model_info.get('hf_id', model_id)

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")
            model = SentenceTransformer(hf_id, device=self._device)
            return model

        self._text_model = model_manager.get_model(f"embed_{hf_id}", loader)
        self._current_model_id = model_id
        print(f"[OK] {model_info['name']} geladen")

    def _load_clip_model(self, model_id: str = "clip-vit-large"):
        """Load CLIP model for image embeddings"""
        model_info = EMBEDDING_MODELS.get(model_id)
        if not model_info:
            model_id = 'clip-vit-large'
            model_info = EMBEDDING_MODELS[model_id]

        hf_id = model_info['hf_id']

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")
            processor = CLIPProcessor.from_pretrained(hf_id)
            model = CLIPModel.from_pretrained(hf_id).to(self._device)
            return {"model": model, "processor": processor}

        result = model_manager.get_model(f"clip_{hf_id}", loader)
        self._clip_model = result["model"]
        self._clip_processor = result["processor"]
        print(f"[OK] {model_info['name']} geladen")

    def embed_text(
        self,
        texts: Union[str, List[str]],
        model: str = "bge-large-en-v1.5",
        normalize: bool = True,
    ) -> dict:
        """Generate embeddings for text"""
        start_time = time.time()

        if self._current_model_id != model or self._text_model is None:
            self._load_text_model(model)

        if isinstance(texts, str):
            texts = [texts]

        # Add prefix for instruction-based models
        if "bge" in model.lower():
            texts = [f"Represent this sentence: {t}" for t in texts]
        elif "e5" in model.lower():
            texts = [f"query: {t}" for t in texts]

        embeddings = self._text_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return {
            "embeddings": embeddings.tolist(),
            "dimensions": embeddings.shape[1],
            "count": len(texts),
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def embed_image(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]],
        model: str = "clip-vit-large",
        normalize: bool = True,
    ) -> dict:
        """Generate embeddings for images"""
        start_time = time.time()

        if self._clip_model is None:
            self._load_clip_model(model)

        # Load images
        if not isinstance(images, list):
            images = [images]

        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        # Process images
        inputs = self._clip_processor(images=pil_images, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            embeddings = self._clip_model.get_image_features(**inputs)

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings_np = embeddings.cpu().numpy()

        return {
            "embeddings": embeddings_np.tolist(),
            "dimensions": embeddings_np.shape[1],
            "count": len(images),
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def embed_text_and_image(
        self,
        text: str,
        image: Union[str, Image.Image],
        model: str = "clip-vit-large",
    ) -> dict:
        """Get both text and image embeddings for comparison"""
        start_time = time.time()

        if self._clip_model is None:
            self._load_clip_model(model)

        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # Process both
        inputs = self._clip_processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._clip_model(**inputs)

            text_embed = F.normalize(outputs.text_embeds, p=2, dim=1)
            image_embed = F.normalize(outputs.image_embeds, p=2, dim=1)

            # Compute similarity
            similarity = (text_embed @ image_embed.T).item()

        return {
            "text_embedding": text_embed.cpu().numpy().tolist()[0],
            "image_embedding": image_embed.cpu().numpy().tolist()[0],
            "similarity": similarity,
            "model": model,
            "gpu_seconds": time.time() - start_time,
        }

    def similarity(
        self,
        embeddings1: List[List[float]],
        embeddings2: List[List[float]],
    ) -> dict:
        """Compute cosine similarity between embeddings"""
        e1 = np.array(embeddings1)
        e2 = np.array(embeddings2)

        # Normalize
        e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)

        # Compute similarity matrix
        similarities = np.dot(e1, e2.T)

        return {
            "similarities": similarities.tolist(),
        }

    def create_index(
        self,
        name: str,
        embeddings: List[List[float]],
        metadata: List[dict] = None,
    ) -> dict:
        """Create a FAISS index for similarity search"""
        if not FAISS_AVAILABLE:
            raise ValueError("FAISS not installed")

        embeddings_np = np.array(embeddings).astype('float32')
        dimension = embeddings_np.shape[1]

        # Create index
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)

        self._vector_stores[name] = {
            "index": index,
            "metadata": metadata or [{}] * len(embeddings),
            "dimension": dimension,
        }

        # Save index
        index_path = Config.DATA_DIR / "indexes" / f"{name}.faiss"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        # Save metadata
        meta_path = Config.DATA_DIR / "indexes" / f"{name}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata or [], f)

        return {
            "name": name,
            "dimension": dimension,
            "count": len(embeddings),
            "path": str(index_path),
        }

    def search(
        self,
        name: str,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> dict:
        """Search for similar embeddings in an index"""
        if name not in self._vector_stores:
            # Try to load from disk
            index_path = Config.DATA_DIR / "indexes" / f"{name}.faiss"
            meta_path = Config.DATA_DIR / "indexes" / f"{name}.json"

            if not index_path.exists():
                raise ValueError(f"Index '{name}' not found")

            index = faiss.read_index(str(index_path))
            metadata = []
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)

            self._vector_stores[name] = {
                "index": index,
                "metadata": metadata,
                "dimension": index.d,
            }

        store = self._vector_stores[name]
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)

        scores, indices = store["index"].search(query, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid result
                result = {
                    "rank": i + 1,
                    "score": float(score),
                    "index": int(idx),
                }
                if idx < len(store["metadata"]):
                    result["metadata"] = store["metadata"][idx]
                results.append(result)

        return {
            "results": results,
            "query_dimension": len(query_embedding),
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process an embedding job"""
        task_type = input_data.get("task", "embed_text")

        self.notify_progress(job_id, 0.1, f"Starting {task_type}")

        if task_type == "embed_text":
            texts = input_data.get("texts") or input_data.get("text")
            model = input_data.get("model", "bge-large-en-v1.5")
            result = self.embed_text(texts, model=model)

        elif task_type == "embed_image":
            images = input_data.get("images") or input_data.get("image")
            model = input_data.get("model", "clip-vit-large")
            result = self.embed_image(images, model=model)

        elif task_type == "similarity":
            e1 = input_data.get("embeddings1")
            e2 = input_data.get("embeddings2")
            result = self.similarity(e1, e2)

        elif task_type == "create_index":
            name = input_data.get("name")
            embeddings = input_data.get("embeddings")
            metadata = input_data.get("metadata")
            result = self.create_index(name, embeddings, metadata)

        elif task_type == "search":
            name = input_data.get("name")
            query = input_data.get("query_embedding")
            top_k = input_data.get("top_k", 10)
            result = self.search(name, query, top_k)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._text_model = None
        self._clip_model = None
        self._clip_processor = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Embedding Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        return EMBEDDING_MODELS


# Singleton Instance
_embedding_worker: Optional[EmbeddingWorker] = None


def get_embedding_worker() -> EmbeddingWorker:
    """Get singleton instance"""
    global _embedding_worker
    if _embedding_worker is None:
        _embedding_worker = EmbeddingWorker()
    return _embedding_worker
