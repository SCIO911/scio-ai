#!/usr/bin/env python3
"""
SCIO - LLM Inference Worker
OpenAI-kompatible Inference-API
Optimiert für RTX 5090 mit 24GB VRAM
"""

import os
import time
import json
import threading
from typing import Optional, Generator, List, Dict, Any

from .base_worker import BaseWorker, WorkerStatus, model_manager
from backend.config import Config

# Try to import inference libraries
try:
    import torch
    TORCH_AVAILABLE = True
    # Setze optimale Thread-Anzahl
    if hasattr(Config, 'TORCH_NUM_THREADS'):
        torch.set_num_threads(Config.TORCH_NUM_THREADS)
    # CUDA Optimierungen
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN]  PyTorch nicht installiert")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import TextIteratorStreamer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN]  Transformers nicht installiert")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Flash Attention Check
FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass


class LLMInferenceWorker(BaseWorker):
    """
    LLM Inference Worker

    Features:
    - OpenAI-kompatible Chat Completions
    - Streaming Support
    - Dynamisches Model-Loading
    - Token-Zählung
    """

    def __init__(self):
        super().__init__("LLM Inference")
        self._current_model_id: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """Initialisiert den Worker"""
        if not TRANSFORMERS_AVAILABLE:
            self._error_message = "Transformers library not available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] LLM Inference Worker bereit (Device: {self._device})")
        return True

    def _load_model(self, model_id: str):
        """Lädt ein Modell - Optimiert für RTX 5090 mit 24GB VRAM"""
        model_info = Config.get_model_info(model_id)
        hf_id = model_info['hf_id']

        def loader():
            print(f"[LOAD] Lade {model_info['name']} mit GPU-Optimierungen...")

            # Quantization nur für sehr große Modelle (>24GB)
            quantization_config = None
            use_quantization = getattr(Config, 'LOAD_IN_4BIT', False) or getattr(Config, 'LOAD_IN_8BIT', False)

            if use_quantization and model_info['vram_gb'] > 20 and self._device == "cuda":
                if getattr(Config, 'LOAD_IN_4BIT', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                elif getattr(Config, 'LOAD_IN_8BIT', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id,
                trust_remote_code=True,
                use_fast=True,  # Schnellerer Tokenizer
            )

            # Optimale Modell-Konfiguration für RTX 5090
            model_kwargs = {
                'device_map': "auto" if self._device == "cuda" else None,
                'torch_dtype': torch.bfloat16 if self._device == "cuda" else torch.float32,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,  # Weniger RAM beim Laden
            }

            # Flash Attention wenn verfügbar
            if FLASH_ATTN_AVAILABLE and getattr(Config, 'USE_FLASH_ATTENTION', True):
                model_kwargs['attn_implementation'] = "flash_attention_2"
                print("[OK] Flash Attention 2 aktiviert")
            elif self._device == "cuda":
                model_kwargs['attn_implementation'] = "sdpa"  # PyTorch SDPA als Fallback
                print("[OK] SDPA Attention aktiviert")

            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config

            # Load model
            model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

            # BetterTransformer für schnellere Inferenz (wenn kein Flash Attention)
            if getattr(Config, 'USE_BETTERTRANSFORMER', True) and not FLASH_ATTN_AVAILABLE:
                try:
                    model = model.to_bettertransformer()
                    print("[OK] BetterTransformer aktiviert")
                except Exception:
                    pass  # Nicht alle Modelle unterstützen BetterTransformer

            if self._device == "cpu":
                model = model.to(self._device)

            # Compile für schnellere Ausführung (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self._device == "cuda":
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    print("[OK] torch.compile aktiviert")
                except Exception:
                    pass

            return {'model': model, 'tokenizer': tokenizer}

        result = model_manager.get_model(hf_id, loader)
        self._model = result['model']
        self._tokenizer = result['tokenizer']
        self._current_model_id = model_id

        print(f"[OK] Modell geladen: {model_info['name']} (VRAM optimiert)")

    def _count_tokens(self, text: str) -> int:
        """Zählt Tokens in Text"""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        elif TIKTOKEN_AVAILABLE:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        else:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

    def _format_messages(self, messages: List[Dict]) -> str:
        """Formatiert Chat-Messages für das Modell"""
        # Default format (Llama-style)
        formatted = ""

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                formatted += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == 'user':
                formatted += f"[INST] {content} [/INST]\n"
            elif role == 'assistant':
                formatted += f"{content}\n"

        return formatted

    def process(self, job_id: str, input_data: dict) -> dict:
        """
        Verarbeitet Inference-Request

        Args:
            job_id: Job identifier
            input_data: Dictionary with:
                - model: Model-ID (optional, default: mistral-7b)
                - messages: Liste von Chat-Messages (required)
                - max_tokens: Max. Ausgabe-Tokens (1-32768)
                - temperature: Sampling-Temperatur (0.0-2.0)
                - top_p: Nucleus Sampling (0.0-1.0)

        Returns:
            dict with response, tokens_input, tokens_output, model, gpu_seconds

        Raises:
            ValueError: If messages are empty or parameters invalid
        """
        start_time = time.time()

        model_id = input_data.get('model', 'mistral-7b')
        messages = input_data.get('messages', [])
        max_tokens = input_data.get('max_tokens', Config.MAX_NEW_TOKENS)
        temperature = input_data.get('temperature', 0.7)
        top_p = input_data.get('top_p', 0.9)

        # Validate messages
        if not messages:
            raise ValueError("No messages provided")

        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if 'role' not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
            if 'content' not in msg:
                raise ValueError(f"Message {i} missing 'content' field")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Message {i} has invalid role: {msg['role']}")

        # Validate and clamp numerical parameters
        max_tokens = max(1, min(32768, max_tokens))
        temperature = max(0.0, min(2.0, temperature))
        top_p = max(0.0, min(1.0, top_p))

        # Load model if needed
        if self._current_model_id != model_id or self._model is None:
            self._load_model(model_id)

        self.notify_progress(job_id, 0.1, "Model loaded")

        # Format input
        if hasattr(self._tokenizer, 'apply_chat_template'):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self._format_messages(messages)

        # Count input tokens
        input_tokens = self._count_tokens(prompt)

        self.notify_progress(job_id, 0.2, "Generating response")

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_CONTEXT_LENGTH - max_tokens,
        ).to(self._device)

        # Generate
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Count output tokens
        output_tokens = len(generated_ids)

        self.notify_progress(job_id, 1.0, "Complete")

        end_time = time.time()

        return {
            'response': response_text,
            'tokens_input': input_tokens,
            'tokens_output': output_tokens,
            'model': model_id,
            'gpu_seconds': end_time - start_time,
            'finish_reason': 'stop',
        }

    def generate_stream(
        self,
        messages: List[Dict],
        model_id: str = None,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, Dict]:
        """
        Streaming-Generation

        Yields:
            Generierte Text-Chunks

        Returns:
            Final statistics dict
        """
        model_id = model_id or 'mistral-7b'
        max_tokens = max_tokens or Config.MAX_NEW_TOKENS

        # Load model if needed
        if self._current_model_id != model_id or self._model is None:
            self._load_model(model_id)

        # Format input
        if hasattr(self._tokenizer, 'apply_chat_template'):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self._format_messages(messages)

        input_tokens = self._count_tokens(prompt)

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_CONTEXT_LENGTH - max_tokens,
        ).to(self._device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation in separate thread
        generation_kwargs = {
            **inputs,
            'max_new_tokens': max_tokens,
            'temperature': temperature if temperature > 0 else 1.0,
            'top_p': top_p,
            'do_sample': temperature > 0,
            'streamer': streamer,
            'pad_token_id': self._tokenizer.eos_token_id,
        }

        thread = threading.Thread(
            target=lambda: self._model.generate(**generation_kwargs)
        )
        thread.start()

        # Yield generated text
        output_text = ""

        for text in streamer:
            output_text += text
            yield text

        thread.join()

        # Count actual output tokens
        output_tokens = self._count_tokens(output_text)

        # Return final stats
        return {
            'tokens_input': input_tokens,
            'tokens_output': output_tokens,
            'model': model_id,
            'finish_reason': 'stop',
        }

    def cleanup(self):
        """Gibt Ressourcen frei"""
        self._model = None
        self._tokenizer = None
        self._current_model_id = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 LLM Inference Worker bereinigt")


# Singleton Instance
_inference_worker: Optional[LLMInferenceWorker] = None


def get_inference_worker() -> LLMInferenceWorker:
    """Gibt Singleton-Instanz zurück"""
    global _inference_worker
    if _inference_worker is None:
        _inference_worker = LLMInferenceWorker()
    return _inference_worker
