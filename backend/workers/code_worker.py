#!/usr/bin/env python3
"""
SCIO - Code Worker
Code Generation, Completion, Analysis
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

# Transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Available Code Models
CODE_MODELS = {
    'deepseek-coder-1.3b': {
        'name': 'DeepSeek Coder 1.3B',
        'hf_id': 'deepseek-ai/deepseek-coder-1.3b-instruct',
        'vram_gb': 4,
        'context_length': 16384,
    },
    'deepseek-coder-6.7b': {
        'name': 'DeepSeek Coder 6.7B',
        'hf_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'vram_gb': 14,
        'context_length': 16384,
    },
    'deepseek-coder-33b': {
        'name': 'DeepSeek Coder 33B',
        'hf_id': 'deepseek-ai/deepseek-coder-33b-instruct',
        'vram_gb': 24,
        'context_length': 16384,
    },
    'codellama-7b': {
        'name': 'Code Llama 7B',
        'hf_id': 'codellama/CodeLlama-7b-Instruct-hf',
        'vram_gb': 14,
        'context_length': 16384,
    },
    'codellama-13b': {
        'name': 'Code Llama 13B',
        'hf_id': 'codellama/CodeLlama-13b-Instruct-hf',
        'vram_gb': 26,
        'context_length': 16384,
    },
    'codellama-34b': {
        'name': 'Code Llama 34B',
        'hf_id': 'codellama/CodeLlama-34b-Instruct-hf',
        'vram_gb': 40,
        'context_length': 16384,
    },
    'starcoder2-3b': {
        'name': 'StarCoder2 3B',
        'hf_id': 'bigcode/starcoder2-3b',
        'vram_gb': 8,
        'context_length': 16384,
    },
    'starcoder2-7b': {
        'name': 'StarCoder2 7B',
        'hf_id': 'bigcode/starcoder2-7b',
        'vram_gb': 14,
        'context_length': 16384,
    },
    'starcoder2-15b': {
        'name': 'StarCoder2 15B',
        'hf_id': 'bigcode/starcoder2-15b',
        'vram_gb': 24,
        'context_length': 16384,
    },
    'codegemma-7b': {
        'name': 'CodeGemma 7B',
        'hf_id': 'google/codegemma-7b-it',
        'vram_gb': 14,
        'context_length': 8192,
    },
    'qwen2.5-coder-7b': {
        'name': 'Qwen2.5 Coder 7B',
        'hf_id': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'vram_gb': 14,
        'context_length': 131072,
    },
}


class CodeWorker(BaseWorker):
    """
    Code Worker - Handles all code AI tasks

    Features:
    - Code Generation
    - Code Completion
    - Code Explanation
    - Code Review
    - Bug Fixing
    - Code Translation
    """

    def __init__(self):
        super().__init__("Code Generation")
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None
        self._current_model_id = None

    def initialize(self) -> bool:
        """Initialize the worker"""
        if not TRANSFORMERS_AVAILABLE:
            self._error_message = "Transformers library not available"
            self.status = WorkerStatus.ERROR
            return False

        self.status = WorkerStatus.READY
        print(f"[OK] Code Worker bereit (Device: {self._device})")
        return True

    def _load_model(self, model_id: str):
        """Load a code model"""
        if model_id not in CODE_MODELS:
            model_id = "deepseek-coder-6.7b"  # Default

        model_info = CODE_MODELS[model_id]
        hf_id = model_info['hf_id']

        def loader():
            print(f"[LOAD] Lade {model_info['name']}...")

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )

            return {"model": model, "tokenizer": tokenizer}

        result = model_manager.get_model(hf_id, loader)
        self._model = result["model"]
        self._tokenizer = result["tokenizer"]
        self._current_model_id = model_id
        print(f"[OK] {model_info['name']} geladen")

    def generate_code(
        self,
        prompt: str,
        model: str = "deepseek-coder-6.7b",
        language: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> dict:
        """
        Generate code from prompt.

        Args:
            prompt: Code generation prompt (required, max 50000 chars)
            model: Model ID (default: deepseek-coder-6.7b)
            language: Optional programming language hint
            max_tokens: Maximum tokens to generate (1-8192)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Top-p sampling (0.0-1.0)

        Returns:
            dict with code, model, language, tokens_generated, gpu_seconds

        Raises:
            ValueError: If prompt is empty or parameters invalid
        """
        start_time = time.time()

        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required")

        prompt = prompt.strip()
        if len(prompt) > 50000:
            raise ValueError("Prompt too long (max 50000 characters)")

        # Validate model
        if model not in CODE_MODELS:
            available = list(CODE_MODELS.keys())
            raise ValueError(f"Unknown model: {model}. Available: {available}")

        # Validate and clamp numerical parameters
        max_tokens = max(1, min(8192, max_tokens))
        temperature = max(0.0, min(2.0, temperature))
        top_p = max(0.0, min(1.0, top_p))

        if self._current_model_id != model or self._model is None:
            self._load_model(model)

        # Format prompt based on model
        if "deepseek" in model.lower():
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        elif "codellama" in model.lower():
            formatted_prompt = f"[INST] {prompt} [/INST]"
        elif "starcoder" in model.lower():
            formatted_prompt = f"<fim_prefix>{prompt}<fim_suffix><fim_middle>"
        else:
            formatted_prompt = prompt

        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return {
            "code": generated.strip(),
            "model": model,
            "language": language,
            "tokens_generated": len(outputs[0]) - inputs['input_ids'].shape[1],
            "gpu_seconds": time.time() - start_time,
        }

    def complete_code(
        self,
        code: str,
        cursor_position: int = None,
        model: str = "deepseek-coder-6.7b",
        max_tokens: int = 256,
    ) -> dict:
        """Complete code at cursor position"""
        if cursor_position is None:
            cursor_position = len(code)

        prefix = code[:cursor_position]
        suffix = code[cursor_position:]

        prompt = f"Complete the following code:\n\n```\n{prefix}"

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=0.1,
        )

        return {
            "completion": result["code"],
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    def explain_code(
        self,
        code: str,
        model: str = "deepseek-coder-6.7b",
        language: str = None,
    ) -> dict:
        """Explain what code does"""
        lang_hint = f" ({language})" if language else ""
        prompt = f"Explain the following code{lang_hint} in detail:\n\n```\n{code}\n```\n\nExplanation:"

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=1024,
            temperature=0.3,
        )

        return {
            "explanation": result["code"],
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    def review_code(
        self,
        code: str,
        model: str = "deepseek-coder-6.7b",
        language: str = None,
    ) -> dict:
        """Review code for issues and improvements"""
        lang_hint = f" ({language})" if language else ""
        prompt = f"""Review the following code{lang_hint} for:
1. Bugs and errors
2. Security issues
3. Performance improvements
4. Code style and best practices

Code:
```
{code}
```

Code Review:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=2048,
            temperature=0.3,
        )

        return {
            "review": result["code"],
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    def fix_code(
        self,
        code: str,
        error_message: str = None,
        model: str = "deepseek-coder-6.7b",
        language: str = None,
    ) -> dict:
        """Fix bugs in code"""
        error_info = f"\n\nError message: {error_message}" if error_message else ""
        lang_hint = f" ({language})" if language else ""

        prompt = f"""Fix the bugs in the following code{lang_hint}:{error_info}

```
{code}
```

Fixed code:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=2048,
            temperature=0.1,
        )

        return {
            "fixed_code": result["code"],
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    def translate_code(
        self,
        code: str,
        source_language: str,
        target_language: str,
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """Translate code from one language to another"""
        prompt = f"""Translate the following {source_language} code to {target_language}:

```{source_language}
{code}
```

{target_language} code:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=2048,
            temperature=0.1,
        )

        return {
            "translated_code": result["code"],
            "source_language": source_language,
            "target_language": target_language,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    def process(self, job_id: str, input_data: dict) -> dict:
        """Process a code job"""
        task_type = input_data.get("task", "generate")
        model = input_data.get("model", "deepseek-coder-6.7b")

        self.notify_progress(job_id, 0.1, f"Starting {task_type}")

        if task_type == "generate":
            prompt = input_data.get("prompt")
            language = input_data.get("language")
            max_tokens = input_data.get("max_tokens", 2048)
            result = self.generate_code(prompt, model=model, language=language, max_tokens=max_tokens)

        elif task_type == "complete":
            code = input_data.get("code")
            cursor_position = input_data.get("cursor_position")
            result = self.complete_code(code, cursor_position, model=model)

        elif task_type == "explain":
            code = input_data.get("code")
            language = input_data.get("language")
            result = self.explain_code(code, model=model, language=language)

        elif task_type == "review":
            code = input_data.get("code")
            language = input_data.get("language")
            result = self.review_code(code, model=model, language=language)

        elif task_type == "fix":
            code = input_data.get("code")
            error_message = input_data.get("error")
            language = input_data.get("language")
            result = self.fix_code(code, error_message, model=model, language=language)

        elif task_type == "translate":
            code = input_data.get("code")
            source_language = input_data.get("source_language")
            target_language = input_data.get("target_language")
            result = self.translate_code(code, source_language, target_language, model=model)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.notify_progress(job_id, 1.0, "Complete")
        return result

    def cleanup(self):
        """Release resources"""
        self._model = None
        self._tokenizer = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] Code Worker bereinigt")

    def get_available_models(self) -> dict:
        """Return available models"""
        return CODE_MODELS


# Singleton Instance
_code_worker: Optional[CodeWorker] = None


def get_code_worker() -> CodeWorker:
    """Get singleton instance"""
    global _code_worker
    if _code_worker is None:
        _code_worker = CodeWorker()
    return _code_worker
