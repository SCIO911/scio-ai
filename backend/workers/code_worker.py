#!/usr/bin/env python3
"""
SCIO - Code Worker (MEGA-UPGRADE v2.0)
Code Generation, Completion, Analysis
Optimiert für RTX 5090 mit 24GB VRAM

MEGA-UPGRADE Features:
- Unit Test Generation (pytest, jest, etc.)
- Documentation Generation (docstrings, JSDoc, etc.)
- Security Analysis (Vulnerabilities)
- Refactoring Suggestions
- Type Hints Generation
- Complexity Analysis
- Multi-File Context
- Repository-Aware Code Review
- Linting Integration
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

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: UNIT TEST GENERATION
    # ═══════════════════════════════════════════════════════════════

    def generate_unit_tests(
        self,
        code: str,
        language: str = "python",
        framework: str = "pytest",
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Generiert Unit Tests für Code

        Args:
            code: Zu testender Code
            language: Programmiersprache
            framework: Test-Framework (pytest, unittest, jest, mocha, etc.)
            model: Code-Modell

        Returns:
            dict mit tests, coverage_estimate, model, gpu_seconds
        """
        framework_hints = {
            'pytest': 'Use pytest with fixtures and parametrize decorators where appropriate.',
            'unittest': 'Use Python unittest.TestCase class with setUp/tearDown methods.',
            'jest': 'Use Jest with describe/it blocks and expect assertions.',
            'mocha': 'Use Mocha with Chai assertions.',
            'rspec': 'Use RSpec with describe/context/it blocks.',
            'junit': 'Use JUnit 5 with @Test annotations.',
        }

        hint = framework_hints.get(framework, '')

        prompt = f"""Generate comprehensive unit tests for the following {language} code.
Use the {framework} testing framework.
{hint}

Include:
1. Test cases for normal operation
2. Edge cases and boundary conditions
3. Error handling tests
4. Mock external dependencies if needed

Code to test:
```{language}
{code}
```

Generate complete, runnable test code:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            language=language,
            max_tokens=4096,
            temperature=0.2,
        )

        return {
            "tests": result["code"],
            "framework": framework,
            "language": language,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: DOCUMENTATION GENERATION
    # ═══════════════════════════════════════════════════════════════

    def generate_documentation(
        self,
        code: str,
        style: str = "google",
        language: str = "python",
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Generiert Dokumentation für Code

        Args:
            code: Code der dokumentiert werden soll
            style: Dokumentationsstil (google, numpy, sphinx, jsdoc, etc.)
            language: Programmiersprache
            model: Code-Modell

        Returns:
            dict mit documented_code, model, gpu_seconds
        """
        style_hints = {
            'google': 'Use Google-style docstrings with Args, Returns, Raises sections.',
            'numpy': 'Use NumPy-style docstrings with Parameters, Returns, Examples sections.',
            'sphinx': 'Use Sphinx/reST-style docstrings with :param:, :returns:, :raises: directives.',
            'jsdoc': 'Use JSDoc comments with @param, @returns, @throws tags.',
            'javadoc': 'Use Javadoc comments with @param, @return, @throws tags.',
        }

        hint = style_hints.get(style, '')

        prompt = f"""Add comprehensive documentation to the following {language} code.
{hint}

Include:
1. Module/class-level documentation
2. Function/method docstrings with all parameters
3. Type hints where applicable
4. Usage examples where helpful
5. Inline comments for complex logic

Code:
```{language}
{code}
```

Return the fully documented code:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            language=language,
            max_tokens=4096,
            temperature=0.2,
        )

        return {
            "documented_code": result["code"],
            "style": style,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: SECURITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    def analyze_security(
        self,
        code: str,
        language: str = None,
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Analysiert Code auf Sicherheitslücken

        Args:
            code: Zu analysierender Code
            language: Programmiersprache
            model: Code-Modell

        Returns:
            dict mit vulnerabilities, severity_summary, recommendations, model, gpu_seconds
        """
        prompt = f"""Analyze the following code for security vulnerabilities.

Check for:
1. SQL Injection
2. Cross-Site Scripting (XSS)
3. Command Injection
4. Path Traversal
5. Insecure Deserialization
6. Hardcoded Credentials/Secrets
7. Improper Input Validation
8. Insecure Cryptography
9. Race Conditions
10. Buffer Overflows (if applicable)

For each vulnerability found, provide:
- Type of vulnerability (OWASP category)
- Severity (Critical/High/Medium/Low)
- Line number(s) affected
- Description of the issue
- Recommended fix

Code:
```
{code}
```

Security Analysis (JSON format):"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=2048,
            temperature=0.1,
        )

        return {
            "analysis": result["code"],
            "language": language,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: REFACTORING SUGGESTIONS
    # ═══════════════════════════════════════════════════════════════

    def suggest_refactoring(
        self,
        code: str,
        language: str = None,
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Schlägt Refactorings vor

        Args:
            code: Zu analysierender Code
            language: Programmiersprache
            model: Code-Modell

        Returns:
            dict mit suggestions, refactored_code, model, gpu_seconds
        """
        prompt = f"""Analyze the following code and suggest improvements:

Consider:
1. Code duplication (DRY principle)
2. Function/method length
3. Naming conventions
4. Design patterns that could be applied
5. SOLID principles violations
6. Performance optimizations
7. Readability improvements
8. Modern language features that could be used

For each suggestion:
- Describe the issue
- Explain the benefit of the change
- Show the refactored code

Code:
```
{code}
```

Refactoring Analysis:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=4096,
            temperature=0.3,
        )

        return {
            "suggestions": result["code"],
            "language": language,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: TYPE HINTS GENERATION
    # ═══════════════════════════════════════════════════════════════

    def generate_type_hints(
        self,
        code: str,
        language: str = "python",
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Fügt Type Hints zu Code hinzu

        Args:
            code: Code ohne/mit teilweisen Type Hints
            language: Programmiersprache (python, typescript)
            model: Code-Modell

        Returns:
            dict mit typed_code, model, gpu_seconds
        """
        if language == "python":
            prompt = f"""Add comprehensive Python type hints to the following code.

Include:
1. Function parameter types
2. Return types
3. Variable annotations where helpful
4. Generic types (List, Dict, Optional, etc.)
5. TypeVar for generic functions
6. Protocol/ABC for duck typing if needed

Code:
```python
{code}
```

Return the code with complete type annotations:"""
        else:
            prompt = f"""Add TypeScript type annotations to the following code.

Include interface/type definitions where needed.

Code:
```
{code}
```

Return the fully typed code:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            language=language,
            max_tokens=4096,
            temperature=0.1,
        )

        return {
            "typed_code": result["code"],
            "language": language,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: COMPLEXITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    def analyze_complexity(
        self,
        code: str,
        language: str = None,
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Analysiert Code-Komplexität

        Args:
            code: Zu analysierender Code
            language: Programmiersprache
            model: Code-Modell

        Returns:
            dict mit metrics (cyclomatic complexity, etc.), model, gpu_seconds
        """
        prompt = f"""Analyze the complexity of the following code.

Calculate/Estimate:
1. Cyclomatic Complexity for each function
2. Cognitive Complexity
3. Lines of Code (LOC)
4. Nesting Depth (max and average)
5. Number of Parameters per function
6. Number of Dependencies/Imports
7. Overall Maintainability Index estimate

Also identify:
- Functions that are too complex (CC > 10)
- Deep nesting that should be refactored
- Long functions that should be split

Code:
```
{code}
```

Complexity Analysis (structured format):"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=2048,
            temperature=0.1,
        )

        return {
            "analysis": result["code"],
            "language": language,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: MULTI-FILE CONTEXT
    # ═══════════════════════════════════════════════════════════════

    def review_with_context(
        self,
        main_file: str,
        context_files: List[Dict[str, str]],
        language: str = None,
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Code Review mit Multi-File Kontext

        Args:
            main_file: Hauptdatei die reviewed werden soll
            context_files: Liste von {'path': str, 'content': str}
            language: Programmiersprache
            model: Code-Modell

        Returns:
            dict mit review, model, gpu_seconds
        """
        # Baue Kontext auf
        context = "## Context Files:\n\n"
        for cf in context_files[:5]:  # Max 5 Kontext-Dateien
            context += f"### {cf.get('path', 'unknown')}\n```\n{cf['content'][:2000]}\n```\n\n"

        prompt = f"""{context}

## File to Review:
```
{main_file}
```

Perform a thorough code review considering the context from related files.

Review aspects:
1. Correctness and bugs
2. Consistency with codebase patterns
3. API usage correctness
4. Dependency handling
5. Error handling
6. Performance considerations
7. Security issues
8. Test coverage suggestions

Detailed Code Review:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=4096,
            temperature=0.3,
        )

        return {
            "review": result["code"],
            "context_files_count": len(context_files),
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: LINTING INTEGRATION
    # ═══════════════════════════════════════════════════════════════

    def generate_lint_config(
        self,
        language: str,
        framework: str = None,
        style: str = "standard",
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Generiert Linting-Konfiguration

        Args:
            language: Programmiersprache (python, javascript, typescript)
            framework: Optional Framework (react, vue, django, etc.)
            style: Style Guide (standard, airbnb, google, etc.)
            model: Code-Modell

        Returns:
            dict mit config_files (dict von Dateiname -> Inhalt)
        """
        configs = {
            'python': ['pyproject.toml', '.ruff.toml', '.flake8'],
            'javascript': ['.eslintrc.json', '.prettierrc'],
            'typescript': ['.eslintrc.json', 'tsconfig.json', '.prettierrc'],
        }

        target_configs = configs.get(language, ['.editorconfig'])

        prompt = f"""Generate linting and formatting configuration for a {language} project.

Language: {language}
Framework: {framework or 'None'}
Style Guide: {style}

Generate configuration files:
{', '.join(target_configs)}

For each config file, provide:
1. The complete configuration
2. Brief explanation of key settings
3. Commands to run the linter

Configuration:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=4096,
            temperature=0.2,
        )

        return {
            "configs": result["code"],
            "language": language,
            "framework": framework,
            "style": style,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }

    def generate_pre_commit_hooks(
        self,
        languages: List[str],
        model: str = "deepseek-coder-6.7b",
    ) -> dict:
        """
        MEGA-UPGRADE: Generiert Pre-Commit Hook Konfiguration

        Args:
            languages: Liste der verwendeten Sprachen
            model: Code-Modell

        Returns:
            dict mit config (.pre-commit-config.yaml)
        """
        prompt = f"""Generate a comprehensive .pre-commit-config.yaml for a project using: {', '.join(languages)}

Include hooks for:
1. Code formatting (black, prettier, etc.)
2. Linting (ruff, eslint, etc.)
3. Type checking (mypy, typescript)
4. Security scanning (bandit, safety)
5. Commit message validation
6. Trailing whitespace
7. Large file checks
8. Secret detection

Generate complete .pre-commit-config.yaml:"""

        result = self.generate_code(
            prompt=prompt,
            model=model,
            max_tokens=2048,
            temperature=0.2,
        )

        return {
            "config": result["code"],
            "languages": languages,
            "model": model,
            "gpu_seconds": result["gpu_seconds"],
        }


# Singleton Instance
_code_worker: Optional[CodeWorker] = None


def get_code_worker() -> CodeWorker:
    """Get singleton instance"""
    global _code_worker
    if _code_worker is None:
        _code_worker = CodeWorker()
    return _code_worker
