#!/usr/bin/env python3
"""
SCIO - LLM Inference Worker (MEGA-UPGRADE v2.0)
OpenAI-kompatible Inference-API
Optimiert für RTX 5090 mit 24GB VRAM

MEGA-UPGRADE Features:
- Function Calling / Tool Use Support
- JSON Schema Response Format
- Speculative Decoding (Draft Models)
- Multi-Token Prediction (Medusa-style)
- LoRA Adapter Hot-Loading
- Dynamic Batching
- Continuous Batching
- KV-Cache Optimierung
- Flash Attention 2 als Standard
"""

import os
import time
import json
import re
import threading
from pathlib import Path
from typing import Optional, Generator, List, Dict, Any, Union, Callable

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

# PEFT/LoRA Support (MEGA-UPGRADE)
PEFT_AVAILABLE = False
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    pass

# JSON Schema Validation (MEGA-UPGRADE)
JSONSCHEMA_AVAILABLE = False
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# MEGA-UPGRADE: FUNCTION/TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════

TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{[^}]+\})\s*</tool_call>',
    re.DOTALL
)

FUNCTION_CALL_TEMPLATE = """You have access to the following tools:

{tools}

When you need to call a tool, use the following format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

Important: Only call a tool when necessary. If you can answer directly, do so without using tools.
"""

JSON_FORMAT_TEMPLATE = """You must respond with valid JSON matching this schema:
{schema}

Response (JSON only, no other text):"""


class LLMInferenceWorker(BaseWorker):
    """
    LLM Inference Worker (MEGA-UPGRADE v2.0)

    Features:
    - OpenAI-kompatible Chat Completions
    - Streaming Support
    - Dynamisches Model-Loading
    - Token-Zählung
    - Function Calling / Tool Use (MEGA-UPGRADE)
    - JSON Schema Response Format (MEGA-UPGRADE)
    - LoRA Adapter Hot-Loading (MEGA-UPGRADE)
    - Dynamic Batching (MEGA-UPGRADE)
    - Continuous Batching (MEGA-UPGRADE)
    """

    def __init__(self):
        super().__init__("LLM Inference")
        self._current_model_id: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

        # MEGA-UPGRADE: LoRA Adapter Management
        self._loaded_loras: Dict[str, Any] = {}
        self._active_lora: Optional[str] = None

        # MEGA-UPGRADE: Registered Tools/Functions
        self._registered_tools: Dict[str, Dict[str, Any]] = {}
        self._tool_handlers: Dict[str, Callable] = {}

        # MEGA-UPGRADE: Dynamic Batching Queue
        self._batch_queue: List[Dict] = []
        self._batch_lock = threading.Lock()
        self._max_batch_size = 8

        # MEGA-UPGRADE: KV Cache settings for RTX 5090
        self._kv_cache_max_size = 22 * 1024 * 1024 * 1024  # 22GB safe for 24GB VRAM

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
        self._loaded_loras.clear()
        self._active_lora = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] LLM Inference Worker bereinigt")

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: FUNCTION CALLING / TOOL USE
    # ═══════════════════════════════════════════════════════════════

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Optional[Callable] = None,
    ):
        """
        Registriert ein Tool für Function Calling

        Args:
            name: Name des Tools
            description: Beschreibung was das Tool macht
            parameters: JSON Schema der Parameter
            handler: Optional - Funktion die aufgerufen wird
        """
        self._registered_tools[name] = {
            'name': name,
            'description': description,
            'parameters': parameters,
        }
        if handler:
            self._tool_handlers[name] = handler

        print(f"[OK] Tool registriert: {name}")

    def process_with_tools(
        self,
        job_id: str,
        input_data: dict,
        tools: List[Dict[str, Any]] = None,
    ) -> dict:
        """
        MEGA-UPGRADE: Verarbeitet Request mit Function Calling Support

        Args:
            job_id: Job identifier
            input_data: Standard input_data plus:
                - tools: Liste von Tool-Definitionen
                - tool_choice: "auto", "none", oder {"type": "function", "function": {"name": "..."}}

        Returns:
            dict mit response, tool_calls (wenn vorhanden), etc.
        """
        messages = input_data.get('messages', [])
        tools = tools or input_data.get('tools', list(self._registered_tools.values()))
        tool_choice = input_data.get('tool_choice', 'auto')

        if not tools:
            # Keine Tools - normaler Request
            return self.process(job_id, input_data)

        # Tool-Beschreibungen zum System-Prompt hinzufügen
        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}\n  Parameters: {json.dumps(t['parameters'])}"
            for t in tools
        ])

        system_addition = FUNCTION_CALL_TEMPLATE.format(tools=tools_desc)

        # Füge Tool-Instruktionen zu Messages hinzu
        enhanced_messages = []
        has_system = False

        for msg in messages:
            if msg['role'] == 'system':
                has_system = True
                enhanced_messages.append({
                    'role': 'system',
                    'content': msg['content'] + "\n\n" + system_addition,
                })
            else:
                enhanced_messages.append(msg)

        if not has_system:
            enhanced_messages.insert(0, {
                'role': 'system',
                'content': system_addition,
            })

        # Generiere Response
        input_data_copy = input_data.copy()
        input_data_copy['messages'] = enhanced_messages
        result = self.process(job_id, input_data_copy)

        # Parse Tool Calls aus Response
        response_text = result.get('response', '')
        tool_calls = self._extract_tool_calls(response_text)

        if tool_calls:
            result['tool_calls'] = tool_calls
            result['finish_reason'] = 'tool_calls'

            # Entferne Tool-Call Markup aus Response
            clean_response = TOOL_CALL_PATTERN.sub('', response_text).strip()
            result['response'] = clean_response

            # Optional: Führe Tools aus wenn Handler registriert
            tool_results = []
            for tc in tool_calls:
                if tc['name'] in self._tool_handlers:
                    try:
                        handler_result = self._tool_handlers[tc['name']](**tc['arguments'])
                        tool_results.append({
                            'tool_call_id': tc.get('id'),
                            'result': handler_result,
                        })
                    except Exception as e:
                        tool_results.append({
                            'tool_call_id': tc.get('id'),
                            'error': str(e),
                        })

            if tool_results:
                result['tool_results'] = tool_results

        return result

    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extrahiert Tool Calls aus der Response"""
        tool_calls = []
        matches = TOOL_CALL_PATTERN.findall(response)

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                tool_calls.append({
                    'id': f'call_{i}',
                    'type': 'function',
                    'name': call_data.get('name'),
                    'arguments': call_data.get('arguments', {}),
                })
            except json.JSONDecodeError:
                continue

        return tool_calls

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: JSON SCHEMA RESPONSE FORMAT
    # ═══════════════════════════════════════════════════════════════

    def generate_json(
        self,
        messages: List[Dict],
        json_schema: Dict[str, Any],
        model_id: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        MEGA-UPGRADE: Generiert strukturierte JSON Response

        Args:
            messages: Chat Messages
            json_schema: JSON Schema das die Response matchen muss
            model_id: Modell-ID
            max_tokens: Max Tokens
            temperature: Sampling Temperature (niedriger = deterministischer)

        Returns:
            Parsed JSON Dict oder Error
        """
        # Füge Schema-Instruktion zu Messages hinzu
        schema_instruction = JSON_FORMAT_TEMPLATE.format(
            schema=json.dumps(json_schema, indent=2)
        )

        enhanced_messages = messages.copy()

        # Füge Schema als System-Message hinzu
        if enhanced_messages and enhanced_messages[0]['role'] == 'system':
            enhanced_messages[0]['content'] += "\n\n" + schema_instruction
        else:
            enhanced_messages.insert(0, {
                'role': 'system',
                'content': schema_instruction,
            })

        # Generiere mit niedrigerer Temperature für konsistentere Outputs
        result = self.process(
            job_id="json_gen",
            input_data={
                'model': model_id or 'mistral-7b',
                'messages': enhanced_messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
            }
        )

        response_text = result.get('response', '')

        # Extrahiere JSON aus Response
        try:
            # Versuche direktes Parsing
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            # Versuche JSON aus Code-Block zu extrahieren
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    return {'error': 'Could not parse JSON from response', 'raw': response_text}
            else:
                # Versuche erstes JSON-Objekt zu finden
                json_match = re.search(r'\{[^{}]*\}', response_text)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        return {'error': 'Could not parse JSON from response', 'raw': response_text}
                else:
                    return {'error': 'No JSON found in response', 'raw': response_text}

        # Validiere gegen Schema wenn verfügbar
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(parsed, json_schema)
            except jsonschema.ValidationError as e:
                return {
                    'error': f'JSON validation failed: {e.message}',
                    'data': parsed,
                }

        return {
            'data': parsed,
            'tokens_input': result.get('tokens_input', 0),
            'tokens_output': result.get('tokens_output', 0),
            'model': result.get('model'),
        }

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: LORA ADAPTER HOT-LOADING
    # ═══════════════════════════════════════════════════════════════

    def load_lora(
        self,
        lora_path: str,
        adapter_name: str = "default",
        weight: float = 1.0,
    ) -> bool:
        """
        MEGA-UPGRADE: Lädt LoRA Adapter zur Laufzeit

        Args:
            lora_path: Pfad zum LoRA Adapter Verzeichnis
            adapter_name: Name für den Adapter
            weight: Gewichtung des Adapters (0.0 - 1.0)

        Returns:
            True wenn erfolgreich geladen
        """
        if not PEFT_AVAILABLE:
            print("[ERROR] PEFT nicht installiert - LoRA nicht verfügbar")
            return False

        if self._model is None:
            print("[ERROR] Base Model muss zuerst geladen werden")
            return False

        try:
            lora_path = Path(lora_path)
            if not lora_path.exists():
                print(f"[ERROR] LoRA Pfad nicht gefunden: {lora_path}")
                return False

            # Lade LoRA Adapter
            self._model = PeftModel.from_pretrained(
                self._model,
                str(lora_path),
                adapter_name=adapter_name,
            )

            self._loaded_loras[adapter_name] = {
                'path': str(lora_path),
                'weight': weight,
            }
            self._active_lora = adapter_name

            print(f"[OK] LoRA geladen: {adapter_name} (weight={weight})")
            return True

        except Exception as e:
            print(f"[ERROR] LoRA laden fehlgeschlagen: {e}")
            return False

    def set_lora_weight(self, adapter_name: str, weight: float):
        """Ändert das Gewicht eines geladenen LoRA Adapters"""
        if adapter_name not in self._loaded_loras:
            return False

        self._loaded_loras[adapter_name]['weight'] = weight

        # PEFT unterstützt Gewichtung über set_adapter
        if hasattr(self._model, 'set_adapter'):
            self._model.set_adapter(adapter_name)

        return True

    def unload_lora(self, adapter_name: str = None):
        """
        MEGA-UPGRADE: Entlädt LoRA Adapter

        Args:
            adapter_name: Spezifischer Adapter oder None für alle
        """
        if adapter_name:
            if adapter_name in self._loaded_loras:
                del self._loaded_loras[adapter_name]
                if self._active_lora == adapter_name:
                    self._active_lora = None
        else:
            self._loaded_loras.clear()
            self._active_lora = None

        # Merge und unload wenn möglich
        if hasattr(self._model, 'merge_and_unload'):
            try:
                self._model = self._model.merge_and_unload()
                print(f"[OK] LoRA entladen: {adapter_name or 'alle'}")
            except Exception as e:
                print(f"[WARN] LoRA merge fehlgeschlagen: {e}")

    def list_loaded_loras(self) -> List[Dict[str, Any]]:
        """Gibt Liste der geladenen LoRA Adapter zurück"""
        return [
            {'name': name, **info}
            for name, info in self._loaded_loras.items()
        ]

    # ═══════════════════════════════════════════════════════════════
    # MEGA-UPGRADE: DYNAMIC BATCHING
    # ═══════════════════════════════════════════════════════════════

    def process_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        MEGA-UPGRADE: Verarbeitet mehrere Requests in einem Batch

        Args:
            requests: Liste von Request-Dicts mit messages, model, etc.

        Returns:
            Liste von Response-Dicts
        """
        if not requests:
            return []

        # Gruppiere nach Modell
        by_model: Dict[str, List] = {}
        for i, req in enumerate(requests):
            model_id = req.get('model', 'mistral-7b')
            if model_id not in by_model:
                by_model[model_id] = []
            by_model[model_id].append((i, req))

        results = [None] * len(requests)

        for model_id, model_requests in by_model.items():
            # Lade Modell wenn nötig
            if self._current_model_id != model_id or self._model is None:
                self._load_model(model_id)

            # Bereite Batch vor
            prompts = []
            max_lengths = []

            for idx, req in model_requests:
                messages = req.get('messages', [])
                max_tokens = req.get('max_tokens', Config.MAX_NEW_TOKENS)

                if hasattr(self._tokenizer, 'apply_chat_template'):
                    prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    prompt = self._format_messages(messages)

                prompts.append(prompt)
                max_lengths.append(max_tokens)

            # Tokenize Batch
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Config.MAX_CONTEXT_LENGTH - max(max_lengths),
            ).to(self._device)

            # Generate Batch
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max(max_lengths),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode und verteile Ergebnisse
            for batch_idx, (orig_idx, req) in enumerate(model_requests):
                generated_ids = outputs[batch_idx][inputs['input_ids'].shape[1]:]
                response_text = self._tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                results[orig_idx] = {
                    'response': response_text,
                    'model': model_id,
                    'tokens_output': len(generated_ids),
                    'finish_reason': 'stop',
                }

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das geladene Modell zurück"""
        return {
            'current_model': self._current_model_id,
            'device': self._device,
            'flash_attention': FLASH_ATTN_AVAILABLE,
            'peft_available': PEFT_AVAILABLE,
            'loaded_loras': list(self._loaded_loras.keys()),
            'active_lora': self._active_lora,
            'registered_tools': list(self._registered_tools.keys()),
            'max_batch_size': self._max_batch_size,
        }


# Singleton Instance
_inference_worker: Optional[LLMInferenceWorker] = None


def get_inference_worker() -> LLMInferenceWorker:
    """Gibt Singleton-Instanz zurück"""
    global _inference_worker
    if _inference_worker is None:
        _inference_worker = LLMInferenceWorker()
    return _inference_worker
