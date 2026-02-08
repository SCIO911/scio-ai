"""
SCIO LLM Agent

Agent für die Integration von Large Language Models.
"""

import os
from typing import Any, Optional

from scio.agents.base import Agent, AgentConfig, AgentContext
from scio.agents.registry import register_agent
from scio.core.exceptions import AgentError
from scio.core.logging import get_logger

logger = get_logger(__name__)


class LLMConfig(AgentConfig):
    """Konfiguration für den LLM Agent."""

    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: Optional[str] = None
    timeout: int = 60


@register_agent("llm")
class LLMAgent(Agent[dict[str, Any], dict[str, Any]]):
    """
    Agent für LLM-Interaktionen.

    Unterstützt: OpenAI, Anthropic, lokale Modelle
    """

    agent_type = "llm"
    version = "1.0"

    def __init__(self, config: LLMConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = LLMConfig(**config)
        super().__init__(config)
        self.config: LLMConfig = config
        self._client = None

    async def execute(
        self, input_data: dict[str, Any], context: AgentContext
    ) -> dict[str, Any]:
        """Führt LLM-Anfrage aus."""

        prompt = input_data.get("prompt")
        if not prompt:
            raise AgentError("Kein Prompt angegeben", agent_id=self.agent_id)

        messages = input_data.get("messages", [])
        system = input_data.get("system", self.config.system_prompt)

        self.logger.info(
            "LLM request",
            provider=self.config.provider,
            model=self.config.model,
            prompt_length=len(prompt),
        )

        # Wähle Provider
        if self.config.provider == "openai":
            response = await self._call_openai(prompt, messages, system)
        elif self.config.provider == "anthropic":
            response = await self._call_anthropic(prompt, messages, system)
        elif self.config.provider == "local":
            response = await self._call_local(prompt, messages, system)
        else:
            raise AgentError(f"Unbekannter Provider: {self.config.provider}")

        return {
            "response": response["content"],
            "model": self.config.model,
            "usage": response.get("usage", {}),
        }

    async def _call_openai(
        self,
        prompt: str,
        messages: list[dict],
        system: Optional[str],
    ) -> dict[str, Any]:
        """Ruft OpenAI API auf."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise AgentError("OpenAI nicht installiert. pip install openai")

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AgentError("OpenAI API Key nicht konfiguriert")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout,
        )

        # Baue Messages
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        msgs.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=msgs,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    async def _call_anthropic(
        self,
        prompt: str,
        messages: list[dict],
        system: Optional[str],
    ) -> dict[str, Any]:
        """Ruft Anthropic API auf."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise AgentError("Anthropic nicht installiert. pip install anthropic")

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise AgentError("Anthropic API Key nicht konfiguriert")

        client = AsyncAnthropic(
            api_key=api_key,
            timeout=self.config.timeout,
        )

        # Baue Messages
        msgs = list(messages)
        msgs.append({"role": "user", "content": prompt})

        response = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system or "",
            messages=msgs,
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    async def _call_local(
        self,
        prompt: str,
        messages: list[dict],
        system: Optional[str],
    ) -> dict[str, Any]:
        """Ruft lokales Modell auf (z.B. via Ollama)."""
        import aiohttp

        base_url = self.config.api_base or "http://localhost:11434"

        # Baue Prompt
        full_prompt = ""
        if system:
            full_prompt += f"System: {system}\n\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role.title()}: {content}\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as resp:
                if resp.status != 200:
                    raise AgentError(f"Lokales Modell Fehler: {resp.status}")

                data = await resp.json()
                return {
                    "content": data.get("response", ""),
                    "usage": {
                        "eval_count": data.get("eval_count", 0),
                    },
                }
