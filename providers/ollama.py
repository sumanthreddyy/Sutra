"""Ollama local model provider."""

from typing import Any

import httpx

from .base import LLMProvider, LLMResponse, Message, ToolDef


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def name(self) -> str:
        return "ollama"

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            if msg.role == "tool":
                # Ollama doesn't have native tool results — inject as user message
                ollama_messages.append({
                    "role": "user",
                    "content": f"[Tool Result: {msg.name}]\n{msg.content}",
                })
            else:
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        payload: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Ollama tool support (if model supports it)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        content = message.get("content", "")

        # Parse tool calls if present
        tool_calls = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append({
                "id": f"ollama_{func.get('name', 'unknown')}",
                "name": func.get("name", ""),
                "arguments": func.get("arguments", {}),
            })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            model=payload["model"],
            stop_reason="stop",
        )
