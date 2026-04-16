"""OpenAI API provider."""

import json
from typing import Any

import openai

from .base import LLMProvider, LLMResponse, Message, ToolDef


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = "", model: str = "gpt-4o"):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key if api_key else None)

    @property
    def name(self) -> str:
        return "openai"

    async def is_available(self) -> bool:
        try:
            return bool(self.client.api_key)
        except Exception:
            return False

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        oai_messages = []
        for msg in messages:
            if msg.role == "tool":
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls
                oai_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                oai_messages.append(oai_msg)
            else:
                oai_messages.append({"role": msg.role, "content": msg.content})

        api_kwargs: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            api_kwargs["tools"] = [
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

        response = await self.client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            model=api_kwargs["model"],
            stop_reason=choice.finish_reason or "",
        )
