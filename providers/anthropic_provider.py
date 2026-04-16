"""Anthropic Claude API provider."""

from typing import Any

import anthropic

from .base import LLMProvider, LLMResponse, Message, ToolDef


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.model = model
        # Uses ANTHROPIC_API_KEY env var if api_key is empty
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key if api_key else None
        )

    @property
    def name(self) -> str:
        return "anthropic"

    async def is_available(self) -> bool:
        try:
            # Quick ping — list models isn't available, just try a tiny call
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
        # Separate system message from conversation
        system_text = ""
        conversation: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_text = msg.content
            elif msg.role == "tool":
                conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls → content blocks
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc.get("arguments", {}),
                    })
                conversation.append({"role": "assistant", "content": content_blocks})
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        api_kwargs: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
        }
        if system_text:
            api_kwargs["system"] = system_text

        if tools:
            api_kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        response = await self.client.messages.create(**api_kwargs)

        # Parse response
        content_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=api_kwargs["model"],
            stop_reason=response.stop_reason or "",
        )
