"""Smart LLM router — routes tasks to local or cloud based on task type and availability."""

import logging
from typing import Any

from config import SutraConfig
from providers.base import LLMProvider, LLMResponse, Message, ToolDef

logger = logging.getLogger(__name__)


class Router:
    """Routes LLM calls to the appropriate provider based on task type."""

    def __init__(self, config: SutraConfig, providers: dict[str, LLMProvider]):
        self.config = config
        self.providers = providers
        self._fallback_name = config.routing.fallback

    def _get_provider_for_task(self, task_type: str) -> str:
        """Get the configured provider name for a task type."""
        routing = self.config.routing
        return getattr(routing, task_type, routing.default)

    async def _get_available_provider(self, preferred: str) -> LLMProvider:
        """Get the preferred provider, falling back if unavailable."""
        if preferred in self.providers:
            provider = self.providers[preferred]
            if await provider.is_available():
                return provider
            logger.warning(f"{preferred} unavailable, trying fallback")

        # Try fallback
        if self._fallback_name in self.providers:
            fallback = self.providers[self._fallback_name]
            if await fallback.is_available():
                return fallback

        # Try any available provider
        for name, provider in self.providers.items():
            if await provider.is_available():
                logger.warning(f"Using {name} as last resort")
                return provider

        raise RuntimeError("No LLM providers available")

    async def complete(
        self,
        messages: list[Message],
        task_type: str = "default",
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Route a completion request to the appropriate provider."""
        preferred_name = self._get_provider_for_task(task_type)
        provider = await self._get_available_provider(preferred_name)

        logger.debug(f"Routing '{task_type}' to {provider.name}")
        return await provider.complete(messages, tools=tools, **kwargs)

    async def complete_with_provider(
        self,
        provider_name: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Force a specific provider."""
        provider = await self._get_available_provider(provider_name)
        return await provider.complete(messages, tools=tools, **kwargs)
