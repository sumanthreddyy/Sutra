"""Web search tool — search the web via DuckDuckGo (free) or Tavily (API)."""

import json
import logging
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    def __init__(self, provider: str = "duckduckgo", tavily_api_key: str = ""):
        self._provider = provider
        self._tavily_key = tavily_api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information. Returns titles, URLs, and snippets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum results (default 5)"},
            },
            "required": ["query"],
        }

    async def _run(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        if self._provider == "tavily" and self._tavily_key:
            return await self._search_tavily(query, max_results)
        return await self._search_duckduckgo(query, max_results)

    async def _search_duckduckgo(self, query: str, max_results: int) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return "No results found."

            output = []
            for r in results:
                output.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}\n")
            return "\n".join(output)

        except ImportError:
            return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
        except Exception as e:
            return f"Search error: {e}"

    async def _search_tavily(self, query: str, max_results: int) -> str:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._tavily_key,
                        "query": query,
                        "max_results": max_results,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            if not results:
                return "No results found."

            output = []
            for r in results:
                output.append(f"**{r.get('title', '')}**\n{r.get('url', '')}\n{r.get('content', '')}\n")
            return "\n".join(output)

        except Exception as e:
            return f"Tavily search error: {e}"
