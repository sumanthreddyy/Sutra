"""Sutra — Personal AI Agent with Persistent Memory.

Entry point. Sets up providers, router, agent, and launches the CLI.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import SutraConfig
from core.agent import Agent
from core.router import Router
from providers.anthropic_provider import AnthropicProvider
from providers.ollama import OllamaProvider
from providers.openai_provider import OpenAIProvider
from interfaces.cli import CLI


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("data/sutra.log"),
            logging.StreamHandler() if debug else logging.NullHandler(),
        ],
    )


def build_providers(config: SutraConfig) -> dict:
    """Initialize all configured LLM providers."""
    providers = {}

    # Anthropic
    anthro_cfg = config.get_provider_config("anthropic")
    if anthro_cfg.api_key:
        providers["anthropic"] = AnthropicProvider(
            api_key=anthro_cfg.api_key,
            model=anthro_cfg.default_model or "claude-sonnet-4-20250514",
        )

    # OpenAI
    oai_cfg = config.get_provider_config("openai")
    if oai_cfg.api_key:
        providers["openai"] = OpenAIProvider(
            api_key=oai_cfg.api_key,
            model=oai_cfg.default_model or "gpt-4o",
        )

    # Ollama (always available — it'll fail gracefully if not running)
    ollama_cfg = config.get_provider_config("ollama")
    providers["ollama"] = OllamaProvider(
        base_url=ollama_cfg.base_url or "http://localhost:11434",
        model=ollama_cfg.default_model or "llama3",
    )

    return providers


async def main() -> None:
    # Ensure data directories exist
    Path("data").mkdir(exist_ok=True)
    Path("data/memory").mkdir(exist_ok=True)
    Path("data/conversations").mkdir(exist_ok=True)
    Path("data/vectors").mkdir(exist_ok=True)

    debug = "--debug" in sys.argv
    setup_logging(debug)

    # Load config
    config = SutraConfig.load()

    # Build vector store
    from memory.vector_store import VectorStore
    oai_cfg = config.get_provider_config("openai")
    vector_store = VectorStore(
        persist_dir=config.memory.vector_dir,
        embedding_provider=config.memory.embedding_provider,
        ollama_model=config.memory.ollama_embed_model,
        openai_api_key=oai_cfg.api_key,
    )

    # Build senses (v0.3)
    from senses.vision import Vision
    from senses.voice_input import VoiceInput
    from senses.voice_output import VoiceOutput

    anthro_cfg = config.get_provider_config("anthropic")
    vision = Vision(
        provider="anthropic" if anthro_cfg.api_key else "openai",
        anthropic_api_key=anthro_cfg.api_key,
        openai_api_key=oai_cfg.api_key,
    )
    voice_in = VoiceInput(
        provider="openai" if oai_cfg.api_key else "local",
        openai_api_key=oai_cfg.api_key,
    )
    voice_out = VoiceOutput(provider="edge")  # Free, no API key needed

    # Build providers
    providers = build_providers(config)

    if not providers:
        print("Error: No LLM providers configured. Set API keys in data/config.yaml or env vars.")
        sys.exit(1)

    # Build router
    router = Router(config, providers)

    # Build agent (with vector store + senses)
    agent = Agent(config, router, vector_store=vector_store, vision=vision, voice_in=voice_in, voice_out=voice_out)

    # Launch CLI
    cli = CLI(agent)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
