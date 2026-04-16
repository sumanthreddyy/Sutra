"""Configuration management for Sutra."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    api_key: str = ""
    default_model: str = ""
    cheap_model: str = ""
    base_url: str = ""


class RoutingConfig(BaseModel):
    default: str = "anthropic"
    summarization: str = "ollama"
    extraction: str = "ollama"
    classification: str = "ollama"
    dream: str = "ollama"
    fallback: str = "anthropic"


class DreamConfig(BaseModel):
    min_hours: int = 24
    min_sessions: int = 5
    max_index_lines: int = 200
    max_index_bytes: int = 25000


class ExtractConfig(BaseModel):
    enabled: bool = True
    min_new_messages: int = 3


class MemoryConfig(BaseModel):
    dream: DreamConfig = Field(default_factory=DreamConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    base_dir: str = "data/memory"
    vector_dir: str = "data/vectors"
    embedding_provider: str = "default"  # "default" (sentence-transformers), "ollama", or "openai"
    ollama_embed_model: str = "nomic-embed-text"


class ConversationConfig(BaseModel):
    transcript_dir: str = "data/conversations"
    max_context_messages: int = 50


class SutraConfig(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    conversations: ConversationConfig = Field(default_factory=ConversationConfig)

    @classmethod
    def load(cls, config_path: str | None = None) -> "SutraConfig":
        """Load config from YAML file, falling back to defaults."""
        base = Path(__file__).parent
        path = Path(config_path) if config_path else base / "data" / "config.yaml"

        raw: dict[str, Any] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        # Merge env vars for API keys
        providers = raw.get("providers", {})
        for name, cfg in providers.items():
            if isinstance(cfg, dict) and not cfg.get("api_key"):
                env_key = f"{name.upper()}_API_KEY"
                cfg["api_key"] = os.environ.get(env_key, "")

        # Parse providers into ProviderConfig objects
        parsed_providers = {}
        for name, cfg in providers.items():
            if isinstance(cfg, dict):
                parsed_providers[name] = ProviderConfig(**cfg)

        return cls(
            providers=parsed_providers,
            routing=RoutingConfig(**raw.get("routing", {})),
            memory=MemoryConfig(**raw.get("memory", {})),
            conversations=ConversationConfig(**raw.get("conversations", {})),
        )

    def get_provider_config(self, name: str) -> ProviderConfig:
        return self.providers.get(name, ProviderConfig())
