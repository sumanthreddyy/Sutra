"""Vision — image understanding via Claude/GPT-4o vision APIs.

Supports:
- Anthropic Claude (claude-sonnet-4-20250514 with vision)
- OpenAI GPT-4o with vision
- Image from file path or URL
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB cap


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read and base64-encode an image file. Returns (base64_data, media_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({size / 1024 / 1024:.1f}MB). Max: 20MB")

    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


class Vision:
    """Image understanding using vision-capable LLMs."""

    def __init__(
        self,
        provider: str = "anthropic",
        anthropic_api_key: str = "",
        anthropic_model: str = "claude-sonnet-4-20250514",
        openai_api_key: str = "",
        openai_model: str = "gpt-4o",
    ):
        self._provider = provider
        self._anthropic_key = anthropic_api_key
        self._anthropic_model = anthropic_model
        self._openai_key = openai_api_key
        self._openai_model = openai_model

    async def analyze(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
    ) -> str:
        """Analyze an image with a vision model."""
        if self._provider == "openai" and self._openai_key:
            return await self._analyze_openai(image_path, prompt)
        if self._anthropic_key:
            return await self._analyze_anthropic(image_path, prompt)
        return "Error: No vision provider configured. Set Anthropic or OpenAI API key."

    async def analyze_url(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail.",
    ) -> str:
        """Analyze an image from URL."""
        if self._provider == "openai" and self._openai_key:
            return await self._analyze_openai_url(image_url, prompt)
        # Anthropic also supports URL — but let's download and use base64 for consistency
        return await self._analyze_url_via_download(image_url, prompt)

    async def _analyze_anthropic(self, image_path: str, prompt: str) -> str:
        """Analyze via Anthropic Claude Vision."""
        try:
            import anthropic

            data, media_type = _encode_image(image_path)
            client = anthropic.AsyncAnthropic(api_key=self._anthropic_key)

            response = await client.messages.create(
                model=self._anthropic_model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic vision failed: {e}")
            return f"Error: {e}"

    async def _analyze_openai(self, image_path: str, prompt: str) -> str:
        """Analyze via OpenAI GPT-4o Vision."""
        try:
            import openai

            data, media_type = _encode_image(image_path)
            client = openai.AsyncOpenAI(api_key=self._openai_key)

            response = await client.chat.completions.create(
                model=self._openai_model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"OpenAI vision failed: {e}")
            return f"Error: {e}"

    async def _analyze_openai_url(self, image_url: str, prompt: str) -> str:
        """Analyze image URL via OpenAI."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self._openai_key)

            response = await client.chat.completions.create(
                model=self._openai_model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"OpenAI vision URL failed: {e}")
            return f"Error: {e}"

    async def _analyze_url_via_download(self, image_url: str, prompt: str) -> str:
        """Download image from URL, then analyze locally."""
        try:
            import httpx
            import tempfile

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()

            suffix = ".png"
            ct = resp.headers.get("content-type", "")
            if "jpeg" in ct or "jpg" in ct:
                suffix = ".jpg"
            elif "webp" in ct:
                suffix = ".webp"
            elif "gif" in ct:
                suffix = ".gif"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

            result = await self.analyze(tmp_path, prompt)
            Path(tmp_path).unlink(missing_ok=True)
            return result

        except Exception as e:
            logger.error(f"URL download + analyze failed: {e}")
            return f"Error: {e}"
