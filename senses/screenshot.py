"""Screenshot tool — capture screen and analyze with vision."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)


class ScreenshotTool(Tool):
    """Capture a screenshot and optionally analyze it."""

    def __init__(self, vision=None):
        self._vision = vision

    @property
    def name(self) -> str:
        return "screenshot"

    @property
    def description(self) -> str:
        return "Take a screenshot of the screen and optionally analyze it with vision AI."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "analyze": {
                    "type": "boolean",
                    "description": "If true, analyze the screenshot with vision AI (default: true)",
                },
                "prompt": {
                    "type": "string",
                    "description": "Custom prompt for vision analysis (default: describe what's on screen)",
                },
            },
        }

    async def _run(
        self,
        analyze: bool = True,
        prompt: str = "Describe what you see on this screen. Focus on the main content and any notable elements.",
        **kwargs: Any,
    ) -> str:
        try:
            from PIL import ImageGrab
        except ImportError:
            return "Error: Pillow not installed. Run: pip install Pillow"

        try:
            screenshot = ImageGrab.grab()

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            screenshot.save(tmp.name)
            tmp.close()
            tmp_path = tmp.name

            result = f"Screenshot saved: {tmp_path}"

            if analyze and self._vision:
                analysis = await self._vision.analyze(tmp_path, prompt)
                result += f"\n\nAnalysis:\n{analysis}"
                Path(tmp_path).unlink(missing_ok=True)
            elif analyze and not self._vision:
                result += "\n\nVision not configured — screenshot saved but not analyzed."

            return result

        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return f"Error: {e}"


class ImageAnalyzeTool(Tool):
    """Analyze an image file with vision AI."""

    def __init__(self, vision=None):
        self._vision = vision

    @property
    def name(self) -> str:
        return "analyze_image"

    @property
    def description(self) -> str:
        return "Analyze an image file or URL using vision AI. Describe, extract text, or answer questions about the image."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path or URL of the image to analyze",
                },
                "prompt": {
                    "type": "string",
                    "description": "What to analyze or ask about the image",
                },
            },
            "required": ["path"],
        }

    async def _run(
        self,
        path: str,
        prompt: str = "Describe this image in detail.",
        **kwargs: Any,
    ) -> str:
        if not self._vision:
            return "Error: Vision not configured. Set Anthropic or OpenAI API key."

        if path.startswith("http://") or path.startswith("https://"):
            return await self._vision.analyze_url(path, prompt)
        return await self._vision.analyze(path, prompt)
