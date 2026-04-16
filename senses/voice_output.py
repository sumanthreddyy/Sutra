"""Voice output — text-to-speech via edge-tts (free) or OpenAI TTS API.

Supports:
- edge-tts: Free, Microsoft Edge neural voices, no API key needed
- OpenAI TTS: High quality, requires API key
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VoiceOutput:
    """Text-to-speech engine."""

    def __init__(
        self,
        provider: str = "edge",
        voice: str = "en-US-AriaNeural",
        openai_api_key: str = "",
        openai_voice: str = "nova",
    ):
        self._provider = provider
        self._voice = voice
        self._openai_key = openai_api_key
        self._openai_voice = openai_voice

    async def speak(self, text: str, output_path: str | None = None) -> str:
        """Convert text to speech. Returns path to audio file."""
        if self._provider == "openai" and self._openai_key:
            return await self._speak_openai(text, output_path)
        return await self._speak_edge(text, output_path)

    async def _speak_edge(self, text: str, output_path: str | None = None) -> str:
        """Generate speech using edge-tts (free)."""
        try:
            import edge_tts

            if not output_path:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                output_path = tmp.name
                tmp.close()

            communicate = edge_tts.Communicate(text, self._voice)
            await communicate.save(output_path)
            logger.debug(f"TTS saved to {output_path}")
            return output_path

        except ImportError:
            logger.error("edge-tts not installed")
            return ""
        except Exception as e:
            logger.error(f"edge-tts failed: {e}")
            return ""

    async def _speak_openai(self, text: str, output_path: str | None = None) -> str:
        """Generate speech using OpenAI TTS API."""
        try:
            import openai

            if not output_path:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                output_path = tmp.name
                tmp.close()

            client = openai.AsyncOpenAI(api_key=self._openai_key)
            response = await client.audio.speech.create(
                model="tts-1",
                voice=self._openai_voice,
                input=text,
            )
            response.stream_to_file(output_path)
            logger.debug(f"OpenAI TTS saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"OpenAI TTS failed: {e}")
            return ""

    async def speak_and_play(self, text: str) -> None:
        """Speak text and play it through speakers."""
        audio_path = await self.speak(text)
        if not audio_path:
            logger.warning("No audio generated, skipping playback")
            return
        await self._play_audio(audio_path)
        Path(audio_path).unlink(missing_ok=True)

    async def _play_audio(self, path: str) -> None:
        """Play an audio file through the default audio device."""
        try:
            import subprocess
            import sys

            if sys.platform == "win32":
                # Windows: use built-in media player
                proc = await asyncio.create_subprocess_exec(
                    "powershell", "-c",
                    f'(New-Object Media.SoundPlayer "{path}").PlaySync()',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            elif sys.platform == "darwin":
                proc = await asyncio.create_subprocess_exec(
                    "afplay", path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            else:
                # Linux: try mpv, then aplay
                for player in ["mpv --no-video", "aplay", "paplay"]:
                    parts = player.split() + [path]
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            *parts,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        await proc.wait()
                        return
                    except FileNotFoundError:
                        continue
                logger.warning("No audio player found")

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
