"""Voice input — speech-to-text via Whisper (local or OpenAI API).

Supports:
- Local: openai-whisper package (free, runs on CPU/GPU)
- Cloud: OpenAI Whisper API (fast, requires API key)
"""

import io
import logging
import tempfile
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VoiceInput:
    """Speech-to-text using Whisper."""

    def __init__(
        self,
        provider: str = "local",
        model: str = "base",
        openai_api_key: str = "",
    ):
        self._provider = provider
        self._model_name = model
        self._openai_key = openai_api_key
        self._local_model = None

    def _load_local_model(self):
        """Lazy-load local Whisper model."""
        if self._local_model is not None:
            return self._local_model
        import whisper
        logger.info(f"Loading Whisper model: {self._model_name}")
        self._local_model = whisper.load_model(self._model_name)
        return self._local_model

    async def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        if self._provider == "openai" and self._openai_key:
            return await self._transcribe_openai(audio_path)
        return self._transcribe_local(audio_path)

    def _transcribe_local(self, audio_path: str) -> str:
        """Transcribe using local Whisper model."""
        try:
            model = self._load_local_model()
            result = model.transcribe(audio_path)
            text = result.get("text", "").strip()
            logger.debug(f"Transcribed ({len(text)} chars): {text[:80]}...")
            return text
        except ImportError:
            return "Error: openai-whisper not installed. Run: pip install openai-whisper"
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return f"Error: {e}"

    async def _transcribe_openai(self, audio_path: str) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self._openai_key)
            with open(audio_path, "rb") as f:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            return response.text.strip()
        except Exception as e:
            logger.error(f"OpenAI Whisper API failed: {e}")
            return f"Error: {e}"

    def record_and_transcribe(self, duration: float = 5.0, sample_rate: int = 16000) -> str:
        """Record from microphone and transcribe. Blocking call."""
        try:
            import sounddevice as sd
            import numpy as np

            logger.info(f"Recording {duration}s of audio...")
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            # Save to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # int16
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio.tobytes())
                tmp_path = tmp.name

            text = self._transcribe_local(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            return text

        except ImportError:
            return "Error: sounddevice not installed. Run: pip install sounddevice"
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return f"Error: {e}"
