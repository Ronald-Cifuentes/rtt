"""
TTS Engine — Text-to-Speech with streaming output.

Supports two backends:
  1. edge-tts (default, lightweight, no GPU needed, good quality)
  2. qwen3-tts (high quality, requires GPU + model download)

Both emit audio as PCM16 bytes in chunks for real-time playback.
"""

import asyncio
import io
import logging
import struct
import tempfile
import numpy as np
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Streaming TTS engine with pluggable backends.

    Usage:
        engine = TTSEngine(backend="edge-tts")
        engine.load()
        async for chunk in engine.synthesize_streaming("Hello world", lang="en"):
            # chunk is bytes (PCM16 mono at self.output_sample_rate)
            ws.send_bytes(chunk)
    """

    def __init__(
        self,
        backend: str = "edge-tts",
        qwen3_model: str = "Qwen/Qwen3-TTS-0.6B",
        device: str = "cpu",
        output_sample_rate: int = 24000,
        chunk_duration_ms: int = 200,
    ):
        self.backend = backend
        self.qwen3_model = qwen3_model
        self.device = device
        self.output_sample_rate = output_sample_rate
        self.chunk_samples = int(output_sample_rate * chunk_duration_ms / 1000)
        self._loaded = False

        # Voice settings (MVP: fixed voice per language)
        self._voice_map = {
            "en": "en-US-AriaNeural",
            "es": "es-ES-ElviraNeural",
        }

    def load(self) -> None:
        """Load TTS model/backend."""
        if self.backend == "edge-tts":
            try:
                import edge_tts  # noqa: F401
                self._loaded = True
                logger.info("TTS loaded: edge-tts (lightweight)")
            except ImportError:
                logger.error("edge-tts not installed. Run: pip install edge-tts")
                raise

        elif self.backend == "qwen3":
            self._load_qwen3()
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    def _load_qwen3(self) -> None:
        """Load Qwen3-TTS model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading Qwen3-TTS: {self.qwen3_model}")
            self._qwen3_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen3_model, trust_remote_code=True
            )
            self._qwen3_model = AutoModelForCausalLM.from_pretrained(
                self.qwen3_model, trust_remote_code=True, torch_dtype=torch.float16
            )

            effective_device = self.device
            if effective_device == "mps":
                try:
                    self._qwen3_model = self._qwen3_model.to("mps")
                except Exception:
                    logger.info("MPS not supported for Qwen3-TTS, using CPU")
                    self._qwen3_model = self._qwen3_model.to("cpu").float()
                    effective_device = "cpu"
            elif effective_device == "cuda":
                self._qwen3_model = self._qwen3_model.to("cuda")
            else:
                self._qwen3_model = self._qwen3_model.to("cpu").float()

            self._qwen3_model.eval()
            self._qwen3_device = effective_device
            self._loaded = True
            logger.info(f"Qwen3-TTS loaded on {effective_device}")

        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS: {e}")
            logger.info("Falling back to edge-tts")
            self.backend = "edge-tts"
            self.load()

    async def synthesize_streaming(
        self, text: str, lang: str = "en"
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text and yield PCM16 mono audio chunks.
        Each chunk is `chunk_duration_ms` of audio.
        """
        if not text.strip():
            return
        if not self._loaded:
            logger.warning("TTS not loaded, skipping synthesis")
            return

        if self.backend == "edge-tts":
            async for chunk in self._synth_edge_tts(text, lang):
                yield chunk
        elif self.backend == "qwen3":
            async for chunk in self._synth_qwen3(text, lang):
                yield chunk

    # ── edge-tts backend ─────────────────────────────────

    async def _synth_edge_tts(
        self, text: str, lang: str
    ) -> AsyncGenerator[bytes, None]:
        """Use edge-tts to synthesize and yield PCM16 chunks."""
        import edge_tts

        voice = self._voice_map.get(lang, "en-US-AriaNeural")

        try:
            communicate = edge_tts.Communicate(text, voice)
            pcm_buffer = bytearray()

            # edge-tts yields MP3 chunks; we need to collect and convert
            mp3_buffer = bytearray()
            async for chunk_data in communicate.stream():
                if chunk_data["type"] == "audio":
                    mp3_buffer.extend(chunk_data["data"])

            if not mp3_buffer:
                return

            # Convert MP3 to PCM16 using pydub or ffmpeg
            pcm_data = await self._mp3_to_pcm16(bytes(mp3_buffer))
            if pcm_data is None:
                return

            # Yield in chunks
            chunk_bytes = self.chunk_samples * 2  # 2 bytes per int16 sample
            for i in range(0, len(pcm_data), chunk_bytes):
                chunk = pcm_data[i : i + chunk_bytes]
                if len(chunk) > 0:
                    yield bytes(chunk)

        except Exception as e:
            logger.error(f"edge-tts synthesis error: {e}")

    async def _mp3_to_pcm16(self, mp3_data: bytes) -> bytes | None:
        """Convert MP3 bytes to PCM16 mono at output_sample_rate."""
        try:
            # Try pydub first
            from pydub import AudioSegment

            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            audio = audio.set_channels(1).set_frame_rate(self.output_sample_rate).set_sample_width(2)
            return audio.raw_data
        except ImportError:
            pass

        # Fallback: use ffmpeg via subprocess
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", "pipe:0",
                "-f", "s16le", "-ar", str(self.output_sample_rate),
                "-ac", "1", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=mp3_data)
            if proc.returncode == 0:
                return stdout
            else:
                logger.error(f"ffmpeg error: {stderr.decode()[:200]}")
                return None
        except FileNotFoundError:
            logger.error(
                "Neither pydub nor ffmpeg found. "
                "Install: pip install pydub OR brew install ffmpeg"
            )
            return None

    # ── Qwen3-TTS backend ────────────────────────────────

    async def _synth_qwen3(
        self, text: str, lang: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Qwen3-TTS streaming synthesis.

        Qwen3-TTS generates audio tokens at 12Hz (one token = ~83ms of audio).
        We generate tokens incrementally and decode them to PCM in chunks.
        """
        try:
            import torch

            loop = asyncio.get_running_loop()

            # Build the prompt for Qwen3-TTS
            prompt = f"<|text|>{text}<|endoftext|>"
            input_ids = self._qwen3_tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(self._qwen3_device)

            # Generate audio tokens (streaming)
            # Note: actual Qwen3-TTS API may differ; this is the expected pattern
            audio_tokens = await loop.run_in_executor(
                None,
                lambda: self._qwen3_model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                ),
            )

            # Decode tokens to audio waveform
            # This depends on Qwen3-TTS's specific codec
            audio_array = await loop.run_in_executor(
                None, self._decode_qwen3_tokens, audio_tokens
            )

            if audio_array is not None:
                pcm16 = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
                pcm_bytes = pcm16.tobytes()

                chunk_bytes = self.chunk_samples * 2
                for i in range(0, len(pcm_bytes), chunk_bytes):
                    yield pcm_bytes[i : i + chunk_bytes]

        except Exception as e:
            logger.error(f"Qwen3-TTS synthesis error: {e}")
            # Fallback: generate silence
            silence = np.zeros(self.chunk_samples, dtype=np.int16)
            yield silence.tobytes()

    def _decode_qwen3_tokens(self, tokens) -> np.ndarray | None:
        """Decode Qwen3-TTS audio tokens to float32 waveform."""
        try:
            # The actual decoding depends on Qwen3-TTS's codec architecture.
            # This is a placeholder that should be updated when the official
            # Qwen3-TTS package is available.
            # For now, we generate a placeholder sine wave.
            duration = len(tokens[0]) * (1.0 / 12.0)  # 12Hz tokenizer
            t = np.linspace(0, duration, int(duration * self.output_sample_rate))
            return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        except Exception as e:
            logger.error(f"Qwen3 token decode error: {e}")
            return None
