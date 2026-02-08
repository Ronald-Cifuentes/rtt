"""
ASR Engine — pseudo-streaming speech recognition.

Uses Qwen3-ASR (qwen-asr package) for multilingual speech recognition.
Operates on a sliding window from the AudioBuffer:
  1. Every ASR_INTERVAL_MS, grab the last WINDOW_SEC seconds of audio.
  2. Run Qwen3-ASR transcription on that window.
  3. Return the full hypothesis string.

Hallucination defences (applied to model output):
  - Audio energy gate: skip transcription if RMS is below threshold.
  - Pattern filter: drop common subtitle/hallucination phrases.
  - Repetition filter: detect repeated hallucination patterns.

The CommitTracker (external) decides what to commit.
"""

import asyncio
import logging
import re
import numpy as np
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hallucination heuristics ────────────────────────────

# Minimum RMS energy to consider audio as containing speech.
_MIN_RMS_ENERGY = 0.008  # ~-42 dB

# Repeated/hallucinated pattern detection
_HALLUCINATION_PATTERNS = re.compile(
    r'(subtitle|subscribe|suscr[ií]bete|suscr[ií]banse|gracias por ver|thank you for watching'
    r'|music|applause|m[uú]sica|aplausos'
    r'|Amara\.org|MoroccoEnglish|Madriman'
    r'|\bwww\.\w+\.\w+\b)',
    re.IGNORECASE,
)

# Qwen3-ASR expects language names (e.g. "Spanish", "English"). Map from our codes.
_LANG_CODE_TO_QWEN: dict[str, Optional[str]] = {
    "es": "Spanish",
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "yue": "Cantonese",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
    "el": "Greek",
    "hu": "Hungarian",
    "fa": "Persian",
    "fil": "Filipino",
    "cs": "Czech",
    "da": "Danish",
    "fi": "Finnish",
    "ro": "Romanian",
    "mk": "Macedonian",
}


def _compute_rms(audio: np.ndarray) -> float:
    """Compute Root Mean Square of audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def _is_repetitive(text: str, threshold: float = 0.5) -> bool:
    """Check if the text is mostly repeated tokens (hallucination pattern)."""
    words = text.lower().split()
    if len(words) < 4:
        return False
    unique = set(words)
    if len(unique) <= 2 and len(words) >= 6:
        return True
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(words) > threshold:
        return True
    return False


def _apply_post_filters(text: str) -> str:
    """Apply hallucination filters to raw ASR output. Returns empty string if rejected."""
    if not text or not text.strip():
        return ""
    t = text.strip()
    if _HALLUCINATION_PATTERNS.search(t):
        logger.debug(f"Dropping hallucination pattern: '{t[:50]}'")
        return ""
    if _is_repetitive(t):
        logger.debug(f"Dropping repetitive hypothesis: '{t[:80]}'")
        return ""
    return t


def _language_for_qwen(code: str) -> Optional[str]:
    """Map client language code (e.g. 'es') to Qwen3-ASR language name, or None for auto."""
    if not code:
        return None
    return _LANG_CODE_TO_QWEN.get(code.strip().lower(), None)


class ASREngine:
    """
    Qwen3-ASR based ASR with sliding-window pseudo-streaming.
    Same interface as before: load(), transcribe(audio, language) -> str.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        device: str = "cpu",
        max_new_tokens: int = 256,
        max_inference_batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_inference_batch_size = max_inference_batch_size
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the Qwen3-ASR model. Call once at startup."""
        if self._loaded:
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            device_map = "cuda:0" if self.device == "cuda" else "cpu"
            if self.device == "mps":
                device_map = "cpu"
                dtype = torch.float32
                logger.info("Qwen3-ASR: MPS not officially supported, using CPU")

            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=self.max_inference_batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            self._loaded = True
            logger.info(
                f"ASR loaded: Qwen3-ASR {self.model_name} on {device_map} ({dtype})"
            )
        except ImportError as e:
            logger.error(
                "qwen-asr not installed. Run: pip install qwen-asr"
            )
            raise ImportError("qwen-asr is required for ASR. pip install qwen-asr") from e

    async def transcribe(
        self,
        audio: np.ndarray,
        language: str = "es",
    ) -> str:
        """
        Transcribe audio (float32, 16kHz) and return the text.
        Runs in a thread pool to avoid blocking the event loop.
        Returns empty string if audio is too quiet or no speech detected.
        """
        if not self._loaded or self._model is None:
            return ""
        if audio is None or len(audio) < 8000:  # < 0.5s
            return ""

        rms = _compute_rms(audio)
        if rms < _MIN_RMS_ENERGY:
            logger.debug(f"Audio too quiet (RMS={rms:.5f}), skipping ASR")
            return ""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._transcribe_sync, audio, language
        )

    def _transcribe_sync(self, audio: np.ndarray, language: str) -> str:
        """Synchronous transcription (called from thread pool)."""
        try:
            # Qwen3-ASR accepts (np.ndarray, sample_rate) or path/URL/base64
            audio_input = (audio.astype(np.float32), 16000)
            lang = _language_for_qwen(language)  # "Spanish" or None for auto

            results = self._model.transcribe(
                audio=audio_input,
                language=lang,
            )
            if not results:
                return ""

            raw = results[0].text
            if not raw:
                return ""

            # Strip optional "lang XXX: " prefix if present (e.g. "lang English: Hi there")
            if raw.lower().startswith("lang "):
                idx = raw.find(":", 5)
                if idx != -1:
                    raw = raw[idx + 1 :].strip()

            return _apply_post_filters(raw)
        except Exception as e:
            logger.error(f"ASR transcription error: {e}", exc_info=True)
            return ""
