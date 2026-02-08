"""
ASR Engine — pseudo-streaming speech recognition.

Uses faster-whisper (CTranslate2) for efficient CPU/GPU inference.
Operates on a sliding window from the AudioBuffer:
  1. Every ASR_INTERVAL_MS, grab the last WINDOW_SEC seconds of audio.
  2. Run Whisper transcription on that window.
  3. Return the full hypothesis string.

Hallucination defences:
  - Audio energy gate: skip transcription if RMS is below threshold.
  - Silero VAD pre-filter: faster-whisper's built-in VAD discards silence.
  - Per-segment filtering: drop segments with high no_speech_prob or low avg_logprob.
  - Repetition filter: detect repeated hallucination patterns.

The CommitTracker (external) decides what to commit.
"""

import asyncio
import logging
import re
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hallucination heuristics ────────────────────────────

# Minimum RMS energy to consider audio as containing speech.
# Below this, we skip ASR entirely (prevents silence hallucinations).
_MIN_RMS_ENERGY = 0.008  # ~-42 dB; typical quiet room noise is ~0.002-0.005

# Whisper segment filters
_MAX_NO_SPEECH_PROB = 0.6    # discard segments above this
_MIN_AVG_LOGPROB = -1.0      # discard segments below this (low confidence)

# Repeated/hallucinated pattern detection
# Added Spanish variants of common YouTube/subtitle hallucinations
_HALLUCINATION_PATTERNS = re.compile(
    r'(subtitle|subscribe|suscr[ií]bete|suscr[ií]banse|gracias por ver|thank you for watching'
    r'|music|applause|m[uú]sica|aplausos'
    r'|Amara\.org|MoroccoEnglish|Madriman'
    r'|\bwww\.\w+\.\w+\b)',
    re.IGNORECASE,
)


def _compute_rms(audio: np.ndarray) -> float:
    """Compute Root Mean Square of audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def _is_repetitive(text: str, threshold: float = 0.5) -> bool:
    """Check if the text is mostly repeated tokens (hallucination pattern)."""
    words = text.lower().split()
    if len(words) < 4:
        return False
    unique = set(words)
    # If more than half the words are just 1-2 unique tokens, it's repetitive
    if len(unique) <= 2 and len(words) >= 6:
        return True
    # Check if any single word appears > threshold of the time
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(words) > threshold:
        return True
    return False


class ASREngine:
    """Whisper-based ASR with sliding-window pseudo-streaming."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the faster-whisper model. Call once at startup."""
        if self._loaded:
            return
        try:
            from faster_whisper import WhisperModel

            compute = "float32"
            fw_device = "cpu"
            if self.device == "cuda":
                fw_device = "cuda"
                compute = "float16"
            elif self.device == "mps":
                # faster-whisper doesn't support MPS directly;
                # it uses CTranslate2 which is CPU or CUDA.
                fw_device = "cpu"
                compute = "float32"
                logger.info("faster-whisper: MPS not supported, using CPU")

            self._model = WhisperModel(
                self.model_size,
                device=fw_device,
                compute_type=compute,
            )
            self._loaded = True
            logger.info(
                f"ASR loaded: faster-whisper {self.model_size} on {fw_device} ({compute})"
            )
        except ImportError:
            logger.error(
                "faster-whisper not installed. "
                "Run: pip install faster-whisper"
            )
            raise

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

        # ── Energy gate: skip ASR if audio is too quiet ──
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
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,              # Increased from 3 for better accuracy
                best_of=3,                # Increased from 1: try 3 candidates, pick best
                temperature=0.0,         # Deterministic (greedy decoding)
                without_timestamps=False, # need timestamps for filtering
                vad_filter=True,          # Silero VAD: suppress silence hallucinations
                vad_parameters=dict(
                    min_silence_duration_ms=300,   # merge pauses < 300ms
                    speech_pad_ms=200,             # padding around speech
                    threshold=0.4,                 # VAD sensitivity: increased from 0.35 to 0.4
                    # Higher threshold = less sensitive = fewer false cuts on speech
                ),
            )
            text_parts = []
            for seg in segments:
                # ── Per-segment hallucination filters ──
                if seg.no_speech_prob > _MAX_NO_SPEECH_PROB:
                    logger.debug(
                        f"Dropping segment (no_speech_prob={seg.no_speech_prob:.2f}): "
                        f"'{seg.text.strip()[:50]}'"
                    )
                    continue

                if seg.avg_logprob < _MIN_AVG_LOGPROB:
                    logger.debug(
                        f"Dropping segment (avg_logprob={seg.avg_logprob:.2f}): "
                        f"'{seg.text.strip()[:50]}'"
                    )
                    continue

                text = seg.text.strip()

                # Pattern-based hallucination filter
                if _HALLUCINATION_PATTERNS.search(text):
                    logger.debug(f"Dropping hallucination pattern: '{text[:50]}'")
                    continue

                if text:
                    text_parts.append(text)

            result = " ".join(text_parts).strip()

            # Final check: repetitive output
            if _is_repetitive(result):
                logger.debug(f"Dropping repetitive hypothesis: '{result[:80]}'")
                return ""

            return result

        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            return ""
