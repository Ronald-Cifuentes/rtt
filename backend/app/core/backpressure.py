"""
Backpressure controller.

Monitors the TTS output queue. When TTS falls behind:
  1. First, increase the commit batch size (merge multiple commits).
  2. If still behind, skip TTS for some commits (text-only).
  3. Never drop committed text — only degrade audio output.
"""

import logging
import time

logger = logging.getLogger(__name__)


class BackpressureController:
    """Adaptive backpressure for the TTS pipeline."""

    def __init__(self, queue_max: int = 5):
        self.queue_max = queue_max
        self._pending_tts: int = 0
        self._batch_mode: bool = False
        self._skip_tts: bool = False
        self._batch_buffer: list[str] = []

    def on_tts_queued(self) -> None:
        self._pending_tts += 1
        self._evaluate()

    def on_tts_completed(self) -> None:
        self._pending_tts = max(0, self._pending_tts - 1)
        self._evaluate()

    def _evaluate(self) -> None:
        if self._pending_tts > self.queue_max:
            if not self._batch_mode:
                logger.warning(
                    f"TTS backpressure: queue={self._pending_tts}, "
                    f"switching to batch mode"
                )
            self._batch_mode = True
        elif self._pending_tts > self.queue_max * 2:
            if not self._skip_tts:
                logger.warning("TTS backpressure: skipping TTS for some commits")
            self._skip_tts = True
        else:
            self._batch_mode = False
            self._skip_tts = False

    def should_skip_tts(self) -> bool:
        """If True, caller should not synthesize this commit."""
        return self._skip_tts

    def should_batch(self) -> bool:
        """If True, caller should accumulate text and synthesize in larger chunks."""
        return self._batch_mode

    def add_to_batch(self, text: str) -> None:
        self._batch_buffer.append(text)

    def flush_batch(self) -> str | None:
        """Return accumulated batch text (if any) and clear buffer."""
        if not self._batch_buffer:
            return None
        merged = " ".join(self._batch_buffer)
        self._batch_buffer.clear()
        return merged

    @property
    def pending_count(self) -> int:
        return self._pending_tts

    def reset(self) -> None:
        self._pending_tts = 0
        self._batch_mode = False
        self._skip_tts = False
        self._batch_buffer.clear()
