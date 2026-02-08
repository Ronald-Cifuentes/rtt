"""
Circular audio buffer that maintains the last N seconds of PCM audio.
Used by the ASR engine for sliding-window transcription.
"""

import numpy as np
import threading
from typing import Optional


class AudioBuffer:
    """Thread-safe circular buffer for PCM float32 audio at a fixed sample rate."""

    def __init__(self, max_duration_sec: float = 10.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)
        self._buf = np.zeros(self.max_samples, dtype=np.float32)
        self._write_pos = 0  # how many samples written total (monotonic)
        self._lock = threading.Lock()

    # ── write ─────────────────────────────────────────────
    def append(self, pcm_float32: np.ndarray) -> None:
        """Append float32 samples. Automatically wraps when buffer is full."""
        with self._lock:
            n = len(pcm_float32)
            if n >= self.max_samples:
                # chunk bigger than buffer → keep only last max_samples
                self._buf[:] = pcm_float32[-self.max_samples:]
                self._write_pos += n
                return
            start = self._write_pos % self.max_samples
            end = start + n
            if end <= self.max_samples:
                self._buf[start:end] = pcm_float32
            else:
                first = self.max_samples - start
                self._buf[start:] = pcm_float32[:first]
                self._buf[: n - first] = pcm_float32[first:]
            self._write_pos += n

    def append_pcm16(self, pcm16_bytes: bytes) -> None:
        """Convenience: convert PCM16 bytes → float32 and append."""
        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        self.append(pcm16.astype(np.float32) / 32768.0)

    # ── read ──────────────────────────────────────────────
    def get_last(self, duration_sec: float) -> Optional[np.ndarray]:
        """Return the last `duration_sec` seconds of audio as float32 array."""
        with self._lock:
            total_written = self._write_pos
            if total_written == 0:
                return None
            n_want = min(int(duration_sec * self.sample_rate), total_written, self.max_samples)
            end = total_written % self.max_samples
            start = end - n_want
            if start >= 0:
                return self._buf[start:end].copy()
            else:
                return np.concatenate([self._buf[start:], self._buf[:end]]).copy()

    @property
    def total_samples_written(self) -> int:
        return self._write_pos

    @property
    def duration_available_sec(self) -> float:
        return min(self._write_pos, self.max_samples) / self.sample_rate

    def reset(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._write_pos = 0
