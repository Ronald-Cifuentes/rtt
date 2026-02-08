"""
Pipeline Orchestrator — connects ASR → CommitTracker → MT → TTS.

Runs the ASR loop at a fixed interval, feeds hypotheses to the CommitTracker,
and dispatches committed text to MT+TTS in sequence.

All output is emitted as events to an asyncio.Queue that the WebSocket
handler reads from.
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Any

from ..core.audio_buffer import AudioBuffer
from ..core.commit_tracker import CommitTracker, CommitEvent
from ..core.backpressure import BackpressureController
from .asr import ASREngine
from .mt import MTEngine
from .tts import TTSEngine
from ..config import (
    WINDOW_SEC,
    ASR_INTERVAL_MS,
    COMMIT_STABILITY_K,
    COMMIT_TIMEOUT_SEC,
    COMMIT_MIN_WORDS,
    TTS_QUEUE_MAX,
    CAPTURE_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# Minimum RMS to consider the window as having speech activity.
# This is a secondary gate (ASR has its own), applied before even calling ASR
# to save CPU cycles.
_SILENCE_RMS_THRESHOLD = 0.005


@dataclass
class PipelineStats:
    asr_ms: float = 0.0
    mt_ms: float = 0.0
    tts_ms: float = 0.0
    e2e_ms: float = 0.0
    commits_total: int = 0
    tts_queue: int = 0


class PipelineOrchestrator:
    """
    Manages the full ASR→MT→TTS pipeline for one WebSocket session.

    Events emitted to `output_queue`:
      - {"type": "partial_transcript", "text": str}
      - {"type": "committed_transcript", "text": str, "segment_id": int}
      - {"type": "translation_committed", "text": str, "source": str, "segment_id": int}
      - {"type": "tts_audio_chunk", "data": bytes, "segment_id": int, "is_last": bool}
      - {"type": "stats", ...}
    """

    def __init__(
        self,
        asr_engine: ASREngine,
        mt_engine: MTEngine,
        tts_engine: TTSEngine,
        source_lang: str = "es",
        target_lang: str = "en",
    ):
        self.asr = asr_engine
        self.mt = mt_engine
        self.tts = tts_engine
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Audio buffer (10s circular)
        self.audio_buffer = AudioBuffer(
            max_duration_sec=max(WINDOW_SEC * 2, 10.0),
            sample_rate=CAPTURE_SAMPLE_RATE,
        )

        # Commit tracker
        self.commit_tracker = CommitTracker(
            stability_k=COMMIT_STABILITY_K,
            timeout_sec=COMMIT_TIMEOUT_SEC,
            min_words=COMMIT_MIN_WORDS,
        )

        # Backpressure
        self.bp = BackpressureController(queue_max=TTS_QUEUE_MAX)

        # Output queue for WS handler
        self.output_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Stats
        self.stats = PipelineStats()

        # Silence tracking — count consecutive silent windows to avoid
        # timeout-commits on silence that slipped past filters.
        self._consecutive_silent_windows = 0

        # Control
        self._running = False
        self._asr_task: asyncio.Task | None = None

    # ── lifecycle ─────────────────────────────────────────

    def start(self) -> None:
        """Start the ASR loop."""
        self._running = True
        self._asr_task = asyncio.create_task(self._asr_loop())
        logger.info(
            f"Pipeline started: {self.source_lang}→{self.target_lang}, "
            f"ASR every {ASR_INTERVAL_MS}ms, window={WINDOW_SEC}s"
        )

    async def stop(self) -> None:
        """Stop the pipeline and force-commit any remaining text."""
        self._running = False
        if self._asr_task:
            self._asr_task.cancel()
            try:
                await self._asr_task
            except asyncio.CancelledError:
                pass

        # Force commit remaining
        events = self.commit_tracker.force_commit()
        for ev in events:
            await self._process_commit(ev)

        logger.info("Pipeline stopped")

    # ── audio input ───────────────────────────────────────

    def feed_audio(self, pcm16_bytes: bytes) -> None:
        """Feed raw PCM16 audio bytes from the WebSocket."""
        self.audio_buffer.append_pcm16(pcm16_bytes)

    # ── ASR loop ──────────────────────────────────────────

    async def _asr_loop(self) -> None:
        """Periodic ASR decode on the sliding window."""
        interval = ASR_INTERVAL_MS / 1000.0

        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break

                # Grab the last WINDOW_SEC seconds
                audio = self.audio_buffer.get_last(WINDOW_SEC)
                if audio is None or len(audio) < CAPTURE_SAMPLE_RATE * 0.5:
                    continue

                # ── Quick silence check (saves CPU) ──
                rms = float(np.sqrt(np.mean(audio ** 2)))
                if rms < _SILENCE_RMS_THRESHOLD:
                    self._consecutive_silent_windows += 1
                    continue
                else:
                    self._consecutive_silent_windows = 0

                # ASR
                t0 = time.monotonic()
                hypothesis = await self.asr.transcribe(audio, language=self.source_lang)
                asr_ms = (time.monotonic() - t0) * 1000
                self.stats.asr_ms = asr_ms

                if not hypothesis:
                    continue

                # Feed to commit tracker FIRST (strips committed prefix)
                commit_events = self.commit_tracker.update(hypothesis)

                # Emit partial transcript with ONLY uncommitted text
                # (the raw hypothesis includes re-transcribed committed text
                #  from the sliding window — we must not show that)
                uncommitted = self.commit_tracker.effective_uncommitted_text
                if uncommitted:
                    await self.output_queue.put({
                        "type": "partial_transcript",
                        "text": uncommitted,
                    })

                for ev in commit_events:
                    await self._process_commit(ev)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ASR loop error: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    # ── commit processing ─────────────────────────────────

    async def _process_commit(self, ev: CommitEvent) -> None:
        """Translate and synthesize a committed segment."""
        e2e_start = time.monotonic()

        # Emit committed transcript
        await self.output_queue.put({
            "type": "committed_transcript",
            "text": ev.text,
            "segment_id": ev.segment_id,
        })

        self.stats.commits_total += 1

        # Check backpressure
        if self.bp.should_skip_tts():
            logger.warning(f"Skipping TTS for segment {ev.segment_id} (backpressure)")
            return

        if self.bp.should_batch():
            self.bp.add_to_batch(ev.text)
            return

        text_to_process = ev.text

        # Flush any batched text
        batched = self.bp.flush_batch()
        if batched:
            text_to_process = batched + " " + text_to_process

        # MT
        t0 = time.monotonic()
        translation = await self.mt.translate(
            text_to_process, self.source_lang, self.target_lang
        )
        mt_ms = (time.monotonic() - t0) * 1000
        self.stats.mt_ms = mt_ms

        await self.output_queue.put({
            "type": "translation_committed",
            "text": translation,
            "source": text_to_process,
            "segment_id": ev.segment_id,
        })

        # TTS
        self.bp.on_tts_queued()
        t0 = time.monotonic()
        chunk_count = 0
        async for audio_chunk in self.tts.synthesize_streaming(
            translation, lang=self.target_lang
        ):
            chunk_count += 1
            await self.output_queue.put({
                "type": "tts_audio_chunk",
                "data": audio_chunk,
                "segment_id": ev.segment_id,
                "is_last": False,
            })

        # Mark last chunk
        if chunk_count > 0:
            # The last item we sent had is_last=False; we send an end marker
            await self.output_queue.put({
                "type": "tts_end",
                "segment_id": ev.segment_id,
            })

        tts_ms = (time.monotonic() - t0) * 1000
        self.stats.tts_ms = tts_ms
        self.bp.on_tts_completed()

        e2e_ms = (time.monotonic() - e2e_start) * 1000
        self.stats.e2e_ms = e2e_ms
        self.stats.tts_queue = self.bp.pending_count

        # Emit stats
        await self.output_queue.put({
            "type": "stats",
            "asr_ms": round(self.stats.asr_ms, 1),
            "mt_ms": round(mt_ms, 1),
            "tts_ms": round(tts_ms, 1),
            "e2e_ms": round(e2e_ms, 1),
            "commits_total": self.stats.commits_total,
            "tts_queue": self.stats.tts_queue,
        })
