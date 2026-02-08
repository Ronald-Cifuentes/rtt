"""
Commit-by-stability algorithm.

Instead of relying on VAD to detect speech boundaries, we compare
consecutive ASR hypotheses.  When a word-level prefix stays identical
for K consecutive ASR runs (or a time-out is reached), that prefix
is "committed" — i.e. sent to MT+TTS exactly once.

Design:
  - Hypotheses are split into words.
  - We track how many consecutive times each word position has been stable.
  - Once a prefix of N words is stable for K runs, those words are committed.
  - The full committed history is used to strip re-transcribed text from
    new hypotheses (the ASR sliding window re-covers committed audio).
"""

import re
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Punctuation pattern for normalization
_PUNCT_RE = re.compile(r'[.,;:!?\-–—¿¡"\'…()\[\]{}]+')


def _normalize(word: str) -> str:
    """Lowercase + strip punctuation for fuzzy comparison."""
    return _PUNCT_RE.sub('', word.lower())


@dataclass
class CommitEvent:
    """Represents newly committed text."""
    text: str
    segment_id: int
    timestamp: float


class CommitTracker:
    """
    Tracks ASR hypothesis stability and produces CommitEvents.

    Parameters
    ----------
    stability_k : int
        Number of consecutive identical hypotheses needed to commit.
    timeout_sec : float
        Force-commit after this many seconds even if stability_k is not met.
    min_words : int
        Minimum number of new words before we allow a commit.
    """

    def __init__(
        self,
        stability_k: int = 3,
        timeout_sec: float = 3.0,
        min_words: int = 2,
        context_tail_words: int = 3,  # kept for API compat but not used for stripping
    ):
        self.stability_k = stability_k
        self.timeout_sec = timeout_sec
        self.min_words = min_words

        # State
        self._prev_words: list[str] = []
        self._stability_counts: list[int] = []  # per word-position
        self._committed_words: list[str] = []
        self._segment_id: int = 0
        self._last_commit_time: float = time.monotonic()
        self._last_hypothesis_time: float = time.monotonic()
        self._last_effective_words: list[str] = []  # exposed for partial transcript

    # ── public API ────────────────────────────────────────

    def update(self, hypothesis: str) -> list[CommitEvent]:
        """
        Feed a new ASR hypothesis string.
        Returns a (possibly empty) list of CommitEvents.
        """
        now = time.monotonic()
        self._last_hypothesis_time = now
        new_words = hypothesis.strip().split()
        events: list[CommitEvent] = []

        # Remove already-committed prefix from hypothesis
        effective_words = self._strip_committed_prefix(new_words)
        self._last_effective_words = effective_words

        # Compare with previous effective hypothesis word-by-word
        self._update_stability(effective_words)
        self._prev_words = effective_words

        # Find the longest stable prefix
        stable_len = self._longest_stable_prefix()

        # Check commit criteria
        time_since_commit = now - self._last_commit_time
        should_commit = False
        commit_len = 0

        if stable_len >= self.min_words and self._all_stable_k(stable_len):
            should_commit = True
            commit_len = stable_len
        elif time_since_commit >= self.timeout_sec and len(effective_words) >= self.min_words:
            # Timeout-based commit: commit whatever we have
            should_commit = True
            commit_len = len(effective_words)
            logger.debug("Timeout commit triggered")

        if should_commit and commit_len > 0:
            words_to_commit = effective_words[:commit_len]
            committed_text = " ".join(words_to_commit)
            self._segment_id += 1
            event = CommitEvent(
                text=committed_text,
                segment_id=self._segment_id,
                timestamp=now,
            )
            events.append(event)

            # Update committed state
            self._committed_words.extend(words_to_commit)
            self._last_commit_time = now

            # Reset stability for the remaining (uncommitted) words
            self._prev_words = effective_words[commit_len:]
            self._stability_counts = [0] * len(self._prev_words)
            self._last_effective_words = self._prev_words[:]

            logger.info(f"Committed [{self._segment_id}]: '{committed_text}'")

        return events

    @property
    def effective_uncommitted_text(self) -> str:
        """The current uncommitted text (after prefix stripping).
        Use this for the partial transcript display instead of the raw hypothesis."""
        return " ".join(self._last_effective_words)

    def force_commit(self) -> list[CommitEvent]:
        """Force-commit any remaining unstable text (e.g. on stop)."""
        if not self._prev_words:
            return []
        text = " ".join(self._prev_words)
        if not text.strip():
            return []
        self._segment_id += 1
        event = CommitEvent(
            text=text,
            segment_id=self._segment_id,
            timestamp=time.monotonic(),
        )
        self._committed_words.extend(self._prev_words)
        self._prev_words = []
        self._stability_counts = []
        self._last_commit_time = time.monotonic()
        logger.info(f"Force-committed [{self._segment_id}]: '{text}'")
        return [event]

    @property
    def context_tail(self) -> str:
        """Return the last few committed words as context for ASR."""
        tail = self._committed_words[-5:]
        return " ".join(tail)

    @property
    def all_committed_text(self) -> str:
        return " ".join(self._committed_words)

    def reset(self) -> None:
        self._prev_words = []
        self._stability_counts = []
        self._committed_words = []
        self._segment_id = 0
        self._last_commit_time = time.monotonic()
        self._last_effective_words = []

    # ── internals ─────────────────────────────────────────

    def _strip_committed_prefix(self, words: list[str]) -> list[str]:
        """
        Remove already-committed words from the beginning of the hypothesis.

        The ASR sliding window covers the last WINDOW_SEC seconds of audio.
        This audio may include content that was already committed. The ASR
        will re-transcribe it, so the hypothesis starts with already-committed
        text. We find the longest overlap between any suffix of committed_words
        and a prefix of the hypothesis, then strip it.

        Uses normalized comparison (lowercase, no punctuation) so minor
        differences like "como..." vs "como" still match.
        """
        if not self._committed_words or not words:
            return words

        committed_norm = [_normalize(w) for w in self._committed_words]
        words_norm = [_normalize(w) for w in words]

        # Limit lookback: the ASR window is ~5s at ~3 words/s = ~15 words.
        # Use generous margin of 40 words to cover fast speech.
        max_lookback = min(len(committed_norm), 40)
        search_committed = committed_norm[-max_lookback:]

        best_strip = 0

        # Try every starting position in the committed suffix.
        # For each start, see how many words from search_committed[start:]
        # match words_norm[0:] contiguously.
        for start in range(len(search_committed)):
            suffix = search_committed[start:]
            match_len = 0
            for j in range(min(len(suffix), len(words_norm))):
                if suffix[j] == words_norm[j]:
                    match_len = j + 1
                else:
                    break
            if match_len > best_strip:
                best_strip = match_len

        # Also check if the ENTIRE hypothesis is a subset of committed text
        # (happens when the user stops talking and ASR just repeats old text)
        if best_strip == len(words_norm):
            logger.debug("Entire hypothesis is already committed — discarding")
            return []

        if best_strip > 0:
            logger.debug(f"Stripped {best_strip} committed words from hypothesis "
                        f"({' '.join(words[:best_strip][:5])}...)")

        return words[best_strip:]

    def _update_stability(self, new_words: list[str]) -> None:
        """Compare new_words with prev_words position by position."""
        new_counts = []
        for i, w in enumerate(new_words):
            if i < len(self._prev_words) and _normalize(w) == _normalize(self._prev_words[i]):
                prev_count = self._stability_counts[i] if i < len(self._stability_counts) else 0
                new_counts.append(prev_count + 1)
            else:
                new_counts.append(1)
        self._stability_counts = new_counts

    def _longest_stable_prefix(self) -> int:
        """Length of the longest prefix where every word has count >= stability_k."""
        length = 0
        for count in self._stability_counts:
            if count >= self.stability_k:
                length += 1
            else:
                break
        return length

    def _all_stable_k(self, up_to: int) -> bool:
        for i in range(up_to):
            if i >= len(self._stability_counts) or self._stability_counts[i] < self.stability_k:
                return False
        return True
