"""Tests for CommitTracker (commit-by-stability logic)."""
import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.core.commit_tracker import CommitTracker, CommitEvent


class TestCommitTracker:
    def test_no_commit_until_k_stable(self):
        """No commit should happen until K consecutive stable hypotheses."""
        tracker = CommitTracker(stability_k=3, timeout_sec=100.0, min_words=2)
        # Different hypotheses each time — no stability
        events1 = tracker.update("hello")
        events2 = tracker.update("hello world")
        assert events1 == []
        assert events2 == []

    def test_commit_on_stable_prefix(self):
        """Should commit when the same prefix is stable K times."""
        tracker = CommitTracker(stability_k=3, timeout_sec=100.0, min_words=2)
        # Same hypothesis 3 times in a row
        tracker.update("hello world")
        tracker.update("hello world")
        events = tracker.update("hello world")
        assert len(events) == 1
        assert events[0].text == "hello world"

    def test_commit_on_timeout(self):
        """Should commit after timeout_sec even if not stable K times."""
        tracker = CommitTracker(stability_k=5, timeout_sec=0.05, min_words=2)
        tracker.update("hello world")
        tracker.update("hello world again")
        time.sleep(0.1)
        events = tracker.update("hello world again more")
        # Timeout triggers commit of all effective words
        assert len(events) == 1

    def test_no_duplicate_commits(self):
        """Should not commit already-committed text again."""
        tracker = CommitTracker(stability_k=3, timeout_sec=100.0, min_words=1)
        # Commit "hello"
        tracker.update("hello")
        tracker.update("hello")
        events1 = tracker.update("hello")
        assert len(events1) == 1
        assert events1[0].text == "hello"

        # Same text again — prefix stripping removes "hello", nothing left
        events2 = tracker.update("hello")
        events3 = tracker.update("hello")
        events4 = tracker.update("hello")
        assert events2 == []
        assert events3 == []
        assert events4 == []

    def test_incremental_commits(self):
        """Should commit new text incrementally as it becomes stable."""
        tracker = CommitTracker(stability_k=3, timeout_sec=100.0, min_words=2)

        # First phrase stabilizes
        tracker.update("hello world")
        tracker.update("hello world")
        events1 = tracker.update("hello world")
        assert len(events1) == 1
        assert events1[0].text == "hello world"

        # Text grows — new suffix should commit when stable
        tracker.update("hello world how are you")
        tracker.update("hello world how are you")
        events2 = tracker.update("hello world how are you")
        assert len(events2) == 1
        # Only the NEW part is committed
        assert "how are you" in events2[0].text
        assert "hello world" not in events2[0].text

    def test_reset(self):
        """Reset should clear all state."""
        tracker = CommitTracker(stability_k=3, timeout_sec=1.0, min_words=2)
        tracker.update("hello world")
        tracker.update("hello world")
        tracker.update("hello world")
        tracker.reset()
        assert tracker.all_committed_text == ""
        assert tracker.effective_uncommitted_text == ""

    # ── Prefix Stripping Tests (the core bug fix) ────────

    def test_strip_full_committed_prefix(self):
        """Should strip all committed words from hypothesis, not just last 3."""
        tracker = CommitTracker(stability_k=2, timeout_sec=100.0, min_words=2)

        # Commit "hola como estás" (2 stable hits)
        tracker.update("hola como estás")
        events1 = tracker.update("hola como estás")
        assert len(events1) == 1
        assert events1[0].text == "hola como estás"

        # ASR re-transcribes the full committed text + new text
        # This is the EXACT bug from the screenshot
        tracker.update("hola como estás es interesante")
        events2 = tracker.update("hola como estás es interesante")
        assert len(events2) == 1
        # Should only contain the NEW part, no duplication
        assert events2[0].text == "es interesante"

    def test_strip_prefix_with_punctuation_variation(self):
        """Punctuation differences (e.g. 'como...' vs 'como') should still match."""
        tracker = CommitTracker(stability_k=2, timeout_sec=100.0, min_words=2)

        # Commit with punctuation
        tracker.update("hola, como...")
        events1 = tracker.update("hola, como...")
        assert len(events1) == 1

        # ASR reproduced the same words without the same punctuation
        tracker.update("hola como estás bien")
        events2 = tracker.update("hola como estás bien")
        assert len(events2) == 1
        # "hola como" should be stripped (matches committed "hola," "como...")
        # leaving only "estás bien"
        assert "estás bien" in events2[0].text
        assert "hola" not in events2[0].text

    def test_strip_long_committed_prefix(self):
        """Should handle many committed words (> old context_tail_words=3)."""
        tracker = CommitTracker(stability_k=2, timeout_sec=100.0, min_words=2)

        # Commit 5 words
        tracker.update("uno dos tres cuatro cinco")
        events1 = tracker.update("uno dos tres cuatro cinco")
        assert len(events1) == 1
        assert events1[0].text == "uno dos tres cuatro cinco"

        # ASR re-transcribes all 5 words + new ones
        tracker.update("uno dos tres cuatro cinco seis siete")
        events2 = tracker.update("uno dos tres cuatro cinco seis siete")
        assert len(events2) == 1
        assert events2[0].text == "seis siete"

    def test_strip_partial_overlap(self):
        """Should handle partial overlap (hypothesis starts mid-committed)."""
        tracker = CommitTracker(stability_k=2, timeout_sec=100.0, min_words=2)

        # Commit "uno dos tres cuatro cinco"
        tracker.update("uno dos tres cuatro cinco")
        events1 = tracker.update("uno dos tres cuatro cinco")
        assert len(events1) == 1

        # ASR window only covers the tail — starts at "tres"
        tracker.update("tres cuatro cinco seis siete")
        events2 = tracker.update("tres cuatro cinco seis siete")
        assert len(events2) == 1
        assert events2[0].text == "seis siete"

    def test_entire_hypothesis_already_committed(self):
        """If the hypothesis is entirely committed text, nothing new is emitted."""
        tracker = CommitTracker(stability_k=2, timeout_sec=100.0, min_words=2)

        tracker.update("hola mundo")
        events1 = tracker.update("hola mundo")
        assert len(events1) == 1

        # ASR just repeats the same committed text
        events2 = tracker.update("hola mundo")
        assert events2 == []
        assert tracker.effective_uncommitted_text == ""

    def test_effective_uncommitted_text(self):
        """effective_uncommitted_text should expose only the non-committed part."""
        tracker = CommitTracker(stability_k=3, timeout_sec=100.0, min_words=2)

        tracker.update("hola mundo")
        tracker.update("hola mundo")
        events = tracker.update("hola mundo")
        assert len(events) == 1

        # Now new text appears
        tracker.update("hola mundo como estás")
        assert "como estás" in tracker.effective_uncommitted_text
        assert "hola" not in tracker.effective_uncommitted_text

    def test_force_commit(self):
        """Force commit should flush remaining uncommitted text."""
        tracker = CommitTracker(stability_k=10, timeout_sec=100.0, min_words=2)
        tracker.update("some unstable text")
        events = tracker.force_commit()
        assert len(events) == 1
        assert events[0].text == "some unstable text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
