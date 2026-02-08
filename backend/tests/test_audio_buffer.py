"""Tests for AudioBuffer (circular buffer logic)."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.core.audio_buffer import AudioBuffer


class TestAudioBuffer:
    def test_add_chunk_basic(self):
        """Test adding a single chunk to the buffer."""
        buf = AudioBuffer(sample_rate=16000, max_duration_sec=5.0)
        chunk = np.ones(1600, dtype=np.float32) * 0.5
        buf.add_chunk(chunk)
        assert len(buf.buffer) == 1600
        assert buf.total_samples_processed == 1600

    def test_circular_eviction(self):
        """Test that buffer evicts old samples when full."""
        buf = AudioBuffer(sample_rate=16000, max_duration_sec=1.0)  # 16000 max samples
        
        # Add 2 seconds worth of data (should evict first second)
        chunk1 = np.ones(16000, dtype=np.float32) * 0.3
        chunk2 = np.ones(16000, dtype=np.float32) * 0.7
        
        buf.add_chunk(chunk1)
        buf.add_chunk(chunk2)
        
        # Buffer should only have 16000 samples (max 1s)
        assert len(buf.buffer) == 16000
        # The remaining samples should be from chunk2
        window = buf.get_latest_window(1.0)
        np.testing.assert_allclose(window, 0.7, atol=0.01)

    def test_get_latest_window(self):
        """Test retrieving the latest audio window."""
        buf = AudioBuffer(sample_rate=16000, max_duration_sec=5.0)
        
        # Add 3 seconds of audio
        for i in range(3):
            chunk = np.ones(16000, dtype=np.float32) * (i + 1) * 0.1
            buf.add_chunk(chunk)
        
        # Get last 1 second
        window = buf.get_latest_window(1.0)
        assert len(window) == 16000
        np.testing.assert_allclose(window, 0.3, atol=0.01)
        
        # Get last 2 seconds
        window2 = buf.get_latest_window(2.0)
        assert len(window2) == 32000

    def test_get_window_when_buffer_smaller(self):
        """Test that window returns available data when buffer is shorter than requested."""
        buf = AudioBuffer(sample_rate=16000, max_duration_sec=5.0)
        chunk = np.ones(8000, dtype=np.float32) * 0.5  # 0.5 seconds
        buf.add_chunk(chunk)
        
        # Request 2 seconds but only 0.5 is available
        window = buf.get_latest_window(2.0)
        assert len(window) == 8000

    def test_reset(self):
        """Test buffer reset."""
        buf = AudioBuffer(sample_rate=16000, max_duration_sec=5.0)
        buf.add_chunk(np.ones(1600, dtype=np.float32))
        buf.reset()
        assert len(buf.buffer) == 0
        assert buf.total_samples_processed == 0

    def test_total_samples_tracks_all(self):
        """Test that total_samples_processed counts all samples, even evicted ones."""
        buf = AudioBuffer(sample_rate=16000, max_duration_sec=1.0)
        
        for _ in range(5):
            buf.add_chunk(np.ones(16000, dtype=np.float32))
        
        # Buffer only keeps 16000, but total should be 80000
        assert len(buf.buffer) == 16000
        assert buf.total_samples_processed == 80000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
