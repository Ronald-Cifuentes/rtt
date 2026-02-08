"""Tests for output assembly (PCM16 conversion, base64 encoding)."""
import numpy as np
import base64
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPCM16Conversion:
    def test_float32_to_pcm16_roundtrip(self):
        """Test float32 → PCM16 → float32 roundtrip."""
        # Create known float32 audio
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        
        # Convert to PCM16 bytes
        pcm16 = (original * 32767).astype(np.int16)
        pcm16_bytes = pcm16.tobytes()
        
        # Convert back
        recovered = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Should be close (quantization error)
        np.testing.assert_allclose(original, recovered, atol=1.0 / 32768.0)

    def test_pcm16_base64_roundtrip(self):
        """Test PCM16 → base64 → PCM16 roundtrip (matching frontend encoding)."""
        # Simulate what the frontend does
        original_pcm16 = np.array([0, 16383, -16384, 32767, -32768], dtype=np.int16)
        pcm16_bytes = original_pcm16.tobytes()
        
        # Encode to base64 (what frontend sends)
        b64_encoded = base64.b64encode(pcm16_bytes).decode('ascii')
        
        # Decode on backend side
        decoded_bytes = base64.b64decode(b64_encoded)
        recovered_pcm16 = np.frombuffer(decoded_bytes, dtype=np.int16)
        
        np.testing.assert_array_equal(original_pcm16, recovered_pcm16)

    def test_pcm16_clamping(self):
        """Test that values are properly clamped to int16 range."""
        # Values that exceed float32 [-1, 1] range
        float_audio = np.array([1.5, -1.5, 2.0, -2.0], dtype=np.float32)
        
        # Clamp first, then convert
        clamped = np.clip(float_audio, -1.0, 1.0)
        pcm16 = (clamped * 32767).astype(np.int16)
        
        assert pcm16[0] == 32767  # Clamped to max
        assert pcm16[1] == -32767  # Clamped to min

    def test_empty_audio_chunk(self):
        """Test handling of empty audio chunks."""
        empty = np.array([], dtype=np.float32)
        pcm16 = (empty * 32767).astype(np.int16)
        assert len(pcm16) == 0
        assert pcm16.tobytes() == b''

    def test_chunk_assembly_no_gaps(self):
        """Test that consecutive audio chunks assemble without gaps."""
        chunks = []
        for i in range(5):
            # Each chunk is 100ms at 24kHz = 2400 samples
            t = np.arange(i * 2400, (i + 1) * 2400, dtype=np.float32) / 24000
            chunk = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            chunks.append(chunk)
        
        # Assemble
        assembled = np.concatenate(chunks)
        
        # Check continuity at boundaries
        for i in range(len(chunks) - 1):
            boundary_idx = (i + 1) * 2400
            # The difference between consecutive samples at boundary should be small
            diff = abs(assembled[boundary_idx] - assembled[boundary_idx - 1])
            # For a 440Hz sine at 24kHz, max sample-to-sample diff is
            # 2*pi*440/24000 ≈ 0.115, so diff should be within that range
            assert diff < 0.2, f"Gap detected at chunk boundary {i}: diff={diff}"

    def test_assembled_audio_length(self):
        """Test that assembled output has the correct total length."""
        chunk_samples = 2400
        n_chunks = 10
        chunks = [np.zeros(chunk_samples, dtype=np.float32) for _ in range(n_chunks)]
        
        assembled = np.concatenate(chunks)
        assert len(assembled) == chunk_samples * n_chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
