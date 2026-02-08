#!/usr/bin/env python3
"""
Test the full pipeline with a .wav file and measure per-stage latencies.

Usage:
    python scripts/test_wav_pipeline.py --input test_audio.wav [--src-lang es] [--tgt-lang en]
    
If no input file is provided, generates a synthetic sine wave for testing.
"""

import argparse
import asyncio
import sys
import time
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment before importing config
os.environ.setdefault("DEVICE", "cpu")


async def test_pipeline(input_file: str, src_lang: str, tgt_lang: str):
    from backend.app.pipeline.asr import ASREngine
    from backend.app.pipeline.mt import MTEngine
    from backend.app.pipeline.tts import TTSEngine
    from backend.app.core.commit_tracker import CommitTracker
    from backend.app.config import Config

    # ── Load audio ──
    if input_file and os.path.exists(input_file):
        print(f"\n📂 Loading audio file: {input_file}")
        try:
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(input_file)
            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                audio_float = audio_data
            else:
                audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
            # Convert to mono if stereo
            if len(audio_float.shape) > 1:
                audio_float = audio_float.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy.signal import resample
                num_samples = int(len(audio_float) * 16000 / sample_rate)
                audio_float = resample(audio_float, num_samples).astype(np.float32)
                sample_rate = 16000
            
            print(f"  Duration: {len(audio_float)/sample_rate:.2f}s, Samples: {len(audio_float)}")
        except Exception as e:
            print(f"❌ Failed to load audio file: {e}")
            return
    else:
        print("\n🔊 Generating synthetic test audio (2s sine wave)...")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_float = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine
        print(f"  Duration: {duration}s, Samples: {len(audio_float)}")

    # ── Initialize engines ──
    print("\n⏳ Loading models...")

    t0 = time.perf_counter()
    asr_engine = ASREngine()
    asr_engine.load_model()
    t_asr_load = time.perf_counter() - t0
    print(f"  ASR loaded in {t_asr_load:.2f}s")

    t0 = time.perf_counter()
    mt_engine = MTEngine()
    mt_engine.load_model()
    t_mt_load = time.perf_counter() - t0
    print(f"  MT loaded in {t_mt_load:.2f}s")

    t0 = time.perf_counter()
    tts_engine = TTSEngine()
    tts_engine.load_model()
    t_tts_load = time.perf_counter() - t0
    print(f"  TTS loaded in {t_tts_load:.2f}s")

    # ── Simulate streaming ──
    print(f"\n🎙️  Simulating streaming pipeline ({src_lang} → {tgt_lang})...")
    print(f"{'='*60}")

    commit_tracker = CommitTracker(Config.COMMIT_STABILITY_K, Config.COMMIT_MIN_TIME_SEC)
    
    chunk_size = int(sample_rate * Config.CHUNK_MS / 1000)  # Samples per chunk
    total_chunks = len(audio_float) // chunk_size
    
    asr_timings = []
    mt_timings = []
    tts_timings = []
    e2e_timings = []
    
    committed_texts = []
    translated_texts = []
    tts_audio_chunks = []

    check_interval_chunks = max(1, int(Config.ASR_CHECK_MS / Config.CHUNK_MS))
    
    for i in range(total_chunks):
        chunk = audio_float[i * chunk_size:(i + 1) * chunk_size]
        asr_engine.add_audio_chunk(chunk)
        commit_tracker.increment_samples(len(chunk))
        
        # Only run ASR every check_interval_chunks
        if (i + 1) % check_interval_chunks == 0:
            chunk_start_time = time.perf_counter()
            
            # ASR
            t0 = time.perf_counter()
            hypothesis = await asr_engine.transcribe_window(src_lang)
            t_asr = (time.perf_counter() - t0) * 1000
            asr_timings.append(t_asr)
            
            if hypothesis:
                current_time = time.time()
                commit_event = commit_tracker.update(hypothesis, current_time)
                
                if commit_event:
                    committed_texts.append(commit_event.text)
                    print(f"\n  📝 Committed: \"{commit_event.text}\"  (ASR: {t_asr:.0f}ms)")
                    
                    # MT
                    t0 = time.perf_counter()
                    translation = await mt_engine.translate(commit_event.text, src_lang, tgt_lang)
                    t_mt = (time.perf_counter() - t0) * 1000
                    mt_timings.append(t_mt)
                    
                    if translation:
                        translated_texts.append(translation)
                        print(f"  🌍 Translated: \"{translation}\"  (MT: {t_mt:.0f}ms)")
                        
                        # TTS
                        t0 = time.perf_counter()
                        chunk_count = 0
                        async for tts_chunk in tts_engine.synthesize_streaming(translation, tgt_lang):
                            tts_audio_chunks.append(tts_chunk)
                            chunk_count += 1
                        t_tts = (time.perf_counter() - t0) * 1000
                        tts_timings.append(t_tts)
                        print(f"  🔊 TTS: {chunk_count} chunks  (TTS: {t_tts:.0f}ms)")
                    
                    t_e2e = (time.perf_counter() - chunk_start_time) * 1000
                    e2e_timings.append(t_e2e)
                    print(f"  ⏱️  E2E: {t_e2e:.0f}ms")

    # ── Report ──
    print(f"\n{'='*60}")
    print("📊 Pipeline Test Results")
    print(f"{'='*60}")
    print(f"\n  Audio duration:     {len(audio_float)/sample_rate:.2f}s")
    print(f"  Chunks processed:   {total_chunks}")
    print(f"  Committed segments: {len(committed_texts)}")
    print(f"  Translated segments:{len(translated_texts)}")
    print(f"  TTS audio chunks:   {len(tts_audio_chunks)}")
    
    if asr_timings:
        print(f"\n  ASR latency:  avg={np.mean(asr_timings):.0f}ms  min={np.min(asr_timings):.0f}ms  max={np.max(asr_timings):.0f}ms")
    if mt_timings:
        print(f"  MT latency:   avg={np.mean(mt_timings):.0f}ms  min={np.min(mt_timings):.0f}ms  max={np.max(mt_timings):.0f}ms")
    if tts_timings:
        print(f"  TTS latency:  avg={np.mean(tts_timings):.0f}ms  min={np.min(tts_timings):.0f}ms  max={np.max(tts_timings):.0f}ms")
    if e2e_timings:
        print(f"  E2E latency:  avg={np.mean(e2e_timings):.0f}ms  min={np.min(e2e_timings):.0f}ms  max={np.max(e2e_timings):.0f}ms")
    
    if not committed_texts:
        print("\n  ⚠️  No segments were committed. For real speech, the pipeline would commit text.")
        print("     With a synthetic sine wave, no speech is detected — this is expected behavior.")
    
    print(f"\n{'='*60}")
    print("✅ Pipeline test complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test RTT pipeline with a WAV file")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input .wav file")
    parser.add_argument("--src-lang", default="es", help="Source language code (default: es)")
    parser.add_argument("--tgt-lang", default="en", help="Target language code (default: en)")
    args = parser.parse_args()
    
    asyncio.run(test_pipeline(args.input, args.src_lang, args.tgt_lang))


if __name__ == "__main__":
    main()
