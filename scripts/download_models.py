#!/usr/bin/env python3
"""
Download all required models ahead of time.
This avoids the first-run delay when the pipeline starts.

Usage:
    python scripts/download_models.py [--device cpu|cuda|mps]
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_asr_model(model_name: str):
    """Download Qwen3-ASR model (via from_pretrained, which caches to HF_HOME/transformers)."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading ASR model: {model_name}")
    print(f"{'='*60}")
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
        print(f"✅ ASR model '{model_name}' downloaded successfully.")
        del model
    except Exception as e:
        print(f"❌ Failed to download ASR model: {e}")


def download_mt_model(model_name: str, cache_dir: str):
    """Download MarianMT model and tokenizer."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading MT model: {model_name}")
    print(f"{'='*60}")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"✅ MT model '{model_name}' downloaded successfully.")
        del tokenizer, model
    except Exception as e:
        print(f"❌ Failed to download MT model: {e}")


def download_tts_model(model_name: str, cache_dir: str):
    """Download TTS model (placeholder for Qwen3-TTS)."""
    print(f"\n{'='*60}")
    print(f"📥 TTS model: {model_name}")
    print(f"{'='*60}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        print(f"✅ TTS model '{model_name}' downloaded successfully.")
        del tokenizer, model
    except Exception as e:
        print(f"⚠️  TTS model download skipped (using mock): {e}")
        print("   The TTS pipeline will use mock audio generation until Qwen3-TTS is available.")


def main():
    parser = argparse.ArgumentParser(description="Download all models for RTT pipeline")
    parser.add_argument("--asr-model", default=os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B"), help="ASR model (Qwen3-ASR HuggingFace id)")
    parser.add_argument("--mt-model", default=os.getenv("MT_MODEL_ES_EN", "Helsinki-NLP/opus-mt-es-en"), help="MT model name")
    parser.add_argument("--tts-model", default=os.getenv("TTS_QWEN3_MODEL", "Qwen/Qwen3-TTS-0.6B"), help="TTS model name")
    parser.add_argument("--cache-dir", default=os.getenv("MODEL_CACHE_DIR", "./models"), help="Model cache directory (MT/TTS)")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"🗂️  Model cache directory (MT/TTS): {os.path.abspath(cache_dir)}")
    print("   (Qwen3-ASR uses HuggingFace cache by default)")

    download_asr_model(args.asr_model)
    download_mt_model(args.mt_model, cache_dir)
    download_tts_model(args.tts_model, cache_dir)

    print(f"\n{'='*60}")
    print("🎉 Model download complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
