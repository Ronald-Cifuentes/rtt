"""
Configuration module - loads from .env with sensible defaults.
All tunables are documented in .env.example at the repo root.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env")
load_dotenv(_root / ".env.example")  # fallback defaults


# ── Device ────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cpu")

# ── ASR (Qwen3-ASR) ────────────────────────────────────────
# HuggingFace model id. Options: Qwen/Qwen3-ASR-0.6B (faster), Qwen/Qwen3-ASR-1.7B (better quality)
# Legacy: if .env has old Whisper size (base/small/medium etc.), use Qwen default
_asr_env = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B").strip()
_LEGACY_WHISPER_SIZES = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
ASR_MODEL: str = (
    "Qwen/Qwen3-ASR-0.6B"
    if _asr_env.lower() in _LEGACY_WHISPER_SIZES
    else _asr_env
)
ASR_MAX_NEW_TOKENS: int = int(os.getenv("ASR_MAX_NEW_TOKENS", "256"))
ASR_MAX_BATCH_SIZE: int = int(os.getenv("ASR_MAX_BATCH_SIZE", "32"))

# ── MT ────────────────────────────────────────────────────
MT_MODELS: dict[str, str] = {
    "es-en": os.getenv("MT_MODEL_ES_EN", "Helsinki-NLP/opus-mt-es-en"),
    "en-es": os.getenv("MT_MODEL_EN_ES", "Helsinki-NLP/opus-mt-en-es"),
}

# ── TTS ───────────────────────────────────────────────────
TTS_ENGINE: str = os.getenv("TTS_ENGINE", "edge-tts")
TTS_QWEN3_MODEL: str = os.getenv("TTS_QWEN3_MODEL", "Qwen/Qwen3-TTS-0.6B")
TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

# ── Audio ─────────────────────────────────────────────────
CAPTURE_SAMPLE_RATE: int = int(os.getenv("CAPTURE_SAMPLE_RATE", "16000"))
CAPTURE_CHUNK_MS: int = int(os.getenv("CAPTURE_CHUNK_MS", "100"))

# ── Commit Algorithm ─────────────────────────────────────
# Increased window from 5.0s to 8.0s for better context on long sentences
WINDOW_SEC: float = float(os.getenv("WINDOW_SEC", "8.0"))
ASR_INTERVAL_MS: int = int(os.getenv("ASR_INTERVAL_MS", "500"))
COMMIT_STABILITY_K: int = int(os.getenv("COMMIT_STABILITY_K", "3"))
COMMIT_TIMEOUT_SEC: float = float(os.getenv("COMMIT_TIMEOUT_SEC", "2.0"))  # Reduced from 4.0 for faster commits
COMMIT_MIN_WORDS: int = int(os.getenv("COMMIT_MIN_WORDS", "1"))  # Allow single words like "hola"

# ── Backpressure ──────────────────────────────────────────
TTS_QUEUE_MAX: int = int(os.getenv("TTS_QUEUE_MAX", "5"))

# ── Server ────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Model Cache ──────────────────────────────────────────
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", str(_root / "models"))
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str | None = None) -> str:
    """Return the best available device, with auto-fallback."""
    dev = requested or DEVICE
    if dev == "mps":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    if dev == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return "cpu"
