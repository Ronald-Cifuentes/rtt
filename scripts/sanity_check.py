#!/usr/bin/env python3
"""
Sanity check script: verifies all dependencies are installed
and models can be loaded.

Usage:
    python scripts/sanity_check.py
"""

import sys
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

checks_passed = 0
checks_failed = 0


def check(name: str, fn):
    global checks_passed, checks_failed
    try:
        result = fn()
        print(f"  ✅ {name}: {result}")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        checks_failed += 1


def main():
    print(f"\n{'='*60}")
    print("🔍 RTT Sanity Check")
    print(f"{'='*60}")
    print(f"\n📋 System Info:")
    print(f"  Python:   {sys.version}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    # ── Python packages ──
    print(f"\n📦 Python Packages:")
    
    check("numpy", lambda: __import__("numpy").__version__)
    check("torch", lambda: __import__("torch").__version__)
    check("transformers", lambda: __import__("transformers").__version__)
    check("faster_whisper", lambda: __import__("faster_whisper").__version__ if hasattr(__import__("faster_whisper"), "__version__") else "installed")
    check("fastapi", lambda: __import__("fastapi").__version__)
    check("uvicorn", lambda: "installed")
    check("websockets", lambda: __import__("websockets").__version__)
    check("scipy", lambda: __import__("scipy").__version__)
    check("dotenv", lambda: __import__("dotenv").__version__ if hasattr(__import__("dotenv"), "__version__") else "installed")

    # ── Device ──
    print(f"\n🖥️  Device:")
    import torch
    check("CUDA available", lambda: f"{torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    check("MPS available", lambda: f"{torch.backends.mps.is_available()}" if hasattr(torch.backends, "mps") else "N/A")
    
    # ── Config ──
    print(f"\n⚙️  Config:")
    try:
        from backend.app.config import Config
        check("Device", lambda: Config.DEVICE)
        check("ASR model", lambda: Config.ASR_MODEL)
        check("MT model", lambda: Config.MT_MODEL)
        check("TTS model", lambda: Config.TTS_MODEL)
        check("Sample rate", lambda: f"{Config.SAMPLE_RATE} Hz")
        check("Model cache dir", lambda: str(Config.MODEL_CACHE_DIR))
    except Exception as e:
        print(f"  ⚠️  Could not load config: {e}")

    # ── Model files ──
    print(f"\n📁 Model Cache:")
    from backend.app.config import Config
    cache_dir = Config.MODEL_CACHE_DIR
    if cache_dir.exists():
        items = list(cache_dir.iterdir())
        if items:
            for item in items[:10]:
                print(f"  📄 {item.name}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")
        else:
            print("  (empty — run scripts/download_models.py first)")
    else:
        print(f"  (directory not found: {cache_dir})")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"📊 Results: {checks_passed} passed, {checks_failed} failed")
    if checks_failed == 0:
        print("🎉 All checks passed! Ready to run.")
    else:
        print("⚠️  Some checks failed. Review errors above.")
    print(f"{'='*60}\n")
    
    sys.exit(0 if checks_failed == 0 else 1)


if __name__ == "__main__":
    main()
