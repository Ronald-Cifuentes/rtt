"""
FastAPI application — Real-Time Speech-to-Speech Translation.

Endpoints:
  GET  /health           → health check
  WS   /ws/stream        → streaming translation WebSocket
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .config import HOST, PORT, LOG_LEVEL, resolve_device, ASR_MODEL_SIZE, TTS_ENGINE, TTS_QWEN3_MODEL, TTS_SAMPLE_RATE
from .pipeline.asr import ASREngine
from .pipeline.mt import MTEngine
from .pipeline.tts import TTSEngine
from .ws.handler import StreamSession

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Shared engines (loaded once, shared across sessions) ──
asr_engine: ASREngine | None = None
mt_engine: MTEngine | None = None
tts_engine: TTSEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global asr_engine, mt_engine, tts_engine

    device = resolve_device()
    logger.info(f"Device: {device}")

    # ASR
    asr_engine = ASREngine(model_size=ASR_MODEL_SIZE, device=device)
    asr_engine.load()

    # MT (lazy-loaded per pair, but we pre-load es-en)
    mt_engine = MTEngine(device=device)
    mt_engine.load_pair("es", "en")
    mt_engine.load_pair("en", "es")

    # TTS
    tts_engine = TTSEngine(
        backend=TTS_ENGINE,
        qwen3_model=TTS_QWEN3_MODEL,
        device=device,
        output_sample_rate=TTS_SAMPLE_RATE,
    )
    tts_engine.load()

    logger.info("All models loaded ✓")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Real-Time Speech-to-Speech Translation",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS (allow frontend dev server) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr": asr_engine is not None and asr_engine._loaded,
        "mt": mt_engine is not None,
        "tts": tts_engine is not None and tts_engine._loaded,
    }


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    session = StreamSession(
        ws=websocket,
        asr_engine=asr_engine,
        mt_engine=mt_engine,
        tts_engine=tts_engine,
    )
    await session.run()


# ── CLI entry point ───────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
