# RTT — Real-Time Speech-to-Speech Translation

A low-latency speech-to-speech translation system that begins playing translated audio **while you're still speaking**. No VAD-based triggers — uses a **commit-by-stability** algorithm for fluid, incremental processing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (React + Vite)                   │
│                                                                 │
│  ┌──────────────┐  PCM16 chunks   ┌──────────────────────────┐  │
│  │ AudioWorklet │ ──────────────► │     WebSocket Client     │  │
│  │ (Capture)    │  base64 JSON    │                          │  │
│  └──────────────┘                 │  {type:"audio", seq,     │  │
│                                   │   pcm16_base64}          │  │
│  ┌──────────────┐  PCM16 binary   │                          │  │
│  │ AudioWorklet │ ◄────────────── │  Receives: JSON events   │  │
│  │ (Playback)   │  ring buffer    │  + binary TTS audio      │  │
│  └──────────────┘                 └──────────┬───────────────┘  │
│                                              │                  │
│  ┌──────────────┐  ┌────────────┐  ┌────────┴────────┐         │
│  │ Language     │  │ Start/Stop │  │ Latency         │         │
│  │ Selectors    │  │ Control    │  │ Indicator       │         │
│  └──────────────┘  └────────────┘  └─────────────────┘         │
└──────────────────────────────────┬──────────────────────────────┘
                                   │ WebSocket
                                   │ ws://host:8000/ws/stream
┌──────────────────────────────────┴──────────────────────────────┐
│                   BACKEND (FastAPI + Python 3.11+)               │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Pipeline Orchestrator                     │  │
│  │                                                            │  │
│  │  ┌─────────┐    ┌──────────┐    ┌──────┐    ┌──────────┐  │  │
│  │  │ Audio   │───►│ ASR      │───►│  MT  │───►│  TTS     │  │  │
│  │  │ Buffer  │    │(Qwen3-ASR)│   │      │    │ (Qwen3)  │  │  │
│  │  │ Circular│    │ Sliding  │    │Marian│    │ Streaming│  │  │
│  │  └─────────┘    │ Window   │    │  MT  │    │ Chunks   │  │  │
│  │                 └────┬─────┘    └──────┘    └──────────┘  │  │
│  │                      │                                     │  │
│  │              ┌───────┴────────┐    ┌───────────────────┐   │  │
│  │              │ Commit Tracker │    │ Backpressure Mgr  │   │  │
│  │              │ (Stability-K)  │    │ (Queue monitor)   │   │  │
│  │              └────────────────┘    └───────────────────┘   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Browser** captures microphone → AudioWorklet resamples to 16kHz mono PCM16 → Base64 → JSON over WebSocket
2. **Backend** receives chunks → appends to circular audio buffer
3. **ASR** (Qwen3-ASR) runs every `ASR_INTERVAL_MS` ms over a sliding window of `WINDOW_SEC` seconds
4. **Commit Tracker** compares consecutive ASR hypotheses → commits stable prefix after K repetitions or timeout
5. **MT** translates only **new committed text** (no re-translation)
6. **TTS** synthesizes translation in streaming chunks (24kHz PCM16)
7. **Browser** receives PCM16 binary → AudioWorklet playback with ring buffer (gapless)

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- ~4GB free disk space for models (Qwen3-ASR, MT, TTS)

### 1. Clone & one-shot setup

From the repository root:

```bash
chmod +x setup.sh
./setup.sh
```

This creates `backend/venv`, installs backend and frontend dependencies, and copies `env.example` to `.env` if missing.

### 2. Run

**Terminal 1 — Backend:**
```bash
cd backend && source venv/bin/activate && python -m app.main
```

**Terminal 2 — Frontend:**
```bash
cd frontend && npm run dev
```

Open **http://localhost:5173**.

### 3. Optional: pre-download models

```bash
PYTHONPATH=. backend/venv/bin/python scripts/download_models.py
```

### 4. Run with Docker

```bash
docker-compose up --build
```
Open **http://localhost:5173**

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cpu` | Compute device: `cpu`, `cuda`, `mps` |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | Qwen3-ASR model (or `Qwen/Qwen3-ASR-1.7B`) |
| `ASR_MAX_NEW_TOKENS` | `256` | Max tokens per ASR output |
| `WINDOW_SEC` | `8.0` | Sliding window duration (seconds) |
| `ASR_INTERVAL_MS` | `500` | Interval between ASR runs (ms) |
| `COMMIT_STABILITY_K` | `3` | Consecutive stable hypotheses to commit |
| `COMMIT_TIMEOUT_SEC` | `2.0` | Force-commit timeout (seconds) |
| `COMMIT_MIN_WORDS` | `1` | Minimum words to commit |
| `MT_MODEL_ES_EN` | `Helsinki-NLP/opus-mt-es-en` | MarianMT Spanish→English |
| `MT_MODEL_EN_ES` | `Helsinki-NLP/opus-mt-en-es` | MarianMT English→Spanish |
| `TTS_ENGINE` | `edge-tts` | TTS backend (`edge-tts` or `qwen3`) |
| `TTS_QWEN3_MODEL` | `Qwen/Qwen3-TTS-0.6B` | TTS model when using qwen3 |
| `CAPTURE_SAMPLE_RATE` | `16000` | Mic capture sample rate (Hz) |
| `MODEL_CACHE_DIR` | `./models` | Local model cache directory |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Backend bind address |
| `LOG_LEVEL` | `INFO` | Logging level |

### Device notes

| Device | Notes |
|---|---|
| **CUDA** | Full GPU acceleration for ASR/MT/TTS. |
| **MPS** (Apple Silicon) | Auto-detected on macOS; Qwen3-ASR may fall back to CPU. |
| **CPU** | Works everywhere; use `ASR_MODEL=Qwen/Qwen3-ASR-0.6B` for speed. |

**Lightweight config (no GPU / low RAM):**
```env
DEVICE=cpu
ASR_MODEL=tiny
ASR_CHECK_MS=800
COMMIT_STABILITY_K=2
```

---

## WebSocket Protocol

### Client → Server

```json
// Initial config (sent first)
{"type": "config", "source_lang": "es", "target_lang": "en"}

// Audio chunks (sent continuously)
{"type": "audio", "seq": 1, "sample_rate": 16000, "pcm16_base64": "..."}

// Stop signal
{"type": "stop"}
```

### Server → Client

```json
// Partial ASR hypothesis (live updating)
{"type": "partial_transcript", "text": "hola cómo est"}

// Committed text (stable, sent to MT)
{"type": "committed_transcript", "text": "hola cómo estás"}

// Translation result
{"type": "translation_committed", "source_text": "hola cómo estás", "translated_text": "hello how are you"}

// Status updates
{"type": "status", "message": "Models loaded, ready for config."}

// Latency stats
{"type": "stats", "asr_latency_ms": 120, "mt_latency_ms": 45, "tts_latency_ms": 200, "e2e_latency_ms": 380}
```

**TTS Audio:** Sent as raw **binary WebSocket frames** (PCM16, 24kHz mono) for efficiency.

---

## Commit-by-Stability Algorithm

Instead of using Voice Activity Detection (VAD) to detect silence and trigger translation, RTT uses **stability tracking**:

1. ASR runs every `ASR_CHECK_MS` milliseconds over the latest `ASR_WINDOW_SEC` seconds
2. Each hypothesis is compared against the previous K hypotheses
3. The **longest common prefix** across K consecutive hypotheses is identified
4. If this stable prefix is longer than the already-committed text, the **new portion** is committed
5. Committed text is translated and synthesized immediately
6. **Time-based fallback**: if `COMMIT_MIN_TIME_SEC` passes without a stability commit, the latest text is committed anyway

This allows:
- ✅ Continuous speech without pauses
- ✅ No waiting for silence
- ✅ Incremental output that starts fast
- ✅ Tolerance for corrections/self-repairs in speech

---

## Scripts

```bash
# Download models ahead of time
python scripts/download_models.py

# Sanity check — verify all dependencies and device
python scripts/sanity_check.py

# Test pipeline with a WAV file
python scripts/test_wav_pipeline.py --input path/to/audio.wav --src-lang es --tgt-lang en

# Test with synthetic audio (no file needed)
python scripts/test_wav_pipeline.py
```

---

## Tests

```bash
cd backend
source venv/bin/activate

# Run all tests
cd backend && source venv/bin/activate && pytest tests/ -v

# Or from repo root with PYTHONPATH
PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/ -v
```

Tests cover:
- **Audio buffer**: circular eviction, window retrieval, reset
- **Commit tracker**: stability-K commits, time-based commits, no duplicates, incremental commits
- **Output assembly**: PCM16 ↔ float32 roundtrip, base64 encoding, chunk continuity

---

## Project Structure

```
rtt/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Centralized configuration
│   │   ├── main.py                # FastAPI app + WS endpoint
│   │   ├── core/
│   │   │   ├── audio_buffer.py    # Circular audio buffer
│   │   │   ├── commit_tracker.py  # Stability-based commit logic
│   │   │   └── backpressure.py    # TTS queue backpressure manager
│   │   ├── pipeline/
│   │   │   ├── asr.py             # Qwen3-ASR engine
│   │   │   ├── mt.py              # MarianMT translation engine
│   │   │   ├── tts.py             # Qwen3-TTS engine (+ mock)
│   │   │   └── orchestrator.py    # Pipeline coordination
│   │   └── ws/
│   │       └── handler.py         # WebSocket connection handler
│   ├── tests/
│   │   ├── test_audio_buffer.py
│   │   ├── test_commit_tracker.py
│   │   └── test_output_assembly.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── public/
│   │   ├── audio-capture-worklet.js   # AudioWorklet for mic capture
│   │   └── audio-playback-worklet.js  # AudioWorklet for TTS playback
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx                    # Main app with full integration
│   │   ├── App.css                    # Modern dark theme styles
│   │   ├── index.css                  # Global styles
│   │   ├── types.ts                   # TypeScript types for WS protocol
│   │   ├── components/
│   │   │   ├── ControlPanel.tsx       # Start/Stop buttons
│   │   │   ├── LanguageSelector.tsx   # Source/target language dropdowns
│   │   │   ├── LatencyIndicator.tsx   # Per-stage latency display
│   │   │   ├── StatusBar.tsx          # Connection status indicator
│   │   │   └── TranscriptPanel.tsx    # Scrolling transcript display
│   │   ├── hooks/
│   │   │   ├── useAudioCapture.ts     # Mic capture with AudioWorklet
│   │   │   ├── useAudioPlayback.ts    # Gapless playback with ring buffer
│   │   │   └── useWebSocket.ts        # WS connection management
│   │   └── utils/
│   │       └── audioUtils.ts          # PCM16↔base64, latency formatting
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   └── nginx.conf
├── scripts/
│   ├── download_models.py            # Pre-download all ML models
│   ├── sanity_check.py               # Verify deps and devices
│   └── test_wav_pipeline.py          # Test pipeline with WAV + latency
├── env.example                        # Configuration template (copy to .env)
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Qwen3-TTS Integration

The TTS engine is designed for **Qwen3-TTS** with streaming synthesis support. Current status:

- **MVP**: Uses a mock TTS that generates audio chunks at realistic timing. The mock simulates the streaming behavior (chunk-by-chunk output with ~100ms intervals).
- **Production**: When Qwen3-TTS is available, replace the mock in `backend/app/pipeline/tts.py`:
  - The `load_model()` method loads the model and processor
  - The `synthesize_streaming()` method yields audio chunks as they're generated
  - The 12Hz tokenizer enables low-latency streaming (~83ms per token)

### Voice Modes

| Mode | Description | Model |
|---|---|---|
| `CustomVoice` | Pre-defined voice presets | `Qwen3-TTS-0.6B-CustomVoice` |
| `VoiceDesign` | Natural language voice description | `Qwen3-TTS-0.6B-VoiceDesign` |
| `Base` | Voice cloning from 3-5s reference audio | `Qwen3-TTS-0.6B-Base` |

---

## Backpressure Control

If TTS synthesis falls behind real-time:
1. **BackpressureManager** monitors the output queue size
2. When queue exceeds `buffer_limit`, degradation mode activates
3. In degradation mode, the system can:
   - Accumulate more text before synthesizing (longer, fewer TTS calls)
   - Reduce commit frequency
4. When queue recovers to 50% of limit, normal mode resumes

This prevents queue overflow and maintains system stability during high load.

---

## Language Pairs (MVP)

| Direction | MT Model |
|---|---|
| Spanish → English | `Helsinki-NLP/opus-mt-es-en` |
| English → Spanish | `Helsinki-NLP/opus-mt-en-es` |
| French → English | `Helsinki-NLP/opus-mt-fr-en` |
| English → French | `Helsinki-NLP/opus-mt-en-fr` |

To change language pair, update `MT_MODEL` in `.env`. The ASR model (Whisper) is multilingual by default.

---

## License

MIT
