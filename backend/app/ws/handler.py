"""
WebSocket handler for /ws/stream.

Protocol (all messages are JSON text frames except audio):

Client → Server:
  Text:  {"type":"config", "source_lang":"es", "target_lang":"en"}
  Text:  {"type":"audio", "seq": N, "sample_rate": 16000, "pcm16_base64": "..."}
  Text:  {"type":"stop"}

Server → Client:
  Text:  {"type":"partial_transcript", "text":"..."}
  Text:  {"type":"committed_transcript", "text":"...", "segment_id": N}
  Text:  {"type":"translation_committed", "text":"...", "source":"...", "segment_id": N}
  Text:  {"type":"tts_audio_chunk", "audio_b64":"...", "segment_id": N, "sample_rate": 24000}
  Text:  {"type":"tts_end", "segment_id": N}
  Text:  {"type":"stats", "asr_ms":..., "mt_ms":..., "tts_ms":..., "e2e_ms":...}
  Text:  {"type":"error", "message":"..."}
  Text:  {"type":"ready"}

Design decision: We use JSON+base64 for audio instead of raw binary frames.
Rationale: simpler debugging, universal browser support, acceptable overhead
for the output audio rate (~3KB/200ms chunk). For the input audio we also
use base64 to keep a uniform protocol.  A future optimization can switch
input to binary frames with a 1-byte type header.
"""

import asyncio
import base64
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect

from ..pipeline.orchestrator import PipelineOrchestrator
from ..pipeline.asr import ASREngine
from ..pipeline.mt import MTEngine
from ..pipeline.tts import TTSEngine
from ..config import TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)


class StreamSession:
    """Manages one WebSocket streaming session."""

    def __init__(
        self,
        ws: WebSocket,
        asr_engine: ASREngine,
        mt_engine: MTEngine,
        tts_engine: TTSEngine,
    ):
        self.ws = ws
        self.asr_engine = asr_engine
        self.mt_engine = mt_engine
        self.tts_engine = tts_engine
        self.pipeline: PipelineOrchestrator | None = None
        self._sender_task: asyncio.Task | None = None

    async def run(self) -> None:
        """Main session loop."""
        try:
            await self.ws.accept()
            logger.info("WebSocket connected")

            # Wait for config message
            config = await self._recv_json()
            if config is None or config.get("type") != "config":
                await self._send_error("First message must be {type:'config', source_lang, target_lang}")
                await self.ws.close()
                return

            source_lang = config.get("source_lang", "es")
            target_lang = config.get("target_lang", "en")

            # Pre-load MT pair
            self.mt_engine.load_pair(source_lang, target_lang)

            # Create pipeline
            self.pipeline = PipelineOrchestrator(
                asr_engine=self.asr_engine,
                mt_engine=self.mt_engine,
                tts_engine=self.tts_engine,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            self.pipeline.start()

            # Start sender task (reads from pipeline output queue → WS)
            self._sender_task = asyncio.create_task(self._sender_loop())

            await self._send_json({"type": "ready"})
            logger.info(f"Session ready: {source_lang} → {target_lang}")

            # Receive loop
            await self._receiver_loop()

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
        finally:
            await self._cleanup()

    async def _receiver_loop(self) -> None:
        """Receive audio chunks and control messages from the client."""
        while True:
            msg = await self._recv_json()
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "audio":
                pcm_b64 = msg.get("pcm16_base64", "")
                if pcm_b64:
                    try:
                        pcm_bytes = base64.b64decode(pcm_b64)
                        self.pipeline.feed_audio(pcm_bytes)
                    except Exception as e:
                        logger.warning(f"Audio decode error: {e}")

            elif msg_type == "stop":
                logger.info("Client sent stop")
                break

            elif msg_type == "config":
                # Runtime language change
                source_lang = msg.get("source_lang", "es")
                target_lang = msg.get("target_lang", "en")
                logger.info(f"Config update: {source_lang} → {target_lang}")
                # Restart pipeline with new config
                if self.pipeline:
                    await self.pipeline.stop()
                self.mt_engine.load_pair(source_lang, target_lang)
                self.pipeline = PipelineOrchestrator(
                    asr_engine=self.asr_engine,
                    mt_engine=self.mt_engine,
                    tts_engine=self.tts_engine,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                self.pipeline.start()

    async def _sender_loop(self) -> None:
        """Read events from the pipeline output queue and send to WS."""
        while True:
            try:
                event = await asyncio.wait_for(
                    self.pipeline.output_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                if event["type"] == "tts_audio_chunk":
                    # Convert audio bytes to base64 for JSON transport
                    audio_data = event.pop("data", b"")
                    event["audio_b64"] = base64.b64encode(audio_data).decode("ascii")
                    event["sample_rate"] = TTS_SAMPLE_RATE

                await self._send_json(event)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Sender error: {e}")
                break

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
        if self.pipeline:
            await self.pipeline.stop()
        logger.info("Session cleaned up")

    # ── helpers ───────────────────────────────────────────

    async def _recv_json(self) -> dict | None:
        try:
            text = await self.ws.receive_text()
            return json.loads(text)
        except (WebSocketDisconnect, RuntimeError):
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from client: {e}")
            return None

    async def _send_json(self, data: dict) -> None:
        try:
            await self.ws.send_json(data)
        except Exception:
            pass

    async def _send_error(self, message: str) -> None:
        await self._send_json({"type": "error", "message": message})
