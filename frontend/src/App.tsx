import './App.css';

import { ConnectionStatus, LatencyStats, ServerEvent } from './types';
import React, { useCallback, useRef, useState } from 'react';

import { ControlPanel } from './components/ControlPanel';
import { LanguageSelector } from './components/LanguageSelector';
import { LatencyIndicator } from './components/LatencyIndicator';
import { StatusBar } from './components/StatusBar';
import { TranscriptPanel } from './components/TranscriptPanel';
import { base64ToPcm16 } from './utils/audioUtils';
import { useAudioCapture } from './hooks/useAudioCapture';
import { useAudioPlayback } from './hooks/useAudioPlayback';
import { useWebSocket } from './hooks/useWebSocket';

const WS_URL = `ws://${window.location.hostname}:8000/ws/stream`;

function App() {
  // ── State ──
  const [sourceLang, setSourceLang] = useState('es');
  const [targetLang, setTargetLang] = useState('en');
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [statusMessage, setStatusMessage] = useState('');
  const [partialTranscript, setPartialTranscript] = useState('');
  const [committedTranscripts, setCommittedTranscripts] = useState<string[]>([]);
  const [partialTranslation, setPartialTranslation] = useState('');
  const [committedTranslations, setCommittedTranslations] = useState<string[]>([]);
  const [latencyStats, setLatencyStats] = useState<LatencyStats>({ asr: -1, mt: -1, tts: -1, e2e: -1 });
  const [captureError, setCaptureError] = useState<string | null>(null);

  const isStreamingRef = useRef(false);

  // ── Audio Playback ──
  const { enqueueAudio, clear: clearPlayback, destroy: destroyPlayback } = useAudioPlayback({
    sourceSampleRate: 24000,
  });

  // ── WebSocket ──
  const handleTextMessage = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case 'partial_transcript':
        setPartialTranscript(event.text);
        break;

      case 'committed_transcript':
        setCommittedTranscripts((prev) => [...prev, event.text]);
        setPartialTranscript(''); // Clear partial on commit
        break;

      case 'translation_committed':
        // Backend sends { text: "translated", source: "original" }
        setCommittedTranslations((prev) => [...prev, event.text]);
        setPartialTranslation('');
        break;

      case 'tts_audio_chunk': {
        // Backend sends audio as base64 JSON: { audio_b64: "...", sample_rate: 24000 }
        if (event.audio_b64) {
          try {
            const pcm16Buffer = base64ToPcm16(event.audio_b64);
            enqueueAudio(pcm16Buffer);
          } catch (e) {
            console.error('[TTS] Failed to decode audio chunk:', e);
          }
        }
        break;
      }

      case 'tts_end':
        // TTS finished for a segment — nothing special needed
        break;

      case 'stats':
        // Backend sends: asr_ms, mt_ms, tts_ms, e2e_ms
        setLatencyStats({
          asr: event.asr_ms,
          mt: event.mt_ms,
          tts: event.tts_ms,
          e2e: event.e2e_ms,
        });
        break;

      case 'ready':
        setStatusMessage('Pipeline ready — speak now!');
        break;

      case 'status':
        setStatusMessage(event.message);
        break;

      case 'error':
        setStatusMessage(`Error: ${event.message}`);
        break;
    }
  }, [enqueueAudio]);

  const handleBinaryMessage = useCallback((data: ArrayBuffer) => {
    // Binary messages are TTS audio chunks (PCM16) — fallback path
    enqueueAudio(data);
  }, [enqueueAudio]);

  const handleStatusChange = useCallback((status: ConnectionStatus) => {
    setConnectionStatus(status);
  }, []);

  const { connect, disconnect, sendJSON, isConnected } = useWebSocket({
    url: WS_URL,
    onTextMessage: handleTextMessage,
    onBinaryMessage: handleBinaryMessage,
    onStatusChange: handleStatusChange,
  });

  // ── Audio Capture ──
  const handleAudioChunk = useCallback((pcm16Base64: string, seq: number) => {
    sendJSON({
      type: 'audio',
      seq,
      sample_rate: 16000,
      pcm16_base64: pcm16Base64,
    });
  }, [sendJSON]);

  const { isCapturing, error: audioError, start: startCapture, stop: stopCapture } = useAudioCapture({
    targetSampleRate: 16000,
    chunkMs: 100,
    onAudioChunk: handleAudioChunk,
  });

  // ── Controls ──
  const handleStart = useCallback(async () => {
    // Reset state
    setPartialTranscript('');
    setCommittedTranscripts([]);
    setPartialTranslation('');
    setCommittedTranslations([]);
    setLatencyStats({ asr: -1, mt: -1, tts: -1, e2e: -1 });
    setStatusMessage('Connecting...');
    setCaptureError(null);

    // Connect WebSocket
    connect();

    // Wait for connection to establish, then send config and start capture
    const waitForConnection = () => {
      return new Promise<void>((resolve) => {
        const interval = setInterval(() => {
          if (isConnected) {
            clearInterval(interval);
            resolve();
          }
        }, 100);
        // Timeout after 10s
        setTimeout(() => {
          clearInterval(interval);
          resolve();
        }, 10000);
      });
    };

    await waitForConnection();

    // Send config
    sendJSON({
      type: 'config',
      source_lang: sourceLang,
      target_lang: targetLang,
    });

    // Start audio capture
    await startCapture();
    isStreamingRef.current = true;
  }, [connect, isConnected, sendJSON, sourceLang, targetLang, startCapture]);

  const handleStop = useCallback(() => {
    isStreamingRef.current = false;
    stopCapture();
    sendJSON({ type: 'stop' });
    clearPlayback();

    // Small delay before disconnecting to ensure stop message is sent
    setTimeout(() => {
      disconnect();
    }, 200);
  }, [stopCapture, sendJSON, clearPlayback, disconnect]);

  // Update capture error
  React.useEffect(() => {
    if (audioError) {
      setCaptureError(audioError);
    }
  }, [audioError]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      destroyPlayback();
    };
  }, [destroyPlayback]);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            <span className="title-icon">🌐</span>
            Real-Time Speech Translation
          </h1>
          <StatusBar status={connectionStatus} statusMessage={statusMessage} />
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Config Row */}
        <div className="config-row">
          <LanguageSelector
            label="Source Language"
            value={sourceLang}
            onChange={setSourceLang}
            disabled={isCapturing}
            excludeLang={targetLang}
          />
          <div className="swap-icon">⇄</div>
          <LanguageSelector
            label="Target Language"
            value={targetLang}
            onChange={setTargetLang}
            disabled={isCapturing}
            excludeLang={sourceLang}
          />
        </div>

        {/* Control + Latency Row */}
        <div className="control-row">
          <ControlPanel
            status={connectionStatus}
            isCapturing={isCapturing}
            onStart={handleStart}
            onStop={handleStop}
          />
          <LatencyIndicator stats={latencyStats} />
        </div>

        {/* Error Display */}
        {captureError && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            {captureError}
          </div>
        )}

        {/* Transcript Panels */}
        <div className="transcripts-row">
          <TranscriptPanel
            title="Source (Transcription)"
            committedSegments={committedTranscripts}
            partialText={partialTranscript}
            accentColor="var(--accent-blue)"
            icon="🎙️"
          />
          <TranscriptPanel
            title="Target (Translation)"
            committedSegments={committedTranslations}
            partialText={partialTranslation}
            accentColor="var(--accent-green)"
            icon="🔊"
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <span>RTT v1.0 — Powered by Faster-Whisper, MarianMT, Qwen3-TTS</span>
      </footer>
    </div>
  );
}

export default App;
