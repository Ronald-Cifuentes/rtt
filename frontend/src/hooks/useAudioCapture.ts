import { useCallback, useRef, useState } from 'react';
import { pcm16ToBase64 } from '../utils/audioUtils';

interface UseAudioCaptureOptions {
  targetSampleRate?: number;
  chunkMs?: number;
  onAudioChunk: (pcm16Base64: string, seq: number) => void;
}

export function useAudioCapture({
  targetSampleRate = 16000,
  chunkMs = 100,
  onAudioChunk,
}: UseAudioCaptureOptions) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const seqRef = useRef(0);

  const start = useCallback(async () => {
    try {
      setError(null);
      seqRef.current = 0;

      // Request microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: { ideal: 48000 },
          channelCount: 1,
        },
      });
      mediaStreamRef.current = stream;

      // Create AudioContext
      const audioContext = new AudioContext({ sampleRate: 48000 });
      audioContextRef.current = audioContext;

      // Load worklet
      await audioContext.audioWorklet.addModule('/audio-capture-worklet.js');

      const bufferSize = Math.round(targetSampleRate * chunkMs / 1000);

      const workletNode = new AudioWorkletNode(audioContext, 'audio-capture-processor', {
        processorOptions: {
          targetSampleRate,
          bufferSize,
        },
      });

      workletNode.port.onmessage = (event: MessageEvent) => {
        if (event.data.type === 'audio') {
          const base64 = pcm16ToBase64(event.data.pcm16);
          seqRef.current++;
          onAudioChunk(base64, seqRef.current);
        }
      };

      // Connect: mic -> worklet
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(workletNode);
      // Don't connect worklet to destination (no self-monitoring)

      workletNodeRef.current = workletNode;
      setIsCapturing(true);

      console.log('[AudioCapture] Started');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to start audio capture';
      setError(errorMsg);
      console.error('[AudioCapture] Error:', err);
    }
  }, [targetSampleRate, chunkMs, onAudioChunk]);

  const stop = useCallback(() => {
    // Stop worklet
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ type: 'stop' });
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setIsCapturing(false);
    console.log('[AudioCapture] Stopped');
  }, []);

  return {
    isCapturing,
    error,
    start,
    stop,
  };
}
