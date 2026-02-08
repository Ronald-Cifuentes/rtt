import { useCallback, useRef, useState } from 'react';

interface UseAudioPlaybackOptions {
  sourceSampleRate?: number;
}

export function useAudioPlayback({ sourceSampleRate = 24000 }: UseAudioPlaybackOptions = {}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [bufferLevel, setBufferLevel] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const isInitializedRef = useRef(false);

  const initialize = useCallback(async () => {
    if (isInitializedRef.current) return;

    try {
      const audioContext = new AudioContext({ sampleRate: 48000 });
      audioContextRef.current = audioContext;

      await audioContext.audioWorklet.addModule('/audio-playback-worklet.js');

      const workletNode = new AudioWorkletNode(audioContext, 'audio-playback-processor', {
        processorOptions: {
          sourceSampleRate,
        },
        outputChannelCount: [1],
      });

      workletNode.port.onmessage = (event: MessageEvent) => {
        if (event.data.type === 'bufferLevel') {
          setBufferLevel(event.data.samples);
        } else if (event.data.type === 'playbackEnded') {
          setIsPlaying(false);
        }
      };

      workletNode.connect(audioContext.destination);
      workletNodeRef.current = workletNode;
      isInitializedRef.current = true;

      console.log('[AudioPlayback] Initialized');
    } catch (err) {
      console.error('[AudioPlayback] Failed to initialize:', err);
    }
  }, [sourceSampleRate]);

  const enqueueAudio = useCallback(async (pcm16Data: ArrayBuffer) => {
    if (!isInitializedRef.current) {
      await initialize();
    }

    // Resume context if suspended (browser autoplay policy)
    if (audioContextRef.current?.state === 'suspended') {
      await audioContextRef.current.resume();
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage(
        { type: 'audio', pcm16: pcm16Data },
        [pcm16Data]
      );
      setIsPlaying(true);
    }
  }, [initialize]);

  const clear = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ type: 'clear' });
    }
    setIsPlaying(false);
    setBufferLevel(0);
  }, []);

  const destroy = useCallback(() => {
    clear();
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    isInitializedRef.current = false;
  }, [clear]);

  return {
    isPlaying,
    bufferLevel,
    initialize,
    enqueueAudio,
    clear,
    destroy,
  };
}
