// ── WebSocket message types ──

export interface AudioMessage {
  type: 'audio';
  seq: number;
  sample_rate: number;
  pcm16_base64: string;
}

export interface ConfigMessage {
  type: 'config';
  source_lang: string;
  target_lang: string;
}

export interface StopMessage {
  type: 'stop';
}

export type ClientMessage = AudioMessage | ConfigMessage | StopMessage;

// ── Server events ──

export interface PartialTranscriptEvent {
  type: 'partial_transcript';
  text: string;
}

export interface CommittedTranscriptEvent {
  type: 'committed_transcript';
  text: string;
  segment_id: number;
}

export interface TranslationCommittedEvent {
  type: 'translation_committed';
  text: string;        // the translated text
  source: string;      // original source text
  segment_id: number;
}

export interface TTSAudioChunkEvent {
  type: 'tts_audio_chunk';
  audio_b64: string;
  segment_id: number;
  sample_rate: number;
}

export interface TTSEndEvent {
  type: 'tts_end';
  segment_id: number;
}

export interface StatsEvent {
  type: 'stats';
  asr_ms: number;
  mt_ms: number;
  tts_ms: number;
  e2e_ms: number;
  commits_total: number;
  tts_queue: number;
}

export interface ReadyEvent {
  type: 'ready';
}

export interface StatusEvent {
  type: 'status';
  message: string;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export type ServerEvent =
  | PartialTranscriptEvent
  | CommittedTranscriptEvent
  | TranslationCommittedEvent
  | TTSAudioChunkEvent
  | TTSEndEvent
  | StatsEvent
  | ReadyEvent
  | StatusEvent
  | ErrorEvent;

// ── App state ──

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'loading_models' | 'ready' | 'streaming' | 'error';

export interface LanguagePair {
  source: string;
  target: string;
}

export interface LatencyStats {
  asr: number;
  mt: number;
  tts: number;
  e2e: number;
}

export const SUPPORTED_LANGUAGES = [
  { code: 'es', name: 'Spanish', flag: '🇪🇸' },
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'fr', name: 'French', flag: '🇫🇷' },
  { code: 'de', name: 'German', flag: '🇩🇪' },
] as const;
