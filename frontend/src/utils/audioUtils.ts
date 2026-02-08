/**
 * Convert an ArrayBuffer of PCM16 (Int16Array) to base64 string.
 */
export function pcm16ToBase64(pcm16Buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(pcm16Buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Convert a base64 string back to an ArrayBuffer of PCM16.
 */
export function base64ToPcm16(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

/**
 * Format milliseconds to a human-readable latency string.
 */
export function formatLatency(ms: number): string {
  if (ms < 0) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Get latency color based on threshold.
 */
export function getLatencyColor(ms: number): string {
  if (ms < 0) return 'var(--text-secondary)';
  if (ms < 300) return 'var(--accent-green)';
  if (ms < 800) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}
