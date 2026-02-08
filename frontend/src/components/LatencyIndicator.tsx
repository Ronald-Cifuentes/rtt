import React from 'react';
import { LatencyStats } from '../types';
import { formatLatency, getLatencyColor } from '../utils/audioUtils';

interface LatencyIndicatorProps {
  stats: LatencyStats;
}

export const LatencyIndicator: React.FC<LatencyIndicatorProps> = ({ stats }) => {
  return (
    <div className="latency-indicator">
      <div className="latency-title">Latency</div>
      <div className="latency-grid">
        <div className="latency-item">
          <span className="latency-label">ASR</span>
          <span className="latency-value" style={{ color: getLatencyColor(stats.asr) }}>
            {formatLatency(stats.asr)}
          </span>
        </div>
        <div className="latency-item">
          <span className="latency-label">MT</span>
          <span className="latency-value" style={{ color: getLatencyColor(stats.mt) }}>
            {formatLatency(stats.mt)}
          </span>
        </div>
        <div className="latency-item">
          <span className="latency-label">TTS</span>
          <span className="latency-value" style={{ color: getLatencyColor(stats.tts) }}>
            {formatLatency(stats.tts)}
          </span>
        </div>
        <div className="latency-item latency-total">
          <span className="latency-label">E2E</span>
          <span className="latency-value" style={{ color: getLatencyColor(stats.e2e) }}>
            {formatLatency(stats.e2e)}
          </span>
        </div>
      </div>
    </div>
  );
};
