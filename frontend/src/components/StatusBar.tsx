import React from 'react';
import { ConnectionStatus } from '../types';

interface StatusBarProps {
  status: ConnectionStatus;
  statusMessage: string;
}

const STATUS_CONFIG: Record<ConnectionStatus, { color: string; label: string; pulse: boolean }> = {
  disconnected: { color: 'var(--text-secondary)', label: 'Disconnected', pulse: false },
  connecting: { color: 'var(--accent-yellow)', label: 'Connecting...', pulse: true },
  connected: { color: 'var(--accent-yellow)', label: 'Connected', pulse: true },
  loading_models: { color: 'var(--accent-yellow)', label: 'Loading Models...', pulse: true },
  ready: { color: 'var(--accent-green)', label: 'Ready', pulse: false },
  streaming: { color: 'var(--accent-green)', label: 'Streaming', pulse: true },
  error: { color: 'var(--accent-red)', label: 'Error', pulse: false },
};

export const StatusBar: React.FC<StatusBarProps> = ({ status, statusMessage }) => {
  const config = STATUS_CONFIG[status];

  return (
    <div className="status-bar">
      <div className="status-indicator-row">
        <span
          className={`status-dot ${config.pulse ? 'pulse' : ''}`}
          style={{ backgroundColor: config.color }}
        />
        <span className="status-label" style={{ color: config.color }}>
          {config.label}
        </span>
      </div>
      {statusMessage && (
        <span className="status-message">{statusMessage}</span>
      )}
    </div>
  );
};
