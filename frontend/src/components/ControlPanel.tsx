import React from 'react';
import { ConnectionStatus } from '../types';

interface ControlPanelProps {
  status: ConnectionStatus;
  isCapturing: boolean;
  onStart: () => void;
  onStop: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  status,
  isCapturing,
  onStart,
  onStop,
}) => {
  const canStart = status === 'disconnected' || status === 'error';
  const canStop = isCapturing || status === 'streaming' || status === 'ready' || status === 'connected' || status === 'connecting' || status === 'loading_models';

  return (
    <div className="control-panel">
      {!isCapturing ? (
        <button
          className="control-btn start-btn"
          onClick={onStart}
          disabled={!canStart && status !== 'disconnected'}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="6" fill="currentColor" />
          </svg>
          Start Translation
        </button>
      ) : (
        <button
          className="control-btn stop-btn"
          onClick={onStop}
          disabled={!canStop}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" />
          </svg>
          Stop
        </button>
      )}
    </div>
  );
};
