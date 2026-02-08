import React, { useEffect, useRef } from 'react';

interface TranscriptPanelProps {
  title: string;
  committedSegments: string[];
  partialText: string;
  accentColor: string;
  icon: string;
}

export const TranscriptPanel: React.FC<TranscriptPanelProps> = ({
  title,
  committedSegments,
  partialText,
  accentColor,
  icon,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [committedSegments, partialText]);

  return (
    <div className="transcript-panel" style={{ borderColor: accentColor }}>
      <div className="transcript-header">
        <span className="transcript-icon">{icon}</span>
        <h3 className="transcript-title">{title}</h3>
      </div>
      <div className="transcript-content" ref={scrollRef}>
        {committedSegments.length === 0 && !partialText && (
          <span className="transcript-placeholder">
            Waiting for speech...
          </span>
        )}
        {committedSegments.map((segment, idx) => (
          <span key={idx} className="committed-segment">
            {segment}{' '}
          </span>
        ))}
        {partialText && (
          <span className="partial-segment" style={{ color: accentColor }}>
            {partialText}
          </span>
        )}
      </div>
    </div>
  );
};
