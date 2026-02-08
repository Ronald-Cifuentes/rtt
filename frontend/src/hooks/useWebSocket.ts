import { ConnectionStatus, ServerEvent } from '../types';
import { useCallback, useEffect, useRef, useState } from 'react';

interface UseWebSocketOptions {
  url: string;
  onTextMessage: (event: ServerEvent) => void;
  onBinaryMessage: (data: ArrayBuffer) => void;
  onStatusChange: (status: ConnectionStatus) => void;
}

export function useWebSocket({ url, onTextMessage, onBinaryMessage, onStatusChange }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const updateStatus = useCallback((newStatus: ConnectionStatus) => {
    setStatus(newStatus);
    onStatusChange(newStatus);
  }, [onStatusChange]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    updateStatus('connecting');

    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log('[WS] Connected');
      updateStatus('connected');
    };

    ws.onmessage = (event: MessageEvent) => {
      if (event.data instanceof ArrayBuffer) {
        onBinaryMessage(event.data);
      } else {
        try {
          const parsed = JSON.parse(event.data) as ServerEvent;
          onTextMessage(parsed);

          // Auto-detect status changes from server events
          if (parsed.type === 'ready') {
            updateStatus('streaming');
          } else if (parsed.type === 'status') {
            if (parsed.message.includes('Models loaded')) {
              updateStatus('ready');
            } else if (parsed.message.includes('Pipeline started')) {
              updateStatus('streaming');
            }
          }
        } catch (e) {
          console.error('[WS] Failed to parse message:', e);
        }
      }
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
      updateStatus('error');
    };

    ws.onclose = (event) => {
      console.log('[WS] Closed:', event.code, event.reason);
      updateStatus('disconnected');
      wsRef.current = null;
    };

    wsRef.current = ws;
  }, [url, onTextMessage, onBinaryMessage, updateStatus]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    updateStatus('disconnected');
  }, [updateStatus]);

  const sendJSON = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send, not connected');
    }
  }, []);

  const sendBinary = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    status,
    connect,
    disconnect,
    sendJSON,
    sendBinary,
    isConnected: status !== 'disconnected' && status !== 'error',
  };
}
