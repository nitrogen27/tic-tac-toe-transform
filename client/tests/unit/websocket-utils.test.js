import { describe, it, expect, beforeEach, vi } from 'vitest'

// Утилиты для работы с WebSocket
function createWebSocketConnection(url) {
  return new WebSocket(url);
}

function parseMessage(data) {
  try {
    return JSON.parse(data);
  } catch (e) {
    return null;
  }
}

function sendMessage(ws, type, payload = {}) {
  if (ws.readyState !== WebSocket.OPEN) {
    throw new Error('WebSocket not connected');
  }
  ws.send(JSON.stringify({ type, payload }));
}

describe('WebSocket Utils', () => {
  let mockWs;

  beforeEach(() => {
    mockWs = {
      readyState: WebSocket.OPEN,
      send: vi.fn(),
      close: vi.fn(),
    };
  });

  describe('parseMessage', () => {
    it('should parse valid JSON message', () => {
      const msg = { type: 'test', payload: { data: 123 } };
      const parsed = parseMessage(JSON.stringify(msg));
      expect(parsed).toEqual(msg);
    });

    it('should return null for invalid JSON', () => {
      const parsed = parseMessage('invalid json');
      expect(parsed).toBe(null);
    });

    it('should handle empty string', () => {
      const parsed = parseMessage('');
      expect(parsed).toBe(null);
    });
  });

  describe('sendMessage', () => {
    it('should send valid message when connected', () => {
      sendMessage(mockWs, 'test', { data: 123 });
      expect(mockWs.send).toHaveBeenCalledWith(JSON.stringify({
        type: 'test',
        payload: { data: 123 }
      }));
    });

    it('should throw error when not connected', () => {
      mockWs.readyState = WebSocket.CLOSED;
      expect(() => sendMessage(mockWs, 'test')).toThrow('WebSocket not connected');
    });

    it('should send message without payload', () => {
      sendMessage(mockWs, 'ping');
      expect(mockWs.send).toHaveBeenCalledWith(JSON.stringify({
        type: 'ping',
        payload: {}
      }));
    });
  });

  describe('Message types', () => {
    it('should handle train message', () => {
      sendMessage(mockWs, 'train', { epochs: 5, nTrain: 1000 });
      expect(mockWs.send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'train', payload: { epochs: 5, nTrain: 1000 } })
      );
    });

    it('should handle predict message', () => {
      sendMessage(mockWs, 'predict', { board: [0,0,0,0,0,0,0,0,0], current: 1 });
      const call = mockWs.send.mock.calls[0][0];
      const parsed = JSON.parse(call);
      expect(parsed.type).toBe('predict');
      expect(parsed.payload.board).toHaveLength(9);
    });

    it('should handle clear_model message', () => {
      sendMessage(mockWs, 'clear_model');
      expect(mockWs.send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'clear_model', payload: {} })
      );
    });
  });
});
