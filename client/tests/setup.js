// Test setup file
import { expect, afterEach } from 'vitest'

// Mock WebSocket
global.WebSocket = class MockWebSocket {
  constructor(url) {
    this.url = url
    this.readyState = WebSocket.CONNECTING
    this.onopen = null
    this.onclose = null
    this.onerror = null
    this.onmessage = null
    this._messages = []
    
    // Simulate connection after a small delay
    setTimeout(() => {
      this.readyState = WebSocket.OPEN
      if (this.onopen) this.onopen()
    }, 10)
  }
  
  send(data) {
    this._messages.push(data)
  }
  
  close() {
    this.readyState = WebSocket.CLOSED
    if (this.onclose) this.onclose({ code: 1000 })
  }
  
  // Helper to simulate message from server
  simulateMessage(data) {
    if (this.onmessage) {
      this.onmessage({ data: typeof data === 'string' ? data : JSON.stringify(data) })
    }
  }
}

WebSocket.CONNECTING = 0
WebSocket.OPEN = 1
WebSocket.CLOSING = 2
WebSocket.CLOSED = 3
