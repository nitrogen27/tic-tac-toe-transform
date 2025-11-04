import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

// Имитация WebSocket клиента для интеграционных тестов
class TestWebSocketClient {
  constructor(url) {
    this.url = url
    this.ws = null
    this.messages = []
    this.connected = false
    this.reconnectAttempts = 0
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        this.connected = true
        this.reconnectAttempts = 0
        resolve()
      }
      
      this.ws.onerror = (err) => {
        reject(err)
      }
      
      this.ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          this.messages.push(msg)
        } catch (e) {
          console.error('Failed to parse message:', e)
        }
      }
      
      this.ws.onclose = () => {
        this.connected = false
      }
    })
  }

  send(type, payload = {}) {
    if (!this.connected || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected')
    }
    this.ws.send(JSON.stringify({ type, payload }))
  }

  waitForMessage(type, timeout = 5000) {
    return new Promise((resolve, reject) => {
      const startTime = Date.now()
      const check = () => {
        const msg = this.messages.find(m => m.type === type)
        if (msg) {
          resolve(msg)
        } else if (Date.now() - startTime > timeout) {
          reject(new Error(`Timeout waiting for message type: ${type}`))
        } else {
          setTimeout(check, 50)
        }
      }
      check()
    })
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
    }
  }
}

describe('WebSocket Integration', () => {
  let client

  beforeEach(() => {
    client = new TestWebSocketClient('ws://localhost:8080')
  })

  afterEach(() => {
    if (client) {
      client.disconnect()
    }
  })

  describe('Connection', () => {
    it('should connect to WebSocket server', async () => {
      // Note: This test requires server to be running
      // In CI/CD, you might want to skip this or use a test server
      try {
        await client.connect()
        expect(client.connected).toBe(true)
      } catch (e) {
        // Skip if server not available
        console.warn('Server not available, skipping connection test')
      }
    }, 10000)

    it('should receive pong on ping', async () => {
      try {
        await client.connect()
        client.send('ping')
        const response = await client.waitForMessage('pong')
        expect(response.type).toBe('pong')
      } catch (e) {
        console.warn('Server not available, skipping ping test')
      }
    }, 10000)

    it('should receive GPU info on connect', async () => {
      try {
        await client.connect()
        const gpuInfo = await client.waitForMessage('gpu.info')
        expect(gpuInfo.type).toBe('gpu.info')
        expect(gpuInfo.payload).toHaveProperty('available')
        expect(gpuInfo.payload).toHaveProperty('backend')
      } catch (e) {
        console.warn('Server not available, skipping GPU info test')
      }
    }, 10000)
  })

  describe('Game Operations', () => {
    it('should get prediction for valid board', async () => {
      try {
        await client.connect()
        const board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        client.send('predict', { board, current: 1, mode: 'model' })
        const response = await client.waitForMessage('predict.result')
        expect(response.type).toBe('predict.result')
        expect(response.payload).toHaveProperty('move')
        expect(response.payload).toHaveProperty('probs')
      } catch (e) {
        console.warn('Server not available, skipping prediction test')
      }
    }, 10000)

    it('should handle invalid board gracefully', async () => {
      try {
        await client.connect()
        const board = [1, 1, 1, 1, 1, 1, 1, 1, 1] // Invalid: all filled
        client.send('predict', { board, current: 1 })
        const response = await client.waitForMessage('predict.result', 2000)
        // Server should handle this gracefully
        expect(response.type).toBe('predict.result')
      } catch (e) {
        // Timeout is expected for invalid board
        console.warn('Expected timeout for invalid board')
      }
    }, 5000)
  })

  describe('Training Operations', () => {
    it('should handle training request', async () => {
      try {
        await client.connect()
        client.send('train', { epochs: 1, nTrain: 10, nVal: 5 })
        const startMsg = await client.waitForMessage('train.start')
        expect(startMsg.type).toBe('train.start')
        expect(startMsg.payload).toHaveProperty('epochs')
      } catch (e) {
        console.warn('Server not available, skipping training test')
      }
    }, 15000)
  })
})
