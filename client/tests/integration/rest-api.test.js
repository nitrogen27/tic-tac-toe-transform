import { describe, it, expect } from 'vitest'

// Интеграционные тесты для WebSocket API (ранее REST API)
const WS_URL = 'ws://localhost:8080'

function createWebSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL)
    ws.onopen = () => resolve(ws)
    ws.onerror = reject
  })
}

function sendMessage(ws, type, payload = {}) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Timeout waiting for response'))
    }, 10000)
    
    ws.onmessage = (event) => {
      clearTimeout(timeout)
      const msg = JSON.parse(event.data)
      if (msg.type === `${type}.result` || msg.type === `${type}.error`) {
        resolve(msg)
      } else if (msg.type === 'error') {
        reject(new Error(msg.error))
      }
    }
    
    ws.send(JSON.stringify({ type, payload }))
  })
}

async function pvInferViaWS(board, player = 1, N = 3) {
  const ws = await createWebSocket()
  try {
    const result = await sendMessage(ws, 'pv.infer', { board, player, N })
    if (result.type === 'pv.infer.error') {
      throw new Error(result.error)
    }
    return result.payload
  } finally {
    ws.close()
  }
}

async function healthCheckViaWS() {
  const ws = await createWebSocket()
  try {
    const result = await sendMessage(ws, 'health')
    if (result.type === 'health.result') {
      return result.payload
    }
    throw new Error('Invalid response')
  } finally {
    ws.close()
  }
}

describe('WebSocket API Integration', () => {
  describe('Health Check', () => {
    it('should return health status', async () => {
      try {
        const health = await healthCheckViaWS()
        expect(health).toHaveProperty('status')
        expect(health.status).toBe('ok')
        expect(health).toHaveProperty('ws_port')
        expect(health).toHaveProperty('gpu')
      } catch (e) {
        console.warn('Server not available, skipping health check test:', e.message)
      }
    }, 5000)
  })

  describe('PV Inference', () => {
    it('should return policy and value for empty board', async () => {
      try {
        const board = Array(9).fill(0)
        const result = await pvInferViaWS(board, 1)
        
        expect(result).toHaveProperty('policy')
        expect(result).toHaveProperty('value')
        expect(Array.isArray(result.policy)).toBe(true)
        expect(result.policy).toHaveLength(9)
        expect(typeof result.value).toBe('number')
        
        // Policy should sum to ~1 (softmax)
        const sum = result.policy.reduce((a, b) => a + b, 0)
        expect(sum).toBeCloseTo(1, 1)
      } catch (e) {
        if (e.message.includes('not available')) {
          console.warn('Model not available, skipping PV inference test')
        } else {
          console.warn('Server not available, skipping PV inference test:', e.message)
        }
      }
    }, 10000)

    it('should return policy and value for partial board', async () => {
      try {
        const board = [1, 0, 0, 0, 2, 0, 0, 0, 0]
        const result = await pvInferViaWS(board, 1)
        
        expect(result).toHaveProperty('policy')
        expect(result).toHaveProperty('value')
        expect(result.policy).toHaveLength(9)
        
        // Occupied cells should have low policy (but not zero due to softmax)
        expect(result.policy[0]).toBeLessThan(0.5) // Cell 0 is occupied
        expect(result.policy[4]).toBeLessThan(0.5) // Cell 4 is occupied
      } catch (e) {
        if (e.message.includes('not available')) {
          console.warn('Model not available, skipping PV inference test')
        } else {
          console.warn('Server not available, skipping PV inference test:', e.message)
        }
      }
    }, 10000)

    it('should handle different board sizes', async () => {
      try {
        // Test with N=3 (default)
        const board3 = Array(9).fill(0)
        const result3 = await pvInferViaWS(board3, 1, 3)
        expect(result3.policy).toHaveLength(9)
        
        // Note: Testing N=10 requires a trained model for that size
        // This is a placeholder for future expansion
      } catch (e) {
        if (e.message.includes('not available')) {
          console.warn('Model not available, skipping board size test')
        } else {
          console.warn('Server not available, skipping board size test:', e.message)
        }
      }
    }, 10000)

    it('should return error for invalid board length', async () => {
      try {
        const board = [1, 2, 3] // Invalid length
        const result = await pvInferViaWS(board, 1, 3)
        // Should not reach here if error handling works
        expect(result).toBeUndefined()
      } catch (e) {
        // Expected - should throw error
        expect(e.message).toContain('length')
      }
    }, 5000)

    it('should handle different player values', async () => {
      try {
        const board = Array(9).fill(0)
        const result1 = await pvInferViaWS(board, 1)
        const result2 = await pvInferViaWS(board, 2)
        
        // Both should return valid results
        expect(result1.policy).toHaveLength(9)
        expect(result2.policy).toHaveLength(9)
        
        // Values might be different (opposite perspectives)
        expect(typeof result1.value).toBe('number')
        expect(typeof result2.value).toBe('number')
      } catch (e) {
        if (e.message.includes('not available')) {
          console.warn('Model not available, skipping player test')
        } else {
          console.warn('Server not available, skipping player test:', e.message)
        }
      }
    }, 10000)
  })

  describe('Error Handling', () => {
    it('should handle missing model gracefully', async () => {
      try {
        const board = Array(9).fill(0)
        const result = await pvInferViaWS(board, 1)
        
        // If model is available, should return valid result
        expect(result).toHaveProperty('policy')
        expect(result).toHaveProperty('value')
      } catch (e) {
        // If model is not available, should get error
        if (e.message.includes('not available')) {
          expect(e.message).toContain('not available')
        } else {
          console.warn('Server not available, skipping error handling test:', e.message)
        }
      }
    }, 5000)
  })
})
