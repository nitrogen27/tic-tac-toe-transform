// E2E тест полного цикла обучения TTT3 через WebSocket
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WebSocket } from 'ws';

const WS_URL = 'ws://localhost:8080';

describe('TTT3 Training E2E via WebSocket', () => {
  let ws;
  let receivedEvents = [];
  let trainingCompleted = false;
  let trainingError = null;
  
  beforeAll(async () => {
    // Ждем подключения к серверу
    await new Promise((resolve, reject) => {
      ws = new WebSocket(WS_URL);
      ws.on('open', resolve);
      ws.on('error', reject);
      setTimeout(() => reject(new Error('Connection timeout')), 10000);
    });
    
    // Собираем все события
    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        receivedEvents.push(msg);
        
        if (msg.type === 'train.done') {
          trainingCompleted = true;
        } else if (msg.type === 'error') {
          trainingError = msg.error;
        }
      } catch (e) {
        console.error('[Test] Error parsing message:', e);
      }
    });
  });
  
  afterAll(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
  });
  
  it('should start training via WebSocket and complete successfully', async () => {
    console.log('[E2E] Starting training via WebSocket...');
    
    // Отправляем запрос на обучение
    ws.send(JSON.stringify({
      type: 'train_ttt3',
      payload: {
        epochs: 2, // Маленькое количество для теста
        batchSize: 32, // Маленький батч для теста
        earlyStop: false // Отключаем early stopping для предсказуемости
      }
    }));
    
    // Ждем завершения обучения (максимум 2 минуты)
    const startTime = Date.now();
    const timeout = 120000; // 2 минуты
    
    while (!trainingCompleted && !trainingError && (Date.now() - startTime) < timeout) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Проверяем результаты
    expect(trainingError).toBeNull();
    expect(trainingCompleted).toBe(true);
    
    // Проверяем, что были получены события
    const startEvents = receivedEvents.filter(e => e.type === 'train.start');
    const progressEvents = receivedEvents.filter(e => e.type === 'train.progress');
    const doneEvents = receivedEvents.filter(e => e.type === 'train.done');
    
    expect(startEvents.length).toBeGreaterThan(0);
    expect(progressEvents.length).toBeGreaterThan(0);
    expect(doneEvents.length).toBe(1);
    
    console.log('[E2E] ✓ Training completed successfully');
    console.log('[E2E] Received events:', {
      start: startEvents.length,
      progress: progressEvents.length,
      done: doneEvents.length
    });
    
    // Проверяем, что прогресс содержит правильные данные
    if (progressEvents.length > 0) {
      const lastProgress = progressEvents[progressEvents.length - 1];
      expect(lastProgress.payload).toHaveProperty('epoch');
      expect(lastProgress.payload).toHaveProperty('loss');
      
      // Для TTT3 должны быть accuracy и mae
      if (lastProgress.payload.accuracy !== undefined) {
        // Могут быть строки из JSON, преобразуем в числа
        const accuracy = typeof lastProgress.payload.accuracy === 'string' 
          ? parseFloat(lastProgress.payload.accuracy) 
          : lastProgress.payload.accuracy;
        const mae = typeof lastProgress.payload.mae === 'string'
          ? parseFloat(lastProgress.payload.mae)
          : lastProgress.payload.mae;
        
        expect(typeof accuracy).toBe('number');
        expect(typeof mae).toBe('number');
        expect(accuracy).toBeGreaterThanOrEqual(0);
        expect(accuracy).toBeLessThanOrEqual(100);
        expect(mae).toBeGreaterThanOrEqual(0);
        
        console.log('[E2E] Final metrics:', {
          accuracy,
          mae
        });
      }
    }
  }, 120000); // 2 минуты таймаут
});

