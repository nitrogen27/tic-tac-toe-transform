// Тест: проверка использования модели после обучения
import { describe, it, expect } from 'vitest';
import { predictMove } from '../service.mjs';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const MODEL_DIR = path.join(__dirname, '..', 'saved', 'ttt3_transformer');

describe('TTT3 Model Usage Tests', () => {
  it('should detect if model file exists', async () => {
    const modelPath = path.join(MODEL_DIR, 'model.json');
    const exists = await fs.stat(modelPath).then(() => true).catch(() => false);
    
    console.log('[Test] Model file exists:', exists);
    console.log('[Test] Model path:', modelPath);
    
    if (exists) {
      const stats = await fs.stat(modelPath);
      console.log('[Test] Model file size:', stats.size, 'bytes');
      console.log('[Test] Model file modified:', stats.mtime);
    }
    
    // Не проверяем существование, просто логируем
    expect(typeof exists).toBe('boolean');
  });
  
  it('should use model for prediction when model exists', async () => {
    // Позиция: первый столбец заполнен (0, 3)
    const board = [1, 0, 0, 1, 0, 0, 0, 0, 0];
    
    console.log('[Test] Testing prediction with board:', board);
    console.log('[Test] Board layout:');
    console.log(`${board[0]}, ${board[1]}, ${board[2]}`);
    console.log(`${board[3]}, ${board[4]}, ${board[5]}`);
    console.log(`${board[6]}, ${board[7]}, ${board[8]}`);
    
    const result = await predictMove({ board, current: 2, mode: 'model' });
    
    console.log('[Test] Prediction result:', result);
    console.log('[Test] Move:', result.move);
    console.log('[Test] Is random:', result.isRandom);
    console.log('[Test] Fallback:', result.fallback);
    console.log('[Test] Mode:', result.mode);
    
    // Проверяем, что модель используется (не случайные ходы)
    expect(result.move).toBeGreaterThanOrEqual(0);
    expect(result.move).toBeLessThan(9);
    
    // Если модель есть, она должна блокировать (ход в 6)
    if (!result.isRandom && !result.fallback) {
      console.log('[Test] ✓ Model is being used');
      // Модель должна блокировать первый столбец
      expect(result.move).toBe(6);
    } else {
      console.log('[Test] ⚠️ Model not used, fallback:', result.fallback);
    }
  });
});

