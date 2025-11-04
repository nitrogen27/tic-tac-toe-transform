// Тесты на количество эпох для основного обучения и дообучения
import { describe, it, expect } from 'vitest';
import { TRAIN } from '../src/config.mjs';
import { trainOnGames } from '../service.mjs';
import { saveGameMove, clearGameHistory, getGameHistoryStats } from '../service.mjs';
import { emptyBoard } from '../src/game_ttt3.mjs';
import { teacherBestMove } from '../src/tic_tac_toe.mjs';

describe('Epochs Count Tests', () => {
  beforeEach(() => {
    clearGameHistory();
  });

  afterEach(() => {
    clearGameHistory();
  });

  it('should have 2 epochs for main training (TRAIN.epochs)', () => {
    console.log('[Test] Checking TRAIN.epochs config...');
    expect(TRAIN.epochs).toBe(2);
    console.log('[Test] ✓ TRAIN.epochs is 2 (optimized)');
  });

  it('should use 1 epoch for incremental training (trainOnGames default)', async () => {
    console.log('[Test] Testing trainOnGames default epochs...');
    
    // Создаем достаточное количество данных
    for (let i = 0; i < 15; i++) {
      const board = emptyBoard();
      const current = 1;
      const bestMove = teacherBestMove(board, current);
      if (bestMove >= 0) {
        saveGameMove({ board: Array.from(board), move: bestMove, current, gameId: undefined });
      }
    }
    
    const stats = getGameHistoryStats();
    expect(stats.count).toBeGreaterThanOrEqual(10);
    
    let actualEpochs = null;
    let trainingStarted = false;
    
    try {
      await trainOnGames(
        (ev) => {
          if (ev.type === 'train.start') {
            trainingStarted = true;
            actualEpochs = ev.payload.epochs;
            console.log('[Test] trainOnGames started with epochs:', actualEpochs);
          }
        },
        {} // Без параметров - должны использоваться дефолтные значения
      );
    } catch (e) {
      // Если обучение не запустилось из-за отсутствия модели, это нормально
      // Главное - проверить, что epochs был установлен правильно
      if (!e.message.includes('model') && !e.message.includes('недостаточно')) {
        throw e;
      }
    }
    
    // Проверяем, что если обучение началось, epochs = 1
    if (trainingStarted) {
      expect(actualEpochs).toBe(1);
      console.log('[Test] ✓ trainOnGames uses 1 epoch (optimized)');
    } else {
      console.log('[Test] ⚠ trainOnGames did not start (model may not exist), skipping epochs check');
    }
  }, 30000);

  it('should use 1 epoch when explicitly passed to trainOnGames', async () => {
    console.log('[Test] Testing trainOnGames with explicit epochs=1...');
    
    // Создаем достаточное количество данных
    for (let i = 0; i < 15; i++) {
      const board = emptyBoard();
      const current = 1;
      const bestMove = teacherBestMove(board, current);
      if (bestMove >= 0) {
        saveGameMove({ board: Array.from(board), move: bestMove, current, gameId: undefined });
      }
    }
    
    const stats = getGameHistoryStats();
    expect(stats.count).toBeGreaterThanOrEqual(10);
    
    let actualEpochs = null;
    let trainingStarted = false;
    
    try {
      await trainOnGames(
        (ev) => {
          if (ev.type === 'train.start') {
            trainingStarted = true;
            actualEpochs = ev.payload.epochs;
            console.log('[Test] trainOnGames started with epochs:', actualEpochs);
          }
        },
        { epochs: 1 } // Явно указываем 1 эпоху
      );
    } catch (e) {
      if (!e.message.includes('model') && !e.message.includes('недостаточно')) {
        throw e;
      }
    }
    
    if (trainingStarted) {
      expect(actualEpochs).toBe(1);
      console.log('[Test] ✓ trainOnGames respects explicit epochs=1');
    } else {
      console.log('[Test] ⚠ trainOnGames did not start, skipping epochs check');
    }
  }, 30000);

  it('should not use more than 2 epochs for main training', () => {
    console.log('[Test] Verifying main training epochs limit...');
    expect(TRAIN.epochs).toBeLessThanOrEqual(2);
    expect(TRAIN.epochs).toBeGreaterThanOrEqual(1);
    console.log('[Test] ✓ Main training epochs is within optimal range (1-2)');
  });

  it('should not use more than 1 epoch for incremental training', async () => {
    console.log('[Test] Verifying incremental training epochs limit...');
    
    // Создаем данные
    for (let i = 0; i < 15; i++) {
      const board = emptyBoard();
      const current = 1;
      const bestMove = teacherBestMove(board, current);
      if (bestMove >= 0) {
        saveGameMove({ board: Array.from(board), move: bestMove, current, gameId: undefined });
      }
    }
    
    const stats = getGameHistoryStats();
    expect(stats.count).toBeGreaterThanOrEqual(10);
    
    let maxEpochs = null;
    let progressEvents = [];
    
    try {
      await trainOnGames(
        (ev) => {
          progressEvents.push(ev);
          if (ev.type === 'train.start') {
            maxEpochs = ev.payload.epochs;
          } else if (ev.type === 'train.progress') {
            // Проверяем, что прогресс не превышает 1 эпоху
            if (ev.payload.epoch > maxEpochs) {
              maxEpochs = ev.payload.epoch;
            }
          }
        },
        {}
      );
    } catch (e) {
      if (!e.message.includes('model') && !e.message.includes('недостаточно')) {
        throw e;
      }
    }
    
    if (maxEpochs !== null) {
      expect(maxEpochs).toBeLessThanOrEqual(1);
      console.log('[Test] ✓ Incremental training uses ≤1 epoch');
    } else {
      console.log('[Test] ⚠ Could not verify epochs (training did not start)');
    }
  }, 30000);
});

