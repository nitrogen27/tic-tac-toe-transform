// Тесты для дообучения на играх (trainOnGames)
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { trainOnGames, clearGameHistory, saveGameMove, startNewGame, finishGame, getGameHistoryStats } from '../service.mjs';
import tfpkg from '../src/tf.mjs';
const tf = tfpkg;
import { emptyBoard, applyMove, winner } from '../src/game_ttt3.mjs';
import { getTeacherValueAndPolicy } from '../src/ttt3_minimax.mjs';
import { relativeCells, teacherBestMove } from '../src/tic_tac_toe.mjs';

describe('Train on Games (Incremental Learning)', () => {
  beforeEach(() => {
    // Очищаем историю перед каждым тестом
    clearGameHistory();
    console.log('[Test] Game history cleared');
  });

  afterEach(async () => {
    // Очищаем историю после теста
    clearGameHistory();
    // Очистка памяти TensorFlow
    await tf.engine().startScope();
    await tf.engine().endScope();
  });

  it('should reject training with insufficient data', async () => {
    console.log('[Test] Testing insufficient data rejection...');
    
    // Создаем только 5 ходов (меньше минимума 10)
    for (let i = 0; i < 5; i++) {
      const board = emptyBoard();
      const move = i % 9;
      const current = 1;
      const rel = relativeCells(board, current);
      const pos = [0, 1, 2, 3, 4, 5, 6, 7, 8];
      const onehot = new Array(9).fill(0);
      onehot[move] = 1;
      
      saveGameMove({ board: Array.from(board), move, current, gameId: undefined });
    }
    
    const stats = getGameHistoryStats();
    expect(stats.count).toBe(5);
    
    // Попытка обучения должна завершиться ошибкой
    let error = null;
    try {
      await trainOnGames(
        (ev) => {
          if (ev.type === 'error') {
            error = ev.error;
          }
        },
        { epochs: 1, batchSize: 64 }
      );
    } catch (e) {
      error = e.message;
    }
    
    expect(error).toContain('Недостаточно данных');
    expect(error).toContain('10');
    console.log('[Test] ✓ Insufficient data correctly rejected');
  });

  it('should train on game history with sufficient data', async () => {
    console.log('[Test] Testing training on game history...');
    
    // Создаем 20 ходов для обучения
    // ВАЖНО: trainOnGames использует старую модель (ensureModel), которая ожидает формат relativeCells
    // Создаем ходы в правильном формате
    let board = emptyBoard();
    const moves = [];
    
    for (let i = 0; i < 20; i++) {
      // Создаем валидную позицию, постепенно заполняя доску
      const current = i % 2 === 0 ? 1 : 2;
      const bestMove = teacherBestMove(board, current);
      
      if (bestMove >= 0 && bestMove < 9 && board[bestMove] === 0) {
        const rel = relativeCells(board, current);
        const pos = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        const onehot = new Array(9).fill(0);
        onehot[bestMove] = 1;
        
        moves.push({ board: Array.from(board), move: bestMove, current, rel, pos, onehot });
        
        // Применяем ход для следующей итерации
        board = applyMove(board, bestMove, current);
        
        // Если игра окончена, создаем новую доску
        if (winner(board) !== null) {
          board = emptyBoard();
        }
      } else {
        // Если нет валидного хода, создаем новую доску
        board = emptyBoard();
      }
    }
    
    // Сохраняем ходы через saveGameMove (автоматически создает правильный формат)
    for (const move of moves) {
      saveGameMove({ board: move.board, move: move.move, current: move.current, gameId: undefined });
    }
    
    // Если данных недостаточно, добавляем больше
    const stats = getGameHistoryStats();
    if (stats.count < 10) {
      // Добавляем дополнительные ходы
      for (let i = 0; i < 15; i++) {
        const testBoard = emptyBoard();
        const current = 1;
        const bestMove = teacherBestMove(testBoard, current);
        if (bestMove >= 0) {
          saveGameMove({ board: Array.from(testBoard), move: bestMove, current, gameId: undefined });
        }
      }
    }
    
    const finalStats = getGameHistoryStats();
    expect(finalStats.count).toBeGreaterThanOrEqual(10);
    console.log('[Test] Created', finalStats.count, 'game moves');
    
    // Запускаем обучение
    let trainingStarted = false;
    let trainingCompleted = false;
    let progressEvents = [];
    let error = null;
    
    try {
      await trainOnGames(
        (ev) => {
          console.log('[Test] Training event:', ev.type);
          progressEvents.push(ev);
          
          if (ev.type === 'train.start') {
            trainingStarted = true;
            expect(ev.payload).toHaveProperty('epochs');
            expect(ev.payload.epochs).toBe(1); // Оптимизировано: 1 эпоха
            expect(ev.payload).toHaveProperty('nTrain');
            expect(ev.payload.nTrain).toBeGreaterThanOrEqual(10);
          } else if (ev.type === 'train.progress') {
            expect(ev.payload).toHaveProperty('epoch');
            expect(ev.payload).toHaveProperty('epochs');
            expect(ev.payload).toHaveProperty('loss');
            expect(ev.payload).toHaveProperty('acc');
            expect(ev.payload.epoch).toBeGreaterThan(0);
            expect(ev.payload.epoch).toBeLessThanOrEqual(ev.payload.epochs);
          } else if (ev.type === 'train.done') {
            trainingCompleted = true;
          } else if (ev.type === 'error') {
            error = ev.error;
          }
        },
        { epochs: 1, batchSize: 64 }
      );
    } catch (e) {
      error = e.message;
      console.error('[Test] Training error:', e);
    }
    
    expect(error).toBeNull();
    expect(trainingStarted).toBe(true);
    expect(trainingCompleted).toBe(true);
    expect(progressEvents.length).toBeGreaterThan(0);
    
    // Проверяем, что были события прогресса
    const progressEvents_ = progressEvents.filter(e => e.type === 'train.progress');
    expect(progressEvents_.length).toBeGreaterThan(0);
    
    console.log('[Test] ✓ Training on game history completed successfully');
    console.log('[Test] Progress events:', progressEvents.length);
  }, 60000); // 60 секунд таймаут

  it('should handle background training with new skills', async () => {
    console.log('[Test] Testing background training simulation...');
    
    // Создаем игру и сохраняем ходы
    const gameId = startNewGame({ playerRole: 2 });
    expect(gameId).toBeDefined();
    
    // Симулируем игру с ошибками модели
    const board = emptyBoard();
    applyMove(board, 0, 1); // Человек (X) ходит в 0
    applyMove(board, 3, 1); // Человек (X) ходит в 3
    
    // Сохраняем ходы
    saveGameMove({ board: Array.from(board), move: 0, current: 1, gameId });
    saveGameMove({ board: Array.from(board), move: 3, current: 1, gameId });
    
    // Завершаем игру с победой человека (модель проиграла)
    const statsBefore = getGameHistoryStats();
    const patternsBefore = statsBefore.count;
    
    // Симулируем finishGame с autoTrain=false (чтобы не запускать реальное фоновое обучение)
    // Но добавляем ходы вручную для теста
    for (let i = 0; i < 15; i++) {
      const testBoard = emptyBoard();
      const current = 1;
      const rel = relativeCells(testBoard, current);
      const pos = [0, 1, 2, 3, 4, 5, 6, 7, 8];
      const onehot = new Array(9).fill(0);
      const bestMove = teacherBestMove(testBoard, current);
      if (bestMove >= 0) {
        onehot[bestMove] = 1;
      }
      saveGameMove({ board: Array.from(testBoard), move: bestMove, current, gameId });
    }
    
    const statsAfter = getGameHistoryStats();
    const patternsAfter = statsAfter.count;
    const newSkills = patternsAfter - patternsBefore;
    
    console.log('[Test] Patterns before:', patternsBefore);
    console.log('[Test] Patterns after:', patternsAfter);
    console.log('[Test] New skills:', newSkills);
    
    expect(patternsAfter).toBeGreaterThan(patternsBefore);
    
    // Теперь можем запустить обучение
    if (statsAfter.count >= 10) {
      let trainingCompleted = false;
      await trainOnGames(
        (ev) => {
          if (ev.type === 'train.done') {
            trainingCompleted = true;
          }
        },
        { epochs: 1, batchSize: 64 }
      );
      
      expect(trainingCompleted).toBe(true);
      console.log('[Test] ✓ Background training simulation completed');
    }
  }, 60000);
});

