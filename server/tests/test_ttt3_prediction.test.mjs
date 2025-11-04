// Тест: проверка предсказания TTT3 с safety-правилами
import { describe, it, expect, beforeAll } from 'vitest';
import { emptyBoard, applyMove, cloneBoard } from '../src/game_ttt3.mjs';
import { safePick, blockingMove } from '../src/safety.mjs';
import { getTeacherValueAndPolicy } from '../src/ttt3_minimax.mjs';

describe('TTT3 Prediction with Safety Rules', () => {
  it('should block first column attack', () => {
    // Человек заполняет первый столбец: 0, 3
    let board = emptyBoard();
    board = applyMove(board, 0, 1); // Человек (X) ходит в 0
    board = applyMove(board, 3, 1); // Человек (X) ходит в 3
    
    // Бот (O) должен заблокировать ход в 6
    // Проверяем, что blockingMove правильно определяет угрозу
    const blockMove = blockingMove(board, -1);
    console.log('[Test] Blocking move (direct):', blockMove);
    expect(blockMove).toBe(6);
    
    // Теперь проверяем через getTeacherValueAndPolicy
    const { policy } = getTeacherValueAndPolicy(board, -1);
    const move = safePick(board, -1, Array.from(policy));
    
    console.log('[Test] Board:', Array.from(board));
    console.log('[Test] Policy:', Array.from(policy).map((p, i) => `${i}:${p.toFixed(3)}`).join(' '));
    console.log('[Test] Selected move:', move);
    
    // Бот должен заблокировать (ход в 6)
    expect(move).toBe(6);
  });
  
  it('should use random fallback when model not available', async () => {
    // Импортируем функцию предсказания
    const { predictMove } = await import('../service.mjs');
    
    // Человек заполняет первый столбец
    const board = [1, 0, 0, 1, 0, 0, 0, 0, 0]; // X в 0 и 3
    
    const result = await predictMove({ board, current: 2, mode: 'model' });
    
    console.log('[Test] Prediction result:', result);
    console.log('[Test] Selected move:', result.move);
    
    // Без модели используется случайный ход (fallback: random)
    // Проверяем что это случайный ход из доступных
    expect(result.isRandom).toBe(true);
    expect(result.fallback).toBe('random');
    expect(result.move).toBeGreaterThanOrEqual(0);
    expect(result.move).toBeLessThan(9);
    // Проверяем что выбранный ход легальный (не занят)
    expect(board[result.move]).toBe(0);
    
    // Для проверки блокировки используем режим 'algorithm'
    const algorithmResult = await predictMove({ board, current: 2, mode: 'algorithm' });
    expect(algorithmResult.move).toBe(6); // minimax должен заблокировать
    expect(algorithmResult.mode).toBe('algorithm');
  });
});

