// Тесты обучаемости модели и качества игры
import { describe, it, expect, beforeAll } from 'vitest';
import { trainTTT3WithProgress } from '../src/train_ttt3_transformer_service.mjs';
import { predictMove } from '../service.mjs';
import { teacherBestMove, legalMoves, getWinner, emptyBoard } from '../src/tic_tac_toe.mjs';

// Играет одну игру: модель (X) vs minimax (O)
async function playGameModelVsMinimax(modelReady = true) {
  const board = emptyBoard();
  let current = 1; // Модель играет за X (1)
  const moves = [];
  
  while (getWinner(board) === null) {
    const legal = legalMoves(board);
    if (legal.length === 0) break;
    
    let move;
    if (current === 1) {
      // Ход модели
      try {
        const result = await predictMove({ board: [...board], current: 1, mode: 'model' });
        move = result.move;
        if (move < 0 || move >= 9 || board[move] !== 0) {
          // Модель сделала недопустимый ход
          return { winner: 2, reason: 'invalid_move', moves };
        }
      } catch (e) {
        return { winner: 2, reason: 'model_error', error: e.message, moves };
      }
    } else {
      // Ход minimax (O)
      move = teacherBestMove(board, 2);
    }
    
    board[move] = current;
    moves.push({ player: current, move, board: [...board] });
    current = current === 1 ? 2 : 1;
    
    // Защита от бесконечного цикла
    if (moves.length > 20) break;
  }
  
  const winner = getWinner(board);
  return { winner: winner || 0, moves };
}

// Играет N игр и возвращает статистику
async function playMultipleGames(nGames = 10, modelReady = true) {
  const results = {
    modelWins: 0,
    minimaxWins: 0,
    draws: 0,
    invalidMoves: 0,
    errors: 0,
    totalMoves: 0
  };
  
  for (let i = 0; i < nGames; i++) {
    const result = await playGameModelVsMinimax(modelReady);
    results.totalMoves += result.moves.length;
    
    if (result.winner === 1) {
      results.modelWins++;
    } else if (result.winner === 2) {
      if (result.reason === 'invalid_move') {
        results.invalidMoves++;
      } else if (result.reason === 'model_error') {
        results.errors++;
      }
      results.minimaxWins++;
    } else {
      results.draws++;
    }
  }
  
  return results;
}

// Проверяет, делает ли модель оптимальные ходы (сравнивает с minimax)
async function checkModelOptimalMoves(nPositions = 20) {
  let optimalMoves = 0;
  let totalMoves = 0;
  let errors = 0;
  
  // Генерируем случайные позиции
  for (let i = 0; i < nPositions; i++) {
    const board = emptyBoard();
    let current = 1;
    let moves = 0;
    
    // Делаем несколько случайных ходов
    while (moves < 3 && getWinner(board) === null) {
      const legal = legalMoves(board);
      if (legal.length === 0) break;
      
      if (current === 1) {
        // Используем minimax для создания позиции
        const move = teacherBestMove(board, current);
        board[move] = current;
      } else {
        // Случайный ход для разнообразия
        const move = legal[Math.floor(Math.random() * legal.length)];
        board[move] = current;
      }
      current = current === 1 ? 2 : 1;
      moves++;
    }
    
    // Проверяем предсказание модели
    try {
      const result = await predictMove({ board: [...board], current: 1, mode: 'model' });
      const modelMove = result.move;
      const optimalMove = teacherBestMove(board, 1);
      
      totalMoves++;
      if (modelMove === optimalMove) {
        optimalMoves++;
      } else {
        // Проверяем, не является ли ход модели допустимым
        const legal = legalMoves(board);
        if (!legal.includes(modelMove)) {
          errors++;
        }
      }
    } catch (e) {
      errors++;
      totalMoves++;
    }
  }
  
  return {
    optimalMoves,
    totalMoves,
    errors,
    accuracy: totalMoves > 0 ? optimalMoves / totalMoves : 0
  };
}

describe('Model Playability Tests', () => {
  let modelTrained = false;
  
  beforeAll(async () => {
    // Проверяем, есть ли обученная модель
    try {
      const result = await predictMove({ 
        board: [0,0,0,0,0,0,0,0,0], 
        current: 1, 
        mode: 'model' 
      });
      if (!result.isRandom) {
        modelTrained = true;
        console.log('[Test] Model is trained and ready');
      }
    } catch (e) {
      console.log('[Test] Model not available or error:', e.message);
    }
  });
  
  it('should make valid moves in all positions', async () => {
    const stats = await checkModelOptimalMoves(30);
    console.log('[Test] Optimal moves check:', stats);
    expect(stats.errors).toBe(0);
    expect(stats.totalMoves).toBeGreaterThan(0);
  }, 60000);
  
  it('should play games without invalid moves', async () => {
    if (!modelTrained) {
      console.log('[Test] Skipping - model not trained');
      return;
    }
    
    const results = await playMultipleGames(5);
    console.log('[Test] Game results:', results);
    expect(results.invalidMoves).toBe(0);
    expect(results.errors).toBe(0);
  }, 120000);
  
  it('should achieve reasonable win rate against minimax', async () => {
    if (!modelTrained) {
      console.log('[Test] Skipping - model not trained');
      return;
    }
    
    const results = await playMultipleGames(10);
    console.log('[Test] Win rate test:', results);
    const winRate = results.modelWins / (results.modelWins + results.minimaxWins + results.draws);
    const drawRate = results.draws / (results.modelWins + results.minimaxWins + results.draws);
    
    // Модель должна выигрывать или делать ничью хотя бы в 30% случаев
    // (minimax идеален, так что модель не должна проигрывать всегда)
    const successRate = winRate + drawRate;
    expect(successRate).toBeGreaterThan(0.3);
  }, 180000);
});

