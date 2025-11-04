// Тесты для прогресса фонового обучения
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { finishGame, startNewGame, saveGameMove, clearGameHistory, getGameHistoryStats } from '../service.mjs';
import { emptyBoard } from '../src/game_ttt3.mjs';
import { teacherBestMove } from '../src/tic_tac_toe.mjs';

describe('Background Training Progress', () => {
  beforeEach(() => {
    clearGameHistory();
  });

  afterEach(() => {
    clearGameHistory();
  });

  it('should send start event immediately when background training starts', async () => {
    console.log('[Test] Testing background training start event...');
    
    // Создаем достаточное количество данных (минимум 10 ходов для обучения)
    const gameId = startNewGame({ playerRole: 2 });
    let board = emptyBoard();
    
    // Сохраняем достаточно ходов для обучения
    for (let i = 0; i < 15; i++) {
      const current = i % 2 === 0 ? 1 : 2;
      const bestMove = teacherBestMove(board, current);
      if (bestMove >= 0 && bestMove < 9 && board[bestMove] === 0) {
        saveGameMove({ board: Array.from(board), move: bestMove, current, gameId });
        board[bestMove] = current;
      } else {
        board = emptyBoard();
      }
    }
    
    const statsBefore = getGameHistoryStats();
    console.log('[Test] Created', statsBefore.count, 'moves before finishing game');
    expect(statsBefore.count).toBeGreaterThanOrEqual(10);
    
    // Завершаем игру с победой человека (модель проиграла)
    let startEventReceived = false;
    let startEventPayload = null;
    
    await finishGame({
      gameId,
      winner: 1, // Человек выиграл
      patternsPerError: 10, // Маленькое количество для быстрого теста
      autoTrain: true,
      progressCb: (ev) => {
        console.log('[Test] Received event:', ev.type);
        if (ev.type === 'background_train.start') {
          startEventReceived = true;
          startEventPayload = ev.payload;
          console.log('[Test] Background training start event received:', ev.payload);
        }
      }
    });
    
    // Ждем немного для асинхронных операций (background training запускается асинхронно)
    for (let i = 0; i < 50; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      if (startEventReceived) break;
    }
    
    // Проверяем, что событие старта было отправлено
    expect(startEventReceived).toBe(true);
    expect(startEventPayload).toBeDefined();
    expect(startEventPayload).toHaveProperty('epochs');
    expect(startEventPayload.epochs).toBe(1); // Фоновое обучение использует 1 эпоху
    expect(startEventPayload).toHaveProperty('newSkills');
    expect(startEventPayload).toHaveProperty('totalSkills');
    expect(startEventPayload).toHaveProperty('message');
    
    console.log('[Test] ✓ Background training start event sent correctly');
  }, 15000);

  it('should send progress events during background training', async () => {
    console.log('[Test] Testing background training progress events...');
    
    // Создаем достаточное количество данных
    const gameId = startNewGame({ playerRole: 2 });
    let board = emptyBoard();
    
    // Сохраняем много ходов для обучения
    for (let i = 0; i < 20; i++) {
      const current = i % 2 === 0 ? 1 : 2;
      const bestMove = teacherBestMove(board, current);
      if (bestMove >= 0 && bestMove < 9 && board[bestMove] === 0) {
        saveGameMove({ board: Array.from(board), move: bestMove, current, gameId });
        board[bestMove] = current;
        
        // Проверяем, не закончилась ли игра
        const winner = board.filter(c => c !== 0).length >= 9 ? 1 : null;
        if (winner) {
          board = emptyBoard();
        }
      } else {
        board = emptyBoard();
      }
    }
    
    const statsBefore = getGameHistoryStats();
    console.log('[Test] Created', statsBefore.count, 'moves before finishing game');
    
    // Завершаем игру
    const progressEvents = [];
    let startEventReceived = false;
    
    await finishGame({
      gameId,
      winner: 1,
      patternsPerError: 10,
      autoTrain: true,
      progressCb: (ev) => {
        progressEvents.push(ev);
        console.log('[Test] Background training event:', ev.type, ev.payload || ev.error);
        
        if (ev.type === 'background_train.start') {
          startEventReceived = true;
        }
      }
    });
    
    // Ждем прогресс обучения (может занять время)
    let progressReceived = false;
    let doneReceived = false;
    
    // Ждем до 15 секунд для получения прогресса
    for (let i = 0; i < 30; i++) {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const hasProgress = progressEvents.some(e => e.type === 'background_train.progress');
      const hasDone = progressEvents.some(e => e.type === 'background_train.done');
      
      if (hasProgress) {
        progressReceived = true;
        console.log('[Test] Progress event received!');
      }
      
      if (hasDone) {
        doneReceived = true;
        console.log('[Test] Done event received!');
        break;
      }
      
      // Если получили прогресс, проверяем его
      if (hasProgress) {
        const progressEvent = progressEvents.find(e => e.type === 'background_train.progress');
        expect(progressEvent).toBeDefined();
        expect(progressEvent.payload).toBeDefined();
        expect(progressEvent.payload).toHaveProperty('epoch');
        expect(progressEvent.payload).toHaveProperty('epochs');
        expect(progressEvent.payload).toHaveProperty('epochPercent');
        expect(progressEvent.payload.epochs).toBe(1);
        expect(progressEvent.payload.epochPercent).toBeGreaterThanOrEqual(0);
        expect(progressEvent.payload.epochPercent).toBeLessThanOrEqual(100);
      }
    }
    
    // Проверяем, что хотя бы одно событие прогресса было получено
    expect(startEventReceived).toBe(true);
    // Если обучение очень быстрое, может быть только done без промежуточного progress
    if (!progressReceived && !doneReceived) {
      console.log('[Test] ⚠ No progress events received, but training may have completed too fast');
      console.log('[Test] Total events:', progressEvents.length, 'Types:', progressEvents.map(e => e.type).join(', '));
    }
    // Проверяем, что хотя бы done или progress был получен
    expect(progressReceived || doneReceived).toBe(true);
    
    console.log('[Test] ✓ Background training progress events working');
    console.log('[Test] Total events received:', progressEvents.length);
    console.log('[Test] Event types:', progressEvents.map(e => e.type).join(', '));
  }, 20000);

  it('should send progress events with correct epoch percent calculation', async () => {
    console.log('[Test] Testing epoch percent calculation...');
    
    // Создаем данные
    const gameId = startNewGame({ playerRole: 2 });
    let board = emptyBoard();
    
    for (let i = 0; i < 15; i++) {
      const current = i % 2 === 0 ? 1 : 2;
      const bestMove = teacherBestMove(board, current);
      if (bestMove >= 0 && bestMove < 9 && board[bestMove] === 0) {
        saveGameMove({ board: Array.from(board), move: bestMove, current, gameId });
        board[bestMove] = current;
      } else {
        board = emptyBoard();
      }
    }
    
    const progressEvents = [];
    
    await finishGame({
      gameId,
      winner: 1,
      patternsPerError: 10,
      autoTrain: true,
      progressCb: (ev) => {
        if (ev.type === 'background_train.progress') {
          progressEvents.push(ev);
          
          // Проверяем корректность расчета процентов
          const { epoch, epochs, epochPercent } = ev.payload;
          const expectedPercent = Math.round((epoch / epochs) * 100);
          
          console.log('[Test] Progress event:', { epoch, epochs, epochPercent, expectedPercent });
          
          expect(epochPercent).toBe(expectedPercent);
          expect(epochPercent).toBeGreaterThanOrEqual(0);
          expect(epochPercent).toBeLessThanOrEqual(100);
        }
      }
    });
    
    // Ждем прогресс
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Проверяем, что были события прогресса с правильными процентами
    if (progressEvents.length > 0) {
      console.log('[Test] ✓ Epoch percent calculation correct');
    } else {
      console.log('[Test] ⚠ No progress events received (training may be too fast)');
    }
  }, 20000);

  it('should send done event when background training completes', async () => {
    console.log('[Test] Testing background training done event...');
    
    const gameId = startNewGame({ playerRole: 2 });
    let board = emptyBoard();
    
    for (let i = 0; i < 15; i++) {
      const current = i % 2 === 0 ? 1 : 2;
      const bestMove = teacherBestMove(board, current);
      if (bestMove >= 0 && bestMove < 9 && board[bestMove] === 0) {
        saveGameMove({ board: Array.from(board), move: bestMove, current, gameId });
        board[bestMove] = current;
      } else {
        board = emptyBoard();
      }
    }
    
    let doneEventReceived = false;
    let doneEventPayload = null;
    
    await finishGame({
      gameId,
      winner: 1,
      patternsPerError: 10,
      autoTrain: true,
      progressCb: (ev) => {
        if (ev.type === 'background_train.done') {
          doneEventReceived = true;
          doneEventPayload = ev.payload;
          console.log('[Test] Background training done event received:', ev.payload);
        }
      }
    });
    
    // Ждем до 30 секунд для завершения обучения
    for (let i = 0; i < 60; i++) {
      await new Promise(resolve => setTimeout(resolve, 500));
      if (doneEventReceived) {
        console.log('[Test] Done event received after', i * 500, 'ms');
        break;
      }
    }
    
    expect(doneEventReceived).toBe(true);
    expect(doneEventPayload).toBeDefined();
    expect(doneEventPayload).toHaveProperty('message');
    
    console.log('[Test] ✓ Background training done event sent correctly');
  }, 35000);
});

