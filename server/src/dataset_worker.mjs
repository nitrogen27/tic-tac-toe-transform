import { parentPort, workerData } from 'worker_threads';
import { emptyBoard, relativeCells, teacherBestMove, getWinner } from './tic_tac_toe.mjs';

function generateGamesChunk(nGames, startIdx = 0) {
  // Оптимизированная версия с TypedArray для минимизации копий
  const estimatedMoves = nGames * 7;
  
  // Используем TypedArray для эффективной передачи через transfer
  const X = new Int32Array(estimatedMoves * 9); // Плоский массив для передачи
  const P = new Int32Array(estimatedMoves * 9); // Позиции
  const Y = new Float32Array(estimatedMoves * 9); // One-hot векторы
  let idx = 0;
  
  const posTemplate = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  let randomCounter = startIdx;
  const randomThreshold = 0.05;
  
  for (let g = 0; g < nGames; g++) {
    let board = emptyBoard();
    let current = 1;
    let step = 0;
    while (getWinner(board) === null) {
      const rel = relativeCells(board, current);
      const best = teacherBestMove(board, current);
      
      // Записываем напрямую в TypedArray
      const baseIdx = idx * 9;
      for (let i = 0; i < 9; i++) {
        X[baseIdx + i] = rel[i];
        P[baseIdx + i] = posTemplate[i];
        Y[baseIdx + i] = i === best ? 1.0 : 0.0;
      }
      idx++;
      
      board[best] = current;
      current = current === 1 ? 2 : 1;
      step++;
      if (step > 9) break;
      randomCounter++;
      if (randomCounter % 20 === 0 && Math.random() < randomThreshold) break;
    }
  }
  
  // Обрезаем до реального размера
  const actualSize = idx * 9;
  
  return { 
    X: X.subarray(0, actualSize), // subarray не копирует, просто ссылка
    P: P.subarray(0, actualSize),
    Y: Y.subarray(0, actualSize),
    count: idx
  };
}

// Генерируем игры
const result = generateGamesChunk(workerData.nGames, workerData.startIdx);

// Отправляем результат с transfer для избежания копий TypedArray
parentPort.postMessage(result, [result.X.buffer, result.P.buffer, result.Y.buffer]);

