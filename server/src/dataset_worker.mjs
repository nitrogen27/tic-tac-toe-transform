import { parentPort, workerData } from 'worker_threads';
import { emptyBoard, relativeCells, teacherBestMove, getWinner } from './tic_tac_toe.mjs';

function generateGamesChunk(nGames, startIdx = 0) {
  const X = [];
  const P = [];
  const Y = [];
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
      const onehot = new Array(9).fill(0);
      onehot[best] = 1;
      X.push(rel); 
      P.push(posTemplate.slice());
      Y.push(onehot);
      board[best] = current;
      current = current === 1 ? 2 : 1;
      step++;
      if (step > 9) break;
      randomCounter++;
      if (randomCounter % 20 === 0 && Math.random() < randomThreshold) break;
    }
  }
  return { X, P, Y };
}

// Генерируем игры
const result = generateGamesChunk(workerData.nGames, workerData.startIdx);

// Отправляем результат обратно
parentPort.postMessage(result);

