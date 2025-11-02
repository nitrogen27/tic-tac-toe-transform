import tfpkg from './tf.mjs';
const tf = tfpkg;
import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import os from 'os';
import { emptyBoard, relativeCells, teacherBestMove, getWinner } from './tic_tac_toe.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Функция генерации игр в одном потоке (для worker)
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

// Параллельная генерация с использованием worker threads
async function playGameSamplesParallel(nGames = 2000) {
  const numWorkers = Math.min(4, os.cpus().length); // Используем до 4 воркеров
  const gamesPerWorker = Math.ceil(nGames / numWorkers);
  console.log(`[Dataset] Generating ${nGames} games using ${numWorkers} workers...`);
  const startTime = Date.now();
  
  // Для небольших датасетов используем синхронную генерацию (быстрее из-за накладных расходов)
  if (nGames < 1000 || numWorkers < 2) {
    return generateGamesChunk(nGames);
  }
  
  // Параллельная генерация через workers
  const workerPath = join(__dirname, 'dataset_worker.mjs');
  const workers = [];
  const promises = [];
  
  for (let i = 0; i < numWorkers; i++) {
    const start = i * gamesPerWorker;
    const count = Math.min(gamesPerWorker, nGames - start);
    if (count <= 0) break;
    
    const worker = new Worker(workerPath, {
      workerData: { nGames: count, startIdx: i * 1000 }
    });
    
    workers.push(worker);
    promises.push(new Promise((resolve, reject) => {
      worker.on('message', (result) => resolve(result));
      worker.on('error', reject);
      worker.on('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker stopped with exit code ${code}`));
      });
    }));
  }
  
  const results = await Promise.all(promises);
  
  // Останавливаем всех workers
  workers.forEach(w => w.terminate());
  
  // Объединяем результаты
  const X = [];
  const P = [];
  const Y = [];
  for (const result of results) {
    X.push(...result.X);
    P.push(...result.P);
    Y.push(...result.Y);
  }
  
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`[Dataset] Generated ${nGames} games in ${elapsed}s (${X.length} moves total) - ${(X.length/elapsed).toFixed(0)} moves/s`);
  return { X, P, Y };
}

// Синхронная версия для случаев когда workers недоступны
function playGameSamples(nGames = 2000) {
  console.log(`[Dataset] Generating ${nGames} game samples (single-threaded)...`);
  const startTime = Date.now();
  
  const posTemplate = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  let randomCounter = 0;
  const randomThreshold = 0.05;
  
  const X = [];
  const P = [];
  const Y = [];
  
  for (let g = 0; g < nGames; g++) {
    if (g % 1000 === 0 && g > 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const rate = (g / elapsed).toFixed(0);
      console.log(`[Dataset] Generated ${g}/${nGames} games (${Math.round((g/nGames)*100)}%) - ${rate} games/s`);
    }
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
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`[Dataset] Generated ${nGames} games in ${elapsed}s (${X.length} moves total) - ${(X.length/elapsed).toFixed(0)} moves/s`);
  return { X, P, Y };
}

export async function loadDataset(nTrain = 4000, nVal = 1000) {
  console.log(`[Dataset] Loading dataset: ${nTrain} train, ${nVal} validation`);
  
  // Пробуем использовать параллельную генерацию, если доступна
  let result;
  try {
    result = await playGameSamplesParallel(nTrain + nVal);
  } catch (e) {
    console.log('[Dataset] Parallel generation failed, using single-threaded:', e.message);
    result = playGameSamples(nTrain + nVal);
  }
  
  const { X, P, Y } = result;
  const Xt = X.slice(0, nTrain), Xv = X.slice(nTrain);
  const Pt = P.slice(0, nTrain), Pv = P.slice(nTrain);
  const Yt = Y.slice(0, nTrain), Yv = Y.slice(nTrain);

  console.log(`[Dataset] Creating tensors: train=${Xt.length}, val=${Xv.length}`);
  const xCells = tf.tensor2d(Xt, [Xt.length, 9], 'int32');
  const xPos   = tf.tensor2d(Pt, [Pt.length, 9], 'int32');
  const yOneHot= tf.tensor2d(Yt, [Yt.length, 9], 'float32');

  const xValCells = tf.tensor2d(Xv, [Xv.length, 9], 'int32');
  const xValPos   = tf.tensor2d(Pv, [Pv.length, 9], 'int32');
  const yVal      = tf.tensor2d(Yv, [Yv.length, 9], 'float32');

  console.log('[Dataset] Dataset loaded successfully');
  return { xCells, xPos, yOneHot, xValCells, xValPos, yVal };
}
