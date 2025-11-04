import tfpkg from './tf.mjs';
const tf = tfpkg;
import { getGpuInfo } from './tf.mjs';
import { emptyBoard, relativeCells, teacherBestMove, getWinner } from './tic_tac_toe.mjs';
import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import os from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Размер пакета для генерации на GPU (генерация пакетов и создание тензоров на GPU)
const GPU_BATCH_SIZE = 2000; // Увеличено для большей эффективности
const PARALLEL_WORKERS = Math.min(8, os.cpus().length); // До 8 воркеров для параллельной генерации

// Генерация одного пакета игр и создание тензоров сразу на GPU
function generateBatchOnGPU(nGames, startIdx = 0) {
  const X = [];
  const P = [];
  const Y = [];
  const posTemplate = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  let randomCounter = startIdx;
  const randomThreshold = 0.05;
  let idx = 0;
  
  for (let g = 0; g < nGames; g++) {
    let board = emptyBoard();
    let current = 1;
    let step = 0;
    while (getWinner(board) === null) {
      const rel = relativeCells(board, current);
      const best = teacherBestMove(board, current);
      
      X[idx] = rel;
      P[idx] = Array.from(posTemplate);
      const onehot = new Array(9).fill(0);
      onehot[best] = 1;
      Y[idx] = onehot;
      idx++;
      
      board[best] = current;
      current = current === 1 ? 2 : 1;
      step++;
      if (step > 9) break;
      randomCounter++;
      if (randomCounter % 20 === 0 && Math.random() < randomThreshold) break;
    }
  }
  
  // Сразу создаём тензоры на GPU из этого пакета
  if (idx === 0) {
    // Пустой батч - возвращаем пустые тензоры на GPU
    return {
      xCells: tf.zeros([0, 9], 'int32'),
      xPos: tf.zeros([0, 9], 'int32'),
      yOneHot: tf.zeros([0, 9], 'float32'),
      count: 0
    };
  }
  
  // Обрезаем до реального размера
  X.length = idx;
  P.length = idx;
  Y.length = idx;
  
  // Создаём тензоры сразу на GPU (tf.tensor2d автоматически размещает на GPU если доступен)
  // Все операции будут выполняться на GPU
  const xCells = tf.tensor2d(X, [idx, 9], 'int32');
  const xPos = tf.tensor2d(P, [idx, 9], 'int32');
  const yOneHot = tf.tensor2d(Y, [idx, 9], 'float32');
  
  // Освобождаем память CPU сразу после создания тензоров
  X.length = 0;
  P.length = 0;
  Y.length = 0;
  
  return { xCells, xPos, yOneHot, count: idx };
}

// Параллельная генерация батчей через worker threads
async function generateBatchesParallel(nGames, progressCb = null) {
  const totalBatches = Math.ceil(nGames / GPU_BATCH_SIZE);
  const batchesPerWorker = Math.ceil(totalBatches / PARALLEL_WORKERS);
  const workerPath = join(__dirname, 'dataset_worker.mjs');
  
  const workers = [];
  const promises = [];
  let completedGames = 0;
  const startTime = Date.now();
  
  // Запускаем воркеров параллельно
  for (let w = 0; w < PARALLEL_WORKERS; w++) {
    const workerStartBatch = w * batchesPerWorker;
    const workerBatches = Math.min(batchesPerWorker, totalBatches - workerStartBatch);
    const workerStartGames = workerStartBatch * GPU_BATCH_SIZE;
    const workerGames = Math.min(workerBatches * GPU_BATCH_SIZE, nGames - workerStartGames);
    
    if (workerGames <= 0) break;
    
    const worker = new Worker(workerPath, {
      workerData: { 
        nGames: workerGames, 
        startIdx: workerStartBatch * 1000,
        workerId: w
      }
    });
    
    workers.push(worker);
    promises.push(new Promise((resolve, reject) => {
      worker.on('message', (result) => {
        completedGames += workerGames;
        const elapsed = ((Date.now() - startTime) / 1000);
        const rate = (completedGames / elapsed).toFixed(0);
        const percent = Math.round((completedGames / nGames) * 100);
        
        // НЕ отправляем промежуточный прогресс - только в onEpochEnd
        // Минимум колбэков для производительности
        resolve(result);
      });
      worker.on('error', reject);
      worker.on('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker ${w} stopped with exit code ${code}`));
      });
    }));
  }
  
  const results = await Promise.all(promises);
  workers.forEach(w => w.terminate());
  
  // Объединяем TypedArrays эффективно
  let totalMoves = 0;
  for (const result of results) {
    totalMoves += result.count || (result.X ? result.X.length / 9 : 0);
  }
  
  // Создаём результирующие TypedArrays
  const allX = new Int32Array(totalMoves * 9);
  const allP = new Int32Array(totalMoves * 9);
  const allY = new Float32Array(totalMoves * 9);
  
  let offset = 0;
  for (const result of results) {
    if (!result.X || !result.P || !result.Y) {
      console.warn('[Dataset] Skipping incomplete result from worker');
      continue;
    }
    const count = result.count || (result.X.length / 9);
    const size = count * 9;
    
    // result.X уже TypedArray после transfer
    allX.set(result.X, offset);
    allP.set(result.P, offset);
    allY.set(result.Y, offset);
    offset += size;
  }
  
  // Возвращаем TypedArrays напрямую - тензоры создадим из них без конвертации
  return {
    X: allX,
    P: allP,
    Y: allY,
    count: totalMoves
  };
}

// Генерация датасета полностью на GPU: пакетами, тензоры создаются сразу на GPU
async function generateDatasetOnGPU(nGames = 2000, progressCb = null) {
  const gpuInfo = getGpuInfo();
  if (!gpuInfo.available) {
    throw new Error('[Dataset] GPU not available! Cannot generate dataset on GPU.');
  }
  
  const startTime = Date.now();
  
  // Используем параллельную генерацию через workers для ускорения
  let gameData;
  let batches = [];
  
  if (nGames >= GPU_BATCH_SIZE * 2 && PARALLEL_WORKERS > 1) {
    // Параллельная генерация для больших датасетов
    gameData = await generateBatchesParallel(nGames, progressCb);
    
    // Создаём тензоры напрямую из TypedArray без промежуточных копий
    // TensorFlow.js поддерживает создание тензоров из TypedArray
    const tensorBatchSize = Math.min(20000, gameData.count); // Большие батчи для тензоров
    for (let i = 0; i < gameData.count; i += tensorBatchSize) {
      const end = Math.min(i + tensorBatchSize, gameData.count);
      const moveStart = i * 9;
      const moveEnd = end * 9;
      
      // Создаём тензоры напрямую из TypedArray без slice (subarray не копирует)
      batches.push({
        xCells: tf.tensor2d(gameData.X.subarray(moveStart, moveEnd), [end - i, 9], 'int32'),
        xPos: tf.tensor2d(gameData.P.subarray(moveStart, moveEnd), [end - i, 9], 'int32'),
        yOneHot: tf.tensor2d(gameData.Y.subarray(moveStart, moveEnd), [end - i, 9], 'float32')
      });
    }
    // Освобождаем память после создания тензоров
    gameData.X = null;
    gameData.P = null;
    gameData.Y = null;
  } else {
    // Последовательная генерация для небольших датасетов
    let totalMoves = 0;
    let startIdx = 0;
    
    for (let i = 0; i < nGames; i += GPU_BATCH_SIZE) {
      const batchSize = Math.min(GPU_BATCH_SIZE, nGames - i);
      const batch = generateBatchOnGPU(batchSize, startIdx);
      
      if (batch.count > 0) {
        batches.push(batch);
        totalMoves += batch.count;
      }
      
      startIdx += batchSize;
    }
  }
  
  if (!batches || batches.length === 0) {
    throw new Error('[Dataset] No batches generated!');
  }
  
  
  // Объединяем тензоры: финальные тензоры создаём вне tidy, чтобы они не были удалены
  const xCellsTensors = batches.map(b => b.xCells);
  const xPosTensors = batches.map(b => b.xPos);
  const yOneHotTensors = batches.map(b => b.yOneHot);
  
  // Конкатенация на GPU
  const xCells = batches.length === 1 
    ? batches[0].xCells 
    : tf.concat(xCellsTensors, 0);
  const xPos = batches.length === 1 
    ? batches[0].xPos 
    : tf.concat(xPosTensors, 0);
  const yOneHot = batches.length === 1 
    ? batches[0].yOneHot 
    : tf.concat(yOneHotTensors, 0);
  
  // Освобождаем промежуточные тензоры
  if (batches.length > 1) {
    batches.forEach(batch => {
      batch.xCells.dispose();
      batch.xPos.dispose();
      batch.yOneHot.dispose();
    });
  }
  
  const tensors = { xCells, xPos, yOneHot };
  
  return tensors;
}

// Резервный метод: если GPU недоступен (не должно произойти в production)
async function generateDatasetFallback(nGames = 2000) {
  console.warn('[Dataset] WARNING: Using CPU fallback (GPU expected)!');
  const X = [];
  const P = [];
  const Y = [];
  const posTemplate = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  let randomCounter = 0;
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
      P.push(Array.from(posTemplate));
      Y.push(onehot);
      board[best] = current;
      current = current === 1 ? 2 : 1;
      step++;
      if (step > 9) break;
      randomCounter++;
      if (randomCounter % 20 === 0 && Math.random() < randomThreshold) break;
    }
  }
  
  // Создаём тензоры (будут на GPU если доступен, но лучше не использовать этот путь)
  return {
    xCells: tf.tensor2d(X, [X.length, 9], 'int32'),
    xPos: tf.tensor2d(P, [P.length, 9], 'int32'),
    yOneHot: tf.tensor2d(Y, [Y.length, 9], 'float32')
  };
}

export async function loadDataset(nTrain = 4000, nVal = 5000, progressCb = null) {
  const totalGames = nTrain + nVal;
  
  // Генерируем весь датасет на GPU
  let dataset;
  try {
    dataset = await generateDatasetOnGPU(totalGames, progressCb);
  } catch (e) {
    console.error('[Dataset] GPU generation failed:', e.message);
    console.log('[Dataset] Falling back to CPU (should not happen!)...');
    dataset = await generateDatasetFallback(totalGames);
  }
  
  // Разделяем на train/val: финальные тензоры создаём вне tidy
  const totalMoves = dataset.xCells.shape[0];
  const trainMoves = Math.floor((totalMoves * nTrain) / totalGames);
  
  // Разделяем тензоры на GPU (операции slice на GPU)
  const xCellsTrain = trainMoves > 0 ? dataset.xCells.slice([0, 0], [trainMoves, 9]) : tf.zeros([0, 9], 'int32');
  const xPosTrain = trainMoves > 0 ? dataset.xPos.slice([0, 0], [trainMoves, 9]) : tf.zeros([0, 9], 'int32');
  const yOneHotTrain = trainMoves > 0 ? dataset.yOneHot.slice([0, 0], [trainMoves, 9]) : tf.zeros([0, 9], 'float32');
  
  const xCellsVal = trainMoves < totalMoves ? dataset.xCells.slice([trainMoves, 0], [totalMoves - trainMoves, 9]) : tf.zeros([0, 9], 'int32');
  const xPosVal = trainMoves < totalMoves ? dataset.xPos.slice([trainMoves, 0], [totalMoves - trainMoves, 9]) : tf.zeros([0, 9], 'int32');
  const yVal = trainMoves < totalMoves ? dataset.yOneHot.slice([trainMoves, 0], [totalMoves - trainMoves, 9]) : tf.zeros([0, 9], 'float32');
  
  // Освобождаем исходные тензоры (они уже разделены)
  dataset.xCells.dispose();
  dataset.xPos.dispose();
  dataset.yOneHot.dispose();
  
  const splitTensors = { 
    xCells: xCellsTrain, 
    xPos: xPosTrain, 
    yOneHot: yOneHotTrain, 
    xValCells: xCellsVal, 
    xValPos: xPosVal, 
    yVal 
  };
  
  return splitTensors;
}
