import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import fs from 'fs/promises';
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import { buildModel } from './src/model_transformer.mjs';
import { BOARD_N, TRANSFORMER_CFG } from './src/config.mjs';
import { loadDataset } from './src/dataset.mjs';
import { relativeCells } from './src/tic_tac_toe.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const _MODEL_DIR = path.resolve(__dirname, 'saved');

let model = null;
let building = false;

// Хранилище игровой истории для дообучения
let gameHistory = [];
const MAX_HISTORY_SIZE = 10000; // Максимум сохраненных ходов

// Хранилище полных игр для анализа ошибок
let gameSequences = []; // [{ moves: [{board, move, current}], winner, playerRole }]
const MAX_GAME_SEQUENCES = 100; // Храним последние 100 игр

async function ensureDir(p) { try { await fs.mkdir(p, { recursive: true }); } catch {} }

// Проверяет, существует ли обученная модель
export async function hasTrainedModel() {
  try {
    const modelPath = path.join(_MODEL_DIR, 'model.json');
    await fs.access(modelPath);
    return true;
  } catch {
    return false;
  }
}

export async function ensureModel({ forceFresh = false } = {}) {
  const force = !!forceFresh;
  if (building) { while (building) await new Promise(r => setTimeout(r, 25)); return model; }
  building = true;
  try {
    if (force && model) { model.dispose?.(); model = null; }
    await ensureDir(_MODEL_DIR);
    if (!model) {
      const exists = await fs.stat(path.join(_MODEL_DIR, 'model.json')).then(()=>true).catch(()=>false);
      if (exists) {
        model = await tf.loadLayersModel(`file://${_MODEL_DIR}/model.json`);
        model.compile({ optimizer: tf.train.adam(1e-3), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
      } else {
        const useBig = process.env.USE_GPU_BIG === '1';
        model = buildModel({  dModel: useBig ? 512 : 64, numLayers: useBig ? 8 : 2 , seqLen: (BOARD_N*BOARD_N), vocabSize: 3 });
        console.log(`[Model] Using ${useBig ? 'BIG GPU' : 'small CPU'} model: dModel=${useBig ? 512 : 64}, numLayers=${useBig ? 8 : 2}`);
      }
    }
    return model;
  } finally { building = false; }
}

// Оптимальные параметры для GPU/CPU: адаптивный batchSize
export async function trainWithProgress(progressCb, { epochs=5, batchSize, nTrain=50000, nVal=5000 } = {}) {
  try {
    // Определяем оптимальный batchSize в зависимости от backend
    const { getGpuInfo } = await import('./src/tf.mjs');
    const gpuInfo = getGpuInfo();
    const isGPU = gpuInfo.available && gpuInfo.backend === 'gpu';
    
    // Адаптивный batchSize: GPU - 8192-16384, CPU - 2048-4096
    // Но для маленьких датасетов (nTrain < 10000) используем меньший batch
    if (batchSize === undefined || batchSize === null) {
      if (nTrain < 100) {
        // Очень маленький датасет - используем очень маленький batch чтобы избежать OOM
        batchSize = isGPU ? 32 : 16;
      } else if (nTrain < 1000) {
        // Маленький датасет - используем маленький batch
        batchSize = isGPU ? 128 : 64;
      } else if (nTrain < 10000) {
        // Средний датасет - используем средний batch
        batchSize = isGPU ? 1024 : 512;
      } else {
        batchSize = isGPU ? 16384 : 4096;
      }
    } else {
      // Если batchSize указан явно, используем его (но ограничиваем для маленьких датасетов)
      if (nTrain < 100) {
        batchSize = Math.min(64, batchSize);  // Максимум 64 для очень маленьких датасетов
      } else if (nTrain < 1000) {
        batchSize = isGPU 
          ? Math.max(32, Math.min(256, batchSize))
          : Math.max(16, Math.min(128, batchSize));
      } else if (nTrain < 10000) {
        batchSize = isGPU 
          ? Math.max(512, Math.min(4096, batchSize))
          : Math.max(256, Math.min(2048, batchSize));
      } else {
        batchSize = isGPU 
          ? Math.max(8192, batchSize)
          : Math.max(2048, Math.min(4096, batchSize));
      }
    }
    
    // Отправляем старт сразу, чтобы клиент знал что запрос получен
    progressCb?.({ type: 'train.start', payload: { epochs, batchSize, nTrain, nVal } });
    
    console.log('[Train] Step 1: Ensuring model...');
    progressCb?.({ type: 'train.status', payload: { message: 'Инициализация модели...' } });
    const m = await ensureModel({ forceFresh: true });
    console.log('[Train] Step 2: Model ensured, loading dataset...');
    
    progressCb?.({ type: 'train.status', payload: { message: `Генерация датасета (${nTrain + nVal} игр)...` } });
    // НЕ передаем progressCb в loadDataset - но добавим статус перед и после
    console.log('[Train] Loading dataset...');
    const { xCells, xPos, yOneHot, xValCells, xValPos, yVal } = await loadDataset(nTrain, nVal, null);
    console.log('[Train] Dataset loaded, starting fit...');
    
    progressCb?.({ type: 'train.status', payload: { message: 'Начало обучения...' } });
    
    // Подаём тензоры целиком - Keras сам правильно разобьёт по батчам
    // НЕ указываем stepsPerEpoch/validationSteps - они дают лишний оверхед
    console.log(`[Train] Starting fit with epochs=${epochs}, batchSize=${batchSize}, data shape=${xCells.shape[0]}`);
    await m.fit([xCells, xPos], yOneHot, {
      epochs, 
      batchSize, // Адаптивный batchSize (GPU: 8192-16384, CPU: 2048-4096)
      shuffle: true,
      validationData: [[xValCells, xValPos], yVal],
      verbose: 0, // Без verbose для производительности
      // НЕ указываем stepsPerEpoch - TensorFlow сам определит из размера тензоров
      callbacks: {
        onTrainBegin: () => {
          console.log('[Train] onTrainBegin called');
          progressCb?.({ type: 'train.status', payload: { message: 'Обучение началось...' } });
        },
        onEpochBegin: (epoch) => {
          console.log(`[Train] onEpochBegin called for epoch ${epoch + 1}/${epochs}`);
          progressCb?.({ type: 'train.status', payload: { message: `Эпоха ${epoch + 1}/${epochs}...` } });
        },
        onEpochEnd: (epoch, logs) => {
          console.log(`[Train] onEpochEnd called for epoch ${epoch + 1}/${epochs}, logs:`, logs);
          // Прогресс ТОЛЬКО в onEpochEnd - без промежуточных колбэков
          progressCb?.({ type:'train.progress',
            payload: {
              epoch: epoch+1, epochs,
              loss: Number(logs.loss ?? 0).toFixed(4),
              acc: Number(logs.acc ?? 0).toFixed(4),
              val_loss: Number(logs.val_loss ?? 0).toFixed(4),
              val_acc: Number(logs.val_acc ?? 0).toFixed(4),
              percent: Math.round(((epoch+1)/epochs)*100),
            }});
        },
        onTrainEnd: () => {
          console.log('[Train] onTrainEnd called');
          progressCb?.({ type:'train.done', payload:{ saved:true } });
        },
      }
    });
    await m.save(`file://${_MODEL_DIR}`);
    xCells.dispose(); xPos.dispose(); yOneHot.dispose();
    xValCells.dispose(); xValPos.dispose(); yVal.dispose();
  } catch (e) {
    console.error('[Train] Error during training:', e);
    progressCb?.({ type: 'error', error: String(e) });
    throw e;
  }
}

// Загрузка TTT3 Transformer модели
let ttt3Model = null;
let ttt3ModelLoading = false;

// Экспортируем функцию для принудительной перезагрузки модели
export function reloadTTT3Model() {
  console.log('[PredictTTT3] Reloading model: clearing cache...');
  if (ttt3Model) {
    try {
      ttt3Model.dispose();
    } catch (e) {
      console.warn('[PredictTTT3] Error disposing old model:', e.message);
    }
  }
  ttt3Model = null;
  ttt3ModelLoading = false;
}

async function ensureTTT3Model() {
  if (ttt3ModelLoading) {
    while (ttt3ModelLoading) await new Promise(r => setTimeout(r, 25));
    return ttt3Model;
  }
  
  // Если модель уже загружена, возвращаем её
  if (ttt3Model) {
    console.log('[PredictTTT3] Using cached TTT3 model from memory');
    return ttt3Model;
  }
  
  // Логируем попытку загрузки для отладки
  console.log('[PredictTTT3] Model not in cache, attempting to load from disk...');
  
  ttt3ModelLoading = true;
  try {
    const ttt3ModelPath = path.join(_MODEL_DIR, 'ttt3_transformer', 'model.json');
    const exists = await fs.stat(ttt3ModelPath).then(() => true).catch(() => false);
    
    if (exists) {
      // Загружаем сохраненную модель
      // ВАЖНО: Модель с кастомными слоями может не загрузиться
      // Если загрузка не удалась, нужно переобучить модель
      try {
        ttt3Model = await tf.loadLayersModel(`file://${ttt3ModelPath}`);
        // Компилируем для инференса (не нужен оптимизатор, но нужны входы/выходы)
        console.log('[PredictTTT3] ✓ Loaded TTT3 Transformer model from', ttt3ModelPath);
        console.log('[PredictTTT3] Model inputs:', ttt3Model.inputs.map(i => i.shape));
        console.log('[PredictTTT3] Model outputs:', ttt3Model.outputs.map(o => o.shape));
      } catch (loadError) {
        console.error('[PredictTTT3] ❌ Failed to load model:', loadError.message);
        console.error('[PredictTTT3] Model file exists but cannot be loaded.');
        console.error('[PredictTTT3] This usually means the model was saved with custom layers that cannot be deserialized.');
        console.error('[PredictTTT3] Solution: Delete the model (click "Clear Model") and retrain.');
        console.error('[PredictTTT3] Will use random moves as fallback');
        ttt3Model = null;
      }
    } else {
      console.log('[PredictTTT3] TTT3 Transformer model not found at', ttt3ModelPath);
      console.log('[PredictTTT3] Will use random moves as fallback');
    }
  } catch (e) {
    console.error('[PredictTTT3] Error loading model:', e);
    console.log('[PredictTTT3] Will use random moves as fallback');
    ttt3Model = null;
  } finally {
    ttt3ModelLoading = false;
  }
  
  return ttt3Model; // null если модель не найдена
}

// Предсказание для TTT3 (3x3) с использованием Transformer модели
async function predictTTT3Move({ board, current = 1 }) {
  const { encodePlanes, maskLegalMoves, legalMoves } = await import('./src/game_ttt3.mjs');
  const { maskLogits } = await import('./src/model_pv_transformer_seq.mjs');
  const { safePick } = await import('./src/safety.mjs');
  
  // Проверяем, что доска правильного размера
  if (board.length !== 9) {
    throw new Error(`Invalid board size: expected 9, got ${board.length}`);
  }
  
  // Проверяем, есть ли обученная модель
  const model = await ensureTTT3Model();
  
  if (!model) {
    // Если модели нет - можно играть, используя случайные ходы или minimax
    // Для удобства используем случайные ходы (можно переключить на minimax)
    const moves = legalMoves(new Int8Array(board));
    if (moves.length === 0) {
      return { move: -1, probs: Array(9).fill(0), isRandom: true, mode: 'model', fallback: 'random' };
    }
    
    // Выбираем случайный ход из доступных
    // Модель можно обучить с нуля, нажав "Обучить"
    const randomMove = moves[Math.floor(Math.random() * moves.length)];
    const probs = Array(9).fill(0);
    probs[randomMove] = 1.0;
    
    console.log('[PredictTTT3] No trained model, using RANDOM move. Use "Train" button to train from scratch.');
    return { 
      move: randomMove, 
      probs, 
      value: 0, 
      isRandom: true, 
      mode: 'model', 
      fallback: 'random' 
    };
  }
  
  // Используем обученную модель
  const player = current === 1 ? 1 : -1; // Преобразуем 1/2 в 1/-1
  const planes = encodePlanes(new Int8Array(board), player);
  const mask = maskLegalMoves(new Int8Array(board));
  
  // Создаем тензоры в правильном формате
  const xFlat = Array.from(planes);
  const x = tf.tensor3d(xFlat, [1, 9, 3]);
  const pos = tf.tensor2d([[0, 1, 2, 3, 4, 5, 6, 7, 8]], [1, 9], 'int32');
  
  // Предсказание
  const [logits, valueTensor] = model.predict([x, pos]);
  const maskedLogits = maskLogits(logits, tf.tensor2d([Array.from(mask)], [1, 9]));
  const probs = tf.softmax(maskedLogits);
  
  const policyArray = Array.from(await probs.data());
  const value = (await valueTensor.data())[0];
  
  // Очистка памяти
  x.dispose();
  pos.dispose();
  logits.dispose();
  maskedLogits.dispose();
  probs.dispose();
  valueTensor.dispose();
  
  // Применяем safety-правила
  const moves = legalMoves(new Int8Array(board));
  if (moves.length === 0) {
    return { move: -1, probs: Array(9).fill(0), value, isRandom: false, mode: 'model' };
  }
  
  // Детальное логирование для отладки
  const policyMax = Math.max(...policyArray);
  const policyMaxIdx = policyArray.indexOf(policyMax);
  
  // Проверяем safety-правила ДО вызова safePick для логирования
  const { winningMove, blockingMove } = await import('./src/safety.mjs');
  const winMove = winningMove(new Int8Array(board), player);
  const blockMove = blockingMove(new Int8Array(board), player);
  
  console.log('[PredictTTT3] Model prediction BEFORE safety:', {
    board: Array.from(board),
    player,
    policyMax: policyMax.toFixed(4),
    policyMaxIdx,
    value: value.toFixed(4),
    hasWinningMove: winMove >= 0,
    hasBlockingMove: blockMove >= 0,
    policy: policyArray.map((p, i) => `${i}:${p.toFixed(3)}`).join(' ')
  });
  
  const move = safePick(new Int8Array(board), player, policyArray);
  
  // Проверяем, был ли ход изменен safety-правилами
  const wasChangedBySafety = move !== policyMaxIdx;
  if (wasChangedBySafety) {
    console.log('[PredictTTT3] ⚠️ Move changed by safety rules!', {
      modelPrediction: policyMaxIdx,
      modelPolicy: policyArray[policyMaxIdx].toFixed(4),
      safetyMove: move,
      reason: winMove >= 0 ? 'winning' : (blockMove >= 0 ? 'blocking' : 'unknown')
    });
  } else {
    console.log('[PredictTTT3] ✓ Using model prediction (no safety override)');
  }
  
  console.log('[PredictTTT3] Final move:', move);
  console.log('[PredictTTT3] Model confidence:', {
    maxProb: policyMax.toFixed(4),
    maxProbIdx: policyMaxIdx,
    value: value.toFixed(4),
    move: move,
    safetyOverride: wasChangedBySafety
  });
  
  return { 
    move, 
    probs: policyArray, 
    value, 
    isRandom: false, 
    mode: 'model',
    confidence: policyMax // Отправляем как число, а не строку
  };
}

export async function predictMove({ board, current = 1, mode = 'model' }) {
  // Если выбран режим алгоритма - используем minimax
  if (mode === 'algorithm') {
    const { teacherBestMove, legalMoves } = await import('./src/tic_tac_toe.mjs');
    const moves = legalMoves(board);
    if (moves.length === 0) {
      return { move: -1, probs: Array(9).fill(0), mode: 'algorithm' };
    }
    const bestMove = teacherBestMove(board, current);
    const probs = Array(9).fill(0);
    probs[bestMove] = 1;
    console.log('[Predict] Using algorithm (minimax), move:', bestMove);
    return { move: bestMove, probs, mode: 'algorithm' };
  }
  
  // Для досок 3x3 используем TTT3 Transformer
  if (board.length === 9) {
    try {
      return await predictTTT3Move({ board, current });
    } catch (e) {
      console.error('[Predict] Error in TTT3 prediction, falling back to old model:', e);
      // Fallback к старой логике
    }
  }
  
  // Режим модели для других размеров досок: проверяем, есть ли обученная модель
  const hasModel = await hasTrainedModel();
  
  if (!hasModel) {
    // Если модели нет - возвращаем случайный ход
    const { legalMoves: getLegalMoves } = await import('./src/tic_tac_toe.mjs');
    const moves = getLegalMoves(board);
    if (moves.length === 0) {
      return { move: -1, probs: Array(9).fill(0), isRandom: true, mode: 'model' };
    }
    const randomMove = moves[Math.floor(Math.random() * moves.length)];
    console.log('[Predict] No trained model, using random move:', randomMove);
    return { move: randomMove, probs: Array(9).fill(1/9), isRandom: true, mode: 'model' };
  }
  
  // Используем обученную модель (для больших досок)
  const m = await ensureModel();
  const rel = relativeCells(board, current);
  const L = board.length; const xCells = tf.tensor2d([rel], [1, L], 'int32');
  const pos = tf.tensor2d([Array.from({length:board.length}, (_,i)=>i)], [1, board.length], 'int32');
  const probs = m.predict([xCells, pos]);
  const pa = await probs.data();
  xCells.dispose(); pos.dispose(); probs.dispose();
  const moves = []; for (let i=0;i<board.length;i++) if (board[i]===0) moves.push(i);
  let best=-1, bestv=-1; for (const mm of moves) if (pa[mm]>bestv){ best=mm; bestv=pa[mm]; }
  console.log('[Predict] Using trained model, move:', best);
  return { move: best, probs: Array.from(pa), isRandom: false, mode: 'model' };
}

// Сохраняет ход игры в историю для дообучения
export function saveGameMove({ board, move, current, gameId }) {
  try {
    const rel = relativeCells(board, current);
    const pos = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const onehot = new Array(9).fill(0);
    if (move >= 0 && move < 9) {
      onehot[move] = 1;
    }
    
    gameHistory.push({ X: rel, P: pos, Y: onehot });
    
    // Также сохраняем в последовательность игры, если указан gameId
    if (gameId !== undefined) {
      const gameSeq = gameSequences.find(g => g.id === gameId);
      if (gameSeq) {
        gameSeq.moves.push({ board: [...board], move, current });
      }
    }
    
    // Ограничиваем размер истории
    if (gameHistory.length > MAX_HISTORY_SIZE) {
      gameHistory = gameHistory.slice(-MAX_HISTORY_SIZE);
    }
    
    console.log(`[GameHistory] Saved move, total: ${gameHistory.length}`);
  } catch (e) {
    console.error('[GameHistory] Error saving move:', e);
  }
}

// Начинает новую игру (для отслеживания последовательности)
export function startNewGame({ playerRole = 1 } = {}) {
  const gameId = Date.now() + Math.random();
  gameSequences.push({
    id: gameId,
    moves: [],
    winner: null,
    playerRole, // 1 = модель играет за X, 2 = модель играет за O
  });
  
  // Ограничиваем размер
  if (gameSequences.length > MAX_GAME_SEQUENCES) {
    gameSequences = gameSequences.slice(-MAX_GAME_SEQUENCES);
  }
  
  return gameId;
}

// Завершает игру и анализирует ошибки
// Флаг для отслеживания фонового обучения
let backgroundTraining = false;
let backgroundTrainingPromise = null;
let backgroundTrainingProgressCb = null; // Callback для отправки прогресса

export async function finishGame({ gameId, winner, patternsPerError = 1000, autoTrain = false, progressCb = null, incrementalBatchSize = 256 }) {
  const gameSeq = gameSequences.find(g => g.id === gameId);
  if (!gameSeq) return;
  
  gameSeq.winner = winner;
  
  // Сохраняем количество паттернов до анализа ошибок
  const patternsBefore = gameHistory.length;
  
  // Если модель проиграла, анализируем ошибки
  if ((gameSeq.playerRole === 1 && winner === 2) || (gameSeq.playerRole === 2 && winner === 1)) {
    console.log(`[ErrorDetection] Model lost game ${gameId}, analyzing mistakes...`);
    await analyzeAndGenerateCorrections(gameSeq, patternsPerError);
  }
  
  // Подсчитываем количество новых навыков (новых паттернов)
  const patternsAfter = gameHistory.length;
  const newSkillsCount = patternsAfter - patternsBefore;
  
  // Если включен авто-режим обучения, запускаем фоновое обучение
  if (autoTrain && !backgroundTraining) {
    console.log('[FinishGame] Auto-train enabled, starting background training...');
    startBackgroundTraining(progressCb, newSkillsCount, incrementalBatchSize);
  }
}

// Анализирует ошибки и генерирует обучающие паттерны
async function analyzeAndGenerateCorrections(gameSeq, patternsPerError = 1000) {
  try {
    const { teacherBestMove, legalMoves, getWinner } = await import('./src/tic_tac_toe.mjs');
    const corrections = [];
    
    // Анализируем каждый ход модели
    for (let i = 0; i < gameSeq.moves.length; i++) {
      const move = gameSeq.moves[i];
      
      // Проверяем, это ход модели?
      const isModelMove = (gameSeq.playerRole === 1 && move.current === 1) || 
                         (gameSeq.playerRole === 2 && move.current === 2);
      
      if (!isModelMove) continue;
      
      // Проверяем, был ли это правильный ход (сравниваем с minimax)
      const correctMove = teacherBestMove(move.board, move.current);
      
      if (correctMove !== move.move) {
        // Модель сделала ошибку!
        console.log(`[ErrorDetection] Found mistake at move ${i}: model chose ${move.move}, should be ${correctMove}`);
        
        // Генерируем вариации этого паттерна (количество из настроек)
        const variations = await generateBoardVariations(move.board, move.current, patternsPerError);
        
        for (const variant of variations) {
          const bestMove = teacherBestMove(variant.board, variant.current);
          const rel = relativeCells(variant.board, variant.current);
          const pos = [0, 1, 2, 3, 4, 5, 6, 7, 8];
          const onehot = new Array(9).fill(0);
          if (bestMove >= 0 && bestMove < 9) {
            onehot[bestMove] = 1;
          }
          
          corrections.push({ X: rel, P: pos, Y: onehot });
        }
      }
    }
    
    // Добавляем исправления в историю
    if (corrections.length > 0) {
      console.log(`[ErrorDetection] Generated ${corrections.length} correction patterns`);
      gameHistory.push(...corrections);
      
      // Ограничиваем размер истории
      if (gameHistory.length > MAX_HISTORY_SIZE) {
        gameHistory = gameHistory.slice(-MAX_HISTORY_SIZE);
      }
    }
  } catch (e) {
    console.error('[ErrorDetection] Error analyzing mistakes:', e);
  }
}

// Генерирует вариации доски для обучения на ошибках
async function generateBoardVariations(board, current, count = 1000) {
  const variations = [];
  const { legalMoves, getWinner } = await import('./src/tic_tac_toe.mjs');
  
  // Базовый паттерн
  variations.push({ board: [...board], current });
  
  // Генерируем вариации:
  // 1. Разные комбинации заполнения незанятых клеток (сохраняя структуру)
  const emptyIndices = [];
  for (let i = 0; i < 9; i++) {
    if (board[i] === 0) emptyIndices.push(i);
  }
  
  // Если слишком мало свободных клеток, просто дублируем базовый паттерн
  if (emptyIndices.length <= 2) {
    for (let i = 0; i < count - 1; i++) {
      variations.push({ board: [...board], current });
    }
    return variations;
  }
  
  // Генерируем вариации несколькими способами для разнообразия
  const maxAttempts = count * 3; // Увеличено для генерации 1000 уникальных вариантов
  const seenBoards = new Set();
  seenBoards.add(board.join(''));
  
  console.log(`[GenerateVariations] Generating ${count} variations from board with ${emptyIndices.length} empty cells`);
  
  for (let i = 0; i < maxAttempts && variations.length < count; i++) {
    const variant = [...board];
    
    // Разные стратегии генерации вариаций для разнообразия
    const strategy = i % 5; // Используем 5 стратегий вместо 3
    
    if (strategy === 0) {
      // Стратегия 1: Меняем одну клетку (консервативно)
      if (emptyIndices.length > 0) {
        const idx = emptyIndices[Math.floor(Math.random() * emptyIndices.length)];
        variant[idx] = current === 1 ? 2 : 1;
      }
    } else if (strategy === 1) {
      // Стратегия 2: Меняем 2 клетки
      const numChanges = Math.min(2, emptyIndices.length);
      const indicesToChange = [];
      const available = [...emptyIndices];
      
      for (let j = 0; j < numChanges && available.length > 0; j++) {
        const idx = Math.floor(Math.random() * available.length);
        indicesToChange.push(available.splice(idx, 1)[0]);
      }
      
      for (const idx of indicesToChange) {
        variant[idx] = current === 1 ? 2 : 1;
      }
    } else if (strategy === 2) {
      // Стратегия 3: Меняем 3 клетки
      const numChanges = Math.min(3, emptyIndices.length);
      const indicesToChange = [];
      const available = [...emptyIndices];
      
      for (let j = 0; j < numChanges && available.length > 0; j++) {
        const idx = Math.floor(Math.random() * available.length);
        indicesToChange.push(available.splice(idx, 1)[0]);
      }
      
      for (const idx of indicesToChange) {
        variant[idx] = current === 1 ? 2 : 1;
      }
    } else if (strategy === 3) {
      // Стратегия 4: Меняем 4 клетки
      const numChanges = Math.min(4, emptyIndices.length);
      const indicesToChange = [];
      const available = [...emptyIndices];
      
      for (let j = 0; j < numChanges && available.length > 0; j++) {
        const idx = Math.floor(Math.random() * available.length);
        indicesToChange.push(available.splice(idx, 1)[0]);
      }
      
      for (const idx of indicesToChange) {
        variant[idx] = current === 1 ? 2 : 1;
      }
    } else {
      // Стратегия 5: Меняем случайное количество (1-5)
      const numChanges = Math.min(Math.max(1, Math.floor(Math.random() * 5) + 1), emptyIndices.length);
      const indicesToChange = [];
      const available = [...emptyIndices];
      
      for (let j = 0; j < numChanges && available.length > 0; j++) {
        const idx = Math.floor(Math.random() * available.length);
        indicesToChange.push(available.splice(idx, 1)[0]);
      }
      
      for (const idx of indicesToChange) {
        variant[idx] = current === 1 ? 2 : 1;
      }
    }
    
    // Проверяем уникальность и валидность
    const variantKey = variant.join('');
    if (!seenBoards.has(variantKey)) {
      if (getWinner(variant) === null && legalMoves(variant).length > 0) {
        variations.push({ board: variant, current });
        seenBoards.add(variantKey);
      }
    }
  }
  
  // Дополняем до нужного количества базовым паттерном (если не удалось сгенерировать достаточно уникальных)
  const uniqueCount = new Set(variations.map(v => v.board.join(''))).size;
  console.log(`[GenerateVariations] Generated ${variations.length} variations (${uniqueCount} unique, ${variations.length - uniqueCount} duplicates)`);
  
  while (variations.length < count) {
    variations.push({ board: [...board], current });
  }
  
  return variations.slice(0, count);
}

// Очищает историю игр
export function clearGameHistory() {
  gameHistory = [];
  console.log('[GameHistory] Cleared');
  return { success: true, count: 0 };
}

// Получает статистику истории
export function getGameHistoryStats() {
  return { count: gameHistory.length, maxSize: MAX_HISTORY_SIZE };
}

// Дообучение модели на реальных играх
export async function trainOnGames(progressCb, { epochs = 1, batchSize = 256, focusOnErrors = true } = {}) {
  try {
    if (gameHistory.length < 10) {
      throw new Error(`Недостаточно данных для обучения. Нужно минимум 10 ходов, есть ${gameHistory.length}`);
    }
    
    // Используем указанное количество эпох (оптимизировано: 1 эпоха для быстрого дообучения)
    let actualEpochs = epochs;
    
    progressCb?.({ type: 'train.start', payload: { epochs: actualEpochs, batchSize, nTrain: gameHistory.length, nVal: 0 } });
    
    console.log(`[TrainOnGames] Training on ${gameHistory.length} game moves with ${actualEpochs} epochs...`);
    progressCb?.({ type: 'train.status', payload: { message: `Подготовка данных (${gameHistory.length} ходов, ${actualEpochs} эпох)...` } });
    
    const m = await ensureModel({ forceFresh: false }); // Не пересоздаем модель, продолжаем обучение
    
    // Подготавливаем данные из истории
    const X = gameHistory.map(h => h.X);
    const P = gameHistory.map(h => h.P);
    const Y = gameHistory.map(h => h.Y);
    
    console.log(`[TrainOnGames] Creating tensors from ${X.length} moves...`);
    const xCells = tf.tensor2d(X, [X.length, 9], 'int32');
    const xPos = tf.tensor2d(P, [P.length, 9], 'int32');
    const yOneHot = tf.tensor2d(Y, [Y.length, 9], 'float32');
    
    progressCb?.({ type: 'train.status', payload: { message: 'Начало дообучения (интенсивное обучение на ошибках)...' } });
    
    // Подсчитываем количество батчей для промежуточных обновлений
    const totalSamples = X.length;
    const batchesPerEpoch = Math.ceil(totalSamples / batchSize);
    let currentBatch = 0;
    let lastProgressUpdate = 0;
    
    await m.fit([xCells, xPos], yOneHot, {
      epochs: actualEpochs,
      batchSize,
      shuffle: true,
      verbose: 0,
      callbacks: {
        onBatchEnd: (batch, logs) => {
          // Обновляем прогресс каждые 10% или каждые 5 батчей, в зависимости от того, что меньше
          currentBatch++;
          const progressUpdateInterval = Math.max(1, Math.floor(batchesPerEpoch / 10)); // Обновляем ~10 раз за эпоху
          
          if (currentBatch % progressUpdateInterval === 0 || currentBatch === batchesPerEpoch) {
            const batchProgress = Math.round((currentBatch / batchesPerEpoch) * 100);
            const epochProgress = Math.round(((batchProgress / 100) + lastProgressUpdate) / actualEpochs * 100);
            
            progressCb?.({ 
              type: 'train.progress',
              payload: {
                epoch: lastProgressUpdate + 1, 
                epochs: actualEpochs,
                loss: Number(logs.loss ?? 0).toFixed(4),
                acc: Number(logs.acc ?? 0).toFixed(4),
                val_loss: 0,
                val_acc: 0,
                percent: Math.min(epochProgress, 99), // Не показываем 100% до завершения эпохи
                batchProgress: batchProgress, // Дополнительный прогресс внутри эпохи
                batchesPerEpoch: batchesPerEpoch,
                currentBatch: currentBatch
              }
            });
          }
        },
        onEpochBegin: (epoch) => {
          lastProgressUpdate = epoch;
          currentBatch = 0;
          console.log(`[TrainOnGames] Starting epoch ${epoch+1}/${actualEpochs}...`);
        },
        onEpochEnd: (epoch, logs) => {
          console.log(`[TrainOnGames] Epoch ${epoch+1}/${actualEpochs}, loss: ${logs.loss}, acc: ${logs.acc}`);
          lastProgressUpdate = epoch + 1;
          progressCb?.({ 
            type:'train.progress',
            payload: {
              epoch: epoch+1, 
              epochs: actualEpochs,
              loss: Number(logs.loss ?? 0).toFixed(4),
              acc: Number(logs.acc ?? 0).toFixed(4),
              val_loss: 0,
              val_acc: 0,
              percent: Math.round(((epoch+1)/actualEpochs)*100),
            }
          });
        },
        onTrainEnd: () => {
          console.log('[TrainOnGames] Training completed');
          progressCb?.({ type:'train.done', payload:{ saved:true } });
        },
      }
    });
    
    console.log('[TrainOnGames] Saving model...');
    await m.save(`file://${_MODEL_DIR}`);
    
    // Перезагружаем модель в памяти после сохранения
    reloadTTT3Model();
    console.log('[TrainOnGames] Model reloaded in memory after training');
    
    xCells.dispose(); xPos.dispose(); yOneHot.dispose();
    console.log('[TrainOnGames] Done');
  } catch (e) {
    console.error('[TrainOnGames] Error:', e);
    progressCb?.({ type: 'error', error: String(e) });
    throw e;
  }
}

// Обучение TTT3 Transformer
export async function trainTTT3WithProgress(progressCb, { epochs, batchSize, earlyStop = true, fromScratch = false } = {}) {
  const { trainTTT3WithProgress: trainTTT3 } = await import('./src/train_ttt3_transformer_service.mjs');
  return await trainTTT3(progressCb, { epochs, batchSize, earlyStop, fromScratch });
}

// Фоновое мини-обучение после игры (быстрое, не блокирующее)
async function startBackgroundTraining(progressCb = null, newSkillsCount = 0, incrementalBatchSize = 256) {
  if (backgroundTraining) {
    console.log('[BackgroundTrain] Training already in progress, skipping...');
    return;
  }
  
  // Проверяем, есть ли данные для обучения
  const stats = getGameHistoryStats();
  if (stats.count < 10) {
    console.log(`[BackgroundTrain] Not enough data (${stats.count} moves), skipping...`);
    return;
  }
  
  backgroundTraining = true;
  backgroundTrainingProgressCb = progressCb;
  console.log('[BackgroundTrain] Starting background training...');
  
  // Отправляем старт фонового обучения
  const totalSkills = stats.count;
  progressCb?.({ 
    type: 'background_train.start', 
    payload: { 
      newSkills: newSkillsCount,
      totalSkills: totalSkills,
      epochs: 1, // Одна эпоха для быстрого усвоения нового навыка
      message: `Новые навыки: ${newSkillsCount} из ${totalSkills} (${Math.round(newSkillsCount / totalSkills * 100)}%)`
    } 
  });
  
  // Запускаем в фоне без блокировки
  backgroundTrainingPromise = (async () => {
    try {
      // Мини-обучение: меньше эпох, меньше батч, без early stopping
      await trainOnGames(
        (ev) => {
          // Преобразуем события обычного обучения в события фонового обучения
          if (ev.type === 'train.start') {
            progressCb?.({ 
              type: 'background_train.start', 
              payload: { 
                newSkills: newSkillsCount,
                totalSkills: totalSkills,
                epochs: ev.payload?.epochs || 1,
                newSkillsPercent: Math.round(newSkillsCount / totalSkills * 100),
                message: `Начало обучения: ${newSkillsCount} новых навыков (${Math.round(newSkillsCount / totalSkills * 100)}%)`
              } 
            });
          } else if (ev.type === 'train.progress') {
            // Отправляем прогресс с информацией о новых навыках
            const epoch = ev.payload?.epoch || 0;
            const epochs = ev.payload?.epochs || 1;
            // Используем percent из payload, если есть, иначе вычисляем
            const percent = ev.payload?.percent || Math.round((epoch / epochs) * 100);
            const epochPercent = percent;
            const newSkillsPercent = Math.round(newSkillsCount / totalSkills * 100);
            
            // Добавляем информацию о батчах, если есть
            const batchInfo = ev.payload?.batchProgress ? 
              ` · Батч ${ev.payload.currentBatch}/${ev.payload.batchesPerEpoch} (${ev.payload.batchProgress}%)` : 
              '';
            
            progressCb?.({ 
              type: 'background_train.progress', 
              payload: { 
                epoch,
                epochs,
                epochPercent: percent, // Используем вычисленный percent
                newSkills: newSkillsCount,
                totalSkills: totalSkills,
                newSkillsPercent,
                loss: ev.payload?.loss,
                acc: ev.payload?.acc,
                batchProgress: ev.payload?.batchProgress,
                currentBatch: ev.payload?.currentBatch,
                batchesPerEpoch: ev.payload?.batchesPerEpoch,
                message: `Эпоха ${epoch}/${epochs} (${percent}%)${batchInfo} · Новые навыки: ${newSkillsCount} (${newSkillsPercent}%)`
              } 
            });
          } else if (ev.type === 'train.done') {
              // Перезагружаем модель после фонового обучения
              reloadTTT3Model();
              console.log('[BackgroundTrain] Model reloaded in memory after background training');
              
              progressCb?.({ 
                type: 'background_train.done', 
                payload: { 
                  newSkills: newSkillsCount,
                  totalSkills: totalSkills,
                  newSkillsPercent: Math.round(newSkillsCount / totalSkills * 100),
                  message: `Обучение завершено: ${newSkillsCount} новых навыков усвоено`
                } 
              });
          } else if (ev.type === 'error') {
            progressCb?.({ 
              type: 'background_train.error', 
              error: ev.error,
              payload: { message: `Ошибка обучения: ${ev.error}` }
            });
          }
        },
        {
          epochs: 1, // Одна эпоха для быстрого усвоения нового навыка (фоновое обучение всегда 1 эпоха)
          batchSize: incrementalBatchSize, // Используем переданный batch size из настроек
          focusOnErrors: true
        }
      );
      console.log('[BackgroundTrain] Background training completed');
    } catch (e) {
      console.error('[BackgroundTrain] Error during background training:', e);
      progressCb?.({ 
        type: 'background_train.error', 
        error: String(e),
        payload: { message: `Ошибка: ${e.message}` }
      });
    } finally {
      backgroundTraining = false;
      backgroundTrainingPromise = null;
      backgroundTrainingProgressCb = null;
    }
  })();
}

// Экспортируем функцию для проверки статуса фонового обучения
export function isBackgroundTraining() {
  return backgroundTraining;
}

export async function clearModel() {
  try {
    console.log('[Clear] Clearing all saved models...');
    
    // Освобождаем текущую модель из памяти
    if (model) {
      model.dispose?.();
      model = null;
    }
    
    // Освобождаем TTT3 модель из памяти и сбрасываем флаг загрузки
    if (ttt3Model) {
      ttt3Model.dispose?.();
      ttt3Model = null;
      console.log('[Clear] TTT3 model cleared from memory');
    }
    // Сбрасываем флаг загрузки, чтобы модель могла быть загружена заново после очистки
    ttt3ModelLoading = false;
    console.log('[Clear] TTT3 model cache reset - model will be reloaded from disk if exists');
    
    // Удаляем файлы старой модели
    const modelJsonPath = path.join(_MODEL_DIR, 'model.json');
    const weightsBinPath = path.join(_MODEL_DIR, 'weights.bin');
    
    let deletedFiles = [];
    try {
      await fs.unlink(modelJsonPath);
      deletedFiles.push('model.json');
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
    }
    
    try {
      await fs.unlink(weightsBinPath);
      deletedFiles.push('weights.bin');
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
    }
    
    // Удаляем TTT3 Transformer модель
    const ttt3ModelDir = path.join(_MODEL_DIR, 'ttt3_transformer');
    try {
      const ttt3ModelPath = path.join(ttt3ModelDir, 'model.json');
      const ttt3WeightsPath = path.join(ttt3ModelDir, 'weights.bin');
      
      try {
        await fs.unlink(ttt3ModelPath);
        deletedFiles.push('ttt3_transformer/model.json');
        console.log('[Clear] Deleted TTT3 model.json');
      } catch (e) {
        if (e.code !== 'ENOENT') throw e;
      }
      
      try {
        await fs.unlink(ttt3WeightsPath);
        deletedFiles.push('ttt3_transformer/weights.bin');
        console.log('[Clear] Deleted TTT3 weights.bin');
      } catch (e) {
        if (e.code !== 'ENOENT') throw e;
      }
      
      // Пробуем удалить другие файлы весов TTT3 (на случай если есть несколько shards)
      try {
        const ttt3Files = await fs.readdir(ttt3ModelDir);
        for (const file of ttt3Files) {
          if (file.startsWith('weights') || file.endsWith('.bin')) {
            await fs.unlink(path.join(ttt3ModelDir, file));
            const filePath = `ttt3_transformer/${file}`;
            if (!deletedFiles.includes(filePath)) deletedFiles.push(filePath);
          }
        }
      } catch (e) {
        // Игнорируем ошибки чтения директории (может не существовать)
      }
    } catch (e) {
      // Игнорируем ошибки при удалении TTT3 модели
      console.warn('[Clear] Error clearing TTT3 model directory:', e.message);
    }
    
    // Пробуем удалить другие файлы весов старой модели (на случай если есть несколько shards)
    try {
      const files = await fs.readdir(_MODEL_DIR);
      for (const file of files) {
        if (file.startsWith('weights') || file.endsWith('.bin')) {
          await fs.unlink(path.join(_MODEL_DIR, file));
          if (!deletedFiles.includes(file)) deletedFiles.push(file);
        }
      }
    } catch (e) {
      // Игнорируем ошибки чтения директории
    }
    
    console.log(`[Clear] All models cleared. Deleted files: ${deletedFiles.join(', ') || 'none'}`);
    return { success: true, deletedFiles };
  } catch (e) {
    console.error('[Clear] Error clearing model:', e);
    throw e;
  }
}
