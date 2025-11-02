import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import fs from 'fs/promises';
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import { buildModel } from './src/model_transformer.mjs';
import { loadDataset } from './src/dataset.mjs';
import { relativeCells } from './src/tic_tac_toe.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const _MODEL_DIR = path.resolve(__dirname, 'saved');

let model = null;
let building = false;

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
        model = buildModel({ dModel: 64, numLayers: 2 });
      }
    }
    return model;
  } finally { building = false; }
}

export async function trainWithProgress(progressCb, { epochs=5, batchSize=256, nTrain=4000, nVal=1000 } = {}) {
  try {
    // Отправляем старт сразу, чтобы клиент знал что запрос получен
    progressCb?.({ type: 'train.start', payload: { epochs, batchSize, nTrain, nVal } });
    
    console.log('[Train] Ensuring model...');
    progressCb?.({ type: 'train.status', payload: { message: 'Инициализация модели...' } });
    const m = await ensureModel({ forceFresh: true });
    
    console.log('[Train] Loading dataset...');
    progressCb?.({ type: 'train.status', payload: { message: `Генерация датасета (${nTrain + nVal} игр)...` } });
    const { xCells, xPos, yOneHot, xValCells, xValPos, yVal } = await loadDataset(nTrain, nVal);
    
    console.log('[Train] Starting training...');
    progressCb?.({ type: 'train.status', payload: { message: 'Начало обучения...' } });
    
    // Оптимизация для M2: используем shuffle для лучшего обучения и verbose для отладки
    await m.fit([xCells, xPos], yOneHot, {
      epochs, 
      batchSize,
      shuffle: true, // Перемешивание данных улучшает обучение
      validationData: [[xValCells, xValPos], yVal],
      verbose: 0, // Отключаем verbose для скорости (прогресс идёт через callbacks)
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`[Train] Epoch ${epoch+1}/${epochs}, loss: ${logs.loss}, acc: ${logs.acc}`);
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
          console.log('[Train] Training completed');
          progressCb?.({ type:'train.done', payload:{ saved:true } });
        },
      }
    });
    console.log('[Train] Saving model...');
    await m.save(`file://${_MODEL_DIR}`);
    xCells.dispose(); xPos.dispose(); yOneHot.dispose();
    xValCells.dispose(); xValPos.dispose(); yVal.dispose();
    console.log('[Train] Done');
  } catch (e) {
    console.error('[Train] Error during training:', e);
    progressCb?.({ type: 'error', error: String(e) });
    throw e;
  }
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
  
  // Режим модели: проверяем, есть ли обученная модель
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
  
  // Используем обученную модель
  const m = await ensureModel();
  const rel = relativeCells(board, current);
  const xCells = tf.tensor2d([rel], [1, 9], 'int32');
  const pos = tf.tensor2d([Array.from({length:9}, (_,i)=>i)], [1, 9], 'int32');
  const probs = m.predict([xCells, pos]);
  const pa = await probs.data();
  xCells.dispose(); pos.dispose(); probs.dispose();
  const moves = []; for (let i=0;i<9;i++) if (board[i]===0) moves.push(i);
  let best=-1, bestv=-1; for (const mm of moves) if (pa[mm]>bestv){ best=mm; bestv=pa[mm]; }
  console.log('[Predict] Using trained model, move:', best);
  return { move: best, probs: Array.from(pa), isRandom: false, mode: 'model' };
}

export async function clearModel() {
  try {
    console.log('[Clear] Clearing saved model...');
    
    // Освобождаем текущую модель из памяти
    if (model) {
      model.dispose?.();
      model = null;
    }
    
    // Удаляем файлы модели
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
    
    // Пробуем удалить другие файлы весов (на случай если есть несколько shards)
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
    
    console.log(`[Clear] Model cleared. Deleted files: ${deletedFiles.join(', ') || 'none'}`);
    return { success: true, deletedFiles };
  } catch (e) {
    console.error('[Clear] Error clearing model:', e);
    throw e;
  }
}
