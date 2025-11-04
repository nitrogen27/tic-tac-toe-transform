// Обертка для обучения TTT3 Transformer через WebSocket API
import tfpkg from './tf.mjs';
const tf = tfpkg;
import { buildPVTransformerSeq, maskLogits } from './model_pv_transformer_seq.mjs';
import { teacherBatches, getTeacherValueAndPolicy } from './ttt3_minimax.mjs';
import { TRAIN, TRANSFORMER_CFG, SEED } from './config.mjs';
import { maskLegalMoves, encodePlanes, winner, legalMoves, applyMove, emptyBoard, cloneBoard } from './game_ttt3.mjs';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Установка seed для воспроизводимости
if (tf.util && tf.util.setSeed) {
  tf.util.setSeed(SEED);
} else {
  console.warn('[Train] tf.util.setSeed not available, reproducibility may be limited');
}

// Создание валидационного набора позиций
function createValidationSet() {
  const positions = [];
  const visited = new Set();
  
  function generatePositions(board, player, depth = 0) {
    const key = Array.from(board).join(',');
    if (visited.has(key) || depth > 9) return;
    visited.add(key);
    
    const w = winner(board);
    if (w !== null || depth >= 5) { // Берем позиции до глубины 5
      positions.push({ board: cloneBoard(board), player });
    }
    
    if (w === null) {
      const moves = legalMoves(board);
      for (const move of moves.slice(0, 3)) { // Ограничиваем для скорости
        const newBoard = applyMove(board, move, player);
        generatePositions(newBoard, -player, depth + 1);
      }
    }
  }
  
  generatePositions(emptyBoard(), 1);
  return positions.slice(0, 100); // Берем первые 100 позиций
}

// Оценка качества модели на валидационном наборе (все операции на GPU)
async function evaluateModel(model, validationSet, progressCb) {
  // Все операции predict выполняются на GPU автоматически
  let correctMoves = 0;
  let valueError = 0;
  let totalPositions = 0;
  
  for (const { board, player } of validationSet) {
    const planes = encodePlanes(board, player);
    const mask = maskLegalMoves(board);
    const { value: teacherValue, policy: teacherPolicy } = getTeacherValueAndPolicy(board, player);
    
    // Инференс модели
    const posIndices = tf.tensor2d([Array.from({ length: 9 }, (_, i) => i)], [1, 9], 'int32');
    // planes это Float32Array(27), преобразуем в плоский массив
    const xFlat = Array.from(planes);
    const x = tf.tensor3d(xFlat, [1, 9, 3]);
    const [logits, value] = model.predict([x, posIndices]);
    const masked = maskLogits(logits, tf.tensor2d([Array.from(mask)], [1, 9]));
    const probs = tf.softmax(masked);
    
    const policyArray = Array.from(await probs.data());
    const valueArray = await value.data();
    
    // Проверяем совпадение оптимальных ходов
    const teacherOptimal = teacherPolicy.findIndex(p => p > 0.01);
    const modelOptimal = policyArray.indexOf(Math.max(...policyArray));
    
    if (teacherPolicy[modelOptimal] > 0.01) {
      correctMoves++;
    }
    
    // Ошибка value
    valueError += Math.abs(valueArray[0] - teacherValue);
    totalPositions++;
    
    x.dispose();
    posIndices.dispose();
    logits.dispose();
    value.dispose();
    masked.dispose();
    probs.dispose();
  }
  
  const accuracy = correctMoves / totalPositions;
  const mae = valueError / totalPositions;
  
  return { accuracy, mae };
}

// Основная функция обучения с прогрессом через callback
export async function trainTTT3WithProgress(progressCb, { 
  epochs = TRAIN.epochs, 
  batchSize = TRAIN.batchSize,
  earlyStop = true 
} = {}) {
  try {
    // Убеждаемся, что epochs не больше 2
    if (epochs > 2) {
      console.warn(`[TrainTTT3] WARNING: epochs=${epochs} is too high, using 2 instead`);
      epochs = 2;
    }
    console.log('[TrainTTT3] Starting training...');
    console.log('[TrainTTT3] Config:', { epochs, batchSize, ...TRANSFORMER_CFG });
    console.log('[TrainTTT3] TRAIN.epochs from config:', TRAIN.epochs);
    
    // Проверяем GPU и явно проверяем backend
    const { getGpuInfo } = await import('./tf.mjs');
    const gpuInfo = getGpuInfo();
    const backend = tf.getBackend();
    const isGPU = backend === 'tensorflow' && gpuInfo.available;
    
    console.log('[TrainTTT3] GPU Info:', gpuInfo);
    console.log('[TrainTTT3] TensorFlow backend:', backend);
    console.log('[TrainTTT3] GPU acceleration:', isGPU ? 'ENABLED ✓' : 'DISABLED ✗');
    
    if (!isGPU) {
      const warning = 'WARNING: GPU not available! Training will be slow on CPU.';
      console.warn(`[TrainTTT3] ${warning}`);
      console.warn(`[TrainTTT3] Backend: ${backend}, GPU available: ${gpuInfo.available}`);
      progressCb?.({ type: 'train.status', payload: { message: warning } });
    } else {
      console.log('[TrainTTT3] GPU acceleration enabled ✓');
      console.log('[TrainTTT3] All tensor operations will run on GPU');
      progressCb?.({ type: 'train.status', payload: { message: 'GPU ускорение активно ✓' } });
    }
    
    // Отправляем старт
    progressCb?.({ type: 'train.start', payload: { epochs, batchSize, modelType: 'ttt3_transformer', gpu: gpuInfo.available } });
    progressCb?.({ type: 'train.status', payload: { message: 'Инициализация модели...' } });
    
    // Создаем модель
    const model = buildPVTransformerSeq({
      dModel: TRANSFORMER_CFG.dModel,
      numLayers: TRANSFORMER_CFG.numLayers,
      heads: TRANSFORMER_CFG.heads,
      dropout: TRANSFORMER_CFG.dropout
    });
    
    // Компилируем модель
    const optimizer = tf.train.adam(TRAIN.lr);
    model.compile({
      optimizer,
      loss: ['categoricalCrossentropy', 'meanSquaredError'],
      lossWeights: [1.0, TRAIN.weightValue]
    });
    
    progressCb?.({ type: 'train.status', payload: { message: 'Создание валидационного набора...' } });
    
    // Создаем валидационный набор
    const validationSet = createValidationSet();
    console.log(`[TrainTTT3] Validation set size: ${validationSet.length}`);
    
    // Генерируем батчи для обучения
    progressCb?.({ type: 'train.status', payload: { message: 'Генерация датасета из всех позиций...' } });
    const batchGen = teacherBatches({ batchSize });
    
    let step = 0;
    let bestAccuracy = 0;
    let bestMae = Infinity;
    const saveDir = path.join(__dirname, '..', 'saved', 'ttt3_transformer');
    
    // Создаем директорию для сохранения
    await fs.mkdir(saveDir, { recursive: true });
    
    progressCb?.({ type: 'train.status', payload: { message: 'Начало обучения...' } });
    
    let totalBatches = 0;
    const batches = [];
    for (const batch of batchGen) {
      batches.push(batch);
      totalBatches++;
    }
    
    console.log(`[TrainTTT3] Total batches: ${totalBatches}`);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      progressCb?.({ type: 'train.status', payload: { message: `Эпоха ${epoch + 1}/${epochs}...` } });
      
      let epochLoss = 0;
      let batchCount = 0;
      
      for (const batch of batches) {
        // Преобразуем batch в тензоры
        const batchSize_actual = batch.count;
        // batch.x это Float32Array с batchSize_actual * 27 элементами (27 = 9*3)
        // Преобразуем в плоский массив и создаем tensor3d напрямую
        const xFlat = Array.from(batch.x); // Преобразуем Float32Array в обычный массив
        const x = tf.tensor3d(xFlat, [batchSize_actual, 9, 3]);
        
        const posIndices = tf.tensor2d(
          Array.from({ length: batchSize_actual }, () => 
            Array.from({ length: 9 }, (_, i) => i)
          ),
          [batchSize_actual, 9],
          'int32'
        );
        
        const yPolicy = tf.tensor2d(
          Array.from({ length: batchSize_actual }, (_, i) => 
            Array.from(batch.yPolicy.slice(i * 9, (i + 1) * 9))
          ),
          [batchSize_actual, 9]
        );
        
        const yValue = tf.tensor2d(
          Array.from(batch.yValue),
          [batchSize_actual, 1]
        );
        
        // Обучение шага (все операции выполняются на GPU автоматически через tfjs-node-gpu)
        // TensorFlow.js автоматически размещает тензоры и операции на GPU если доступен
        const history = await model.fit(
          [x, posIndices],
          [yPolicy, yValue],
          {
            epochs: 1,
            batchSize: batchSize_actual,
            verbose: 0
          }
        );
        
        // Проверяем, что операции действительно на GPU (для логирования)
        if (step % 50 === 0) {
          const currentBackend = tf.getBackend();
          const isGpuBackend = currentBackend === 'tensorflow';
          if (isGpuBackend) {
            console.log(`[TrainTTT3] Step ${step}: ✓ Operations running on GPU (backend: ${currentBackend})`);
          } else {
            console.warn(`[TrainTTT3] Step ${step}: ⚠ Operations running on ${currentBackend} (not GPU!)`);
          }
        }
        
        const loss = history.history.loss[0];
        epochLoss += loss;
        batchCount++;
        
        // Очистка памяти
        x.dispose();
        posIndices.dispose();
        yPolicy.dispose();
        yValue.dispose();
        
        step++;
      }
      
      const avgLoss = epochLoss / batchCount;
      
      // Оценка каждую эпоху
      progressCb?.({ type: 'train.status', payload: { message: `Оценка модели (эпоха ${epoch + 1})...` } });
      const { accuracy, mae } = await evaluateModel(model, validationSet, progressCb);
      
      // Отправляем прогресс
      progressCb?.({ 
        type: 'train.progress',
        payload: {
          epoch: epoch + 1,
          epochs,
          loss: avgLoss.toFixed(4),
          acc: (accuracy * 100).toFixed(2),
          val_loss: mae.toFixed(4),
          val_acc: (accuracy * 100).toFixed(2),
          percent: Math.round(((epoch + 1) / epochs) * 100),
          accuracy: (accuracy * 100).toFixed(2),
          mae: mae.toFixed(4)
        }
      });
      
      console.log(`[TrainTTT3] Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}, Accuracy: ${(accuracy * 100).toFixed(2)}%, MAE: ${mae.toFixed(4)}`);
      
      // Ранний стоп: accuracy >= 99.9% и MAE <= 1e-3
      if (earlyStop && accuracy >= 0.999 && mae <= 1e-3) {
        console.log('[TrainTTT3] Early stopping: model reached perfection!');
        await model.save(`file://${saveDir}`);
        progressCb?.({ type: 'train.status', payload: { message: 'Модель достигла идеальности! Сохранение...' } });
        progressCb?.({ type: 'train.done', payload: { saved: true, earlyStop: true, accuracy, mae } });
        return;
      }
      
      // Сохраняем лучший чекпоинт
      if (accuracy > bestAccuracy || (Math.abs(accuracy - bestAccuracy) < 1e-6 && mae < bestMae)) {
        bestAccuracy = accuracy;
        bestMae = mae;
        await model.save(`file://${saveDir}`);
        console.log(`[TrainTTT3] Best checkpoint saved (acc: ${(accuracy * 100).toFixed(2)}%, mae: ${mae.toFixed(4)})`);
      }
    }
    
    // Сохраняем финальную модель
    await model.save(`file://${saveDir}`);
    progressCb?.({ type: 'train.status', payload: { message: 'Обучение завершено! Сохранение модели...' } });
    progressCb?.({ type: 'train.done', payload: { saved: true, accuracy: bestAccuracy, mae: bestMae } });
    
    console.log('[TrainTTT3] Training completed!');
  } catch (e) {
    console.error('[TrainTTT3] Error during training:', e);
    progressCb?.({ type: 'error', error: String(e) });
    throw e;
  }
}
