// Обучение PV Transformer для крестики-нолики 3×3
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

// Оценка качества модели на валидационном наборе
async function evaluateModel(model, validationSet) {
  let correctMoves = 0;
  let valueError = 0;
  let totalPositions = 0;
  
  for (const { board, player } of validationSet) {
    const planes = encodePlanes(board, player);
    const mask = maskLegalMoves(board);
    const { value: teacherValue, policy: teacherPolicy } = getTeacherValueAndPolicy(board, player);
    
    // Инференс модели
    const posIndices = tf.tensor2d([Array.from({ length: 9 }, (_, i) => i)], [1, 9], 'int32');
    const x = tf.tensor3d([Array.from(planes)], [1, 9, 3]);
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

// Основная функция обучения
async function train() {
  console.log('[Train] Starting training...');
  console.log('[Train] Config:', { ...TRAIN, ...TRANSFORMER_CFG });
  
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
  
  // Создаем валидационный набор
  console.log('[Train] Creating validation set...');
  const validationSet = createValidationSet();
  console.log(`[Train] Validation set size: ${validationSet.length}`);
  
  // Генерируем батчи для обучения
  const batchGen = teacherBatches({ batchSize: TRAIN.batchSize });
  
  let step = 0;
  let bestAccuracy = 0;
  let bestMae = Infinity;
  const saveDir = path.join(__dirname, '..', 'saved', 'ttt3_transformer');
  
  // Создаем директорию для сохранения
  await fs.mkdir(saveDir, { recursive: true });
  
  console.log('[Train] Starting training loop...');
  
  for (const batch of batchGen) {
    // Преобразуем batch в тензоры
    const batchSize = batch.count;
    
    // batch.x это Float32Array с batchSize * 27 элементами (27 = 9*3)
    // Преобразуем в плоский массив и создаем tensor3d напрямую
    const xFlat = Array.from(batch.x); // Преобразуем Float32Array в обычный массив
    const x = tf.tensor3d(xFlat, [batchSize, 9, 3]);
    
    const posIndices = tf.tensor2d(
      Array.from({ length: batchSize }, () => 
        Array.from({ length: 9 }, (_, i) => i)
      ),
      [batchSize, 9],
      'int32'
    );
    
    const yPolicy = tf.tensor2d(
      Array.from({ length: batchSize }, (_, i) => 
        Array.from(batch.yPolicy.slice(i * 9, (i + 1) * 9))
      ),
      [batchSize, 9]
    );
    
    const yValue = tf.tensor2d(
      Array.from(batch.yValue),
      [batchSize, 1]
    );
    
    // Обучение шага
    const history = await model.fit(
      [x, posIndices],
      [yPolicy, yValue],
      {
        epochs: 1,
        batchSize: TRAIN.batchSize,
        verbose: 0
      }
    );
    
    const loss = history.history.loss[0];
    
    // Очистка памяти
    x.dispose();
    posIndices.dispose();
    yPolicy.dispose();
    yValue.dispose();
    
    step++;
    
    // Оценка каждые N шагов
    if (step % 10 === 0) {
      console.log(`[Train] Step ${step}, loss: ${loss.toFixed(4)}`);
      
      const { accuracy, mae } = await evaluateModel(model, validationSet);
      console.log(`[Train] Validation - Accuracy: ${(accuracy * 100).toFixed(2)}%, MAE: ${mae.toFixed(4)}`);
      
      // Ранний стоп: accuracy >= 99.9% и MAE <= 1e-3
      if (accuracy >= 0.999 && mae <= 1e-3) {
        console.log('[Train] Early stopping: model reached perfection!');
        await model.save(`file://${saveDir}`);
        console.log(`[Train] Model saved to ${saveDir}`);
        break;
      }
      
      // Сохраняем лучший чекпоинт
      if (accuracy > bestAccuracy || (Math.abs(accuracy - bestAccuracy) < 1e-6 && mae < bestMae)) {
        bestAccuracy = accuracy;
        bestMae = mae;
        await model.save(`file://${saveDir}`);
        console.log(`[Train] Best checkpoint saved (acc: ${(accuracy * 100).toFixed(2)}%, mae: ${mae.toFixed(4)})`);
      }
    }
  }
  
  console.log('[Train] Training completed!');
}

// Запуск
train().catch(e => {
  console.error('[Train] Error:', e);
  process.exit(1);
});
