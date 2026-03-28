// Обертка для обучения TTT3 Transformer через WebSocket API
import tfpkg from './tf.mjs';
const tf = tfpkg;
import { buildPVTransformerSeq, maskLogits } from './model_pv_transformer_seq.mjs';
import { teacherBatches, getTeacherValueAndPolicy } from './ttt3_minimax.mjs';
import { TRAIN, TRANSFORMER_CFG, SEED } from './config.mjs';
import { maskLegalMoves, encodePlanes, winner, legalMoves, applyMove, emptyBoard, cloneBoard } from './game_ttt3.mjs';
import { SYMMETRY_MAPS, transformPlanes, transformPolicy } from './ttt3_symmetry.mjs';
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

// ===== Кастомная loss: softmax cross-entropy из логитов =====
// Модель выдаёт RAW LOGITS, а categoricalCrossentropy ожидает probabilities!
// Использование categoricalCrossentropy с логитами → clip(logits, eps, 1-eps)
// → все логиты > 1 маппятся в ~1 → loss ≈ 0 → нет градиента → модель не учится.
// Правильный подход: logSoftmax → cross entropy
function policyLossFromLogits(yTrue, yPred) {
  // yTrue: [B, 9] target probability distribution (from minimax)
  // yPred: [B, 9] raw logits from model
  const logProbs = tf.logSoftmax(yPred, -1); // numerically stable log(softmax(x))
  return tf.neg(tf.sum(tf.mul(yTrue, logProbs), -1)).mean();
}

// Fisher-Yates shuffle
function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function augmentSampleBySymmetry(planes, policy, value) {
  return SYMMETRY_MAPS.map(({ map }) => ({
    planes: transformPlanes(planes, map),
    policy: transformPolicy(policy, map),
    value,
  }));
}

function getEffectiveBatchSize(requestedBatchSize, totalSamples) {
  let maxUsefulBatch = 256;

  if (totalSamples <= 8192) {
    maxUsefulBatch = 64;
  } else if (totalSamples <= 32768) {
    maxUsefulBatch = 128;
  }

  return Math.max(32, Math.min(requestedBatchSize, maxUsefulBatch, totalSamples));
}

function buildLrPhases(baseLR, epochs) {
  if (epochs <= 8) {
    return [
      { fraction: 0.8, lr: baseLR },
      { fraction: 0.2, lr: baseLR / 4 },
    ];
  }

  if (epochs <= 20) {
    return [
      { fraction: 0.6, lr: baseLR },
      { fraction: 0.3, lr: baseLR / 4 },
      { fraction: 0.1, lr: baseLR / 10 },
    ];
  }

  return [
    { fraction: 0.4, lr: baseLR },
    { fraction: 0.3, lr: baseLR / 4 },
    { fraction: 0.3, lr: baseLR / 20 },
  ];
}

// Создание валидационного набора — стратифицированная выборка из ВСЕХ позиций
function createValidationSet() {
  const allPositions = [];

  // Собираем ВСЕ уникальные позиции через teacherBatches
  for (const batch of teacherBatches({ batchSize: 10000 })) {
    for (let i = 0; i < batch.count; i++) {
      const planes = batch.x.slice(i * 27, (i + 1) * 27);
      const policy = batch.yPolicy.slice(i * 9, (i + 1) * 9);
      const value = batch.yValue[i];
      allPositions.push(...augmentSampleBySymmetry(planes, policy, value));
    }
  }

  console.log(`[TrainTTT3] Total positions for validation pool: ${allPositions.length}`);

  // Перемешиваем и берём 500 позиций (или 10%, что меньше)
  shuffleArray(allPositions);
  const valSize = Math.min(500, Math.floor(allPositions.length * 0.1));
  return allPositions.slice(0, valSize);
}

// Оценка качества модели на валидационном наборе
async function evaluateModel(model, validationSet) {
  let correctMoves = 0;
  let valueError = 0;
  let totalPositions = 0;

  // Батчевый инференс для скорости
  const N = validationSet.length;
  const planesArr = new Float32Array(N * 27);
  const posArr = new Int32Array(N * 9);

  for (let i = 0; i < N; i++) {
    planesArr.set(validationSet[i].planes, i * 27);
    for (let j = 0; j < 9; j++) {
      posArr[i * 9 + j] = j;
    }
  }

  const x = tf.tensor3d(Array.from(planesArr), [N, 9, 3]);
  const pos = tf.tensor2d(Array.from(posArr), [N, 9], 'int32');

  // Оборачиваем predict в tidy для очистки промежуточных тензоров MHA
  const { logitsData, valueData } = tf.tidy(() => {
    const [logits, valueTensor] = model.predict([x, pos]);
    return {
      logitsData: logits.dataSync(),
      valueData: valueTensor.dataSync()
    };
  });

  x.dispose();
  pos.dispose();

  for (let i = 0; i < N; i++) {
    const teacherPolicy = validationSet[i].policy;
    const teacherValue = validationSet[i].value;

    // Получаем policy модели для этой позиции
    const modelLogits = logitsData.slice(i * 9, (i + 1) * 9);

    // Маскируем нелегальные ходы (где teacherPolicy > 0 — легальные оптимальные)
    // Но нужна маска легальных ходов — восстановим из planes
    const planes = validationSet[i].planes;
    const mask = new Float32Array(9);
    for (let j = 0; j < 9; j++) {
      // planes[j*3 + 2] = 1.0 если клетка пустая
      mask[j] = planes[j * 3 + 2];
    }

    // Argmax модели среди легальных ходов
    let bestModelMove = -1;
    let bestModelLogit = -Infinity;
    for (let j = 0; j < 9; j++) {
      if (mask[j] > 0.5 && modelLogits[j] > bestModelLogit) {
        bestModelLogit = modelLogits[j];
        bestModelMove = j;
      }
    }

    // Проверяем, выбрала ли модель оптимальный ход (teacher даёт prob > 0 для оптимальных)
    if (bestModelMove >= 0 && teacherPolicy[bestModelMove] > 0.01) {
      correctMoves++;
    }

    // Ошибка value
    valueError += Math.abs(valueData[i] - teacherValue);
    totalPositions++;
  }

  const accuracy = totalPositions > 0 ? correctMoves / totalPositions : 0;
  const mae = totalPositions > 0 ? valueError / totalPositions : 0;

  return { accuracy, mae };
}

// Основная функция обучения с прогрессом через callback
export async function trainTTT3WithProgress(progressCb, {
  epochs = TRAIN.epochs,
  batchSize = TRAIN.batchSize,
  earlyStop = true
} = {}) {
  try {
    // Валидируем epochs: максимум 50
    if (epochs > 50) {
      console.warn(`[TrainTTT3] WARNING: epochs=${epochs} exceeds maximum of 50, using 50 instead`);
      epochs = 50;
    }
    if (epochs < 1) {
      console.warn(`[TrainTTT3] WARNING: epochs=${epochs} is less than 1, using 1 instead`);
      epochs = 1;
    }

    console.log('[TrainTTT3] Starting training...');
    console.log('[TrainTTT3] Config:', { epochs, batchSize, ...TRANSFORMER_CFG });

    // Проверяем GPU
    const { getGpuInfo } = await import('./tf.mjs');
    const gpuInfo = getGpuInfo();
    const backend = tf.getBackend();
    const isGPU = backend === 'tensorflow' && gpuInfo.available;

    console.log('[TrainTTT3] TensorFlow backend:', backend);
    console.log('[TrainTTT3] GPU acceleration:', isGPU ? 'ENABLED ✓' : 'DISABLED ✗');

    if (!isGPU) {
      progressCb?.({ type: 'train.status', payload: { message: 'CPU режим (GPU недоступен)' } });
    } else {
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

    // Multi-phase LR schedule:
    // Phase 1 (40% эпох): LR = baseLR → быстрая конвергенция
    // Phase 2 (30% эпох): LR = baseLR/4 → точная настройка
    // Phase 3 (30% эпох): LR = baseLR/20 → финальная полировка
    const baseLR = TRAIN.lr;
    const lrPhases = buildLrPhases(baseLR, epochs);

    function getLRForEpoch(epoch) {
      let accumulated = 0;
      for (const phase of lrPhases) {
        accumulated += Math.round(phase.fraction * epochs);
        if (epoch < accumulated) return phase.lr;
      }
      return lrPhases[lrPhases.length - 1].lr;
    }

    // Начальная компиляция
    function compileWithLR(lr) {
      model.compile({
        optimizer: tf.train.adam(lr),
        loss: [policyLossFromLogits, 'meanSquaredError'],
        lossWeights: [1.0, TRAIN.weightValue]
      });
    }

    let currentLR = baseLR;
    compileWithLR(currentLR);
    console.log(`[TrainTTT3] LR schedule: ${lrPhases.map(p => `${Math.round(p.fraction*100)}%@${p.lr}`).join(' → ')}`);

    progressCb?.({ type: 'train.status', payload: { message: 'Создание валидационного набора...' } });

    // Создаем валидационный набор (500 позиций)
    const validationSet = createValidationSet();
    console.log(`[TrainTTT3] Validation set size: ${validationSet.length}`);

    // Генерируем ВСЕ обучающие данные из minimax
    progressCb?.({ type: 'train.status', payload: { message: 'Генерация датасета из всех позиций...' } });
    const batchGen = teacherBatches({ batchSize: 100000 }); // один большой батч

    // Собираем все данные в один набор
    const allX = [];
    const allYPolicy = [];
    const allYValue = [];
    let totalSamples = 0;

    for (const batch of batchGen) {
      for (let i = 0; i < batch.count; i++) {
        const planes = batch.x.slice(i * 27, (i + 1) * 27);
        const policy = batch.yPolicy.slice(i * 9, (i + 1) * 9);
        const value = batch.yValue[i];
        const augmented = augmentSampleBySymmetry(planes, policy, value);

        for (const sample of augmented) {
          allX.push(...sample.planes);
          allYPolicy.push(...sample.policy);
          allYValue.push(sample.value);
          totalSamples++;
        }
      }
    }

    console.log(`[TrainTTT3] Total training samples: ${totalSamples}`);
    const effectiveBatchSize = getEffectiveBatchSize(batchSize, totalSamples);
    console.log(`[TrainTTT3] Effective batch size: ${effectiveBatchSize} (requested: ${batchSize})`);

    // Создаём тензоры
    const xTensor = tf.tensor3d(new Float32Array(allX), [totalSamples, 9, 3]);
    const posTensor = tf.tensor2d(
      Array.from({ length: totalSamples }, () => [0, 1, 2, 3, 4, 5, 6, 7, 8]).flat(),
      [totalSamples, 9],
      'int32'
    );
    const yPolicyTensor = tf.tensor2d(new Float32Array(allYPolicy), [totalSamples, 9]);
    const yValueTensor = tf.tensor2d(new Float32Array(allYValue), [totalSamples, 1]);

    const saveDir = path.join(__dirname, '..', 'saved', 'ttt3_transformer');
    await fs.mkdir(saveDir, { recursive: true });

    progressCb?.({
      type: 'train.status',
      payload: {
        message: `Начало обучения (${totalSamples} позиций, ${epochs} эпох, batch ${effectiveBatchSize})...`
      }
    });

    let bestAccuracy = 0;
    let bestMae = Infinity;

    // Обучение через model.fit — Keras правильно разобьёт на батчи
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Обновляем LR при переходе между фазами
      const epochLR = getLRForEpoch(epoch);
      if (Math.abs(epochLR - currentLR) > 1e-8) {
        currentLR = epochLR;
        compileWithLR(currentLR);
        console.log(`[TrainTTT3] LR changed to ${currentLR} at epoch ${epoch + 1}`);
      }

      progressCb?.({ type: 'train.status', payload: { message: `Эпоха ${epoch + 1}/${epochs} (LR: ${currentLR.toExponential(1)})...` } });

      const history = await model.fit(
        [xTensor, posTensor],
        [yPolicyTensor, yValueTensor],
        {
          epochs: 1,
          batchSize: effectiveBatchSize,
          shuffle: true,
          verbose: 0
        }
      );

      const loss = history.history.loss[0];

      // Оценка на валидационном наборе
      progressCb?.({ type: 'train.status', payload: { message: `Оценка модели (эпоха ${epoch + 1})...` } });
      const { accuracy, mae } = await evaluateModel(model, validationSet);

      // Отправляем прогресс
      progressCb?.({
        type: 'train.progress',
        payload: {
          epoch: epoch + 1,
          epochs,
          loss: Number(loss).toFixed(4),
          acc: (accuracy * 100).toFixed(2),
          val_loss: mae.toFixed(4),
          val_acc: (accuracy * 100).toFixed(2),
          percent: Math.round(((epoch + 1) / epochs) * 100),
          accuracy: (accuracy * 100).toFixed(2),
          mae: mae.toFixed(4)
        }
      });

      console.log(`[TrainTTT3] Epoch ${epoch + 1}/${epochs} - Loss: ${Number(loss).toFixed(4)}, Accuracy: ${(accuracy * 100).toFixed(2)}%, MAE: ${mae.toFixed(4)}`);

      // Ранний стоп: accuracy >= 99.9% и MAE <= 1e-3
      if (earlyStop && accuracy >= 0.999 && mae <= 1e-3) {
        console.log('[TrainTTT3] Early stopping: model reached perfection!');
        await model.save(`file://${saveDir}`);
        progressCb?.({ type: 'train.status', payload: { message: 'Модель достигла идеальности! Сохранение...' } });

        const { reloadTTT3Model } = await import('../service.mjs');
        reloadTTT3Model();

        progressCb?.({ type: 'train.done', payload: { saved: true, earlyStop: true, accuracy, mae } });

        // Cleanup
        xTensor.dispose();
        posTensor.dispose();
        yPolicyTensor.dispose();
        yValueTensor.dispose();
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

    // Cleanup тензоров
    xTensor.dispose();
    posTensor.dispose();
    yPolicyTensor.dispose();
    yValueTensor.dispose();

    // Сохраняем финальную модель
    await model.save(`file://${saveDir}`);
    progressCb?.({ type: 'train.status', payload: { message: 'Обучение завершено! Сохранение модели...' } });

    const { reloadTTT3Model } = await import('../service.mjs');
    reloadTTT3Model();

    progressCb?.({ type: 'train.done', payload: { saved: true, accuracy: bestAccuracy, mae: bestMae } });
    console.log('[TrainTTT3] Training completed!');
  } catch (e) {
    console.error('[TrainTTT3] Error during training:', e);
    progressCb?.({ type: 'error', error: String(e) });
    throw e;
  }
}
