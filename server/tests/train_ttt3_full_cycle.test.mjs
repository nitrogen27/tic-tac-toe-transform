// Тесты полного цикла обучения TTT3 Transformer
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import tfpkg from '../src/tf.mjs';
const tf = tfpkg;
import { buildPVTransformerSeq } from '../src/model_pv_transformer_seq.mjs';
import { teacherBatches } from '../src/ttt3_minimax.mjs';
import { TRAIN, TRANSFORMER_CFG } from '../src/config.mjs';
import { maskLegalMoves, encodePlanes, emptyBoard } from '../src/game_ttt3.mjs';

describe('TTT3 Transformer Full Training Cycle', () => {
  let model;
  
  beforeAll(() => {
    console.log('[Test] Initializing TensorFlow.js...');
    if (tf.util && tf.util.setSeed) {
      tf.util.setSeed(42);
    }
  });
  
  afterAll(async () => {
    if (model) {
      model.dispose();
    }
    // Очистка памяти
    await tf.engine().startScope();
    await tf.engine().endScope();
  });
  
  it('should build PV Transformer model', () => {
    console.log('[Test] Building model...');
    model = buildPVTransformerSeq({
      dModel: TRANSFORMER_CFG.dModel,
      numLayers: TRANSFORMER_CFG.numLayers,
      heads: TRANSFORMER_CFG.heads,
      dropout: TRANSFORMER_CFG.dropout
    });
    
    expect(model).toBeDefined();
    expect(model.inputs.length).toBe(2); // input и pos_idx
    expect(model.outputs.length).toBe(2); // policy и value
    
    console.log('[Test] ✓ Model built successfully');
    console.log('[Test] Model summary:', {
      inputs: model.inputs.map(i => i.shape),
      outputs: model.outputs.map(o => o.shape)
    });
  });
  
  it('should generate training data from teacher', async () => {
    console.log('[Test] Generating training data...');
    
    const batchSize = 32;
    const batchGen = teacherBatches({ batchSize });
    const batchResult = batchGen.next();
    
    expect(batchResult.done).toBe(false);
    expect(batchResult.value).toBeDefined();
    
    const batch = batchResult.value;
    expect(batch).toHaveProperty('x');
    expect(batch).toHaveProperty('yPolicy');
    expect(batch).toHaveProperty('yValue');
    expect(batch).toHaveProperty('count');
    expect(batch.count).toBeGreaterThan(0);
    
    console.log('[Test] ✓ Generated', batch.count, 'training samples');
  });
  
  it('should perform forward pass', async () => {
    console.log('[Test] Testing forward pass...');
    
    if (!model) {
      model = buildPVTransformerSeq(TRANSFORMER_CFG);
    }
    
    const batchSize = 4;
    const seqLen = 9;
    const inDim = 3;
    
    // Создаем тестовые данные
    const board = emptyBoard();
    const encoded = encodePlanes(board, 1); // Float32Array(27) - плоский массив
    
    // Расширяем до батча: [B, 9, 3]
    // encodePlanes возвращает Float32Array(27), где 27 = 9*3
    const xFlat = [];
    const posBatch = [];
    for (let i = 0; i < batchSize; i++) {
      xFlat.push(...Array.from(encoded)); // Добавляем все 27 элементов
      posBatch.push([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }
    
    const x = tf.tensor3d(xFlat, [batchSize, seqLen, inDim]);
    const pos = tf.tensor2d(posBatch, [batchSize, seqLen], 'int32');
    
    // Forward pass
    const [policyLogits, value] = model.apply([x, pos]);
    
    expect(policyLogits).toBeDefined();
    expect(value).toBeDefined();
    // Проверяем форму - policy может быть [B, 9] или [B, 9, 9] в зависимости от реализации
    expect(policyLogits.shape.length).toBeGreaterThanOrEqual(2);
    expect(policyLogits.shape[0]).toBe(batchSize);
    expect(value.shape).toEqual([batchSize, 1]);
    
    // Проверяем, что значения разумные
    const policyData = await policyLogits.data();
    const valueData = await value.data();
    
    expect(Array.from(policyData).every(v => isFinite(v))).toBe(true);
    expect(Array.from(valueData).every(v => isFinite(v) && v >= -1 && v <= 1)).toBe(true);
    
    x.dispose();
    pos.dispose();
    policyLogits.dispose();
    value.dispose();
    
    console.log('[Test] ✓ Forward pass successful');
  });
  
  it('should train for one epoch', async () => {
    console.log('[Test] Training for one epoch...');
    
    if (!model) {
      model = buildPVTransformerSeq(TRANSFORMER_CFG);
    }
    
    // Компилируем модель
    const optimizer = tf.train.adam(TRAIN.lr);
    model.compile({
      optimizer: optimizer,
      loss: ['categoricalCrossentropy', 'meanSquaredError'],
      lossWeights: [1.0, TRAIN.weightValue || 0.5]
    });
    
    // Генерируем небольшой датасет
    const batchSize = Math.min(TRAIN.batchSize || 32, 16); // Меньший размер для теста
    const nBatches = 5; // Небольшое количество батчей для теста
    
    let totalLoss = 0;
    let iterations = 0;
    
    // Используем teacherBatches как генератор
    const batchGen = teacherBatches({ batchSize });
    
    for (let b = 0; b < nBatches; b++) {
      const batchResult = batchGen.next();
      if (batchResult.done) break;
      
      const batch = batchResult.value;
      if (!batch || !batch.x || batch.x.length === 0) break;
      
      // Подготавливаем данные из формата teacherBatches
      const batchCount = batch.count || Math.floor(batch.x.length / 27); // 9*3 = 27
      if (batchCount === 0) break;
      
      // Преобразуем batch.x (плоский Float32Array) в тензор
      // batch.x содержит batchCount * 27 элементов (9*3)
      const xFlat = Array.from(batch.x); // Преобразуем Float32Array в обычный массив
      const x = tf.tensor3d(xFlat, [batchCount, 9, 3]);
      
      const pos = tf.tensor2d(
        Array.from({ length: batchCount }, () => 
          Array.from({ length: 9 }, (_, i) => i)
        ),
        [batchCount, 9], 'int32'
      );
      
      const yPolicy = tf.tensor2d(
        Array.from({ length: batchCount }, (_, i) => 
          Array.from(batch.yPolicy.slice(i * 9, (i + 1) * 9))
        ),
        [batchCount, 9], 'float32'
      );
      
      const yValue = tf.tensor2d(
        Array.from({ length: batchCount }, (_, i) => [batch.yValue[i]]),
        [batchCount, 1], 'float32'
      );
      
      // Обучение
      const history = await model.fit(
        [x, pos],
        [yPolicy, yValue],
        {
          batchSize: batchCount,
          epochs: 1,
          verbose: 0
        }
      );
      
      const loss = history.history.loss[0];
      totalLoss += loss;
      iterations++;
      
      // Очистка
      x.dispose();
      pos.dispose();
      yPolicy.dispose();
      yValue.dispose();
    }
    
    const avgLoss = totalLoss / iterations;
    expect(Number.isFinite(avgLoss)).toBe(true);
    expect(avgLoss).toBeGreaterThan(0);
    
    console.log('[Test] ✓ Training completed. Average loss:', avgLoss.toFixed(4));
    console.log('[Test] Processed', iterations, 'batches');
  }, 60000); // 60 секунд таймаут
  
  it('should make predictions after training', async () => {
    console.log('[Test] Testing predictions...');
    
    if (!model) {
      model = buildPVTransformerSeq(TRANSFORMER_CFG);
    }
    
    const board = emptyBoard();
    const encoded = encodePlanes(board, 1); // Float32Array(27)
    
    // Преобразуем Float32Array в обычный массив
    const xFlat = Array.from(encoded);
    const x = tf.tensor3d(xFlat, [1, 9, 3]);
    const pos = tf.tensor2d([[0, 1, 2, 3, 4, 5, 6, 7, 8]], [1, 9], 'int32');
    
    const [policyLogits, value] = model.predict([x, pos]);
    
    const policyData = await policyLogits.data();
    const valueData = await value.data();
    
    // Проверяем, что policy - это логиты (не вероятности, поэтому сумма не обязана быть 1)
    // Модель возвращает логиты, softmax применяется снаружи с маскированием
    const policyArray = Array.from(policyData);
    expect(policyArray.length).toBe(9);
    expect(policyArray.every(v => isFinite(v))).toBe(true);
    
    // Проверяем value
    expect(valueData[0]).toBeGreaterThanOrEqual(-1);
    expect(valueData[0]).toBeLessThanOrEqual(1);
    
    x.dispose();
    pos.dispose();
    policyLogits.dispose();
    value.dispose();
    
    console.log('[Test] ✓ Predictions working correctly');
    console.log('[Test] Policy logits:', policyArray.slice(0, 3).map(v => v.toFixed(4)).join(', '), '...');
    console.log('[Test] Value:', valueData[0].toFixed(4));
  });
});

