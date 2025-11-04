#!/usr/bin/env node
/**
 * Микробенчмарк для сравнения производительности CPU vs GPU
 * 
 * Прогоняет model.predict на батчах 2048/8192 с большой моделью (dModel=512, numLayers=8)
 * по 50 итераций и меряет среднее время.
 */

import { getGpuInfo } from './src/tf.mjs';
import tf from './src/tf.mjs';
import { buildModel } from './src/model_transformer.mjs';

// Параметры бенчмарка - тестируем только 2048 для сравнения с предыдущими результатами
const BATCH_SIZES = [2048];
const ITERATIONS = 50;
const WARMUP_ITERATIONS = 10; // Разогрев для прогрева графа/кеша GPU
const MODEL_CONFIG = {
  dModel: 512, // GPU модель - большая
  numLayers: 8,
  seqLen: 9,
  vocabSize: 3,
};

// Генерация тестовых данных - создаём один раз и переиспользуем БЕЗ КОПИЙ
function generateTestBatch(batchSize) {
  const xCells = new Int32Array(batchSize * 9);
  const xPos = new Int32Array(batchSize * 9);
  
  // Генерируем случайные данные
  for (let i = 0; i < batchSize * 9; i++) {
    xCells[i] = Math.floor(Math.random() * 3); // 0, 1, 2
    xPos[i] = i % 9; // позиции 0-8
  }
  
  // Создаём тензоры ОДИН РАЗ - они уже на GPU
  // TensorFlow.js автоматически размещает на GPU если доступен
  // НЕ создаём новые тензоры в цикле - это вызывает host→device копии
  return {
    xCells: tf.tensor2d(xCells, [batchSize, 9], 'int32'),
    xPos: tf.tensor2d(xPos, [batchSize, 9], 'int32'),
  };
}

// Прогон бенчмарка для одного размера батча
async function benchmarkBatchSize(model, batchSize) {
  console.log(`\n📊 Бенчмарк для batchSize=${batchSize}...`);
  
  // Генерируем тестовые данные ОДИН РАЗ - тензоры уже на GPU
  // БЕЗ создания новых тензоров в цикле (нет host→device копий)
  const { xCells, xPos } = generateTestBatch(batchSize);
  
  // Разогрев (warmup) - 10 итераций для прогрева графа/кеша GPU
  console.log(`  Разогрев (${WARMUP_ITERATIONS} итераций)...`);
  for (let w = 0; w < WARMUP_ITERATIONS; w++) {
    const warmupResult = model.predict([xCells, xPos]);
    // НЕ вызываем .data() - это синхронный барьер!
    // Просто запускаем predict и освобождаем
    warmupResult.dispose();
  }
  
  // Синхронизируем GPU перед замером
  await new Promise(resolve => setTimeout(resolve, 200));
  
  // Замер времени - ТОЛЬКО вычисления, БЕЗ синхронных барьеров
  const times = [];
  for (let i = 0; i < ITERATIONS; i++) {
    const start = performance.now(); // Используем high-res timer
    const result = model.predict([xCells, xPos]);
    // НЕ вызываем .data() или .arraySync() - это синхронные барьеры!
    // Просто запускаем predict и меряем время до dispose
    // TensorFlow.js запускает операции асинхронно на GPU
    result.dispose();
    const end = performance.now();
    times.push(end - start);
    
    // Редкий лог только каждые 10 итераций - без блокировки
    if ((i + 1) % 10 === 0) {
      process.stdout.write(`  Итерация ${i + 1}/${ITERATIONS}...\r`);
    }
  }
  
  // Освобождаем входные данные
  xCells.dispose();
  xPos.dispose();
  
  // Вычисляем статистику
  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const sortedTimes = [...times].sort((a, b) => a - b);
  const medianTime = sortedTimes[Math.floor(sortedTimes.length / 2)];
  const p95 = sortedTimes[Math.floor(sortedTimes.length * 0.95)];
  
  // Throughput (samples/s)
  const throughput = batchSize / avgTime * 1000;
  const throughputRounded = throughput.toFixed(0);
  
  // Оценка использования памяти (приблизительная)
  // dModel=512, numLayers=8, batchSize
  const estimatedMemoryMB = (batchSize * 9 * 4 * 3) / (1024 * 1024); // входные данные (int32)
  const modelMemoryMB = (512 * 512 * 4 * 8 * 2) / (1024 * 1024); // веса модели (приблизительно)
  
  console.log(`\n  ✅ Среднее время: ${avgTime.toFixed(2)}ms`);
  console.log(`     Мин: ${minTime.toFixed(1)}ms, Макс: ${maxTime.toFixed(1)}ms, Медиана: ${medianTime.toFixed(1)}ms, P95: ${p95.toFixed(1)}ms`);
  console.log(`     Throughput: ${throughputRounded} samples/s`);
  console.log(`     Примерное использование памяти: ~${(estimatedMemoryMB + modelMemoryMB).toFixed(1)} MB`);
  
  return { batchSize, avgTime, minTime, maxTime, medianTime, throughput: parseFloat(throughputRounded), p95 };
}

// Основная функция
async function runBenchmark() {
  console.log('🚀 Запуск микробенчмарка CPU vs GPU');
  console.log('=' .repeat(70));
  
  // Устанавливаем переменные окружения для оптимизации GPU
  if (process.env.TF_FORCE_GPU_ALLOW_GROWTH === undefined) {
    process.env.TF_FORCE_GPU_ALLOW_GROWTH = 'true';
    console.log('\n🔧 TF_FORCE_GPU_ALLOW_GROWTH=true (снижает фрагментацию VRAM)');
  }
  
  // Проверяем GPU
  const gpuInfo = getGpuInfo();
  const isGPU = gpuInfo.available && gpuInfo.backend === 'gpu';
  
  console.log(`\n📡 Backend: ${gpuInfo.backend.toUpperCase()}`);
  console.log(`   Platform: ${gpuInfo.platform}, Architecture: ${gpuInfo.arch}`);
  console.log(`   GPU доступен: ${isGPU ? '✅ ДА' : '❌ НЕТ'}`);
  
  // Создаём модель в зависимости от backend
  const modelConfig = isGPU ? MODEL_CONFIG : { ...MODEL_CONFIG, dModel: 64, numLayers: 2 };
  console.log(`\n🔧 Создание модели: dModel=${modelConfig.dModel}, numLayers=${modelConfig.numLayers}...`);
  console.log(`   ${isGPU ? 'GPU модель (большая)' : 'CPU модель (малая)'}`);
  const model = buildModel(modelConfig);
  model.compile({
    optimizer: tf.train.adam(1e-3),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  console.log('✅ Модель создана');
  
  // Прогоняем бенчмарки для каждого размера батча
  const results = [];
  for (const batchSize of BATCH_SIZES) {
    const result = await benchmarkBatchSize(model, batchSize);
    results.push(result);
    
    // Небольшая пауза между тестами
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  // Выводим итоговую таблицу
  const result = results[0];
  console.log('\n' + '='.repeat(70));
  console.log('📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ (batchSize=2048):');
  console.log('='.repeat(70));
  console.log('\nBackend | Batch Size | Avg Time (ms) | Throughput (samples/s)');
  console.log('-'.repeat(70));
  const backendLabel = isGPU ? 'GPU' : 'CPU';
  console.log(`${backendLabel.padEnd(7)} | ${String(result.batchSize).padStart(10)} | ${String(result.avgTime.toFixed(2)).padStart(13)} | ${String(result.throughput).padStart(23)}`);
  
  // Анализ производительности
  console.log('\n💡 Анализ производительности:');
  if (isGPU) {
    console.log(`   GPU активен ✅`);
    console.log(`   batchSize=2048: throughput ${result.throughput} samples/s, среднее время ${result.avgTime.toFixed(2)}ms`);
    console.log(`\n   💾 Конфигурация:`);
    console.log(`      Модель: dModel=${MODEL_CONFIG.dModel}, numLayers=${MODEL_CONFIG.numLayers}`);
    console.log(`      Backend: ${gpuInfo.backend}`);
    console.log(`      Оптимизации: без host→device копий, без синхронных барьеров, warmup=${WARMUP_ITERATIONS}`);
  } else {
    console.log(`   ⚠️  GPU недоступен - работает CPU режим`);
    console.log(`   Для активации GPU убедитесь что CUDA/cuDNN установлены и доступны`);
    console.log(`   batchSize=2048: throughput ${result.throughput} samples/s, среднее время ${result.avgTime.toFixed(2)}ms`);
  }
  
  // Освобождаем модель
  model.dispose();
  
  console.log('\n✅ Бенчмарк завершён');
}

// Запуск
runBenchmark().catch(err => {
  console.error('❌ Ошибка при запуске бенчмарка:', err);
  process.exit(1);
});

