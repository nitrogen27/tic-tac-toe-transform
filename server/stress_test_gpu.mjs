// Стресс-тест GPU для проверки загрузки и потребления мощности
// Выполняет умножение больших матриц на обратные матрицы

import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import { execSync } from 'child_process';

console.log('=== GPU Stress Test ===\n');

// Проверяем backend
const backend = tf.getBackend();
const isGPU = backend === 'tensorflow';
console.log(`Backend: ${backend} ${isGPU ? '(GPU)' : '(CPU)'}\n`);

if (!isGPU) {
  console.error('ERROR: GPU backend not available!');
  process.exit(1);
}

// Функция для получения мощности GPU
function getGpuPower() {
  try {
    const output = execSync('nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits', { encoding: 'utf-8' });
    const parts = output.trim().split(', ').map(p => p.trim());
    return {
      draw: parseFloat(parts[0]) || 0,
      limit: parseFloat(parts[1]) || 0
    };
  } catch (e) {
    return { draw: 0, limit: 0 };
  }
}

// Функция для получения загрузки GPU
function getGpuUtilization() {
  try {
    const output = execSync('nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader', { encoding: 'utf-8' });
    const parts = output.trim().split(', ').map(p => p.trim());
    return {
      gpu: parseInt(parts[0]) || 0,
      memory: parseInt(parts[1]) || 0
    };
  } catch (e) {
    return { gpu: 0, memory: 0 };
  }
}

// Начальное состояние
console.log('Initial GPU state:');
let initialPower = getGpuPower();
let initialUtil = getGpuUtilization();
// Ждем немного для стабилизации
await new Promise(resolve => setTimeout(resolve, 500));
initialPower = getGpuPower();
initialUtil = getGpuUtilization();
console.log(`  Power: ${initialPower.draw.toFixed(2)} W / ${initialPower.limit > 0 ? initialPower.limit.toFixed(2) : 'N/A'} W`);
console.log(`  GPU Utilization: ${initialUtil.gpu}%`);
console.log(`  Memory Utilization: ${initialUtil.memory}%\n`);

// Параметры теста
const MATRIX_SIZE = 4096; // Размер матрицы (4096x4096) - увеличен для большей нагрузки
const ITERATIONS = 50; // Количество итераций
const WARMUP_ITERATIONS = 5; // Прогрев

console.log(`Test configuration:`);
console.log(`  Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}`);
console.log(`  Warmup iterations: ${WARMUP_ITERATIONS}`);
console.log(`  Test iterations: ${ITERATIONS}`);
console.log(`  Expected power draw: > 50 W\n`);

// Создаем большие матрицы
console.log('Creating matrices...');
const matrixA = tf.randomNormal([MATRIX_SIZE, MATRIX_SIZE]);
const matrixB = tf.randomNormal([MATRIX_SIZE, MATRIX_SIZE]);

console.log(`  Matrix A: ${MATRIX_SIZE}x${MATRIX_SIZE} (${MATRIX_SIZE * MATRIX_SIZE * 4 / 1024 / 1024} MB)`);
console.log(`  Matrix B: ${MATRIX_SIZE}x${MATRIX_SIZE} (${MATRIX_SIZE * MATRIX_SIZE * 4 / 1024 / 1024} MB)\n`);

// Прогрев
console.log('Warming up GPU...');
for (let i = 0; i < WARMUP_ITERATIONS; i++) {
  const result = tf.matMul(matrixA, matrixB);
  await result.data();
  result.dispose();
  if ((i + 1) % 5 === 0) {
    process.stdout.write(`  ${i + 1}/${WARMUP_ITERATIONS}\r`);
  }
}
console.log(`  Warmup complete\n`);

// Основной тест
console.log('Starting stress test...\n');

let maxPower = initialPower.draw;
let maxUtilGpu = initialUtil.gpu;
let maxUtilMem = initialUtil.memory;
let totalPower = 0;
let powerSamples = 0;
const powerReadings = [];

const startTime = Date.now();

// Запускаем мониторинг мощности в фоне
// Используем более частый опрос для лучшего сбора данных
const monitorInterval = setInterval(() => {
  const power = getGpuPower();
  const util = getGpuUtilization();
  
  if (power && power.draw > 0 && !isNaN(power.draw) && isFinite(power.draw)) {
    powerReadings.push(power.draw);
    totalPower += power.draw;
    powerSamples++;
    if (power.draw > maxPower) {
      maxPower = power.draw;
    }
    if (util.gpu > maxUtilGpu) {
      maxUtilGpu = util.gpu;
    }
    if (util.memory > maxUtilMem) {
      maxUtilMem = util.memory;
    }
  }
}, 100); // Обновляем каждые 100мс для лучшего сбора данных

// Выполняем интенсивные вычисления
for (let i = 0; i < ITERATIONS; i++) {
  // Создаем более сложные операции для максимальной нагрузки GPU
  // A @ B (матричное умножение)
  const temp1 = tf.matMul(matrixA, matrixB);
  
  // (A @ B) @ A^T (транспонированная матрица A)
  const temp2 = tf.matMul(temp1, matrixA, false, true);
  
  // ((A @ B) @ A^T) @ B^T (транспонированная матрица B)
  const temp3 = tf.matMul(temp2, matrixB, false, true);
  
  // Добавляем еще одну операцию для максимальной нагрузки
  const temp4 = tf.matMul(temp3, matrixA);
  const result = tf.matMul(temp4, matrixB);
  
  // Принудительно выполняем вычисления
  await result.data();
  
  // Очистка
  temp1.dispose();
  temp2.dispose();
  temp3.dispose();
  temp4.dispose();
  result.dispose();
  
  // Прогресс - обновляем статистику из мониторинга
  if ((i + 1) % 5 === 0) {
    const currentPower = getGpuPower();
    const currentUtil = getGpuUtilization();
    const elapsed = (Date.now() - startTime) / 1000;
    const avgPower = powerSamples > 0 ? totalPower / powerSamples : currentPower.draw;
    
    console.log(`  Iteration ${i + 1}/${ITERATIONS} | Power: ${currentPower.draw.toFixed(2)} W | GPU: ${currentUtil.gpu}% | Mem: ${currentUtil.memory}% | Avg: ${avgPower.toFixed(2)} W | Max: ${maxPower.toFixed(2)} W | Samples: ${powerSamples} | Time: ${elapsed.toFixed(1)}s`);
  }
}

// Продолжаем мониторинг еще немного после завершения вычислений
await new Promise(resolve => setTimeout(resolve, 500));

// Останавливаем мониторинг
clearInterval(monitorInterval);

// Финальные измерения - ждем немного после остановки мониторинга
await new Promise(resolve => setTimeout(resolve, 500));
const finalPower = getGpuPower();
const finalUtil = getGpuUtilization();

const totalTime = (Date.now() - startTime) / 1000;
const avgPower = powerSamples > 0 ? totalPower / powerSamples : 0;
const minPower = powerReadings.length > 0 ? Math.min(...powerReadings) : 0;

// Очистка
matrixA.dispose();
matrixB.dispose();

// Результаты
console.log('\n=== Test Results ===\n');

console.log('Power Consumption:');
console.log(`  Initial: ${initialPower.draw.toFixed(2)} W`);
console.log(`  Maximum: ${maxPower.toFixed(2)} W`);
console.log(`  Average: ${avgPower.toFixed(2)} W (from ${powerSamples} samples)`);
console.log(`  Minimum: ${minPower.toFixed(2)} W`);
console.log(`  Final: ${finalPower.draw.toFixed(2)} W`);
console.log(`  Limit: ${initialPower.limit > 0 ? initialPower.limit.toFixed(2) : 'N/A'} W`);

console.log('\nGPU Utilization:');
console.log(`  Initial GPU: ${initialUtil.gpu}%`);
console.log(`  Maximum GPU: ${maxUtilGpu}%`);
console.log(`  Final GPU: ${finalUtil.gpu}%`);
console.log(`  Maximum Memory: ${maxUtilMem}%`);

console.log('\nPerformance:');
console.log(`  Total time: ${totalTime.toFixed(2)}s`);
console.log(`  Iterations: ${ITERATIONS}`);
console.log(`  Time per iteration: ${(totalTime / ITERATIONS).toFixed(3)}s`);
console.log(`  Matrix operations per second: ${(ITERATIONS / totalTime).toFixed(2)}`);

console.log('\n=== Test Status ===\n');

// Проверяем результаты
if (maxPower >= 50) {
  console.log(`✅ SUCCESS: Maximum power draw (${maxPower.toFixed(2)} W) is >= 50 W`);
} else {
  console.log(`❌ FAILED: Maximum power draw (${maxPower.toFixed(2)} W) is < 50 W`);
  console.log(`   Power limit may be too low or GPU not fully utilized`);
}

if (avgPower >= 50) {
  console.log(`✅ SUCCESS: Average power draw (${avgPower.toFixed(2)} W) is >= 50 W`);
} else {
  console.log(`⚠️  WARNING: Average power draw (${avgPower.toFixed(2)} W) is < 50 W`);
}

if (maxUtilGpu >= 80) {
  console.log(`✅ SUCCESS: GPU utilization (${maxUtilGpu}%) is >= 80%`);
} else {
  console.log(`⚠️  WARNING: GPU utilization (${maxUtilGpu}%) is < 80%`);
}

console.log('\n=== Recommendations ===\n');

if (maxPower < 50) {
  console.log('To increase power consumption:');
  console.log('1. Increase power limit: sudo nvidia-smi -pl 85 (or 95 for max)');
  console.log('2. Increase matrix size (currently 2048x2048)');
  console.log('3. Increase number of iterations');
}

if (maxUtilGpu < 80) {
  console.log('To increase GPU utilization:');
  console.log('1. Increase matrix size');
  console.log('2. Use more complex operations');
  console.log('3. Ensure no CPU bottlenecks');
}

console.log('\n');

