// Проверка загрузки GPU и потребления энергии
import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import { execSync } from 'child_process';

console.log('=== GPU Power and Utilization Check ===\n');

// Проверяем backend TensorFlow
const backend = tf.getBackend();
console.log(`TensorFlow Backend: ${backend}`);

// Проверяем информацию о GPU через nvidia-smi
try {
  console.log('\n=== NVIDIA-SMI Output ===');
  const nvidiaSmi = execSync('nvidia-smi --query-gpu=name,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader', { encoding: 'utf-8' });
  console.log('GPU Name | Power Draw | Power Limit | GPU Util | Memory Util | Memory Used | Memory Total | Temperature');
  console.log(nvidiaSmi);
  
  // Парсим значения
  const lines = nvidiaSmi.trim().split('\n');
  lines.forEach(line => {
    const parts = line.split(', ').map(p => p.trim());
    if (parts.length >= 8) {
      const [name, powerDraw, powerLimit, gpuUtil, memUtil, memUsed, memTotal, temp] = parts;
      console.log(`\n📊 GPU Statistics:`);
      console.log(`   Name: ${name}`);
      console.log(`   Power Draw: ${powerDraw} / ${powerLimit} (${(parseFloat(powerDraw) / parseFloat(powerLimit) * 100).toFixed(1)}%)`);
      console.log(`   GPU Utilization: ${gpuUtil}`);
      console.log(`   Memory Utilization: ${memUtil}`);
      console.log(`   Memory: ${memUsed} / ${memTotal}`);
      console.log(`   Temperature: ${temp}`);
      
      // Предупреждение если мощность низкая
      const powerPercent = (parseFloat(powerDraw) / parseFloat(powerLimit)) * 100;
      if (powerPercent < 50) {
        console.log(`\n⚠️  WARNING: GPU power usage is only ${powerPercent.toFixed(1)}% of limit!`);
        console.log(`   GPU may not be fully utilized.`);
      }
    }
  });
} catch (e) {
  console.error('Error running nvidia-smi:', e.message);
  console.log('Make sure nvidia-smi is available and GPU is accessible');
}

// Проверяем, действительно ли операции выполняются на GPU
console.log('\n=== TensorFlow GPU Operations Test ===');
try {
  const testTensor = tf.ones([1000, 1000]);
  const result = tf.matMul(testTensor, testTensor);
  const data = await result.data();
  console.log('✓ GPU operations working');
  console.log(`   Test tensor shape: [1000, 1000]`);
  console.log(`   Result sum: ${Array.from(data).reduce((a, b) => a + b, 0).toFixed(2)}`);
  testTensor.dispose();
  result.dispose();
} catch (e) {
  console.error('✗ Error testing GPU operations:', e.message);
}

// Проверяем настройки TensorFlow для GPU
console.log('\n=== TensorFlow GPU Configuration ===');
try {
  const gpuInfo = tf.engine().backend;
  console.log(`Backend: ${gpuInfo}`);
  
  // Проверяем переменные окружения
  console.log('\nEnvironment variables:');
  console.log(`   TF_FORCE_GPU_ALLOW_GROWTH: ${process.env.TF_FORCE_GPU_ALLOW_GROWTH || 'not set'}`);
  console.log(`   TF_ENABLE_ONEDNN_OPTS: ${process.env.TF_ENABLE_ONEDNN_OPTS || 'not set'}`);
  console.log(`   CUDA_VISIBLE_DEVICES: ${process.env.CUDA_VISIBLE_DEVICES || 'not set'}`);
  
  // Проверяем, есть ли доступ к GPU через tfjs-node-gpu
  if (backend === 'tensorflow') {
    console.log('\n✓ TensorFlow.js is using GPU backend');
    console.log('   To maximize GPU utilization:');
    console.log('   1. Use larger batch sizes (512-1024)');
    console.log('   2. Ensure operations are not synchronous');
    console.log('   3. Use tf.tidy() to manage memory efficiently');
    console.log('   4. Consider increasing model complexity');
  } else {
    console.log('\n⚠️  TensorFlow.js is NOT using GPU backend');
    console.log(`   Current backend: ${backend}`);
    console.log('   Install @tensorflow/tfjs-node-gpu for GPU support');
  }
} catch (e) {
  console.error('Error checking TensorFlow configuration:', e.message);
}

// Проверяем текущие процессы на GPU
console.log('\n=== GPU Processes ===');
try {
  const processes = execSync('nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader', { encoding: 'utf-8' });
  if (processes.trim()) {
    console.log('PID | Process Name | Memory Used');
    console.log(processes);
  } else {
    console.log('No compute processes found on GPU');
  }
} catch (e) {
  console.error('Error checking GPU processes:', e.message);
}

console.log('\n=== Recommendations ===');
console.log('To maximize GPU power usage:');
console.log('1. Increase batch size in training (currently 512 for main, 64 for incremental)');
console.log('2. Use larger models (increase dModel, numLayers)');
console.log('3. Ensure all tensor operations are on GPU (use tf.tidy())');
console.log('4. Avoid CPU-GPU data transfers');
console.log('5. Use mixed precision training if available');

