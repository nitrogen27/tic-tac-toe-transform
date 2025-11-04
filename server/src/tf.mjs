// Пытаемся загрузить GPU версию TensorFlow.js, если не получается - fallback на CPU
let tf;
let backendType = 'cpu';
let backend = 'cpu';

try {
  console.log('[TFJS] Attempting to load GPU version...');
  tf = await import('@tensorflow/tfjs-node-gpu');
  
  // Ждём готовности backend
  await tf.ready?.();
  
  // Проверяем backend
  backend = tf.getBackend?.() || 'cpu';
  backendType = backend === 'tensorflow' ? 'gpu' : 'cpu';
  
  if (backendType === 'gpu') {
    console.log('[TFJS] ✓ GPU backend loaded successfully');
    console.log('[TFJS] Using tfjs-node-gpu backend (CUDA support)');
  } else {
    console.warn('[TFJS] GPU package loaded but backend is not GPU, falling back to CPU...');
    throw new Error('GPU backend not available');
  }
} catch (e) {
  console.warn('[TFJS] GPU version failed to load:', e.message);
  console.log('[TFJS] Falling back to CPU version...');
  
  try {
    tf = await import('@tensorflow/tfjs-node');
    await tf.ready?.();
    backend = tf.getBackend?.() || 'cpu';
    backendType = 'cpu';
    console.log('[TFJS] ✓ CPU backend loaded successfully');
    console.log('[TFJS] Using tfjs-node backend (CPU only)');
  } catch (cpuError) {
    console.error('[TFJS] WARNING: Both GPU and CPU versions failed to load!');
    console.error('[TFJS] GPU error:', e.message);
    console.error('[TFJS] CPU error:', cpuError.message);
    console.error('[TFJS] This usually means native bindings are missing.');
    console.error('[TFJS] Attempting to use CPU backend anyway...');
    
    // Попробуем использовать базовый tfjs без нативных модулей
    try {
      const tfjsCore = await import('@tensorflow/tfjs');
      tf = tfjsCore;
      backend = 'cpu';
      backendType = 'cpu';
      console.log('[TFJS] ✓ Using pure JavaScript TensorFlow.js (slower, but functional)');
      console.warn('[TFJS] ⚠️  Performance will be significantly slower without native bindings');
      console.warn('[TFJS] ⚠️  Consider installing Visual C++ Redistributable or rebuilding native modules');
    } catch (fallbackError) {
      console.error('[TFJS] FATAL: Even fallback failed:', fallbackError.message);
      throw new Error('Failed to load TensorFlow.js: ' + cpuError.message);
    }
  }
}

console.log('[TFJS] Backend:', backend, `(${backendType})`);

// Определение архитектуры процессора
const arch = process.arch;
const platform = process.platform;
console.log(`[TFJS] Platform: ${platform}, Architecture: ${arch}`);

// Экспортируем информацию о GPU для клиента
// Используем функцию, так как переменные могут быть не готовы из-за top-level await
function getGpuInfo() {
  return {
    backend: backendType,
    available: backendType === 'gpu',
    platform,
    arch,
  };
}

// Экспорт константы после инициализации
const gpuInfo = {
  get backend() { return backendType; },
  get available() { return backendType === 'gpu'; },
  get platform() { return platform; },
  get arch() { return arch; },
};

export default tf;
export { backendType, gpuInfo, getGpuInfo };
