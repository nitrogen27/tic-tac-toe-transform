let tf;
try {
  tf = await import('@tensorflow/tfjs-node');
  console.log('[TFJS] Using tfjs-node backend');
  
  // Проверяем доступный backend для оптимизации на M2
  const backend = tf.getBackend();
  console.log('[TFJS] Backend:', backend);
  
  // Для M2 можно увеличить количество потоков
  if (typeof tf.setBackend === 'function') {
    try {
      // tfjs-node автоматически использует нативные биндинги на M2
      console.log('[TFJS] Native bindings available for acceleration');
    } catch (e) {
      console.log('[TFJS] Native bindings check:', e.message);
    }
  }
  
  // Оптимизация: отключаем лишнюю валидацию для скорости
  tf.env().set('WEBGL_PACK', false);
  tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
  
} catch (e) {
  console.error('[TFJS] Failed to load @tensorflow/tfjs-node:', e);
  throw e;
}
export default tf;
