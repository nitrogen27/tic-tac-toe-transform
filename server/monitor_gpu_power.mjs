// Мониторинг мощности GPU во время обучения
import { execSync } from 'child_process';

console.log('=== GPU Power Monitor ===\n');
console.log('Monitoring GPU power during training...');
console.log('Press Ctrl+C to stop\n');

let maxPower = 0;
let maxUtil = 0;
let maxMemUtil = 0;
let samples = 0;

setInterval(() => {
  try {
    const output = execSync('nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader', { encoding: 'utf-8' });
    const parts = output.trim().split(', ').map(p => p.trim());
    
    if (parts.length >= 7) {
      const [powerDraw, powerLimit, gpuUtil, memUtil, memUsed, memTotal, temp] = parts;
      const powerNum = parseFloat(powerDraw);
      const utilNum = parseFloat(gpuUtil);
      const memUtilNum = parseFloat(memUtil);
      
      maxPower = Math.max(maxPower, powerNum);
      maxUtil = Math.max(maxUtil, utilNum);
      maxMemUtil = Math.max(maxMemUtil, memUtilNum);
      samples++;
      
      const timestamp = new Date().toLocaleTimeString();
      console.log(`[${timestamp}] Power: ${powerDraw}${powerLimit !== '[N/A]' ? `/${powerLimit}` : ''} | GPU: ${gpuUtil}% | Memory: ${memUtil}% | Mem: ${memUsed}/${memTotal} | Temp: ${temp}`);
      
      if (samples % 10 === 0) {
        console.log(`\n📊 Max so far: Power: ${maxPower.toFixed(2)}W | GPU Util: ${maxUtil}% | Memory Util: ${maxMemUtil}%`);
        if (powerLimit !== '[N/A]') {
          const powerLimitNum = parseFloat(powerLimit);
          const powerPercent = (maxPower / powerLimitNum) * 100;
          console.log(`   Power usage: ${powerPercent.toFixed(1)}% of limit\n`);
        }
      }
    }
  } catch (e) {
    console.error('Error reading GPU stats:', e.message);
  }
}, 1000); // Обновляем каждую секунду

