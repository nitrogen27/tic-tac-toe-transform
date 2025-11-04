// Кроссплатформенный скрипт для остановки всех процессов и контейнеров
import { execSync } from 'child_process';
import { platform } from 'os';

const isWindows = platform() === 'win32';

function exec(command, ignoreErrors = true) {
  try {
    if (isWindows) {
      execSync(command, { stdio: 'inherit', shell: true });
    } else {
      execSync(command, { stdio: 'inherit', shell: '/bin/bash' });
    }
  } catch (e) {
    if (!ignoreErrors) {
      console.error(`Error executing: ${command}`, e.message);
    }
  }
}

console.log('[Stop] Stopping all containers and processes...');

// Останавливаем Docker контейнеры
console.log('[Stop] Stopping Docker containers...');
if (isWindows) {
  exec('docker stop tic-tac-toe-server tic-tac-toe-client 2>nul', true);
  exec('docker-compose -f docker-compose.gpu.yml stop 2>nul', true);
  exec('docker-compose -f docker-compose.yml stop 2>nul', true);
} else {
  exec('docker stop tic-tac-toe-server tic-tac-toe-client 2>/dev/null', true);
  exec('docker-compose -f docker-compose.gpu.yml stop 2>/dev/null', true);
  exec('docker-compose -f docker-compose.yml stop 2>/dev/null', true);
}

// Примечание: Локальные процессы Node.js (запущенные через npm start)
// должны останавливаться вручную через Ctrl+C в терминале, где они запущены.
// Docker контейнеры останавливаются выше.
console.log('[Stop] Note: If you started with "npm start", stop it with Ctrl+C in that terminal.');

console.log('[Stop] All stopped ✓');
