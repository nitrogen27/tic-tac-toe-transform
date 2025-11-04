// Кроссплатформенный скрипт для перезапуска проекта
import { execSync } from 'child_process';
import { spawn } from 'child_process';
import { platform } from 'os';

const isWindows = platform() === 'win32';

console.log('[Restart] Stopping all...');

// Сначала останавливаем всё
try {
  execSync('node scripts/stop-all.js', { stdio: 'inherit' });
} catch (e) {
  console.warn('[Restart] Warning during stop:', e.message);
}

// Небольшая задержка
console.log('[Restart] Waiting 2 seconds...');
if (isWindows) {
  try {
    execSync('timeout /t 2 /nobreak >nul 2>&1', { stdio: 'ignore' });
  } catch (e) {
    // Игнорируем ошибки
  }
} else {
  try {
    execSync('sleep 2', { stdio: 'ignore' });
  } catch (e) {
    // Игнорируем ошибки
  }
}

console.log('[Restart] Starting project...');

// Запускаем заново
const child = spawn(isWindows ? 'npm.cmd' : 'npm', ['start'], {
  stdio: 'inherit',
  shell: true,
  cwd: process.cwd()
});

child.on('error', (err) => {
  console.error('[Restart] Error starting:', err);
  process.exit(1);
});

child.on('exit', (code) => {
  if (code !== 0 && code !== null) {
    console.error(`[Restart] Process exited with code ${code}`);
    process.exit(code);
  }
});
