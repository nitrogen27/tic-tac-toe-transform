import { WebSocketServer } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fs from 'fs/promises';
import { trainWithProgress, trainTTT3WithProgress, predictMove, clearModel, saveGameMove, trainOnGames, clearGameHistory, getGameHistoryStats, startNewGame, finishGame } from './service.mjs';
import { getGpuInfo } from './src/tf.mjs';
import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import { BOARD_N } from './src/config.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// WebSocket сервер
const WS_PORT = 8080;

const wss = new WebSocketServer({ port: WS_PORT });
console.log('[WS] Listening on ws://localhost:' + WS_PORT);

// Это обработчик событий WebSocket-сервера, который обрабатывает сообщения от клиентов
// Он обрабатывает три типа запросов:
// 1. 'ping' - отвечает 'pong' для проверки соединения
// 2. 'train' - запускает обучение модели с обновлениями о прогрессе
// 3. 'predict' - вычисляет следующий лучший ход в игре крестики-нолики
wss.on('connection', (ws) => {
  console.log('[WS] Client connected');
  
  // Отправляем информацию о GPU после небольшой задержки, чтобы клиент успел установить обработчики
  const sendGpuInfo = () => {
    try {
      const gpuInfoData = getGpuInfo();
      console.log('[WS] Sending GPU info:', JSON.stringify(gpuInfoData));
      const message = JSON.stringify({ type: 'gpu.info', payload: gpuInfoData });
      if (ws.readyState === ws.OPEN) {
        ws.send(message);
        console.log('[WS] GPU info sent successfully');
      } else {
        console.warn('[WS] WebSocket not ready, GPU info not sent');
      }
    } catch (e) {
      console.error('[WS] Error sending GPU info:', e.message);
    }
  };
  
  // Отправляем информацию о GPU сразу после подключения
  console.log('[WS] About to send GPU info...');
  sendGpuInfo();
  console.log('[WS] sendGpuInfo called');
  
  ws.on('message', async (msg) => {
    try {
      const m = JSON.parse(msg.toString());
      console.log('[WS] Received message type:', m.type);
      if (m.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
      } else if (m.type === 'train') {
        console.log('[WS] Starting training with payload:', m.payload);
        console.log('[WS] Sending train.start immediately...');
        await trainWithProgress((ev)=>{
          console.log('[WS] Sending progress:', ev.type);
          try {
            ws.send(JSON.stringify(ev));
          } catch (sendErr) {
            console.error('[WS] Error sending message:', sendErr);
          }
        }, m.payload || {});
      } else if (m.type === 'train_ttt3') {
        console.log('[WS] Starting TTT3 Transformer training with payload:', m.payload);
        console.log('[WS] Sending train.start immediately...');
        await trainTTT3WithProgress((ev)=>{
          console.log('[WS] Sending progress:', ev.type);
          try {
            ws.send(JSON.stringify(ev));
          } catch (sendErr) {
            console.error('[WS] Error sending message:', sendErr);
          }
        }, m.payload || {});
      } else if (m.type === 'predict') {
        const payload = m.payload || { board: Array(9).fill(0), current: 1 };
        const res = await predictMove({ 
          board: payload.board, 
          current: payload.current, 
          mode: payload.mode || 'model' 
        });
        ws.send(JSON.stringify({ type: 'predict.result', payload: res }));
      } else if (m.type === 'clear_model') {
        console.log('[WS] Clearing model...');
        try {
          const result = await clearModel();
          ws.send(JSON.stringify({ type: 'clear_model.success', payload: result }));
        } catch (e) {
          ws.send(JSON.stringify({ type: 'error', error: String(e) }));
        }
      } else if (m.type === 'save_move') {
        // Сохраняем ход для обучения
        saveGameMove(m.payload || { board: [], move: -1, current: 1, gameId: undefined });
        ws.send(JSON.stringify({ type: 'move.saved', payload: getGameHistoryStats() }));
      } else if (m.type === 'start_game') {
        // Начинаем новую игру для отслеживания последовательности
        const gameId = startNewGame({ playerRole: m.payload?.playerRole || 1 });
        ws.send(JSON.stringify({ type: 'game.started', payload: { gameId } }));
      } else if (m.type === 'finish_game') {
        // Завершаем игру и анализируем ошибки
        const patternsPerError = m.payload?.patternsPerError || 1000;
        await finishGame({ gameId: m.payload?.gameId, winner: m.payload?.winner, patternsPerError });
        ws.send(JSON.stringify({ type: 'game.finished', payload: getGameHistoryStats() }));
      } else if (m.type === 'train_on_games') {
        console.log('[WS] Training on game history...', m.payload);
        await trainOnGames((ev)=>ws.send(JSON.stringify(ev)), m.payload || {});
      } else if (m.type === 'clear_history') {
        const result = clearGameHistory();
        ws.send(JSON.stringify({ type: 'history.cleared', payload: result }));
      } else if (m.type === 'get_history_stats') {
        const stats = getGameHistoryStats();
        ws.send(JSON.stringify({ type: 'history.stats', payload: stats }));
        // Также отправляем GPU info при запросе статистики (клиент запрашивает её при подключении)
        try {
          const gpuInfoData = getGpuInfo();
          ws.send(JSON.stringify({ type: 'gpu.info', payload: gpuInfoData }));
          console.log('[WS] GPU info sent with history stats');
        } catch (e) {
          console.error('[WS] Error sending GPU info:', e.message);
        }
      } else if (m.type === 'get_gpu_info') {
        // Отправляем GPU info по запросу клиента
        try {
          const gpuInfoData = getGpuInfo();
          ws.send(JSON.stringify({ type: 'gpu.info', payload: gpuInfoData }));
        } catch (e) {
          console.error('[WS] Error getting GPU info:', e);
        }
      } else if (m.type === 'pv.infer') {
        // Policy+Value inference через WebSocket
        try {
          const model = await ensurePVModel();
          if (!model) {
            ws.send(JSON.stringify({ type: 'pv.infer.error', error: 'PV model not available. Please train a model first.' }));
            return;
          }
          
          const N = m.payload.N || BOARD_N;
          const board = m.payload.board;
          const player = m.payload.player || 1;
          const L = N * N;
          
          if (!board || board.length !== L) {
            ws.send(JSON.stringify({ type: 'pv.infer.error', error: `Board must be array of length ${L} (N=${N})` }));
            return;
          }
          
          const xs = new Float32Array(1 * L * 3);
          const ms = new Float32Array(1 * L);
          
          // Конвертируем board в 3 плоскости (my/opponent/empty)
          for (let i = 0; i < L; i++) {
            const base = i * 3;
            const v = board[i];
            // Поддерживаем оба формата: {-1,0,1} и {0,1,2}
            const normalized = v === -1 ? 2 : (v === 0 ? 0 : 1);
            xs[base + 0] = (normalized === player) ? 1 : 0; // my
            xs[base + 1] = (normalized !== 0 && normalized !== player) ? 1 : 0; // opponent
            xs[base + 2] = (normalized === 0) ? 1 : 0; // empty
            ms[i] = (normalized === 0) ? 1 : 0; // mask (legal moves)
          }
          
          const x = tf.tensor4d(xs, [1, N, N, 3]);
          const maskTensor = tf.tensor2d(ms, [1, L]);
          const [logits, v] = model.apply(x);
          const masked = logits.add(maskTensor.mul(-1).add(1).mul(-1e9));
          const probs = tf.softmax(masked);
          const pa = Array.from((await probs.data()));
          const val = (await v.data())[0];
          
          x.dispose();
          maskTensor.dispose();
          logits.dispose();
          masked.dispose();
          probs.dispose();
          v.dispose();
          
          ws.send(JSON.stringify({ type: 'pv.infer.result', payload: { policy: pa, value: val } }));
        } catch (e) {
          console.error('[WS] PV inference error:', e);
          ws.send(JSON.stringify({ type: 'pv.infer.error', error: String(e) }));
        }
      } else if (m.type === 'health') {
        // Health check через WebSocket
        ws.send(JSON.stringify({ 
          type: 'health.result',
          payload: { 
            status: 'ok', 
            ws_port: WS_PORT,
            gpu: getGpuInfo()
          }
        }));
      } else {
        ws.send(JSON.stringify({ type: 'error', error: 'unknown_type' }));
      }
    } catch (e) {
      console.error('[WS] Error handling message:', e);
      ws.send(JSON.stringify({ type: 'error', error: String(e) }));
    }
  });
  ws.on('error', (err) => {
    console.error('[WS] Connection error:', err);
  });
  ws.on('close', () => {
    console.log('[WS] Client disconnected');
  });
});

// ===== Policy+Value inference через WebSocket =====
let pvModel = null;
async function ensurePVModel(){
  if (!pvModel){
    // Пробуем сначала PV модель, потом обычную
    const savedPvPath = path.join(__dirname, 'saved_pv', `N${BOARD_N}_value`, 'model.json');
    const savedPath = path.join(__dirname, 'saved', 'model.json');
    
    let url;
    try {
      await fs.access(savedPvPath);
      url = 'file://' + savedPvPath;
      console.log('[PV WS] Found PV model, using:', url);
    } catch {
      url = 'file://' + savedPath;
      console.log('[PV WS] Using standard model:', url);
    }
    
    try {
      pvModel = await tf.loadLayersModel(url);
      console.log('[PV WS] Model loaded successfully');
    } catch (e) {
      console.warn('[PV WS] Failed to load model:', e.message);
      // Не выбрасываем ошибку - модель может быть недоступна
    }
  }
  return pvModel;
}
