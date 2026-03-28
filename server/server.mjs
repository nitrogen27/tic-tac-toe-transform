import { WebSocketServer } from 'ws';
import { trainTTT3WithProgress, trainTTT5WithProgress, trainGomokuWithProgress, predictMove, clearModel, saveGameMove, trainOnGames, clearGameHistory, getGameHistoryStats, startNewGame, finishGame } from './service.mjs';
import { getGpuInfo } from './src/tf.mjs';

// WebSocket сервер
const WS_PORT = 8080;

const wss = new WebSocketServer({ port: WS_PORT });
console.log('[WS] Listening on ws://localhost:' + WS_PORT);

wss.on('connection', (ws) => {
  console.log('[WS] Client connected');

  // Отправляем GPU info при подключении
  const sendGpuInfo = () => {
    try {
      const gpuInfoData = getGpuInfo();
      if (ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({ type: 'gpu.info', payload: gpuInfoData }));
      }
    } catch (e) {
      console.error('[WS] Error sending GPU info:', e.message);
    }
  };
  sendGpuInfo();

  ws.on('message', async (msg) => {
    try {
      const m = JSON.parse(msg.toString());
      console.log('[WS] Received:', m.type);

      if (m.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));

      } else if (m.type === 'train_ttt3' || m.type === 'train') {
        // Единый обработчик обучения — всегда TTT3 Transformer
        console.log('[WS] Starting TTT3 Transformer training:', m.payload);
        const trainingPayload = m.payload || {};
        if (trainingPayload.epochs === undefined) {
          trainingPayload.epochs = 30; // из config.mjs TRAIN.epochs
        } else if (trainingPayload.epochs > 50) {
          trainingPayload.epochs = 50;
        }
        if (trainingPayload.batchSize !== undefined) {
          trainingPayload.batchSize = Math.max(64, Math.min(4096, trainingPayload.batchSize));
        }
        console.log('[WS] Training config:', trainingPayload);

        await trainTTT3WithProgress((ev) => {
          try {
            if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(ev));
          } catch (sendErr) {
            console.error('[WS] Error sending:', sendErr);
          }
        }, trainingPayload);

      } else if (m.type === 'train_ttt5') {
        // TTT5 Transformer training (bootstrap + MCTS self-play)
        console.log('[WS] Starting TTT5 Transformer training:', m.payload);
        const trainingPayload = m.payload || {};
        const useBigGpuMode = process.env.USE_GPU_BIG === '1';

        if (trainingPayload.epochs === undefined) {
          trainingPayload.epochs = 25;
        } else {
          trainingPayload.epochs = Math.max(1, Math.min(60, Math.round(trainingPayload.epochs)));
        }

        const minTtt5Batch = useBigGpuMode ? 512 : 128;
        if (trainingPayload.batchSize === undefined) {
          trainingPayload.batchSize = useBigGpuMode ? 1024 : 256;
        } else {
          trainingPayload.batchSize = Math.max(minTtt5Batch, Math.min(4096, trainingPayload.batchSize));
          trainingPayload.batchSize = Math.round(trainingPayload.batchSize / 32) * 32;
        }

        if (trainingPayload.bootstrapGames !== undefined) {
          trainingPayload.bootstrapGames = Math.max(20, Math.min(400, Math.round(trainingPayload.bootstrapGames)));
        }
        if (trainingPayload.mctsIterations !== undefined) {
          trainingPayload.mctsIterations = Math.max(1, Math.min(8, Math.round(trainingPayload.mctsIterations)));
        }
        if (trainingPayload.mctsGamesPerIter !== undefined) {
          trainingPayload.mctsGamesPerIter = Math.max(8, Math.min(200, Math.round(trainingPayload.mctsGamesPerIter)));
        }
        if (trainingPayload.mctsTrainingSims !== undefined) {
          trainingPayload.mctsTrainingSims = Math.max(16, Math.min(1024, Math.round(trainingPayload.mctsTrainingSims)));
        }
        console.log('[WS] TTT5 training config:', trainingPayload);

        await trainTTT5WithProgress((ev) => {
          try {
            if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(ev));
          } catch (sendErr) {
            console.error('[WS] Error sending:', sendErr);
          }
        }, trainingPayload);

      } else if (m.type === 'predict') {
        const payload = m.payload || { board: Array(9).fill(0), current: 1 };
        // Auto-detect variant from board size
        let variant = payload.variant;
        if (!variant) {
          const len = payload.board?.length || 9;
          if (len === 9) variant = 'ttt3';
          else if (len === 25) variant = 'ttt5';
          else {
            const N = Math.round(Math.sqrt(len));
            if (N >= 7 && N <= 16 && N * N === len) variant = `gomoku${N}`;
            else variant = 'ttt3';
          }
        }
        const res = await predictMove({
          board: payload.board,
          current: payload.current,
          mode: payload.mode || 'model',
          variant,
        });
        ws.send(JSON.stringify({ type: 'predict.result', payload: res }));

      } else if (m.type === 'train_gomoku') {
        console.log('[WS] Starting Gomoku Engine V2 training:', m.payload);
        const trainingPayload = m.payload || {};

        await trainGomokuWithProgress((ev) => {
          try {
            if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(ev));
          } catch (sendErr) {
            console.error('[WS] Error sending:', sendErr);
          }
        }, trainingPayload);

      } else if (m.type === 'clear_model') {
        const variant = m.payload?.variant || 'all';
        console.log(`[WS] Clearing model (variant=${variant})...`);
        try {
          const result = await clearModel(variant);
          ws.send(JSON.stringify({ type: 'clear_model.success', payload: result }));
        } catch (e) {
          ws.send(JSON.stringify({ type: 'error', error: String(e) }));
        }

      } else if (m.type === 'save_move') {
        await saveGameMove(m.payload || { board: [], move: -1, current: 1, gameId: undefined });
        ws.send(JSON.stringify({ type: 'move.saved', payload: getGameHistoryStats() }));

      } else if (m.type === 'start_game') {
        const gameId = startNewGame({
          playerRole: m.payload?.playerRole || 1,
          variant: m.payload?.variant || 'ttt3',
        });
        ws.send(JSON.stringify({ type: 'game.started', payload: { gameId } }));

      } else if (m.type === 'finish_game') {
        const patternsPerError = Math.max(10, Math.min(500, m.payload?.patternsPerError || 100));
        const autoTrain = m.payload?.autoTrain || false;
        const incrementalBatchSize = Math.max(32, Math.min(1024, m.payload?.incrementalBatchSize || 256));

        await finishGame({
          gameId: m.payload?.gameId,
          winner: m.payload?.winner,
          patternsPerError,
          autoTrain,
          incrementalBatchSize,
          progressCb: (ev) => {
            try {
              if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(ev));
            } catch (e) {
              console.error('[WS] Error sending progress:', e);
            }
          }
        });
        ws.send(JSON.stringify({ type: 'game.finished', payload: getGameHistoryStats() }));

      } else if (m.type === 'train_on_games') {
        console.log('[WS] Training on game history:', m.payload);
        const incrementalPayload = m.payload || {};
        if (incrementalPayload.epochs !== undefined) {
          incrementalPayload.epochs = Math.max(1, Math.min(10, incrementalPayload.epochs));
        }
        if (incrementalPayload.batchSize !== undefined) {
          incrementalPayload.batchSize = Math.max(32, Math.min(1024, incrementalPayload.batchSize));
        }
        await trainOnGames((ev) => {
          try {
            if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(ev));
          } catch (e) {
            console.error('[WS] Error sending:', e);
          }
        }, incrementalPayload);

      } else if (m.type === 'clear_history') {
        const result = clearGameHistory();
        ws.send(JSON.stringify({ type: 'history.cleared', payload: result }));

      } else if (m.type === 'get_history_stats') {
        const stats = getGameHistoryStats();
        ws.send(JSON.stringify({ type: 'history.stats', payload: stats }));
        sendGpuInfo();

      } else if (m.type === 'get_gpu_info') {
        sendGpuInfo();

      } else if (m.type === 'health') {
        ws.send(JSON.stringify({
          type: 'health.result',
          payload: { status: 'ok', ws_port: WS_PORT, gpu: getGpuInfo() }
        }));

      } else {
        ws.send(JSON.stringify({ type: 'error', error: 'unknown_type' }));
      }
    } catch (e) {
      console.error('[WS] Error:', e);
      try {
        if (ws.readyState === ws.OPEN) {
          ws.send(JSON.stringify({ type: 'error', error: String(e) }));
        }
      } catch (sendErr) {
        // Client disconnected
      }
    }
  });

  ws.on('error', (err) => {
    console.error('[WS] Connection error:', err);
  });
  ws.on('close', () => {
    console.log('[WS] Client disconnected');
  });
});
