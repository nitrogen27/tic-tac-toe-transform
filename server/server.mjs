import { WebSocketServer } from 'ws';
import { trainWithProgress, predictMove, clearModel, saveGameMove, trainOnGames, clearGameHistory, getGameHistoryStats, startNewGame, finishGame } from './service.mjs';

const wss = new WebSocketServer({ port: 8080 });
console.log('[WS] Listening on ws://localhost:8080');

// Это обработчик событий WebSocket-сервера, который обрабатывает сообщения от клиентов
// Он обрабатывает три типа запросов:
// 1. 'ping' - отвечает 'pong' для проверки соединения
// 2. 'train' - запускает обучение модели с обновлениями о прогрессе
// 3. 'predict' - вычисляет следующий лучший ход в игре крестики-нолики
wss.on('connection', (ws) => {
  console.log('[WS] Client connected');
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
