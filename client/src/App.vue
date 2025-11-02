<template>
  <div class="container">
    <h1>Трансформер для крестики-нолики 3×3</h1>

    <section class="panel">
      <h2>Тренировка</h2>
      <div class="controls">
        <div class="button-group">
          <button :disabled="training || clearing" @click="startTrain">Обучить</button>
          <button :disabled="training || clearing" @click="trainOnGames" :class="historyCount >= 10 ? '' : 'disabled-btn'">Дообучить на играх</button>
          <button :disabled="training || clearing" @click="clearModel" class="clear-btn">Очистить модель</button>
        </div>
        <div class="history-info" v-if="historyCount > 0">
          <span>Сохранено ходов: {{historyCount}}</span>
          <button @click="clearHistory" class="small-btn">Очистить историю</button>
        </div>
        <div class="checkbox-group">
          <label>
            <input type="checkbox" v-model="autoTrainAfterGame" />
            <span>Автоматически дообучать после каждой игры</span>
          </label>
        </div>
        <div class="training-settings">
          <h3>Настройки дообучения</h3>
          <div class="setting-row">
            <label>
              <span>Количество эпох:</span>
              <input type="number" v-model.number="trainingEpochs" min="1" max="20" step="1" :disabled="training || clearing" />
            </label>
          </div>
          <div class="setting-row">
            <label>
              <span>Вариаций паттерна на ошибку:</span>
              <input type="number" v-model.number="patternsPerError" min="10" max="2000" step="10" :disabled="training || clearing" />
            </label>
          </div>
          <div class="setting-hint">
            <small>Больше вариаций = лучшее обучение, но медленнее генерация</small>
          </div>
        </div>
        <div class="progress" v-if="progress">
          <div class="bar" :style="{ width: (progress.percent||0)+'%' }"></div>
        </div>
        <div class="logs" v-if="progress">
          Эпоха {{progress.epoch}} / {{progress.epochs}} ·
          loss: {{progress.loss}} · acc: {{progress.acc}} ·
          val_loss: {{progress.val_loss}} · val_acc: {{progress.val_acc}}
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Игра</h2>
      <div class="game-type-selector">
        <label class="mode-label">
          <input type="radio" v-model="gameType" value="human" :disabled="training || clearing || autoPlaying" />
          <span>Человек vs Бот</span>
        </label>
        <label class="mode-label">
          <input type="radio" v-model="gameType" value="auto" :disabled="training || clearing || autoPlaying" />
          <span>Модель vs Бот (minimax)</span>
        </label>
      </div>
      <div class="mode-selector" v-if="gameType === 'human'">
        <label class="mode-label">
          <input type="radio" v-model="gameMode" value="model" :disabled="training || clearing" />
          <span>Модель</span>
        </label>
        <label class="mode-label">
          <input type="radio" v-model="gameMode" value="algorithm" :disabled="training || clearing" />
          <span>Алгоритм (minimax)</span>
        </label>
      </div>
      <div class="pause-control" v-if="gameType === 'auto'">
        <label class="pause-label">
          Пауза между ходами (мс):
          <input type="number" v-model.number="pauseMs" :disabled="autoPlaying" min="0" max="5000" step="100" />
        </label>
      </div>
      <div class="board">
        <button v-for="(cell, idx) in board" :key="idx" class="cell" 
                :disabled="cell!==0 || waiting || gameOver || autoPlaying"
                @click="humanMove(idx)">
          {{ renderCell(cell) }}
        </button>
      </div>
      <div class="row">
        <button @click="reset" :disabled="autoPlaying">Сброс</button>
        <button v-if="gameType === 'auto' && !autoPlaying && !gameOver" @click="startAutoGame">Начать игру</button>
        <button v-if="autoPlaying" @click="stopAutoGame">Остановить</button>
        <span class="status" :class="{ 'game-over': gameOver }">{{ status }}</span>
      </div>
    </section>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'

const ws = ref(null)
const training = ref(false)
const clearing = ref(false)
const progress = ref(null)
const board = ref(Array(9).fill(0))
const current = ref(1) // 1 = X (человек/модель), 2 = O (бот/алгоритм)
const waiting = ref(false) // Флаг ожидания ответа от сервера
const status = ref('Ваш ход (X)')
const gameMode = ref('model') // 'model' или 'algorithm' (для режима human)
const gameType = ref('human') // 'human' или 'auto'
const gameOver = ref(false) // Игра завершена
const autoPlaying = ref(false) // Автоматическая игра идет
const pauseMs = ref(1000) // Пауза между ходами в мс
let autoGameInterval = null
const historyCount = ref(0) // Количество сохраненных ходов
const autoTrainAfterGame = ref(false) // Автоматическое дообучение после игры
const trainingEpochs = ref(5) // Количество эпох при дообучении
const patternsPerError = ref(1000) // Количество вариаций паттерна для каждой ошибки
let currentGameMoves = [] // Ходы текущей игры
let currentGameId = null // ID текущей игры для отслеживания последовательности
let reconnectAttempts = 0
let reconnectTimeout = null
let isConnecting = false

function connectWS() {
  // Предотвращаем множественные попытки подключения
  if (isConnecting) {
    console.log('[WS] Already connecting, skipping...')
    return
  }
  
  // Если уже есть открытое соединение, не подключаемся снова
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    console.log('[WS] Already connected')
    return
  }
  
  // Закрываем старое соединение если оно есть в неправильном состоянии
  if (ws.value) {
    try {
      ws.value.close()
    } catch (e) {
      // Игнорируем ошибки закрытия
    }
  }
  
  isConnecting = true
  reconnectAttempts++
  
  console.log(`[WS] Attempting to connect (attempt ${reconnectAttempts})...`)
  
  try {
    ws.value = new WebSocket('ws://localhost:8080')
    
    let hasHandledClose = false
    
    ws.value.onopen = () => { 
      console.log('[WS] Connected successfully')
      isConnecting = false
      reconnectAttempts = 0
      status.value = 'Подключено'
      hasHandledClose = false
      // Запрашиваем статистику истории
      if (ws.value.readyState === WebSocket.OPEN) {
        ws.value.send(JSON.stringify({ type: 'get_history_stats' }))
      }
    }
    
    ws.value.onerror = (err) => { 
      console.error('[WS] Connection error:', err)
      // Не устанавливаем isConnecting = false здесь, чтобы onclose мог обработать переподключение
      
      // Показываем сообщение только при первой попытке
      if (reconnectAttempts === 1) {
        status.value = '⚠️ Сервер не запущен. Запустите: npm start'
      }
    }
    
    ws.value.onclose = (event) => { 
      if (hasHandledClose) return
      hasHandledClose = true
      
      console.log('[WS] Connection closed', event.code, event.reason)
      isConnecting = false
      
      // Не переподключаемся если это нормальное закрытие
      if (event.code === 1000 || event.code === 1001) {
        console.log('[WS] Normal closure, not reconnecting')
        return
      }
      
      // Очищаем старый таймаут
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
        reconnectTimeout = null
      }
      
      // Задержка для переподключения
      const delay = Math.min(1000 * Math.pow(2, Math.min(reconnectAttempts - 1, 3)), 5000)
      
      if (reconnectAttempts <= 5) {
        status.value = `Переподключение через ${(delay/1000).toFixed(1)}с...`
      } else {
        status.value = `Ожидание сервера... (попытка ${reconnectAttempts})`
      }
      
      reconnectTimeout = setTimeout(() => {
        reconnectTimeout = null
        connectWS()
      }, delay)
    }
    
    ws.value.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        console.log('[WS] Received:', msg.type, msg.payload || '')
        
        if (msg.type === 'train.progress') {
          progress.value = msg.payload
          status.value = `Эпоха ${msg.payload.epoch}/${msg.payload.epochs}`
        }
        if (msg.type === 'train.start') { 
          training.value = true
          progress.value = { percent: 0, epoch: 0, epochs: msg.payload.epochs }
          status.value = 'Подготовка к обучению...'
          console.log('[WS] Training started, epochs:', msg.payload.epochs)
        }
        if (msg.type === 'train.status') {
          status.value = msg.payload.message || 'Обработка...'
          console.log('[WS] Status:', msg.payload.message)
        }
      if (msg.type === 'train.done') { 
        training.value = false
        status.value = 'Обучение завершено'
        console.log('[WS] Training completed')
        // Обновляем статистику истории после обучения
        if (ws.value && ws.value.readyState === WebSocket.OPEN) {
          ws.value.send(JSON.stringify({ type: 'get_history_stats' }))
        }
      }
        if (msg.type === 'predict.result') {
          console.log('[WS] Received predict.result, resetting waiting')
          waiting.value = false
          const move = msg.payload.move
          
          // Проверяем, что игра еще не закончилась (может быть состояние изменилось)
          if (gameOver.value) {
            console.log('[WS] Game already over, ignoring move')
            return
          }
          
          if (move !== -1 && board.value[move] === 0) {
            // Определяем текущего игрока по режиму игры
            if (gameType.value === 'auto') {
              // В автоматическом режиме: current определяет кто сейчас ходит
              board.value[move] = current.value
              const playerName = current.value === 1 ? 'Модель (X)' : 'Бот (O, minimax)'
              status.value = `Ход ${playerName}`
              
              // Проверяем победу
              checkGameOver()
              
              // Продолжаем автоматическую игру
              if (autoPlaying.value && !gameOver.value) {
                continueAutoGame()
              } else if (gameOver.value) {
                stopAutoGame()
              }
            } else {
              // В режиме человек vs бот
              // Сохраняем ход бота для обучения
              saveMove(board.value, move, 2)
              
              board.value[move] = 2
              current.value = 1
              
              // Проверяем победу после хода бота
              if (!checkGameOver()) {
                // Игра продолжается
                if (msg.payload.mode === 'algorithm') {
                  status.value = 'Ход бота (алгоритм minimax) - Ваш ход (X)'
                } else if (msg.payload.isRandom) {
                  status.value = 'Ход бота (случайный) - Ваш ход (X)'
                } else {
                  status.value = 'Ваш ход (X)'
                }
              } else {
                // Игра окончена - проверяем авто-дообучение
                autoTrainAfterGameIfEnabled()
              }
              // Если checkGameOver вернул true, статус уже установлен
            }
          } else {
            status.value = 'Нет хода'
            if (gameType.value === 'auto') {
              stopAutoGame()
            }
          }
        }
        if (msg.type === 'clear_model.success') {
          clearing.value = false
          progress.value = null
          status.value = 'Модель очищена'
          console.log('[WS] Model cleared:', msg.payload)
        }
        if (msg.type === 'move.saved') {
          historyCount.value = msg.payload.count || 0
          console.log('[WS] Move saved, history:', msg.payload)
        }
        if (msg.type === 'history.cleared') {
          historyCount.value = 0
          currentGameMoves = []
          console.log('[WS] History cleared')
        }
        if (msg.type === 'history.stats') {
          historyCount.value = msg.payload.count || 0
        }
        if (msg.type === 'game.started') {
          currentGameId = msg.payload.gameId
          console.log('[WS] Game tracking started:', currentGameId)
        }
        if (msg.type === 'game.finished') {
          historyCount.value = msg.payload.count || 0
          console.log('[WS] Game finished, error patterns generated')
          currentGameId = null
        }
        if (msg.type === 'error') {
          console.error('[WS] Server error:', msg.error)
          training.value = false
          clearing.value = false
          waiting.value = false // Сбрасываем ожидание при ошибке
          status.value = 'Ошибка: ' + msg.error
        }
      } catch (e) {
        console.error('[WS] Parse error:', e, 'Raw data:', ev.data)
        status.value = 'Ошибка обработки сообщения'
      }
    }
    
  } catch (e) {
    console.error('[WS] Error creating WebSocket:', e)
    isConnecting = false
    status.value = 'Ошибка создания WebSocket соединения'
    
    // Пытаемся переподключиться через некоторое время
    const delay = 2000
    reconnectTimeout = setTimeout(() => {
      reconnectTimeout = null
      connectWS()
    }, delay)
  }
}

function startTrain() {
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
    status.value = 'Ошибка: WebSocket не подключен. Проверьте, запущен ли сервер на порту 8080.'
    console.error('[Train] WebSocket not ready, state:', ws.value?.readyState)
    // Пытаемся переподключиться
    if (!isConnecting) {
      connectWS()
    }
    return
  }
  training.value = true
  status.value = 'Отправка запроса на обучение...'
  try {
    ws.value.send(JSON.stringify({ type: 'train', payload: { epochs: 5, batchSize: 256, nTrain: 4000, nVal: 1000 } }))
    console.log('[Train] Training request sent')
  } catch (e) {
    console.error('[Train] Send error:', e)
    training.value = false
    status.value = 'Ошибка отправки запроса: ' + e.message
  }
}

function clearModel() {
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
    status.value = 'Ошибка: WebSocket не подключен. Проверьте, запущен ли сервер на порту 8080.'
    console.error('[Clear] WebSocket not ready, state:', ws.value?.readyState)
    if (!isConnecting) {
      connectWS()
    }
    return
  }
  if (!confirm('Вы уверены, что хотите удалить сохраненную модель? Это действие нельзя отменить.')) {
    return
  }
  clearing.value = true
  status.value = 'Очистка модели...'
  try {
    ws.value.send(JSON.stringify({ type: 'clear_model' }))
    console.log('[Clear] Clear model request sent')
  } catch (e) {
    console.error('[Clear] Send error:', e)
    clearing.value = false
    status.value = 'Ошибка отправки запроса: ' + e.message
  }
}

// Проверка победы (скопировано с сервера)
function getWinner(board) {
  const lines = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6],
  ];
  for (const [a,b,c] of lines) {
    if (board[a] && board[a] === board[b] && board[b] === board[c]) return board[a];
  }
  if (board.every(v => v !== 0)) return 0; // Ничья
  return null; // Игра продолжается
}

function checkGameOver() {
  const winner = getWinner(board.value);
  if (winner === 1) {
    gameOver.value = true;
    if (gameType.value === 'auto') {
      status.value = '🎉 Модель выиграла! (X)';
    } else {
      status.value = '🎉 Вы выиграли! (X)';
      // Если модель проиграла, анализируем ошибки
      if (gameMode.value === 'model') {
        finishGameTracking(1) // Человек (1) выиграл, модель (2) проиграла
      }
      // Сохраняем финальные ходы если игра окончена
      autoTrainAfterGameIfEnabled()
    }
    return true;
  } else if (winner === 2) {
    gameOver.value = true;
    if (gameType.value === 'auto') {
      status.value = '❌ Бот (minimax) выиграл! (O)';
    } else {
      status.value = '❌ Бот выиграл! (O)';
      // Если модель выиграла, ничего особенного не делаем
      if (gameMode.value === 'model') {
        finishGameTracking(2) // Модель (2) выиграла
      }
      autoTrainAfterGameIfEnabled()
    }
    return true;
  } else if (winner === 0) {
    gameOver.value = true;
    status.value = '🤝 Ничья!';
    if (gameType.value === 'human') {
      if (gameMode.value === 'model') {
        finishGameTracking(0) // Ничья
      }
      autoTrainAfterGameIfEnabled()
    }
    return true;
  }
  gameOver.value = false;
  return false;
}

function renderCell(v) { return v===0?'·':(v===1?'X':'O') }

function reset() {
  stopAutoGame()
  board.value = Array(9).fill(0)
  current.value = 1
  status.value = gameType.value === 'auto' ? 'Готово к игре' : 'Ваш ход (X)'
  gameOver.value = false
  waiting.value = false // Сбрасываем флаг ожидания
  currentGameMoves = []
  currentGameId = null
}

function saveMove(board, move, currentPlayer) {
  if (gameType.value === 'human' && move >= 0) {
    // Сохраняем только ходы в режиме человек vs бот (ходы человека и бота)
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify({
        type: 'save_move',
        payload: { board: [...board], move, current: currentPlayer, gameId: currentGameId }
      }))
      currentGameMoves.push({ board: [...board], move, current: currentPlayer })
    } else {
      console.warn('[WS] Cannot save move - WebSocket not connected')
    }
  }
}

function startGameTracking() {
  // Начинаем отслеживание новой игры только в режиме человек vs бот с моделью
  if (gameType.value === 'human' && gameMode.value === 'model' && ws.value && ws.value.readyState === WebSocket.OPEN) {
    try {
      ws.value.send(JSON.stringify({ 
        type: 'start_game', 
        payload: { playerRole: 2 } // Модель играет за O (2)
      }))
      console.log('[GameTracking] Start game request sent')
    } catch (e) {
      console.error('[GameTracking] Error sending start_game:', e)
    }
  }
}

function finishGameTracking(winner) {
  // Завершаем отслеживание игры и анализируем ошибки
  if (currentGameId && ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({ 
      type: 'finish_game', 
      payload: { 
        gameId: currentGameId, 
        winner,
        patternsPerError: patternsPerError.value // Передаем настройку количества паттернов
      }
    }))
  }
}

async function autoTrainAfterGameIfEnabled() {
  if (autoTrainAfterGame.value && historyCount.value >= 10 && !training.value) {
    status.value = 'Автоматическое дообучение...'
    setTimeout(() => {
      trainOnGames()
    }, 500)
  }
}

function trainOnGames() {
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
    status.value = 'Ошибка: WebSocket не подключен. Переподключение...'
    if (!isConnecting) {
      connectWS()
    }
    return
  }
  if (historyCount.value < 10) {
    alert(`Недостаточно данных для обучения. Нужно минимум 10 ходов, есть ${historyCount.value}`)
    return
  }
  training.value = true
  status.value = 'Дообучение на реальных играх (с фокусом на ошибках)...'
  try {
    // Используем настройки из интерфейса
    ws.value.send(JSON.stringify({ 
      type: 'train_on_games', 
      payload: { 
        epochs: trainingEpochs.value,
        batchSize: 32,
        focusOnErrors: true, // Включаем автоматическое увеличение эпох для паттернов ошибок
        patternsPerError: patternsPerError.value // Передаем настройку количества паттернов
      } 
    }))
    console.log(`[TrainOnGames] Request sent: ${trainingEpochs.value} epochs, ${patternsPerError.value} patterns per error`)
  } catch (e) {
    console.error('[TrainOnGames] Send error:', e)
    training.value = false
    status.value = 'Ошибка: ' + e.message
  }
}

function clearHistory() {
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
    status.value = 'Ошибка: WebSocket не подключен.'
    return
  }
  if (confirm('Очистить историю игр?')) {
    ws.value.send(JSON.stringify({ type: 'clear_history' }))
  }
}

function stopAutoGame() {
  if (autoGameInterval) {
    clearTimeout(autoGameInterval)
    autoGameInterval = null
  }
  autoPlaying.value = false
  waiting.value = false
}

async function startAutoGame() {
  if (gameType.value !== 'auto') return
  
  // Сбрасываем игру
  board.value = Array(9).fill(0)
  current.value = 1
  gameOver.value = false
  autoPlaying.value = true
  status.value = 'Игра началась: Модель (X) vs Бот (O)'
  
  // Начинаем автоматическую игру
  await makeAutoMove()
}

async function makeAutoMove() {
  if (!autoPlaying.value || gameOver.value) {
    stopAutoGame()
    return
  }
  
  // Проверяем текущего игрока
  if (current.value === 1) {
    // Ход модели (X)
    status.value = 'Ход модели (X)...'
    waiting.value = true
    
    try {
      ws.value.send(JSON.stringify({ 
        type: 'predict', 
        payload: { 
          board: board.value, 
          current: 1, 
          mode: 'model' 
        } 
      }))
    } catch (e) {
      console.error('[Auto] Error making model move:', e)
      stopAutoGame()
    }
  } else {
    // Ход алгоритма (O)
    status.value = 'Ход бота (O, minimax)...'
    waiting.value = true
    
    try {
      ws.value.send(JSON.stringify({ 
        type: 'predict', 
        payload: { 
          board: board.value, 
          current: 2, 
          mode: 'algorithm' 
        } 
      }))
    } catch (e) {
      console.error('[Auto] Error making algorithm move:', e)
      stopAutoGame()
    }
  }
}

async function continueAutoGame() {
  if (!autoPlaying.value) return
  
  // Проверяем окончание игры после хода
  if (gameOver.value) {
    stopAutoGame()
    return
  }
  
  // Пауза перед следующим ходом
  await new Promise(resolve => setTimeout(resolve, pauseMs.value))
  
  // Меняем игрока и делаем следующий ход
  current.value = current.value === 1 ? 2 : 1
  await makeAutoMove()
}

function humanMove(idx) {
  if (board.value[idx] !== 0 || waiting.value || gameOver.value || gameType.value !== 'human') {
    return
  }
  
  // Проверяем подключение WebSocket
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
    if (reconnectAttempts === 0 || (reconnectAttempts > 0 && !isConnecting)) {
      status.value = '⚠️ Сервер не запущен. Запустите сервер командой: npm start'
      if (!isConnecting) {
        connectWS()
      }
    } else {
      status.value = 'Ожидание подключения к серверу...'
    }
    return
  }
  
  // Начинаем отслеживание игры при первом ходе (только для модели)
  if (!currentGameId && gameMode.value === 'model') {
    startGameTracking()
  }
  
  // Сохраняем ход человека для обучения
  saveMove(board.value, idx, 1)
  
  // Делаем ход
  board.value[idx] = 1
  current.value = 2
  
  // Проверяем победу после хода человека
  if (checkGameOver()) {
    return; // Игра окончена, бот не должен ходить
  }
  
  // Если игра продолжается - ждем ход бота
  console.log('[HumanMove] Setting waiting=true, sending predict request')
  waiting.value = true
  status.value = 'Ожидание хода бота...'
  try {
    ws.value.send(JSON.stringify({ 
      type: 'predict', 
      payload: { 
        board: board.value, 
        current: 2, 
        mode: gameMode.value 
      } 
    }))
    console.log('[HumanMove] Predict request sent')
  } catch (e) {
    console.error('[HumanMove] Error sending predict:', e)
    waiting.value = false
    status.value = 'Ошибка отправки запроса'
  }
}

// Останавливаем автоматическую игру при смене типа игры
watch(gameType, () => {
  stopAutoGame()
  reset()
})

onMounted(() => {
  // Даем небольшую задержку перед первым подключением
  setTimeout(() => {
    connectWS()
  }, 100)
})
</script>

<style>
:root { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.container { max-width: 760px; margin: 24px auto; padding: 0 16px; }
.panel { background: #fafafa; border: 1px solid #eaeaea; padding: 16px; border-radius: 12px; margin-bottom: 16px; }
.controls button { padding: 8px 12px; }
.button-group { display: flex; gap: 8px; margin-bottom: 8px; }
.clear-btn { background: #ff5722; color: white; border: none; }
.clear-btn:hover:not(:disabled) { background: #e64a19; }
.clear-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.game-type-selector { display: flex; gap: 16px; margin-bottom: 12px; padding: 8px; background: white; border-radius: 8px; border: 1px solid #ddd; }
.mode-selector { display: flex; gap: 16px; margin-bottom: 12px; padding: 8px; background: white; border-radius: 8px; border: 1px solid #ddd; }
.mode-label { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.mode-label input[type="radio"] { cursor: pointer; }
.mode-label:has(input:disabled) { opacity: 0.5; cursor: not-allowed; }
.pause-control { margin-bottom: 12px; padding: 8px; background: white; border-radius: 8px; border: 1px solid #ddd; }
.pause-label { display: flex; align-items: center; gap: 8px; }
.pause-label input[type="number"] { width: 80px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }
.progress { width: 100%; height: 10px; background: #eee; border-radius: 6px; margin-top: 8px; overflow: hidden; }
.progress .bar { height: 100%; background: #4caf50; transition: width .25s ease; }
.board { display: grid; grid-template-columns: repeat(3, 80px); grid-gap: 8px; margin: 12px 0; }
.cell { width: 80px; height: 80px; font-size: 28px; border: 1px solid #ccc; border-radius: 8px; background: white; cursor: pointer; }
.cell:disabled { background: #f3f3f3; cursor: not-allowed; }
.row { display:flex; align-items:center; gap: 12px; }
.history-info { display: flex; align-items: center; gap: 12px; margin-top: 8px; padding: 8px; background: #f0f0f0; border-radius: 4px; font-size: 0.9em; }
.small-btn { padding: 4px 8px; font-size: 0.85em; }
.disabled-btn { opacity: 0.5; cursor: not-allowed; }
.checkbox-group { margin-top: 8px; padding: 8px; background: #f9f9f9; border-radius: 4px; }
.checkbox-group label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.checkbox-group input[type="checkbox"] { cursor: pointer; }
.training-settings { margin-top: 12px; padding: 12px; background: #f0f7ff; border-radius: 6px; border: 1px solid #cce5ff; }
.training-settings h3 { margin: 0 0 12px 0; font-size: 1em; color: #0066cc; }
.setting-row { margin-bottom: 10px; }
.setting-row label { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.setting-row label span { flex: 1; }
.setting-row input[type="number"] { width: 100px; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; text-align: center; }
.setting-row input[type="number"]:disabled { background: #f5f5f5; cursor: not-allowed; }
.setting-hint { margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; }
.setting-hint small { color: #666; font-size: 0.85em; }
.status { opacity: .8; }
.status.game-over { 
  font-weight: bold; 
  font-size: 1.1em;
  opacity: 1;
}
</style>
