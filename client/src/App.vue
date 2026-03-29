<template>
  <div class="container">
    <h1>Трансформер для крестики-нолики {{ variant === 'ttt5' ? '5×5' : '3×3' }}</h1>

    <section class="panel">
      <h2>Тренировка</h2>
      <div class="gpu-status" :class="{ 'gpu-active': gpuAvailable, 'gpu-inactive': !gpuAvailable }">
        <span class="gpu-icon">{{ gpuAvailable ? '🚀' : '💻' }}</span>
        <span class="gpu-text">{{ gpuAvailable ? 'CUDA GPU ускорение активно' : 'CPU режим' }}</span>
        <span class="gpu-backend" v-if="gpuBackend">({{ gpuBackend }})</span>
      </div>
      <div v-if="gpuTelemetry" class="gpu-live-stats">
        <span v-if="gpuTelemetry.gpuUtilization !== null && gpuTelemetry.gpuUtilization !== undefined">GPU {{ gpuTelemetry.gpuUtilization }}%</span>
        <span v-if="gpuTelemetry.gpuPowerW !== null && gpuTelemetry.gpuPowerW !== undefined">{{ gpuTelemetry.gpuPowerW }} W</span>
        <span v-if="gpuTelemetry.gpuMemoryUsedMB !== null && gpuTelemetry.gpuMemoryUsedMB !== undefined">
          VRAM {{ gpuTelemetry.gpuMemoryUsedMB }}/{{ gpuTelemetry.gpuMemoryTotalMB || '?' }} MB
        </span>
        <span v-if="gpuTelemetry.gpuTemperatureC !== null && gpuTelemetry.gpuTemperatureC !== undefined">{{ gpuTelemetry.gpuTemperatureC }}°C</span>
      </div>
      <div v-if="trainingMeta && training" class="training-meta-strip">
        <span>{{ trainingMeta.variant || variant }}</span>
        <span v-if="trainingMeta.boardSize">{{ trainingMeta.boardSize }}×{{ trainingMeta.boardSize }}</span>
        <span v-if="trainingMeta.batchSize">batch {{ trainingMeta.batchSize }}</span>
        <span v-if="trainingMeta.deviceName">{{ trainingMeta.deviceName }}</span>
        <span v-if="trainingMeta.modelParams">params {{ formatParamCount(trainingMeta.modelParams) }}</span>
      </div>
      <div class="controls">
        <div class="button-group">
          <button :disabled="training || clearing" @click="startTrain">Обучить с нуля</button>
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
            <span>Автоматически дообучать после каждой игры (фоновое мини-обучение)</span>
          </label>
          <small v-if="autoTrainAfterGame" style="display: block; margin-top: 4px; color: #666;">
            После каждой игры будет запускаться быстрое фоновое обучение на ошибках (1 эпоха, не блокирует игру, выполняется за несколько секунд)
          </small>
        </div>
        <div class="training-settings">
          <h3>Настройки основного обучения</h3>
          <!-- Presets -->
          <div class="preset-row">
            <button class="preset-btn" :class="{ active: activePreset === 'light' }" :disabled="training || clearing" @click="applyPreset('light')">
              <span class="preset-icon">⚡</span>
              <span class="preset-name">Лёгкое</span>
              <span class="preset-time">~2 мин</span>
            </button>
            <button class="preset-btn" :class="{ active: activePreset === 'medium' }" :disabled="training || clearing" @click="applyPreset('medium')">
              <span class="preset-icon">⚙️</span>
              <span class="preset-name">Среднее</span>
              <span class="preset-time">~5 мин</span>
            </button>
            <button class="preset-btn" :class="{ active: activePreset === 'deep' }" :disabled="training || clearing" @click="applyPreset('deep')">
              <span class="preset-icon">🧠</span>
              <span class="preset-name">Глубокое</span>
              <span class="preset-time">~15 мин</span>
            </button>
            <button class="preset-btn preset-custom" :class="{ active: activePreset === 'custom' }" :disabled="training || clearing" @click="activePreset = 'custom'">
              <span class="preset-icon">🔧</span>
              <span class="preset-name">Своё</span>
            </button>
          </div>
          <!-- Custom settings (always visible but collapsed unless custom) -->
          <div v-if="activePreset === 'custom'" class="custom-settings">
            <div class="setting-row">
              <label>
                <span>Количество эпох:</span>
                <input type="number" v-model.number="mainTrainingEpochs" min="1" max="50" step="1" :disabled="training || clearing" @blur="validateMainTrainingEpochs" @input="validateMainTrainingEpochs" />
              </label>
              <span v-if="validationErrors.mainTrainingEpochs" class="validation-error">{{ validationErrors.mainTrainingEpochs }}</span>
            </div>
            <div class="setting-row">
              <label>
                <span>Размер батча:</span>
                <input type="number" v-model.number="mainTrainingBatchSize" min="32" max="4096" step="32" :disabled="training || clearing" @blur="validateMainTrainingBatchSize" @input="validateMainTrainingBatchSize" />
              </label>
              <span v-if="validationErrors.mainTrainingBatchSize" class="validation-error">{{ validationErrors.mainTrainingBatchSize }}</span>
            </div>
            <div class="setting-hint">
              <small>Больше батч = быстрее, но больше памяти GPU</small>
            </div>
          </div>
          <div v-else class="preset-summary">
            <small>{{ presetDescription }}</small>
          </div>
        </div>
        <div class="training-settings">
          <h3>Настройки дообучения</h3>
          <div class="setting-row">
            <label>
              <span>Количество эпох:</span>
              <input type="number" v-model.number="trainingEpochs" min="1" max="10" step="1" :disabled="training || clearing" @blur="validateTrainingEpochs" @input="validateTrainingEpochs" />
            </label>
            <span v-if="validationErrors.trainingEpochs" class="validation-error">{{ validationErrors.trainingEpochs }}</span>
          </div>
          <div class="setting-row">
            <label>
              <span>Размер батча:</span>
              <input type="number" v-model.number="incrementalBatchSize" min="32" max="1024" step="32" :disabled="training || clearing" @blur="validateIncrementalBatchSize" @input="validateIncrementalBatchSize" />
            </label>
            <span v-if="validationErrors.incrementalBatchSize" class="validation-error">{{ validationErrors.incrementalBatchSize }}</span>
          </div>
          <div class="setting-row">
            <label>
              <span>Вариаций паттерна на ошибку:</span>
              <input type="number" v-model.number="patternsPerError" min="10" max="2000" step="10" :disabled="training || clearing" @blur="validatePatternsPerError" @input="validatePatternsPerError" />
            </label>
            <span v-if="validationErrors.patternsPerError" class="validation-error">{{ validationErrors.patternsPerError }}</span>
          </div>
          <div class="setting-hint">
            <small>Больше вариаций = лучшее обучение, но медленнее генерация</small>
          </div>
        </div>
        <!-- Прогресс генерации датасета -->
        <div v-if="datasetProgress" class="dataset-progress">
          <div class="dataset-header">
            <strong>Генерация датасета</strong>
            <span v-if="datasetProgress.workers" class="workers-badge">{{ datasetProgress.workers }} воркеров</span>
          </div>
          <div class="progress">
            <div class="bar dataset-bar" :style="{ width: (datasetProgress.percent||0)+'%' }"></div>
          </div>
          <div class="dataset-info">
            <span>{{ datasetProgress.generated || 0 }} / {{ datasetProgress.total || 0 }} {{ datasetProgress.unit || 'samples' }} ({{ datasetProgress.percent || 0 }}%)</span>
            <span v-if="datasetProgress.games" class="rate-info">· games: {{ datasetProgress.games }}</span>
            <span v-if="datasetProgress.rate" class="rate-info">· {{ datasetProgress.rate }} {{ datasetProgress.unit || 'samples' }}/с</span>
            <span v-if="datasetProgress.elapsed" class="time-info">· {{ datasetProgress.elapsed }}с</span>
          </div>
          <div v-if="datasetProgress.message" class="dataset-stage">
            {{ datasetProgress.message }}
          </div>
        </div>
        <!-- TTT5 Interactive Training Panel -->
        <div v-if="training && ttt5Progress && ttt5Progress.phase && !datasetProgress && !backgroundProgress" class="ttt5-training-panel">
          <!-- Phase badges -->
          <div class="phase-badges">
            <template v-if="isLegacyStructuredPhase(ttt5Progress.phase)">
              <span class="phase-badge"
                    :class="{
                      'phase-completed': ttt5Progress.completedPhases && ttt5Progress.completedPhases.includes('tactical'),
                      'phase-active': ttt5Progress.phase === 'tactical',
                      'phase-pending': !ttt5Progress.completedPhases?.includes('tactical') && ttt5Progress.phase !== 'tactical'
                    }">
                {{ ttt5Progress.completedPhases?.includes('tactical') ? '\u2713 ' : '' }}Tactical
              </span>
              <span class="phase-badge"
                    :class="{
                      'phase-completed': ttt5Progress.completedPhases && ttt5Progress.completedPhases.includes('bootstrap'),
                      'phase-active': ttt5Progress.phase === 'bootstrap',
                      'phase-pending': !ttt5Progress.completedPhases?.includes('bootstrap') && ttt5Progress.phase !== 'bootstrap'
                    }">
                {{ ttt5Progress.completedPhases?.includes('bootstrap') ? '\u2713 ' : '' }}Bootstrap
              </span>
              <span v-for="i in (ttt5Progress.totalIterations || 1)" :key="'mcts'+i"
                    class="phase-badge"
                    :class="{
                      'phase-completed': ttt5Progress.phase === 'mcts' && ttt5Progress.iteration > i || (ttt5Progress.phase === 'mcts_game' && ttt5Progress.iteration > i) || (ttt5Progress.phase === 'mcts_train' && ttt5Progress.iteration > i),
                      'phase-active': (ttt5Progress.phase === 'mcts' || ttt5Progress.phase === 'mcts_game' || ttt5Progress.phase === 'mcts_train') && ttt5Progress.iteration === i,
                      'phase-pending': (!['mcts','mcts_game','mcts_train'].includes(ttt5Progress.phase)) || ttt5Progress.iteration < i,
                    }">
                MCTS {{ i }}
              </span>
            </template>
            <template v-else>
              <span class="phase-badge phase-active">{{ getPhaseLabel(ttt5Progress.phase) }}</span>
            </template>
          </div>

          <!-- Overall progress bar -->
          <div class="progress">
            <div class="bar ttt5-bar" :style="{ width: (ttt5Progress.percent || 0) + '%' }"></div>
          </div>

          <!-- Counters row -->
          <div class="training-counters">
            <div class="counter" v-if="ttt5Progress.totalGames > 0">
              <span class="counter-label">Games</span>
              <span class="counter-value">{{ ttt5Progress.game || 0 }}<span class="counter-total">/{{ ttt5Progress.totalGames }}</span></span>
            </div>
            <div class="counter">
              <span class="counter-label">{{ ttt5Progress.samplesTotal ? 'Samples' : 'Positions' }}</span>
              <span class="counter-value">{{ formatPositionCount(ttt5Progress) }}</span>
            </div>
            <div class="counter">
              <span class="counter-label">Elapsed</span>
              <span class="counter-value">{{ formatTime(ttt5Progress.elapsed) }}</span>
            </div>
            <div class="counter">
              <span class="counter-label">ETA</span>
              <span class="counter-value">{{ formatTime(ttt5Progress.eta) }}</span>
            </div>
            <div class="counter" v-if="ttt5Progress.speed > 0">
              <span class="counter-label">Speed</span>
              <span class="counter-value">{{ ttt5Progress.speed }} <span class="counter-unit">{{ ttt5Progress.speedUnit || 'g/s' }}</span></span>
            </div>
          </div>

          <!-- Epoch/batch detail -->
          <div v-if="ttt5Progress.epoch > 0" class="epoch-detail">
            <span class="epoch-text">Epoch {{ ttt5Progress.epoch }}/{{ ttt5Progress.totalEpochs }}</span>
            <span v-if="ttt5Progress.batch > 0" class="batch-text">(batch {{ ttt5Progress.batch }}/{{ ttt5Progress.totalBatches }})</span>
            <span v-if="ttt5Progress.loss" class="metric-text">loss: {{ ttt5Progress.loss }}</span>
            <span v-if="ttt5Progress.accuracy" class="metric-text acc-text">acc: {{ ttt5Progress.accuracy }}%</span>
            <span v-if="ttt5Progress.mae" class="metric-text">MAE: {{ ttt5Progress.mae }}</span>
          </div>
          <div v-if="hasRuntimeDiagnostics(ttt5Progress)" class="runtime-grid">
            <span v-if="ttt5Progress.deviceName" class="runtime-pill">device: {{ ttt5Progress.deviceName }}</span>
            <span v-if="ttt5Progress.batchSize" class="runtime-pill">batch: {{ ttt5Progress.batchSize }}</span>
            <span v-if="ttt5Progress.learningRate" class="runtime-pill">lr: {{ ttt5Progress.learningRate }}</span>
            <span v-if="ttt5Progress.samplesPerSec" class="runtime-pill">samples/s: {{ ttt5Progress.samplesPerSec }}</span>
            <span v-if="ttt5Progress.batchesPerSec" class="runtime-pill">batches/s: {{ ttt5Progress.batchesPerSec }}</span>
            <span v-if="ttt5Progress.batchTimeMs" class="runtime-pill">batch: {{ ttt5Progress.batchTimeMs }} ms</span>
            <span v-if="ttt5Progress.modelParams" class="runtime-pill">params: {{ formatParamCount(ttt5Progress.modelParams) }}</span>
            <span class="runtime-pill">AMP: {{ ttt5Progress.mixedPrecision ? 'on' : 'off' }}</span>
            <span class="runtime-pill">TF32: {{ ttt5Progress.tf32 ? 'on' : 'off' }}</span>
            <span v-if="ttt5Progress.gpuMemoryUsedMB" class="runtime-pill">VRAM: {{ ttt5Progress.gpuMemoryUsedMB }}/{{ ttt5Progress.gpuMemoryTotalMB || '?' }} MB</span>
            <span v-if="ttt5Progress.gpuPowerW" class="runtime-pill">Power: {{ ttt5Progress.gpuPowerW }} W</span>
          </div>

          <!-- Self-play stats -->
          <div v-if="ttt5Progress.selfPlayStats && (ttt5Progress.selfPlayStats.wins > 0 || ttt5Progress.selfPlayStats.losses > 0 || ttt5Progress.selfPlayStats.draws > 0)" class="selfplay-stats">
            Self-play: <span class="sp-win">W {{ ttt5Progress.selfPlayStats.wins }}</span> /
            <span class="sp-loss">L {{ ttt5Progress.selfPlayStats.losses }}</span> /
            <span class="sp-draw">D {{ ttt5Progress.selfPlayStats.draws }}</span>
          </div>

          <!-- Training Charts (uPlot) -->
          <div v-if="metricsHistory.length > 1" class="training-charts">
            <div ref="lossChartEl" class="uplot-chart"></div>
          </div>
        </div>

        <!-- Legacy progress (TTT3 or when no structured data) -->
        <div class="progress" v-if="progress && !ttt5Progress?.phase && !datasetProgress && !backgroundProgress">
          <div class="bar" :style="{ width: (progress.percent||0)+'%' }"></div>
        </div>
        <div class="logs" v-if="progress && !ttt5Progress?.phase && !datasetProgress && !backgroundProgress">
          Эпоха {{progress.epoch}} / {{progress.epochs}} ·
          loss: {{progress.loss}} · acc: {{progress.acc}} ·
          <span v-if="progress.accuracy">Accuracy: {{progress.accuracy}}% · MAE: {{progress.mae}}</span>
          <span v-if="trainingElapsed > 0" class="training-timer">
            · ⏱ {{ formatTime(trainingElapsed) }}
            <span v-if="trainingETA > 0"> · осталось ~{{ formatTime(trainingETA) }}</span>
          </span>
          <span v-else>val_loss: {{progress.val_loss}} · val_acc: {{progress.val_acc}}</span>
        </div>
        <!-- Прогресс фонового дообучения -->
        <div v-if="backgroundProgress" class="background-training" style="margin-top: 12px; padding: 12px; background: #e3f2fd; border-radius: 8px; border: 1px solid #90caf9;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong style="color: #1976d2;">🔄 Фоновое дообучение</strong>
            <span style="font-size: 0.85em; color: #666;">{{backgroundProgress.epoch}}/{{backgroundProgress.epochs}} эпох</span>
          </div>
          <div class="progress" style="margin-bottom: 8px;">
            <div class="bar" style="background: #2196F3;" :style="{ width: (backgroundProgress.epochPercent||0)+'%' }"></div>
          </div>
          <div style="font-size: 0.9em; color: #555;">
            <div style="margin-bottom: 4px;">
              <strong>Новые навыки:</strong> {{backgroundProgress.newSkills}} из {{backgroundProgress.totalSkills}} 
              <span style="color: #1976d2;">({{backgroundProgress.newSkillsPercent}}%)</span>
            </div>
            <div>
              <strong>Прогресс обучения:</strong> {{backgroundProgress.epochPercent}}%
              <span v-if="backgroundProgress.batchProgress && backgroundProgress.batchesPerEpoch" style="margin-left: 8px; color: #666;">
                (батч {{backgroundProgress.currentBatch}}/{{backgroundProgress.batchesPerEpoch}}: {{backgroundProgress.batchProgress}}%)
              </span>
              <span v-if="backgroundProgress.loss" style="margin-left: 12px;">loss: {{backgroundProgress.loss}}</span>
              <span v-if="backgroundProgress.acc" style="margin-left: 8px;">acc: {{backgroundProgress.acc}}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Игра</h2>
      <div class="variant-selector">
        <label class="mode-label">
          <input type="radio" v-model="variant" value="ttt3" :disabled="training || clearing || autoPlaying" />
          <span>3×3 (классика)</span>
        </label>
        <label class="mode-label">
          <input type="radio" v-model="variant" value="ttt5" :disabled="training || clearing || autoPlaying" />
          <span>5×5 (4 в ряд)</span>
        </label>
      </div>
      <div class="game-type-selector">
        <label class="mode-label">
          <input type="radio" v-model="gameType" value="human" :disabled="training || clearing || autoPlaying" />
          <span>Человек vs Бот</span>
        </label>
        <label class="mode-label" v-if="variant === 'ttt3'">
          <input type="radio" v-model="gameType" value="auto" :disabled="training || clearing || autoPlaying" />
          <span>Модель vs Бот (minimax)</span>
        </label>
      </div>
      <div class="mode-selector" v-if="gameType === 'human'">
        <label class="mode-label">
          <input type="radio" v-model="gameMode" value="model" :disabled="training || clearing" />
          <span>Модель</span>
        </label>
        <label class="mode-label" v-if="variant === 'ttt3'">
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
      <div class="board" :style="{ gridTemplateColumns: `repeat(${gridN}, ${cellSize}px)` }">
        <button v-for="(cell, idx) in board" :key="idx" class="cell"
                :style="{ width: cellSize + 'px', height: cellSize + 'px', fontSize: variant === 'ttt5' ? '20px' : '28px' }"
                :disabled="cell!==0 || waiting || gameOver || autoPlaying || training"
                @click="humanMove(idx)">
          {{ renderCell(cell) }}
        </button>
      </div>
      <!-- Policy Heatmap -->
      <div v-if="lastProbs && !gameOver" class="policy-heatmap-container">
        <div class="heatmap-label">Policy Heatmap</div>
        <canvas ref="policyCanvas" class="policy-heatmap"
          :width="gridN * cellSize" :height="gridN * cellSize"
          :style="{ width: gridN * cellSize + 'px', height: gridN * cellSize + 'px' }"></canvas>
      </div>
      <div class="row">
        <button @click="reset" :disabled="autoPlaying">Сброс</button>
        <button v-if="gameType === 'auto' && !autoPlaying && !gameOver" @click="startAutoGame">Начать игру</button>
        <button v-if="autoPlaying" @click="stopAutoGame">Остановить</button>
        <span class="status" :class="{ 'game-over': gameOver }">{{ status }}</span>
        <span v-if="modelConfidence !== null && modelConfidence !== undefined && (gameMode === 'model' || (gameType === 'auto' && current === 1))" class="confidence-indicator" :class="getConfidenceClass(modelConfidence)">
          Уверенность: {{ (modelConfidence * 100).toFixed(1) }}%
        </span>
      </div>
    </section>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, computed, watch, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

const ws = ref(null)
const training = ref(false)
const clearing = ref(false)
const progress = ref(null)
const variant = ref('ttt3') // 'ttt3' или 'ttt5'
const boardSize = computed(() => variant.value === 'ttt5' ? 25 : 9)
const gridN = computed(() => variant.value === 'ttt5' ? 5 : 3)
const cellSize = computed(() => variant.value === 'ttt5' ? 60 : 80)
const winLen = computed(() => variant.value === 'ttt5' ? 4 : 3)
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
// Настройки основного обучения
const mainTrainingEpochs = ref(30)
const mainTrainingBatchSize = ref(1024)
const activePreset = ref('medium') // 'light', 'medium', 'deep', 'custom'

const PRESETS = {
  light:  { epochs: 10, batchSize: 512,  bootstrapGames: 50,  mctsIterations: 2, mctsGamesPerIter: 20,  mctsTrainingSims: 64,  label: 'Лёгкое (~2 мин)', desc: '10 эпох · batch 512 · 50 bootstrap · 2×20 MCTS · 64 sims/ход' },
  medium: { epochs: 25, batchSize: 1024, bootstrapGames: 100, mctsIterations: 3, mctsGamesPerIter: 40,  mctsTrainingSims: 96,  label: 'Среднее (~5 мин)', desc: '25 эпох · batch 1024 · 100 bootstrap · 3×40 MCTS · 96 sims/ход' },
  deep:   { epochs: 50, batchSize: 2048, bootstrapGames: 200, mctsIterations: 5, mctsGamesPerIter: 100, mctsTrainingSims: 160, label: 'Глубокое (~15 мин)', desc: '50 эпох · batch 2048 · 200 bootstrap · 5×100 MCTS · 160 sims/ход' },
}

const presetDescription = computed(() => PRESETS[activePreset.value]?.desc || '')

function applyPreset(name) {
  activePreset.value = name
  const p = PRESETS[name]
  if (p) {
    mainTrainingEpochs.value = p.epochs
    mainTrainingBatchSize.value = p.batchSize
  }
}

// Init with medium preset
applyPreset('medium')
// Настройки дообучения
const trainingEpochs = ref(3) // Количество эпох при дообучении
const incrementalBatchSize = ref(256) // Размер батча для дообучения
const patternsPerError = ref(100) // Количество вариаций паттерна для каждой ошибки

// Ошибки валидации
const validationErrors = ref({
  mainTrainingEpochs: null,
  mainTrainingBatchSize: null,
  trainingEpochs: null,
  incrementalBatchSize: null,
  patternsPerError: null
})

// Валидация настроек основного обучения
function validateMainTrainingSettings() {
  validationErrors.value.mainTrainingEpochs = null
  validationErrors.value.mainTrainingBatchSize = null
  
  if (mainTrainingEpochs.value < 1 || mainTrainingEpochs.value > 50 || !Number.isInteger(mainTrainingEpochs.value)) {
    validationErrors.value.mainTrainingEpochs = 'Количество эпох должно быть от 1 до 50'
    return false
  }

  if (mainTrainingBatchSize.value < 32 || mainTrainingBatchSize.value > 4096 || !Number.isInteger(mainTrainingBatchSize.value)) {
    validationErrors.value.mainTrainingBatchSize = 'Размер батча должен быть от 32 до 4096'
    return false
  }

  if (mainTrainingBatchSize.value % 32 !== 0) {
    validationErrors.value.mainTrainingBatchSize = 'Размер батча должен быть кратен 32'
    return false
  }
  
  return true
}

// Валидация настроек дообучения
function validateIncrementalTrainingSettings() {
  validationErrors.value.trainingEpochs = null
  validationErrors.value.incrementalBatchSize = null
  validationErrors.value.patternsPerError = null
  
  if (trainingEpochs.value < 1 || trainingEpochs.value > 10 || !Number.isInteger(trainingEpochs.value)) {
    validationErrors.value.trainingEpochs = 'Количество эпох должно быть от 1 до 10'
    return false
  }
  
  if (incrementalBatchSize.value < 32 || incrementalBatchSize.value > 1024 || !Number.isInteger(incrementalBatchSize.value)) {
    validationErrors.value.incrementalBatchSize = 'Размер батча должен быть от 32 до 1024'
    return false
  }
  
  if (incrementalBatchSize.value % 32 !== 0) {
    validationErrors.value.incrementalBatchSize = 'Размер батча должен быть кратен 32'
    return false
  }
  
  if (patternsPerError.value < 10 || patternsPerError.value > 2000 || !Number.isInteger(patternsPerError.value)) {
    validationErrors.value.patternsPerError = 'Вариаций паттерна должно быть от 10 до 2000'
    return false
  }
  
  if (patternsPerError.value % 10 !== 0) {
    validationErrors.value.patternsPerError = 'Вариаций паттерна должно быть кратно 10'
    return false
  }
  
  return true
}

// Валидация при изменении значений
function validateMainTrainingEpochs() {
  if (mainTrainingEpochs.value < 1) mainTrainingEpochs.value = 1
  if (mainTrainingEpochs.value > 50) mainTrainingEpochs.value = 50
  if (!Number.isInteger(mainTrainingEpochs.value)) mainTrainingEpochs.value = Math.round(mainTrainingEpochs.value)
  validateMainTrainingSettings()
}

function validateMainTrainingBatchSize() {
  if (mainTrainingBatchSize.value < 32) mainTrainingBatchSize.value = 32
  if (mainTrainingBatchSize.value > 4096) mainTrainingBatchSize.value = 4096
  // Округляем до ближайшего кратного 32
  mainTrainingBatchSize.value = Math.round(mainTrainingBatchSize.value / 32) * 32
  validateMainTrainingSettings()
}

function validateTrainingEpochs() {
  if (trainingEpochs.value < 1) trainingEpochs.value = 1
  if (trainingEpochs.value > 10) trainingEpochs.value = 10
  if (!Number.isInteger(trainingEpochs.value)) trainingEpochs.value = Math.round(trainingEpochs.value)
  validateIncrementalTrainingSettings()
}

function validateIncrementalBatchSize() {
  if (incrementalBatchSize.value < 32) incrementalBatchSize.value = 32
  if (incrementalBatchSize.value > 1024) incrementalBatchSize.value = 1024
  // Округляем до ближайшего кратного 32
  incrementalBatchSize.value = Math.round(incrementalBatchSize.value / 32) * 32
  validateIncrementalTrainingSettings()
}

function validatePatternsPerError() {
  if (patternsPerError.value < 10) patternsPerError.value = 10
  if (patternsPerError.value > 2000) patternsPerError.value = 2000
  // Округляем до ближайшего кратного 10
  patternsPerError.value = Math.round(patternsPerError.value / 10) * 10
  validateIncrementalTrainingSettings()
}
let currentGameMoves = [] // Ходы текущей игры
let currentGameId = null // ID текущей игры для отслеживания последовательности
let reconnectAttempts = 0
let reconnectTimeout = null
let isConnecting = false
const gpuAvailable = ref(false)
const gpuBackend = ref('')
const gpuTelemetry = ref(null)
const datasetProgress = ref(null)
const backgroundProgress = ref(null) // Прогресс фонового обучения
const modelConfidence = ref(null) // Уверенность модели (0-1)
const ttt5Progress = ref(null) // Structured TTT5 training progress
const trainingMeta = ref(null)
const trainingStartTime = ref(0) // When training started (ms)
const trainingElapsed = ref(0) // Elapsed seconds
const trainingETA = ref(0) // Estimated remaining seconds
let trainingTimerInterval = null
let gpuPollInterval = null

const metricsHistory = ref([]) // For training charts
const lossChartEl = ref(null) // uPlot loss chart container
const accChartEl = ref(null) // uPlot accuracy chart container
let lossChart = null
let accChart = null
const lastProbs = ref(null) // Last policy probabilities from bot
const policyCanvas = ref(null) // Canvas for policy heatmap

function formatTime(seconds) {
  if (!seconds || seconds <= 0) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatPositionCount(progress) {
  if (!progress) return '0'
  if (progress.samplesTotal) return `${progress.samplesDone || 0}/${progress.samplesTotal}`
  const base = progress.positions || 0
  const effective = progress.effectivePositions || 0
  if (effective > base) return `${base} (${effective})`
  return String(base)
}

function getPhaseLabel(name) {
  const labels = { tactical: 'Tactical', bootstrap: 'Bootstrap', mcts: 'MCTS', training: 'PyTorch Training', preparing: 'Preparing', encoding: 'Encoding', evaluating: 'Evaluating' }
  return labels[name] || name
}

function isLegacyStructuredPhase(name) {
  return ['tactical', 'bootstrap', 'mcts', 'mcts_game', 'mcts_train'].includes(name)
}

function formatParamCount(value) {
  if (!value) return '0'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return String(value)
}

function hasRuntimeDiagnostics(progress) {
  return !!(progress?.deviceName || progress?.batchSize || progress?.samplesPerSec || progress?.gpuPowerW || progress?.gpuMemoryUsedMB)
}

function extractGpuTelemetry(payload) {
  if (!payload) return null
  const source = payload.gpu || payload
  const telemetry = source.telemetry || {}
  const vram = source.vram || {}
  const snapshot = {
    gpuUtilization: payload.gpuUtilization ?? telemetry.utilizationGpu ?? null,
    gpuMemoryUtilization: payload.gpuMemoryUtilization ?? telemetry.utilizationMemory ?? null,
    gpuPowerW: payload.gpuPowerW ?? telemetry.powerDrawW ?? null,
    gpuPowerLimitW: payload.gpuPowerLimitW ?? telemetry.powerLimitW ?? null,
    gpuMemoryUsedMB: payload.gpuMemoryUsedMB ?? telemetry.memoryUsedMB ?? vram.usedMB ?? null,
    gpuMemoryTotalMB: payload.gpuMemoryTotalMB ?? telemetry.memoryTotalMB ?? vram.totalMB ?? null,
    gpuTemperatureC: payload.gpuTemperatureC ?? telemetry.temperatureC ?? null,
    gpuClockSmMHz: payload.gpuClockSmMHz ?? telemetry.clockSmMHz ?? null,
    gpuClockMemMHz: payload.gpuClockMemMHz ?? telemetry.clockMemMHz ?? null,
    gpuAllocatedMB: payload.gpuAllocatedMB ?? vram.allocatedMB ?? null,
    gpuReservedMB: payload.gpuReservedMB ?? vram.reservedMB ?? null,
    gpuTelemetryTimestamp: payload.gpuTelemetryTimestamp ?? telemetry.timestamp ?? null,
    name: source.name || payload.deviceName || null,
  }
  return Object.values(snapshot).some(v => v !== null && v !== undefined) ? snapshot : null
}

// ===== uPlot Training Charts =====
function createLossChart() {
  if (!lossChartEl.value || lossChart) return
  const opts = {
    width: lossChartEl.value.offsetWidth || 600,
    height: 180,
    title: 'Loss',
    cursor: { show: true },
    scales: { x: { time: false }, y: { auto: true } },
    axes: [
      { label: 'Epoch', size: 30, stroke: '#888', font: '11px system-ui' },
      { label: 'Loss', size: 50, stroke: '#888', font: '11px system-ui' },
    ],
    series: [
      { label: 'Epoch' },
      { label: 'Total', stroke: '#e74c3c', width: 2 },
      { label: 'Accuracy', stroke: '#4caf50', width: 1.5, scale: 'acc' },
    ],
    scales: { x: { time: false }, y: { auto: true }, acc: { auto: true, range: [0, 100] } },
    axes: [
      { stroke: '#666', font: '11px system-ui', size: 28 },
      { stroke: '#e74c3c', font: '11px system-ui', size: 45, label: 'Loss' },
      { side: 1, scale: 'acc', stroke: '#4caf50', font: '11px system-ui', size: 45, label: 'Acc%', grid: { show: false } },
    ],
  }
  lossChart = new uPlot(opts, [[0], [null], [null]], lossChartEl.value)
}

function updateCharts() {
  const data = metricsHistory.value
  if (!data || data.length < 1) return
  const epochs = data.map((_, i) => i + 1)
  const losses = data.map(d => parseFloat(d.loss) || null)
  const accs = data.map(d => parseFloat(d.acc) || null)

  if (lossChart) {
    lossChart.setData([epochs, losses, accs])
  }
}

function destroyCharts() {
  if (lossChart) { lossChart.destroy(); lossChart = null }
  if (accChart) { accChart.destroy(); accChart = null }
}

// ===== Policy Heatmap =====
function drawPolicyHeatmap() {
  const canvas = policyCanvas.value
  const probs = lastProbs.value
  if (!canvas || !probs) return

  const N = gridN.value
  const ctx = canvas.getContext('2d')
  const cellW = canvas.width / N
  const cellH = canvas.height / N

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const maxProb = Math.max(...probs, 0.01)

  for (let i = 0; i < probs.length; i++) {
    const col = i % N
    const row = Math.floor(i / N)
    const prob = probs[i] || 0
    const norm = prob / maxProb

    // Color: transparent (low) -> blue -> yellow -> red (high)
    let r, g, b
    if (norm < 0.33) {
      const t = norm / 0.33
      r = Math.round(50 * t); g = Math.round(100 * t); b = Math.round(200 + 55 * t)
    } else if (norm < 0.66) {
      const t = (norm - 0.33) / 0.33
      r = Math.round(50 + 205 * t); g = Math.round(100 + 155 * t); b = Math.round(255 - 155 * t)
    } else {
      const t = (norm - 0.66) / 0.34
      r = 255; g = Math.round(255 - 200 * t); b = Math.round(100 - 100 * t)
    }

    const alpha = Math.max(0.15, norm * 0.85)
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
    ctx.fillRect(col * cellW + 1, row * cellH + 1, cellW - 2, cellH - 2)

    // Label
    if (prob > 0.01) {
      ctx.fillStyle = norm > 0.5 ? '#fff' : '#333'
      ctx.font = `bold ${Math.round(cellW * 0.28)}px system-ui`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText((prob * 100).toFixed(0) + '%', col * cellW + cellW / 2, row * cellH + cellH / 2)
    }
  }
}

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
      // Запрашиваем статистику истории и GPU информацию
      if (ws.value.readyState === WebSocket.OPEN) {
        ws.value.send(JSON.stringify({ type: 'get_history_stats' }))
        ws.value.send(JSON.stringify({ type: 'get_gpu_info' }))
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
        
        // Обработка событий фонового обучения
        if (msg.type === 'background_train.start') {
          backgroundProgress.value = {
            epoch: 0,
            epochs: msg.payload.epochs || 1,
            epochPercent: 0,
            newSkills: msg.payload.newSkills || 0,
            totalSkills: msg.payload.totalSkills || 0,
            newSkillsPercent: msg.payload.newSkillsPercent || 0,
            message: msg.payload.message
          }
          status.value = msg.payload.message || 'Фоновое обучение началось...'
          console.log('[WS] Background training started:', msg.payload)
        }
        if (msg.type === 'background_train.progress') {
          backgroundProgress.value = {
            epoch: msg.payload.epoch || 0,
            epochs: msg.payload.epochs || 1,
            epochPercent: msg.payload.epochPercent || 0,
            newSkills: msg.payload.newSkills || 0,
            totalSkills: msg.payload.totalSkills || 0,
            newSkillsPercent: msg.payload.newSkillsPercent || 0,
            loss: msg.payload.loss,
            acc: msg.payload.acc,
            batchProgress: msg.payload.batchProgress,
            currentBatch: msg.payload.currentBatch,
            batchesPerEpoch: msg.payload.batchesPerEpoch,
            message: msg.payload.message
          }
          // Используем epochPercent для статуса, если есть batchProgress - показываем его
          const progressText = msg.payload.batchProgress ? 
            `${msg.payload.epochPercent}% (батч ${msg.payload.currentBatch}/${msg.payload.batchesPerEpoch})` : 
            `${msg.payload.epochPercent}%`
          status.value = msg.payload.message || `Фоновое обучение: ${progressText}`
          console.log('[WS] Background training progress:', msg.payload)
        }
        if (msg.type === 'background_train.done') {
          status.value = msg.payload.message || 'Фоновое обучение завершено'
          console.log('[WS] Background training done:', msg.payload)
          // Очищаем прогресс через 3 секунды
          setTimeout(() => {
            backgroundProgress.value = null
          }, 3000)
        }
        if (msg.type === 'background_train.error') {
          status.value = msg.payload?.message || msg.error || 'Ошибка фонового обучения'
          backgroundProgress.value = null
          console.error('[WS] Background training error:', msg.error)
        }
        
        if (msg.type === 'train.progress') {
          // Игнорируем прогресс обычного обучения, если идет фоновое
          if (!backgroundProgress.value) {
            datasetProgress.value = null
            progress.value = msg.payload

            // Structured TTT5 progress
            if (msg.payload.phase) {
              ttt5Progress.value = msg.payload
              trainingMeta.value = { ...(trainingMeta.value || {}), ...msg.payload }
              gpuTelemetry.value = extractGpuTelemetry(msg.payload)
              if (msg.payload.metricsHistory) {
                metricsHistory.value = msg.payload.metricsHistory
              }
              // Build status text from structured data
              const p = msg.payload
              let statusText = ''
              if (p.phase === 'tactical') statusText = `Tactical curriculum`
              else if (p.phase === 'bootstrap') statusText = `Bootstrap`
              else if (p.phase === 'mcts') statusText = `MCTS ${p.iteration || ''}/${p.totalIterations || ''}`
              if (p.epoch > 0) statusText += ` | Epoch ${p.epoch}/${p.totalEpochs}`
              if (p.accuracy) statusText += ` | Acc: ${p.accuracy}%`
              if (p.loss) statusText += ` | Loss: ${p.loss}`
              status.value = statusText
            } else {
              // Legacy TTT3 progress
              ttt5Progress.value = null
              if (msg.payload.accuracy !== undefined) {
                status.value = `Эпоха ${msg.payload.epoch}/${msg.payload.epochs} - Accuracy: ${msg.payload.accuracy}%, MAE: ${msg.payload.mae}`
              } else {
                status.value = `Эпоха ${msg.payload.epoch}/${msg.payload.epochs}`
              }
            }
          }
        }
        if (msg.type === 'train.start') { 
          training.value = true
          trainingMeta.value = msg.payload
          gpuTelemetry.value = extractGpuTelemetry(msg.payload)
          progress.value = { percent: 0, epoch: 0, epochs: msg.payload.epochs }
          ttt5Progress.value = { ...msg.payload, phase: 'preparing', percent: 0, elapsed: 0, eta: 0 }
          // Не сбрасываем datasetProgress здесь - генерация еще не началась
          status.value = 'Подготовка к обучению...'
          console.log('[WS] Training started, epochs:', msg.payload.epochs)
        }
        if (msg.type === 'train.status') {
          status.value = msg.payload.message || 'Обработка...'
          console.log('[WS] Status:', msg.payload.message)
        }
        if (msg.type === 'dataset.progress') {
          // Прогресс генерации датасета
          const p = msg.payload
          console.log('[WS] Dataset progress received:', JSON.stringify(p))
          datasetProgress.value = p
          if (p.stage === 'concatenating') {
            status.value = p.message || 'Объединение тензоров на GPU...'
          } else {
            status.value = `Генерация датасета: ${p.generated || 0}/${p.total || 0} игр (${p.percent || 0}%) - ${p.rate || 0} игр/с`
          }
          console.log('[WS] Dataset progress updated:', p.generated, '/', p.total, `(${p.percent}%)`)
        }
      if (msg.type === 'train.done') {
        training.value = false
        datasetProgress.value = null
        // Stop timer & show final time
        if (trainingTimerInterval) { clearInterval(trainingTimerInterval); trainingTimerInterval = null }
        const totalTime = formatTime(Math.floor((Date.now() - trainingStartTime.value) / 1000))
        ttt5Progress.value = null
        trainingMeta.value = null
        metricsHistory.value = []
        status.value = `Обучение завершено за ${totalTime}`
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

          // Store policy probabilities for heatmap
          if (msg.payload.probs && Array.isArray(msg.payload.probs)) {
            lastProbs.value = msg.payload.probs
            nextTick(drawPolicyHeatmap)
          }
          
          // Обновляем уверенность модели
          console.log('[WS] Predict result payload:', { 
            mode: msg.payload.mode, 
            confidence: msg.payload.confidence, 
            hasProbs: !!msg.payload.probs,
            isRandom: msg.payload.isRandom,
            currentValue: current.value,
            gameType: gameType.value,
            gameMode: gameMode.value
          })
          
          if (msg.payload.mode === 'model' && !msg.payload.isRandom && msg.payload.confidence !== undefined && msg.payload.confidence !== null) {
            // Уверенность приходит как число от сервера
            modelConfidence.value = typeof msg.payload.confidence === 'number' ? msg.payload.confidence : parseFloat(msg.payload.confidence)
            console.log('[WS] ✓ Model confidence set to:', modelConfidence.value, '(', (modelConfidence.value * 100).toFixed(1), '%)')
          } else if (msg.payload.mode === 'model' && !msg.payload.isRandom && msg.payload.probs && Array.isArray(msg.payload.probs)) {
            // Вычисляем уверенность как максимальную вероятность если confidence нет
            const maxProb = Math.max(...msg.payload.probs)
            modelConfidence.value = maxProb
            console.log('[WS] ✓ Model confidence computed from probs:', modelConfidence.value, '(', (modelConfidence.value * 100).toFixed(1), '%)')
          } else {
            // Для minimax или случайных ходов - скрываем индикатор
            modelConfidence.value = null
            console.log('[WS] ✗ No confidence (mode:', msg.payload.mode, 'isRandom:', msg.payload.isRandom, ')')
          }
          
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
                if (msg.payload.fallback === 'heuristic') {
                  status.value = '⚠️ Модель не обучена — бот играет по эвристике. Ваш ход (X)'
                } else if (msg.payload.mode === 'algorithm') {
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
          ttt5Progress.value = null
          metricsHistory.value = []
          modelConfidence.value = null
          // Сбросить игру — старая модель больше не существует
          reset()
          status.value = '🗑️ Модель очищена. Обучите новую модель для игры.'
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
        if (msg.type === 'gpu.info') {
          gpuAvailable.value = msg.payload.available || false
          gpuBackend.value = msg.payload.backend || 'cpu'
          gpuTelemetry.value = extractGpuTelemetry(msg.payload)
          console.log('[WS] GPU info:', msg.payload)
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
          trainingMeta.value = null
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
  
  // Валидация настроек перед отправкой
  if (!validateMainTrainingSettings()) {
    status.value = 'Ошибка валидации: проверьте настройки основного обучения'
    return
  }
  
  training.value = true
  trainingStartTime.value = Date.now()
  trainingElapsed.value = 0
  trainingETA.value = 0
  // Start timer
  if (trainingTimerInterval) clearInterval(trainingTimerInterval)
  trainingTimerInterval = setInterval(() => {
    trainingElapsed.value = Math.floor((Date.now() - trainingStartTime.value) / 1000)
    // Estimate ETA from progress percent
    const pct = progress.value?.percent || ttt5Progress.value?.percent || 0
    if (pct > 2 && trainingElapsed.value > 3) {
      trainingETA.value = Math.floor(trainingElapsed.value * (100 - pct) / pct)
    }
  }, 1000)
  const trainType = variant.value === 'ttt5' ? 'train_ttt5' : 'train_ttt3'
  status.value = `Отправка запроса на обучение ${variant.value === 'ttt5' ? 'TTT5' : 'TTT3'} Transformer...`
  try {
    const preset = PRESETS[activePreset.value]
    const payload = {
      epochs: mainTrainingEpochs.value,
      batchSize: mainTrainingBatchSize.value,
      earlyStop: true,
    }
    // Pass preset-specific params for TTT5
    if (variant.value === 'ttt5' && preset) {
      payload.bootstrapGames = preset.bootstrapGames
      payload.mctsIterations = preset.mctsIterations
      payload.mctsGamesPerIter = preset.mctsGamesPerIter
      payload.mctsTrainingSims = preset.mctsTrainingSims
    }
    ws.value.send(JSON.stringify({ type: trainType, payload }))
    console.log(`[Train] ${trainType} sent:`, payload)
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
    ws.value.send(JSON.stringify({ type: 'clear_model', payload: { variant: variant.value } }))
    console.log(`[Clear] Clear model request sent (variant=${variant.value})`)
  } catch (e) {
    console.error('[Clear] Send error:', e)
    clearing.value = false
    status.value = 'Ошибка отправки запроса: ' + e.message
  }
}

// Проверка победы (скопировано с сервера)
function getWinner(board) {
  const N = gridN.value
  const wLen = winLen.value
  const DIRS = [[1,0],[0,1],[1,1],[1,-1]]

  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const who = board[r * N + c]
      if (!who) continue
      for (const [dr, dc] of DIRS) {
        let k = 1, rr = r + dr, cc = c + dc
        while (rr >= 0 && cc >= 0 && rr < N && cc < N && board[rr * N + cc] === who) {
          k++; rr += dr; cc += dc
          if (k >= wLen) return who
        }
      }
    }
  }
  if (board.every(v => v !== 0)) return 0 // Ничья
  return null // Игра продолжается
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
  board.value = Array(boardSize.value).fill(0)
  current.value = 1
  status.value = gameType.value === 'auto' ? 'Готово к игре' : 'Ваш ход (X)'
  gameOver.value = false
  waiting.value = false
  modelConfidence.value = null
  currentGameMoves = []
  currentGameId = null
}

// Определяет класс CSS для уверенности модели
function getConfidenceClass(confidence) {
  if (confidence === null || confidence === undefined) return 'confidence-unknown'
  if (confidence >= 0.7) return 'confidence-high' // Зеленый - высокая уверенность
  if (confidence >= 0.4) return 'confidence-medium' // Желтый - средняя уверенность
  return 'confidence-low' // Красный - низкая уверенность
}

function saveMove(board, move, currentPlayer) {
  if (gameType.value === 'human' && move >= 0) {
    // Сохраняем только ходы в режиме человек vs бот (ходы человека и бота)
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify({
        type: 'save_move',
        payload: { board: [...board], move, current: currentPlayer, gameId: currentGameId, variant: variant.value }
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
        payload: { playerRole: 2, variant: variant.value } // Модель играет за O (2)
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
                patternsPerError: patternsPerError.value, // Передаем настройку количества паттернов
                autoTrain: autoTrainAfterGame.value, // Передаем флаг автоматического обучения
                incrementalBatchSize: incrementalBatchSize.value, // Передаем настройку batch size для дообучения
                variant: variant.value
              }
            }))
  }
}

async function autoTrainAfterGameIfEnabled() {
  // Фоновое обучение теперь запускается автоматически на сервере
  // после finish_game, если autoTrainAfterGame включен
  // Эта функция больше не нужна, но оставляем для совместимости
  if (autoTrainAfterGame.value) {
    console.log('[AutoTrain] Background training will be triggered after game finish');
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
  
  // Валидация настроек перед отправкой
  if (!validateIncrementalTrainingSettings()) {
    status.value = 'Ошибка валидации: проверьте настройки дообучения'
    return
  }
  
  training.value = true
  status.value = 'Дообучение на реальных играх (с фокусом на ошибках)...'
  try {
    // Используем настройки из интерфейса (после валидации)
    ws.value.send(JSON.stringify({ 
      type: 'train_on_games', 
      payload: { 
        epochs: trainingEpochs.value, // Используем настройку из UI
        batchSize: incrementalBatchSize.value, // Используем настройку из UI
        focusOnErrors: true, // Включаем автоматическое увеличение эпох для паттернов ошибок
        patternsPerError: patternsPerError.value, // Передаем настройку количества паттернов
        variant: variant.value
      } 
    }))
    console.log(`[TrainOnGames] Request sent: epochs=${trainingEpochs.value}, batchSize=${incrementalBatchSize.value}, patternsPerError=${patternsPerError.value}`)
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
  board.value = Array(boardSize.value).fill(0)
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
          mode: 'model',
          variant: variant.value,
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
          mode: 'algorithm',
          variant: variant.value,
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
        mode: gameMode.value,
        variant: variant.value,
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

// Update training charts when metrics arrive
watch(metricsHistory, () => {
  if (metricsHistory.value.length > 0) {
    nextTick(() => {
      if (!lossChart && lossChartEl.value) createLossChart()
      updateCharts()
    })
  }
}, { deep: true })

// Clear heatmap on board reset
watch(board, () => {
  if (board.value.every(v => v === 0)) {
    lastProbs.value = null
    const canvas = policyCanvas.value
    if (canvas) {
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    }
  }
})

// Destroy charts when training ends
watch(training, (val) => {
  if (!val) {
    setTimeout(destroyCharts, 500)
  }
})

onUnmounted(() => {
  destroyCharts()
  if (gpuPollInterval) clearInterval(gpuPollInterval)
})

// Сбрасываем игру и настройки при смене варианта
watch(variant, (newVariant) => {
  stopAutoGame()
  // При переключении на TTT5 отключаем minimax
  if (newVariant === 'ttt5') {
    gameMode.value = 'model'
    gameType.value = 'human'
  }
  board.value = Array(newVariant === 'ttt5' ? 25 : 9).fill(0)
  current.value = 1
  gameOver.value = false
  waiting.value = false
  modelConfidence.value = null
  currentGameMoves = []
  currentGameId = null
  status.value = 'Ваш ход (X)'
})

onMounted(() => {
  // Даем небольшую задержку перед первым подключением
  setTimeout(() => {
    connectWS()
  }, 100)
  gpuPollInterval = setInterval(() => {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify({ type: 'get_gpu_info' }))
    }
  }, 3000)
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
.dataset-progress { 
  margin-top: 12px; 
  padding: 12px; 
  background: white; 
  border-radius: 8px; 
  border: 1px solid #e0e0e0;
}
.dataset-header { 
  display: flex; 
  justify-content: space-between; 
  align-items: center; 
  margin-bottom: 8px;
}
.workers-badge { 
  background: #2196F3; 
  color: white; 
  padding: 2px 8px; 
  border-radius: 12px; 
  font-size: 12px; 
  font-weight: normal;
}
.dataset-bar { 
  background: #2196F3 !important; 
}
.dataset-info { 
  margin-top: 6px; 
  font-size: 13px; 
  color: #666; 
  display: flex; 
  gap: 8px; 
  flex-wrap: wrap;
}
.gpu-live-stats {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 8px 0 12px;
  font-size: 13px;
  color: #444;
}
.training-meta-strip {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 8px 0 12px;
  font-size: 12px;
  color: #555;
}
.training-meta-strip span,
.gpu-live-stats span,
.runtime-pill {
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 999px;
  padding: 4px 8px;
}
.runtime-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.rate-info { 
  color: #4caf50; 
  font-weight: 500;
}
.time-info { 
  color: #666;
}
.dataset-stage { 
  margin-top: 8px; 
  padding: 6px; 
  background: #f5f5f5; 
  border-radius: 4px; 
  font-size: 12px; 
  color: #666;
}
.board { display: grid; grid-gap: 8px; margin: 12px 0; }
.cell { border: 1px solid #ccc; border-radius: 8px; background: white; cursor: pointer; }
.variant-selector { display: flex; gap: 16px; margin-bottom: 12px; padding: 8px; background: #e8f5e9; border-radius: 8px; border: 1px solid #a5d6a7; }
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
.validation-error { display: block; color: #d32f2f; font-size: 0.85em; margin-top: 4px; }
.setting-row { margin-bottom: 10px; }
.setting-row label { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-direction: row; }
.setting-row label span { flex: 1; }
.setting-row input[type="number"] { width: 100px; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; text-align: center; }
.setting-row input[type="number"]:invalid { border-color: #d32f2f; }
.setting-row input[type="number"]:disabled { background: #f5f5f5; cursor: not-allowed; }
.setting-hint { margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; }
.setting-hint small { color: #666; font-size: 0.85em; }
.status { opacity: .8; }
.status.game-over { 
  font-weight: bold; 
  font-size: 1.1em;
  opacity: 1;
}
.confidence-indicator { 
  display: inline-block; 
  padding: 4px 12px; 
  border-radius: 12px; 
  font-size: 0.85em; 
  font-weight: 500;
  margin-left: 8px;
  transition: all 0.3s ease;
}
.confidence-high { 
  background-color: #4caf50; 
  color: white; 
}
.confidence-medium { 
  background-color: #ffc107; 
  color: #333; 
}
.confidence-low { 
  background-color: #f44336; 
  color: white; 
}
.confidence-unknown { 
  background-color: #9e9e9e; 
  color: white; 
}
.gpu-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  margin-bottom: 12px;
  border-radius: 6px;
  font-size: 0.95em;
  font-weight: 500;
}
.gpu-status.gpu-active {
  background: #e8f5e9;
  border: 1px solid #4caf50;
  color: #2e7d32;
}
.gpu-status.gpu-inactive {
  background: #fff3e0;
  border: 1px solid #ff9800;
  color: #e65100;
}
.gpu-icon {
  font-size: 1.2em;
}
.gpu-text {
  flex: 1;
}
.gpu-backend {
  font-size: 0.85em;
  opacity: 0.8;
  font-family: monospace;
}

/* ===== TTT5 Interactive Training Panel ===== */
.ttt5-training-panel {
  margin-top: 12px;
  padding: 14px;
  background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
  border-radius: 10px;
  border: 1px solid #c5cae9;
}

.phase-badges {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.phase-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 14px;
  font-size: 0.8em;
  font-weight: 600;
  letter-spacing: 0.3px;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.phase-pending {
  background: #e0e0e0;
  color: #999;
}

.phase-active {
  background: #1976d2;
  color: white;
  animation: pulseBadge 1.5s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(25, 118, 210, 0.4);
}

.phase-completed {
  background: #4caf50;
  color: white;
}

@keyframes pulseBadge {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.85; transform: scale(1.03); }
}

.ttt5-bar {
  background: linear-gradient(90deg, #1976d2, #42a5f5) !important;
  transition: width 0.4s ease;
}

.training-counters {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-variant-numeric: tabular-nums;
}

.counter {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 56px;
}

.counter-label {
  font-size: 0.7em;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #888;
  margin-bottom: 2px;
}

.counter-value {
  font-size: 1.05em;
  font-weight: 700;
  color: #333;
  font-variant-numeric: tabular-nums;
}

.counter-total {
  font-weight: 400;
  color: #999;
  font-size: 0.9em;
}

.counter-unit {
  font-weight: 400;
  font-size: 0.8em;
  color: #888;
}

.epoch-detail {
  margin-top: 8px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.7);
  border-radius: 6px;
  font-size: 0.88em;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
}

.epoch-text {
  font-weight: 600;
  color: #333;
}

.batch-text {
  color: #777;
}

.metric-text {
  color: #555;
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 0.95em;
}

.acc-text {
  color: #2e7d32;
  font-weight: 600;
}

.selfplay-stats {
  margin-top: 8px;
  padding: 5px 10px;
  background: rgba(255,255,255,0.7);
  border-radius: 6px;
  font-size: 0.88em;
  color: #555;
}

.sp-win { color: #2e7d32; font-weight: 600; }
.sp-loss { color: #c62828; font-weight: 600; }
.sp-draw { color: #f57c00; font-weight: 600; }

.sparkline-container {
  margin-top: 10px;
}

.sparkline-legend {
  display: flex;
  gap: 14px;
  font-size: 0.72em;
  color: #888;
  margin-bottom: 4px;
}

.legend-acc { color: #4caf50; }
.legend-loss { color: #f44336; }

.sparkline-canvas {
  display: block;
  width: 320px;
  height: 50px;
  background: rgba(255,255,255,0.5);
  border-radius: 4px;
  border: 1px solid #e0e0e0;
}

.logs {
  margin-top: 8px;
  font-size: 0.9em;
  color: #555;
}

/* Training Charts */
.training-charts {
  margin-top: 10px;
  padding: 8px;
  background: white;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}
.uplot-chart {
  width: 100%;
  max-width: 640px;
}

/* Policy Heatmap */
.policy-heatmap-container {
  margin-top: 8px;
  text-align: center;
}
.heatmap-label {
  font-size: 11px;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.policy-heatmap {
  border: 1px solid #ddd;
  border-radius: 8px;
  display: block;
  margin: 0 auto;
}

/* Training Presets */
.preset-row {
  display: flex;
  gap: 6px;
  margin-bottom: 10px;
}
.preset-btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 8px 4px;
  border: 2px solid #e0e0e0;
  border-radius: 10px;
  background: white;
  cursor: pointer;
  transition: all 0.15s ease;
}
.preset-btn:hover:not(:disabled) {
  border-color: #4caf50;
  background: #f8fff8;
}
.preset-btn.active {
  border-color: #4caf50;
  background: #e8f5e9;
  box-shadow: 0 0 0 1px #4caf50;
}
.preset-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.preset-icon { font-size: 18px; }
.preset-name { font-size: 12px; font-weight: 600; color: #333; }
.preset-time { font-size: 10px; color: #888; }
.preset-custom .preset-name { color: #666; }
.preset-summary {
  padding: 6px 10px;
  background: #f5f5f5;
  border-radius: 6px;
  margin-bottom: 8px;
}
.preset-summary small { color: #666; }
.custom-settings {
  padding: 8px 0;
}

/* Training timer */
.training-timer {
  color: #1976d2;
  font-weight: 500;
}
</style>
