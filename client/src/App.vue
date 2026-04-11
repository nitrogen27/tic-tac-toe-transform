<template>
  <div class="app-shell">
    <!-- Header -->
    <header class="app-header">
      <div class="logo">Gomoku AI</div>
      <div class="variant-tabs">
        <button @click="setVariant('ttt3')" :class="{active: variant==='ttt3'}">3x3</button>
        <button @click="setVariant('ttt5')" :class="{active: variant==='ttt5'}">5x5</button>
      </div>
      <div class="header-right">
        <div class="gpu-pill" :class="{ 'gpu-active': gpuAvailable, 'gpu-inactive': !gpuAvailable }">
          <span class="gpu-icon">{{ gpuAvailable ? '🚀' : '💻' }}</span>
          <span class="gpu-text">{{ gpuAvailable ? 'GPU' : 'CPU' }}</span>
          <span class="gpu-backend" v-if="gpuBackend">({{ gpuBackend }})</span>
        </div>
        <div class="theme-toggle">
          <button @click="setTheme('light')" :class="{active: currentTheme==='light'}" title="Light">☀️</button>
          <button @click="setTheme('dark')" :class="{active: currentTheme==='dark'}" title="Dark">🌙</button>
          <button @click="setTheme('auto')" :class="{active: currentTheme==='auto'}" title="System">⚙️</button>
        </div>
      </div>
    </header>

    <!-- Main content: board + sidebar -->
    <div class="main-layout">
      <!-- Board area (LEFT on desktop, TOP on mobile) -->
      <section class="board-area">
        <!-- Commentary panel (separate card) -->
        <div class="card" v-if="gameType === 'human' && commentaryEnabled">
          <div class="commentary-panel">
            <div class="commentary-header">Коуч позиции</div>
            <div v-if="latestCommentary" class="commentary-card" :class="getCommentaryClass(latestCommentary.mood)">
              <div class="commentary-title">
                {{ latestCommentary.actor === 'bot' ? 'Разбор ответа бота' : 'Разбор вашего хода' }}
              </div>
              <div class="commentary-text">{{ latestCommentary.text }}</div>
              <div class="commentary-meta">
                <span v-if="latestCommentary.advantageLabel">Оценка: {{ latestCommentary.advantageLabel }}</span>
                <span v-if="latestCommentary.bestMoveLabel && latestCommentary.bestMoveLabel !== latestCommentary.moveLabel">Сильнее: {{ latestCommentary.bestMoveLabel }}</span>
                <span v-if="latestCommentary.opponentThreatsAfter > 0">Угроз после хода: {{ latestCommentary.opponentThreatsAfter }}</span>
                <span v-if="latestCommentary.forcingThreatsAfter > 0">Ваше давление: {{ latestCommentary.forcingThreatsAfter }}</span>
              </div>
            </div>
            <div v-else class="commentary-empty">
              После ходов здесь появится живой разбор: опасности, потеря темпа, перехват инициативы и подсказки.
            </div>
          </div>
        </div>
        <div class="card">
          <!-- Board grid with inline heatmap -->
          <div class="board-wrapper">
            <div class="board" :style="{ gridTemplateColumns: `repeat(${gridN}, 1fr)` }">
              <button v-for="(cell, idx) in board" :key="idx" class="cell"
                      :class="{
                        'cell-black': cell === 1,
                        'cell-white': cell === 2,
                        'cell-empty': cell === 0,
                        'cell-last': idx === lastMoveIdx,
                        'cell-win': winLine && winLine.includes(idx),
                      }"
                      :style="cellHeatStyle(idx)"
                      :disabled="cell!==0 || waiting || gameOver || autoPlaying || training"
                      @click="humanMove(idx)">
                <!-- X mark (SVG cross) -->
                <svg v-if="cell === 1" class="mark mark-x" viewBox="0 0 100 100">
                  <line x1="20" y1="20" x2="80" y2="80" stroke="currentColor" stroke-width="12" stroke-linecap="round"/>
                  <line x1="80" y1="20" x2="20" y2="80" stroke="currentColor" stroke-width="12" stroke-linecap="round"/>
                </svg>
                <!-- O mark (SVG circle) -->
                <svg v-else-if="cell === 2" class="mark mark-o" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="32" fill="none" stroke="currentColor" stroke-width="10" stroke-linecap="round"/>
                </svg>
                <!-- Ghost on hover -->
                <svg v-else class="mark mark-ghost" :class="{ 'ghost-visible': !(waiting || gameOver || autoPlaying || training) }" viewBox="0 0 100 100">
                  <line x1="20" y1="20" x2="80" y2="80" stroke="currentColor" stroke-width="12" stroke-linecap="round"/>
                  <line x1="80" y1="20" x2="20" y2="80" stroke="currentColor" stroke-width="12" stroke-linecap="round"/>
                </svg>
                <!-- Inline probability label (fades out) -->
                <span v-if="showHeatmap && cellProb(idx) > 0.01 && cell === 0 && !gameOver"
                      class="cell-prob" :class="{ 'prob-fade': probFading }">
                  {{ (cellProb(idx) * 100).toFixed(0) }}%
                </span>
              </button>
            </div>
            <!-- Win strike-through overlay -->
            <svg v-if="winLine && winLine.length > 0" class="win-overlay" :viewBox="`0 0 ${gridN} ${gridN}`" preserveAspectRatio="none">
              <line :x1="winLineCoords.x1" :y1="winLineCoords.y1"
                    :x2="winLineCoords.x2" :y2="winLineCoords.y2"
                    stroke="var(--accent-red)" stroke-width="0.12" stroke-linecap="round" opacity="0.85"/>
            </svg>
          </div>

          <!-- Heatmap toggle -->
          <label class="heatmap-toggle" v-if="lastProbs">
            <input type="checkbox" v-model="showHeatmap"> Policy Heatmap на доске
          </label>

          <!-- Game controls row -->
          <div class="game-controls-row">
            <button class="btn btn-secondary" @click="reset" :disabled="autoPlaying">Сброс</button>
            <button class="btn btn-primary" v-if="gameType === 'auto' && !autoPlaying && !gameOver" @click="startAutoGame">Начать игру</button>
            <button class="btn btn-danger" v-if="autoPlaying" @click="stopAutoGame">Остановить</button>
            <span class="status" :class="{ 'game-over': gameOver }">{{ status }}</span>
            <span v-if="modelConfidence !== null && modelConfidence !== undefined && (gameMode === 'model' || (gameType === 'auto' && current === 1))" class="confidence-indicator" :class="getConfidenceClass(modelConfidence)">
              Уверенность: {{ (modelConfidence * 100).toFixed(1) }}%
            </span>
          </div>
        </div>

        <!-- Game settings card -->
        <div class="card">
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
            <label class="mode-label">
              <input type="radio" v-model="gameMode" value="algorithm" :disabled="training || clearing" />
              <span>Алгоритм (minimax)</span>
            </label>
          </div>
          <div class="mode-selector" v-if="gameMode === 'model' && gameType === 'human'">
            <label class="mode-label">
              <input type="radio" v-model="modelDecisionMode" value="mcts" :disabled="training || clearing || autoPlaying" />
              <span>MCTS (думает)</span>
            </label>
            <label class="mode-label">
              <input type="radio" v-model="modelDecisionMode" value="hybrid" :disabled="training || clearing || autoPlaying" />
              <span>Гибрид (быстрый)</span>
            </label>
            <label class="mode-label">
              <input type="radio" v-model="modelDecisionMode" value="pure" :disabled="training || clearing || autoPlaying" />
              <span>Чистая модель</span>
            </label>
          </div>
          <div v-if="gameMode === 'model'" class="setting-hint game-hint">
            <small>
              <template v-if="modelDecisionMode === 'mcts'">
                MCTS: модель просчитывает 50 вариантов вперёд. Сильнее, но чуть медленнее (~100ms).
              </template>
              <template v-else-if="modelDecisionMode === 'hybrid'">
                Гибридный режим: сеть выбирает ход, но сервер страхует от простых тактических зевков.
              </template>
              <template v-else>
                Чистый режим: ход выбирается только моделью (policy + value) без программных tactical overrides.
              </template>
            </small>
          </div>
          <div class="pause-control" v-if="gameType === 'auto'">
            <label class="pause-label">
              Пауза между ходами (мс):
              <input type="number" v-model.number="pauseMs" :disabled="autoPlaying" min="0" max="5000" step="100" />
            </label>
          </div>
          <div class="commentary-controls" v-if="gameType === 'human'">
            <label class="mode-label">
              <input type="checkbox" v-model="commentaryEnabled" />
              <span>Комментарии к ходам</span>
            </label>
            <label class="pause-label" v-if="commentaryEnabled">
              Стиль:
              <select v-model="commentaryStyle">
                <option value="coach">Коуч</option>
                <option value="emotional">Эмоции</option>
                <option value="hint">Подсказки</option>
              </select>
            </label>
          </div>
        </div>

      </section>

      <!-- Sidebar area (RIGHT on desktop, BOTTOM on mobile) -->
      <aside class="sidebar-area">
        <!-- GPU telemetry card -->
        <div class="card" v-if="gpuTelemetry">
          <div class="gpu-live-stats">
            <span v-if="gpuTelemetry.gpuUtilization !== null && gpuTelemetry.gpuUtilization !== undefined">GPU {{ gpuTelemetry.gpuUtilization }}%</span>
            <span v-if="gpuTelemetry.gpuPowerW !== null && gpuTelemetry.gpuPowerW !== undefined">{{ gpuTelemetry.gpuPowerW }} W</span>
            <span v-if="gpuTelemetry.gpuMemoryUsedMB !== null && gpuTelemetry.gpuMemoryUsedMB !== undefined">
              VRAM {{ gpuTelemetry.gpuMemoryUsedMB }}/{{ gpuTelemetry.gpuMemoryTotalMB || '?' }} MB
            </span>
            <span v-if="gpuTelemetry.gpuTemperatureC !== null && gpuTelemetry.gpuTemperatureC !== undefined">{{ gpuTelemetry.gpuTemperatureC }}°C</span>
          </div>
        </div>

        <!-- Training meta strip -->
        <div class="card" v-if="trainingMeta && training">
          <div class="training-meta-strip">
            <span>{{ trainingMeta.variant || variant }}</span>
            <span v-if="trainingMeta.boardSize">{{ trainingMeta.boardSize }}x{{ trainingMeta.boardSize }}</span>
            <span v-if="trainingMeta.batchSize">batch {{ trainingMeta.batchSize }}</span>
            <span v-if="trainingMeta.deviceName">{{ trainingMeta.deviceName }}</span>
            <span v-if="trainingMeta.modelParams">params {{ formatParamCount(trainingMeta.modelParams) }}</span>
          </div>
        </div>

        <!-- Training controls + presets (compact single card) -->
        <div class="card">
          <div class="compact-train-header">
            <div class="button-group">
              <button class="btn btn-primary btn-sm" :disabled="training || clearing" @click="startTrain">Обучить</button>
              <button class="btn btn-danger btn-sm" v-if="training" :disabled="clearing || cancellingTraining" @click="cancelTraining">
                {{ cancellingTraining ? 'Остановка...' : 'Стоп' }}
              </button>
              <button class="btn btn-secondary btn-sm" :disabled="training || clearing" @click="trainOnGames" :class="historyCount >= 10 ? '' : 'disabled-btn'">Дообучить</button>
              <button class="btn btn-danger btn-sm" :disabled="training || clearing" @click="clearModel">Очистить</button>
              <button class="btn btn-secondary btn-sm" v-if="variant === 'ttt5'" :disabled="training || generatingDataset" @click="generateDataset">{{ generatingDataset ? '...' : 'Dataset' }}</button>
            </div>
            <div class="preset-row-compact">
              <button class="preset-pill" :class="{ active: activePreset === 'light' }" :disabled="training || clearing" @click="applyPreset('light')">⚡ 2м</button>
              <button class="preset-pill" :class="{ active: activePreset === 'medium' }" :disabled="training || clearing" @click="applyPreset('medium')">⚙️ 5м</button>
              <button class="preset-pill" :class="{ active: activePreset === 'deep' }" :disabled="training || clearing" @click="applyPreset('deep')">🧠 15м</button>
              <button class="preset-pill preset-selfplay" :class="{ active: activePreset === 'selfplay' }" :disabled="training || clearing" @click="applyPreset('selfplay')">⚔️ Self-Play</button>
              <button class="preset-pill" :class="{ active: activePreset === 'custom' }" :disabled="training || clearing" @click="activePreset = 'custom'">🔧</button>
            </div>
          </div>
          <div class="compact-train-meta">
            <small>{{ presetDescription }}</small>
            <span v-if="historyCount > 0" class="history-badge">{{ historyCount }} ходов <a href="#" @click.prevent="clearHistory">×</a></span>
          </div>
          <label class="auto-train-check">
            <input type="checkbox" v-model="autoTrainAfterGame" />
            <small>Авто-дообучение после игр</small>
          </label>
          <div v-if="activePreset === 'custom'" class="custom-settings-compact">
            <label>Эпохи: <input type="number" v-model.number="mainTrainingEpochs" min="1" max="50" :disabled="training" /></label>
            <label>Батч: <input type="number" v-model.number="mainTrainingBatchSize" min="32" max="4096" step="32" :disabled="training" /></label>
          </div>
        </div>

        <!-- Fine-tuning settings (collapsed by default) -->
        <details class="card details-card">
          <summary class="card-title-sm">Настройки дообучения</summary>
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
        </details>

        <!-- User Game Corpus status -->
        <div v-if="corpusStatus || corpusAnalyzing" class="card corpus-card">
          <div class="corpus-header">
            <span class="corpus-title">Корпус партий</span>
            <span v-if="corpusAnalyzing" class="corpus-spinner">⏳</span>
            <span v-else class="corpus-ready">✓</span>
          </div>
          <div v-if="corpusStatus" class="corpus-buckets">
            <div class="corpus-bucket">
              <span class="bucket-count">{{ corpusStatus.recentCount }}</span>
              <span class="bucket-label">Недавние</span>
            </div>
            <div class="corpus-bucket bucket-mistakes">
              <span class="bucket-count">{{ corpusStatus.hardMistakeCount }}</span>
              <span class="bucket-label">Ошибки</span>
            </div>
            <div class="corpus-bucket bucket-conversion">
              <span class="bucket-count">{{ corpusStatus.conversionCount }}</span>
              <span class="bucket-label">Упущения</span>
            </div>
            <div class="corpus-bucket bucket-weak">
              <span class="bucket-count">{{ corpusStatus.weakSideCount }}</span>
              <span class="bucket-label">Слабая сторона</span>
            </div>
          </div>
          <div v-if="corpusLastEvent" class="corpus-event">
            <small>{{ corpusLastEvent }}</small>
          </div>
        </div>

        <!-- Dataset progress card -->
        <div v-if="datasetProgress" class="card">
          <div class="dataset-header">
            <strong>Генерация датасета</strong>
            <span v-if="datasetProgress.workers" class="workers-badge">{{ datasetProgress.workers }} воркеров</span>
          </div>
          <div class="progress">
            <div class="bar dataset-bar" :style="{ width: (datasetProgress.percent||0)+'%' }"></div>
          </div>
          <div class="dataset-info">
            <span>{{ datasetProgress.generated || 0 }} / {{ datasetProgress.total || 0 }} {{ datasetProgress.unit || 'samples' }} ({{ datasetProgress.percent || 0 }}%)</span>
            <span v-if="datasetProgress.games" class="rate-info">games: {{ datasetProgress.games }}</span>
            <span v-if="datasetProgress.rate" class="rate-info">{{ datasetProgress.rate }} {{ datasetProgress.unit || 'samples' }}/s</span>
            <span v-if="datasetProgress.elapsed" class="time-info">{{ datasetProgress.elapsed }}s</span>
          </div>
          <div v-if="datasetProgress.message" class="dataset-stage">
            {{ datasetProgress.message }}
          </div>
        </div>

        <!-- TTT5 Training progress card -->
        <div v-if="ttt5Progress && ttt5Progress.phase && !datasetProgress && !backgroundProgress" class="card ttt5-training-panel">
          <div class="overall-progress-panel">
            <div class="overall-progress-header">
              <strong>Общий прогресс</strong>
              <span class="heartbeat-pill" :class="isHeartbeatFresh(ttt5Progress) ? 'heartbeat-fresh' : 'heartbeat-stale'">
                {{ getHeartbeatLabel(ttt5Progress) }}
              </span>
            </div>
            <div class="progress overall-progress">
              <div class="bar ttt5-bar overall-bar" :style="{ width: getOverallPercent(ttt5Progress) + '%' }"></div>
            </div>
            <div class="overall-progress-meta">
              <span class="overall-progress-value">{{ getOverallPercent(ttt5Progress).toFixed(1) }}%</span>
              <span class="overall-progress-text">{{ getOverallDetailText(ttt5Progress) }}</span>
            </div>
          </div>

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

          <div class="progress">
            <div class="bar ttt5-bar" :style="{ width: (ttt5Progress.percent || 0) + '%' }"></div>
          </div>

          <div class="phase-detail" v-if="getPhaseDetail(ttt5Progress)">
            {{ getPhaseDetail(ttt5Progress) }}
          </div>

          <div class="training-counters">
            <div class="counter" v-if="ttt5Progress.cycle > 0 || ttt5Progress.iteration > 0">
              <span class="counter-label">Cycle</span>
              <span class="counter-value">{{ ttt5Progress.cycle || ttt5Progress.iteration || 0 }}<span class="counter-total">/{{ ttt5Progress.totalCycles || ttt5Progress.totalIterations || 1 }}</span></span>
            </div>
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

          <div v-if="ttt5Progress.epoch > 0" class="epoch-detail">
            <span class="epoch-text">Epoch {{ ttt5Progress.epoch }}/{{ ttt5Progress.totalEpochs }}</span>
            <span v-if="ttt5Progress.batch > 0" class="batch-text">(batch {{ ttt5Progress.batch }}/{{ ttt5Progress.totalBatches }})</span>
            <span v-if="ttt5Progress.loss" class="metric-text">loss: {{ ttt5Progress.loss }}</span>
            <span v-if="ttt5Progress.policyTop1Acc" class="metric-text acc-text">policyAcc: {{ ttt5Progress.policyTop1Acc }}%</span>
            <span v-else-if="ttt5Progress.accuracy" class="metric-text acc-text">policyAcc: {{ ttt5Progress.accuracy }}%</span>
            <span v-if="ttt5Progress.teacherMassOnPred" class="metric-text">tMass: {{ ttt5Progress.teacherMassOnPred }}</span>
            <span v-if="ttt5Progress.policyKL" class="metric-text">KL: {{ ttt5Progress.policyKL }}</span>
            <span v-if="ttt5Progress.mae" class="metric-text">MAE: {{ ttt5Progress.mae }}</span>
            <span v-if="ttt5Progress.valueSignAgreement" class="metric-text">vSign: {{ ttt5Progress.valueSignAgreement }}%</span>
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

          <div v-if="ttt5Progress.selfPlayStats && hasSelfPlayStats(ttt5Progress.selfPlayStats)" class="selfplay-stats">
            Data gen:
            <span class="sp-win">P1 {{ getSelfPlayStat(ttt5Progress.selfPlayStats, 'wins') }}</span> /
            <span class="sp-loss">P2 {{ getSelfPlayStat(ttt5Progress.selfPlayStats, 'losses') }}</span> /
            <span class="sp-draw">D {{ getSelfPlayStat(ttt5Progress.selfPlayStats, 'draws') }}</span>
          </div>

          <!-- Strength metrics -->
          <div v-if="
            ttt5Progress.winrateVsChampion != null ||
            ttt5Progress.winrateVsAlgorithm != null ||
            ttt5Progress.promotionDecision != null ||
            (ttt5Progress.holdoutPolicyAcc ?? trainingMeta?.holdoutPolicyAcc) != null ||
            (ttt5Progress.frozenBlockAcc ?? trainingMeta?.frozenBlockAcc) != null
          " class="strength-block">
            <div class="strength-header">Strength</div>
            <div class="strength-metrics">
              <span v-if="ttt5Progress.winrateVsChampion != null" class="strength-pill">
                vs champion: {{ (ttt5Progress.winrateVsChampion * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.winrateVsAlgorithm != null" class="strength-pill">
                vs engine: {{ (ttt5Progress.winrateVsAlgorithm * 100).toFixed(1) }}%
              </span>
              <span v-if="(ttt5Progress.servingSource ?? trainingMeta?.servingSource) != null" class="strength-pill">
                serving: {{ ttt5Progress.servingSource ?? trainingMeta?.servingSource }}
                <template v-if="(ttt5Progress.servingGeneration ?? trainingMeta?.servingGeneration) != null">
                  #{{ ttt5Progress.servingGeneration ?? trainingMeta?.servingGeneration }}
                </template>
              </span>
              <span v-if="ttt5Progress.decisiveWinRate != null" class="strength-pill">
                decisive: {{ (ttt5Progress.decisiveWinRate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.drawRate != null" class="strength-pill">
                draw: {{ (ttt5Progress.drawRate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.winrateAsP1 != null" class="strength-pill">
                as P1: {{ (ttt5Progress.winrateAsP1 * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.winrateAsP2 != null" class="strength-pill">
                as P2: {{ (ttt5Progress.winrateAsP2 * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.balancedSideWinrate != null" class="strength-pill">
                balanced: {{ (ttt5Progress.balancedSideWinrate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.tacticalOverrideRate != null" class="strength-pill">
                overrides: {{ (ttt5Progress.tacticalOverrideRate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.valueGuidedRate != null" class="strength-pill">
                value-guided: {{ (ttt5Progress.valueGuidedRate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.modelPolicyRate != null" class="strength-pill">
                model-policy: {{ (ttt5Progress.modelPolicyRate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.deltaWinrate != null" class="strength-pill">
                delta: {{ (ttt5Progress.deltaWinrate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.progressTrend" class="strength-pill">
                trend: {{ ttt5Progress.progressTrend }}
              </span>
              <span v-if="ttt5Progress.arenaWins != null" class="strength-pill">
                arena: W{{ ttt5Progress.arenaWins }}/L{{ ttt5Progress.arenaLosses }}/D{{ ttt5Progress.arenaDraws }}
              </span>
              <span v-if="ttt5Progress.failureBankSize != null" class="strength-pill">
                failures: {{ ttt5Progress.failureBankSize }}
              </span>
              <span v-if="ttt5Progress.fixedErrors != null" class="strength-pill">
                fixed: {{ ttt5Progress.fixedErrors }}
              </span>
              <span v-if="ttt5Progress.regressedErrors != null" class="strength-pill">
                regressed: {{ ttt5Progress.regressedErrors }}
              </span>
              <span v-if="ttt5Progress.correctedRate != null" class="strength-pill">
                corrected: {{ (ttt5Progress.correctedRate * 100).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.tacticalAccuracy != null" class="strength-pill">
                tactical: {{ ttt5Progress.tacticalAccuracy }}%
              </span>
              <span v-if="(ttt5Progress.holdoutPolicyAcc ?? trainingMeta?.holdoutPolicyAcc) != null" class="strength-pill">
                holdout: {{ (ttt5Progress.holdoutPolicyAcc ?? trainingMeta?.holdoutPolicyAcc).toFixed(1) }}%
              </span>
              <span v-if="(ttt5Progress.holdoutPolicyKL ?? trainingMeta?.holdoutPolicyKL) != null" class="strength-pill">
                holdout KL: {{ (ttt5Progress.holdoutPolicyKL ?? trainingMeta?.holdoutPolicyKL).toFixed(4) }}
              </span>
              <span v-if="(ttt5Progress.frozenBlockAcc ?? trainingMeta?.frozenBlockAcc) != null" class="strength-pill">
                frozen block: {{ (ttt5Progress.frozenBlockAcc ?? trainingMeta?.frozenBlockAcc).toFixed(1) }}%
              </span>
              <span v-if="(ttt5Progress.frozenWinAcc ?? trainingMeta?.frozenWinAcc) != null" class="strength-pill">
                frozen win: {{ (ttt5Progress.frozenWinAcc ?? trainingMeta?.frozenWinAcc).toFixed(1) }}%
              </span>
              <span v-if="(ttt5Progress.frozenExactAcc ?? trainingMeta?.frozenExactAcc) != null" class="strength-pill">
                frozen exact: {{ (ttt5Progress.frozenExactAcc ?? trainingMeta?.frozenExactAcc).toFixed(1) }}%
              </span>
              <span v-if="(ttt5Progress.frozenMidAcc ?? trainingMeta?.frozenMidAcc) != null" class="strength-pill">
                frozen mid: {{ (ttt5Progress.frozenMidAcc ?? trainingMeta?.frozenMidAcc).toFixed(1) }}%
              </span>
              <span v-if="(ttt5Progress.frozenLateAcc ?? trainingMeta?.frozenLateAcc) != null" class="strength-pill">
                frozen late: {{ (ttt5Progress.frozenLateAcc ?? trainingMeta?.frozenLateAcc).toFixed(1) }}%
              </span>
              <span v-if="ttt5Progress.promotionDecision != null" class="strength-pill" :class="ttt5Progress.promoted ? 'promoted' : 'rejected'">
                {{ ttt5Progress.promoted ? 'PROMOTED' : 'not promoted' }}
              </span>
            </div>
            <div v-if="
              (ttt5Progress.pureFrozenWinRecall ?? trainingMeta?.pureFrozenWinRecall) != null ||
              (ttt5Progress.pureFrozenBlockRecall ?? trainingMeta?.pureFrozenBlockRecall) != null ||
              (ttt5Progress.pureExactTrapRecall ?? trainingMeta?.pureExactTrapRecall) != null
            " class="strength-subblock">
              <div class="strength-subheader">Pure</div>
              <div class="strength-metrics">
                <span v-if="(ttt5Progress.pureFrozenWinRecall ?? trainingMeta?.pureFrozenWinRecall) != null" class="strength-pill">
                  win recall: {{ (ttt5Progress.pureFrozenWinRecall ?? trainingMeta?.pureFrozenWinRecall).toFixed(1) }}%
                </span>
                <span v-if="(ttt5Progress.pureFrozenBlockRecall ?? trainingMeta?.pureFrozenBlockRecall) != null" class="strength-pill">
                  block recall: {{ (ttt5Progress.pureFrozenBlockRecall ?? trainingMeta?.pureFrozenBlockRecall).toFixed(1) }}%
                </span>
                <span v-if="(ttt5Progress.pureExactTrapRecall ?? trainingMeta?.pureExactTrapRecall) != null" class="strength-pill">
                  exact traps: {{ (ttt5Progress.pureExactTrapRecall ?? trainingMeta?.pureExactTrapRecall).toFixed(1) }}%
                </span>
              </div>
            </div>
            <div v-if="
              (ttt5Progress.hybridFrozenWinRecall ?? trainingMeta?.hybridFrozenWinRecall) != null ||
              (ttt5Progress.hybridFrozenBlockRecall ?? trainingMeta?.hybridFrozenBlockRecall) != null ||
              (ttt5Progress.hybridExactTrapRecall ?? trainingMeta?.hybridExactTrapRecall) != null
            " class="strength-subblock">
              <div class="strength-subheader">Hybrid</div>
              <div class="strength-metrics">
                <span v-if="(ttt5Progress.hybridFrozenWinRecall ?? trainingMeta?.hybridFrozenWinRecall) != null" class="strength-pill">
                  win recall: {{ (ttt5Progress.hybridFrozenWinRecall ?? trainingMeta?.hybridFrozenWinRecall).toFixed(1) }}%
                </span>
                <span v-if="(ttt5Progress.hybridFrozenBlockRecall ?? trainingMeta?.hybridFrozenBlockRecall) != null" class="strength-pill">
                  block recall: {{ (ttt5Progress.hybridFrozenBlockRecall ?? trainingMeta?.hybridFrozenBlockRecall).toFixed(1) }}%
                </span>
                <span v-if="(ttt5Progress.hybridExactTrapRecall ?? trainingMeta?.hybridExactTrapRecall) != null" class="strength-pill">
                  exact traps: {{ (ttt5Progress.hybridExactTrapRecall ?? trainingMeta?.hybridExactTrapRecall).toFixed(1) }}%
                </span>
              </div>
            </div>
          </div>

          <!-- Training Charts -->
          <div v-if="metricsHistory.length > 1" class="training-charts">
            <div ref="lossChartEl" class="uplot-chart"></div>
          </div>

          <!-- Winrate Chart -->
          <div v-if="hasWinrateChartData" class="training-charts winrate-chart-block">
            <div class="winrate-chart-meta">
              <span class="winrate-legend-item">
                <span class="winrate-dot quick"></span>
                Quick probe
              </span>
              <span class="winrate-legend-item">
                <span class="winrate-dot confirm"></span>
                Confirm exam
              </span>
            </div>
            <div v-if="latestQuickProbe || latestConfirmExam" class="winrate-chart-summary">
              <span v-if="latestQuickProbe" class="strength-pill">
                {{ formatExamSummary(latestQuickProbe, 'quick') }}
              </span>
              <span v-if="latestConfirmExam" class="strength-pill promoted">
                {{ formatExamSummary(latestConfirmExam, 'confirm') }}
              </span>
            </div>
            <div ref="winrateChartEl" class="uplot-chart"></div>
          </div>
        </div>

        <!-- Legacy progress card -->
        <div class="card" v-if="progress && !ttt5Progress?.phase && !datasetProgress && !backgroundProgress">
          <div class="progress">
            <div class="bar" :style="{ width: (progress.percent||0)+'%' }"></div>
          </div>
          <div class="logs">
            Эпоха {{progress.epoch}} / {{progress.epochs}} ·
            loss: {{progress.loss}} · acc: {{progress.acc}} ·
            <span v-if="progress.accuracy">Accuracy: {{progress.accuracy}}% · MAE: {{progress.mae}}</span>
            <span v-if="trainingElapsed > 0" class="training-timer">
              · {{ formatTime(trainingElapsed) }}
              <span v-if="trainingETA > 0"> · ETA ~{{ formatTime(trainingETA) }}</span>
            </span>
            <span v-else>val_loss: {{progress.val_loss}} · val_acc: {{progress.val_acc}}</span>
          </div>
        </div>

        <!-- Background training card -->
        <div v-if="backgroundProgress" class="card background-training">
          <div class="bg-train-header">
            <strong>Фоновое дообучение</strong>
            <span class="bg-train-epochs">{{backgroundProgress.epoch}}/{{backgroundProgress.epochs}} эпох</span>
          </div>
          <div class="overall-progress-panel">
            <div class="overall-progress-header">
              <strong>Общий прогресс</strong>
              <span class="heartbeat-pill" :class="isHeartbeatFresh(backgroundProgress) ? 'heartbeat-fresh' : 'heartbeat-stale'">
                {{ getHeartbeatLabel(backgroundProgress) }}
              </span>
            </div>
            <div class="progress overall-progress">
              <div class="bar bg-train-bar overall-bar" :style="{ width: getOverallPercent(backgroundProgress) + '%' }"></div>
            </div>
            <div class="overall-progress-meta">
              <span class="overall-progress-value">{{ getOverallPercent(backgroundProgress).toFixed(1) }}%</span>
              <span class="overall-progress-text">{{ getOverallDetailText(backgroundProgress) }}</span>
            </div>
          </div>
          <div class="progress">
            <div class="bar bg-train-bar" :style="{ width: (backgroundProgress.epochPercent||0)+'%' }"></div>
          </div>
          <div class="bg-train-details">
            <div>
              <strong>Новые навыки:</strong> {{backgroundProgress.newSkills}} из {{backgroundProgress.totalSkills}}
              <span class="accent-text">({{backgroundProgress.newSkillsPercent}}%)</span>
            </div>
            <div>
              <strong>Прогресс:</strong> {{backgroundProgress.epochPercent}}%
              <span v-if="backgroundProgress.batchProgress && backgroundProgress.batchesPerEpoch" class="secondary-text">
                (батч {{backgroundProgress.currentBatch}}/{{backgroundProgress.batchesPerEpoch}}: {{backgroundProgress.batchProgress}}%)
              </span>
              <span v-if="backgroundProgress.loss" class="secondary-text">loss: {{backgroundProgress.loss}}</span>
              <span v-if="backgroundProgress.acc" class="secondary-text">acc: {{backgroundProgress.acc}}</span>
            </div>
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, computed, watch, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

const currentTheme = ref(localStorage.getItem('gomoku-theme') || 'auto')

function setTheme(theme) {
  currentTheme.value = theme
  localStorage.setItem('gomoku-theme', theme)
  applyTheme()
}

function applyTheme() {
  const html = document.documentElement
  if (currentTheme.value === 'dark') {
    html.setAttribute('data-theme', 'dark')
  } else if (currentTheme.value === 'light') {
    html.setAttribute('data-theme', 'light')
  } else {
    html.removeAttribute('data-theme')
  }
}

function setVariant(v) {
  variant.value = v
  const knownState = trainingStateByVariant.value?.[v]
  if (knownState) {
    applyTrainingStatus(knownState)
  } else {
    requestTrainingStatus(v)
  }
}

const ws = ref(null)
const training = ref(false)
const cancellingTraining = ref(false)
const clearing = ref(false)
const generatingDataset = ref(false)
const progress = ref(null)
const variant = ref('ttt3') // 'ttt3' или 'ttt5'
const trainingStateByVariant = ref({})
const boardSize = computed(() => variant.value === 'ttt5' ? 25 : 9)
const gridN = computed(() => variant.value === 'ttt5' ? 5 : 3)
const cellSize = computed(() => variant.value === 'ttt5' ? 60 : 80)
const winLen = computed(() => variant.value === 'ttt5' ? 4 : 3)
const board = ref(Array(9).fill(0))
const current = ref(1) // 1 = X (человек/модель), 2 = O (бот/алгоритм)
const waiting = ref(false) // Флаг ожидания ответа от сервера
const status = ref('Ваш ход (X)')
const gameMode = ref('model') // 'model' или 'algorithm' (для режима human)
const modelDecisionMode = ref('hybrid') // 'mcts', 'hybrid' или 'pure' для model-mode
const gameType = ref('human') // 'human' или 'auto'
const gameOver = ref(false) // Игра завершена
const lastMoveIdx = ref(-1) // Last played move index
const winLine = ref(null) // Array of winning cell indices
const showHeatmap = ref(true) // Inline heatmap on board
const probFading = ref(false) // Fade-out animation trigger

// Corpus status
const corpusStatus = ref(null) // { recentCount, hardMistakeCount, conversionCount, weakSideCount, ... }
const corpusAnalyzing = ref(false) // analysis in progress
const corpusLastEvent = ref('') // last event description

function requestTrainingStatus(targetVariant = variant.value) {
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({ type: 'get_training_status', payload: { variant: targetVariant } }))
  }
}

function applyTrainingStatus(payload) {
  if (!payload || !payload.variant) return

  trainingStateByVariant.value = {
    ...trainingStateByVariant.value,
    [payload.variant]: payload,
  }

  if (payload.variant !== variant.value) return

  const trainActive = Boolean(payload.active)
  const backgroundActive = Boolean(payload.backgroundActive)
  const phase = payload.phase || payload.payload?.phase || ''
  const stage = payload.stage || payload.payload?.stage || ''
  const nextProgress = { ...(payload.payload || {}) }
  for (const key of ['overallPercent', 'overallEta', 'heartbeatTs', 'secondsSinceUpdate', 'heartbeatFresh', 'eventTs']) {
    if (payload[key] != null) nextProgress[key] = payload[key]
  }
  if (payload.message && !nextProgress.message) nextProgress.message = payload.message

  training.value = trainActive
  if (!trainActive) {
    cancellingTraining.value = false
  }

  if (Object.keys(nextProgress).length > 0) {
    ttt5Progress.value = { ...(ttt5Progress.value || {}), ...nextProgress }
    trainingMeta.value = { ...(trainingMeta.value || {}), ...nextProgress }
    applyGpuTelemetry(nextProgress)
  }

  if (backgroundActive && Object.keys(nextProgress).length > 0) {
    backgroundProgress.value = {
      ...(backgroundProgress.value || {}),
      ...nextProgress,
      epoch: nextProgress.epoch || backgroundProgress.value?.epoch || 1,
      epochs: nextProgress.epochs || backgroundProgress.value?.epochs || 1,
      epochPercent: nextProgress.epochPercent ?? nextProgress.overallPercent ?? backgroundProgress.value?.epochPercent ?? 0,
      message: nextProgress.message || backgroundProgress.value?.message,
    }
  }

  if (Array.isArray(payload.metricsHistory) && payload.metricsHistory.length > 0) {
    metricsHistory.value = [...payload.metricsHistory]
  }

  if (Array.isArray(payload.winrateHistory) && payload.winrateHistory.length > 0) {
    winrateHistory.value = [...payload.winrateHistory]
      .map(item => ({
        cycle: Number(item.cycle || 0),
        winrate: Number(item.winrate || 0),
        decisiveWinRate: item.decisiveWinRate != null ? Number(item.decisiveWinRate || 0) : null,
        drawRate: item.drawRate != null ? Number(item.drawRate || 0) : null,
        winrateAsP1: item.winrateAsP1 != null ? Number(item.winrateAsP1 || 0) : null,
        winrateAsP2: item.winrateAsP2 != null ? Number(item.winrateAsP2 || 0) : null,
        balancedSideWinrate: item.balancedSideWinrate != null ? Number(item.balancedSideWinrate || 0) : null,
        tacticalOverrideRate: item.tacticalOverrideRate != null ? Number(item.tacticalOverrideRate || 0) : null,
        valueGuidedRate: item.valueGuidedRate != null ? Number(item.valueGuidedRate || 0) : null,
        modelPolicyRate: item.modelPolicyRate != null ? Number(item.modelPolicyRate || 0) : null,
        wins: Number(item.wins || 0),
        losses: Number(item.losses || 0),
        draws: Number(item.draws || 0),
      }))
      .sort((a, b) => a.cycle - b.cycle)
  }

  if (Array.isArray(payload.confirmWinrateHistory) && payload.confirmWinrateHistory.length > 0) {
    confirmWinrateHistory.value = [...payload.confirmWinrateHistory]
      .map(item => ({
        cycle: Number(item.cycle || 0),
        winrate: Number(item.winrate || 0),
        decisiveWinRate: item.decisiveWinRate != null ? Number(item.decisiveWinRate || 0) : null,
        drawRate: item.drawRate != null ? Number(item.drawRate || 0) : null,
        winrateAsP1: item.winrateAsP1 != null ? Number(item.winrateAsP1 || 0) : null,
        winrateAsP2: item.winrateAsP2 != null ? Number(item.winrateAsP2 || 0) : null,
        balancedSideWinrate: item.balancedSideWinrate != null ? Number(item.balancedSideWinrate || 0) : null,
        tacticalOverrideRate: item.tacticalOverrideRate != null ? Number(item.tacticalOverrideRate || 0) : null,
        valueGuidedRate: item.valueGuidedRate != null ? Number(item.valueGuidedRate || 0) : null,
        modelPolicyRate: item.modelPolicyRate != null ? Number(item.modelPolicyRate || 0) : null,
        wins: Number(item.wins || 0),
        losses: Number(item.losses || 0),
        draws: Number(item.draws || 0),
      }))
      .sort((a, b) => a.cycle - b.cycle)
  }

  if (trainActive) {
    if (!trainingStartTime.value) trainingStartTime.value = Date.now()
    status.value = payload.message || getOverallDetailText(nextProgress) || `Обучение активно${phase ? `: ${phase}` : ''}`
    return
  }

  if (backgroundActive) {
    status.value = payload.message || getOverallDetailText(nextProgress) || `Фоновая оценка активна${stage ? `: ${stage}` : ''}`
    return
  }

  if (payload.message) {
    status.value = payload.message
  }
}

// Get winning line indices
function getWinningLine(brd) {
  const N = gridN.value
  const wLen = winLen.value
  const DIRS = [[1,0],[0,1],[1,1],[1,-1]]
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const who = brd[r * N + c]
      if (!who) continue
      for (const [dr, dc] of DIRS) {
        const cells = [r * N + c]
        let rr = r + dr, cc = c + dc
        while (rr >= 0 && cc >= 0 && rr < N && cc < N && brd[rr * N + cc] === who) {
          cells.push(rr * N + cc)
          rr += dr; cc += dc
          if (cells.length >= wLen) return cells
        }
      }
    }
  }
  return null
}

// Win line SVG coordinates (center of first and last cell)
const winLineCoords = computed(() => {
  if (!winLine.value || winLine.value.length === 0) return { x1: 0, y1: 0, x2: 0, y2: 0 }
  const N = gridN.value
  const first = winLine.value[0]
  const last = winLine.value[winLine.value.length - 1]
  return {
    x1: (first % N) + 0.5,
    y1: Math.floor(first / N) + 0.5,
    x2: (last % N) + 0.5,
    y2: Math.floor(last / N) + 0.5,
  }
})

// Cell probability from lastProbs
function cellProb(idx) {
  if (!lastProbs.value) return 0
  return lastProbs.value[idx] || 0
}

// Cell heatmap background style
function cellHeatStyle(idx) {
  if (!showHeatmap.value || !lastProbs.value || gameOver.value) return {}
  const prob = cellProb(idx)
  if (prob < 0.01 || board.value[idx] !== 0) return {}
  const maxP = Math.max(...lastProbs.value, 0.01)
  const norm = prob / maxP
  // Green gradient: low=transparent, high=green glow
  const alpha = Math.max(0.05, norm * 0.35)
  const g = Math.round(120 + 135 * norm)
  return { backgroundColor: `rgba(40, ${g}, 80, ${alpha})` }
}

// Trigger prob fade-out after bot move
function triggerProbFade() {
  probFading.value = false
  setTimeout(() => { probFading.value = true }, 2000) // start fading after 2s
}
const autoPlaying = ref(false) // Автоматическая игра идет
const pauseMs = ref(1000) // Пауза между ходами в мс
const commentaryEnabled = ref(true)
const commentaryStyle = ref('coach')
const commentaryEntries = ref([])
let autoGameInterval = null
const historyCount = ref(0) // Количество сохраненных ходов
const autoTrainAfterGame = ref(false) // Автоматическое дообучение после игры
// Настройки основного обучения
const mainTrainingEpochs = ref(30)
const mainTrainingBatchSize = ref(1024)
const activePreset = ref('medium') // 'light', 'medium', 'deep', 'custom'

const PRESETS = {
  light:  { epochs: 10, batchSize: 512,  bootstrapGames: 50,  mctsIterations: 2, mctsGamesPerIter: 20,  label: 'Лёгкое', desc: '10 эпох · batch 512 · 50 teacher-pos · 2 коротких exam cycles' },
  medium: { epochs: 25, batchSize: 1024, bootstrapGames: 100, mctsIterations: 3, mctsGamesPerIter: 40,  label: 'Среднее', desc: '25 эпох · batch 1024 · 100 teacher-pos · 3 exam/repair cycles' },
  deep:   { epochs: 50, batchSize: 2048, bootstrapGames: 200, mctsIterations: 5, mctsGamesPerIter: 100, label: 'Глубокое', desc: '50 эпох · batch 2048 · 200 teacher-pos · 5 exam/repair cycles' },
  selfplay: { epochs: 25, batchSize: 1024, bootstrapGames: 100, mctsIterations: 3, mctsGamesPerIter: 40, selfPlay: true, selfPlayIterations: 15, selfPlayGames: 200, selfPlaySims: 200, selfPlayTrainSteps: 200, label: 'Self-Play', desc: 'MCTS self-play · 15 итераций × 200 игр × 200 sims · ~60-90 мин' },
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
let trainingStatusPollInterval = null

const metricsHistory = ref([]) // For training charts
const winrateHistory = ref([]) // Winrate vs engine per exam cycle
const confirmWinrateHistory = ref([]) // Confirm exam winrate (post-repair / final)
const lossChartEl = ref(null) // uPlot loss chart container
const accChartEl = ref(null) // uPlot accuracy chart container
const winrateChartEl = ref(null) // uPlot winrate chart container
let lossChart = null
let winrateChart = null
let accChart = null
const lastProbs = ref(null) // Last policy probabilities from bot
const policyCanvas = ref(null) // Canvas for policy heatmap

const hasWinrateChartData = computed(() => winrateHistory.value.length > 0 || confirmWinrateHistory.value.length > 0)
const latestQuickProbe = computed(() => winrateHistory.value.length ? winrateHistory.value[winrateHistory.value.length - 1] : null)
const latestConfirmExam = computed(() => confirmWinrateHistory.value.length ? confirmWinrateHistory.value[confirmWinrateHistory.value.length - 1] : null)
const latestCommentary = computed(() => commentaryEntries.value.length ? commentaryEntries.value[0] : null)

function hasExamCounts(point) {
  if (!point) return false
  return (Number(point.wins || 0) + Number(point.losses || 0) + Number(point.draws || 0)) > 0
}

function formatExamSummary(point, label) {
  if (!point) return ''
  const base = `${label}: ${(Number(point.winrate || 0) * 100).toFixed(1)}%`
  const decisive = point.decisiveWinRate != null ? ` · DW${(Number(point.decisiveWinRate || 0) * 100).toFixed(1)}%` : ''
  const draw = point.drawRate != null ? ` · DR${(Number(point.drawRate || 0) * 100).toFixed(1)}%` : ''
  const sideSplit = point.winrateAsP1 != null && point.winrateAsP2 != null
    ? ` · P1 ${(Number(point.winrateAsP1 || 0) * 100).toFixed(0)} / P2 ${(Number(point.winrateAsP2 || 0) * 100).toFixed(0)}`
    : ''
  if (!hasExamCounts(point)) return `${base}${decisive}${draw}${sideSplit}`
  return `${base}${decisive}${draw}${sideSplit} · W${point.wins}/L${point.losses}/D${point.draws}`
}

function getCommentaryClass(mood) {
  if (mood === 'positive') return 'commentary-positive'
  if (mood === 'warning') return 'commentary-warning'
  if (mood === 'danger') return 'commentary-danger'
  return 'commentary-neutral'
}

function pushCommentaryEntry(entry) {
  commentaryEntries.value = [entry, ...commentaryEntries.value].slice(0, 12)
}

function requestMoveCommentary(boardBefore, move, currentPlayer, actor = 'player') {
  if (!commentaryEnabled.value) return
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return
  try {
    ws.value.send(JSON.stringify({
      type: 'comment_move',
      payload: {
        boardBefore,
        move,
        current: currentPlayer,
        variant: variant.value,
        style: commentaryStyle.value,
        actor,
      },
    }))
  } catch (e) {
    console.warn('[Commentary] Failed to request commentary:', e)
  }
}

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

function clampPercent(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return 0
  return Math.min(100, Math.max(0, num))
}

function extractPhaseFraction(progress) {
  if (!progress) return null
  const pairs = [
    ['step', 'totalSteps'],
    ['game', 'totalGames'],
    ['generated', 'total'],
    ['currentBatch', 'batchesPerEpoch'],
    ['batch', 'totalBatches'],
    ['samplesDone', 'samplesTotal'],
    ['cycle', 'totalCycles'],
    ['iteration', 'totalIterations'],
    ['epoch', 'totalEpochs'],
  ]
  for (const [currentKey, totalKey] of pairs) {
    const currentValue = Number(progress[currentKey])
    const totalValue = Number(progress[totalKey])
    if (Number.isFinite(currentValue) && Number.isFinite(totalValue) && totalValue > 0) {
      return Math.min(1, Math.max(0, currentValue / totalValue))
    }
  }
  return null
}

function estimateOverallPercent(progress) {
  if (!progress) return 0
  if (progress.overallPercent != null) return clampPercent(progress.overallPercent)
  if (progress.percent != null) return clampPercent(progress.percent)
  if (progress.epochPercent != null) return clampPercent(progress.epochPercent)

  const ranges = {
    preparing: [0, 2],
    foundation: [2, 55],
    tactical: [2, 15],
    bootstrap: [15, 30],
    turbo_train: [30, 62],
    exam: [62, 72],
    repair: [72, 79],
    repair_eval: [79, 82],
    holdout: [82, 86],
    checkpoint_selection: [86, 88],
    arena: [88, 90],
    self_play_gen: [90, 92.5],
    self_play_warmup: [92.5, 93.5],
    self_play_train: [93.5, 97],
    self_play_exam: [97, 98.5],
    self_play_acceptance: [98.5, 99],
    confirm_exam: [99, 99.7],
    promotion: [99.7, 100],
    done: [100, 100],
    background_done: [100, 100],
  }
  const range = ranges[progress.phase]
  if (!range) return 0
  const fraction = extractPhaseFraction(progress)
  if (fraction == null) return clampPercent(range[0])
  return clampPercent(range[0] + (range[1] - range[0]) * fraction)
}

function getOverallPercent(progress) {
  return estimateOverallPercent(progress)
}

function getHeartbeatAgeSeconds(progress) {
  if (!progress) return null
  if (progress.secondsSinceUpdate != null) {
    const num = Number(progress.secondsSinceUpdate)
    if (Number.isFinite(num)) return Math.max(0, num)
  }
  if (progress.heartbeatTs) {
    const parsed = Date.parse(progress.heartbeatTs)
    if (!Number.isNaN(parsed)) {
      return Math.max(0, Math.floor((Date.now() - parsed) / 1000))
    }
  }
  return null
}

function isHeartbeatFresh(progress) {
  const age = getHeartbeatAgeSeconds(progress)
  return age == null || age <= 5
}

function getHeartbeatLabel(progress) {
  if (!progress) return 'нет данных'
  const age = getHeartbeatAgeSeconds(progress)
  if (age == null) return 'живой статус'
  if (age <= 1) return 'обновлено только что'
  return `обновлено ${age}с назад`
}

function getOverallEta(progress) {
  if (!progress) return null
  const direct = Number(progress.overallEta ?? progress.eta)
  if (Number.isFinite(direct) && direct > 0) return direct
  const elapsed = Number(progress.elapsed)
  const pct = getOverallPercent(progress)
  if (Number.isFinite(elapsed) && elapsed > 3 && pct > 1) {
    return Math.floor(elapsed * (100 - pct) / pct)
  }
  return null
}

function getOverallDetailText(progress) {
  if (!progress) return ''
  const phase = getPhaseLabel(progress.phase)
  const stage = getStageLabel(progress.stage)
  const eta = getOverallEta(progress)
  const parts = []
  if (phase) parts.push(phase)
  if (stage) parts.push(stage)
  if (progress.totalIterations) parts.push(`итерация ${progress.iteration || 0}/${progress.totalIterations}`)
  else if (progress.totalCycles) parts.push(`цикл ${progress.cycle || 0}/${progress.totalCycles}`)
  if (progress.totalGames) parts.push(`игра ${progress.game || 0}/${progress.totalGames}`)
  else if (progress.totalSteps) parts.push(`шаг ${progress.step || 0}/${progress.totalSteps}`)
  if (eta && eta > 0) parts.push(`ETA ~${formatTime(eta)}`)
  return parts.join(' · ')
}

function getPhaseLabel(name) {
  const labels = {
    tactical: 'Tactical',
    supervised: 'Supervised',
    bootstrap: 'Bootstrap',
    foundation: 'Foundation',
    turbo_train: 'Turbo Train',
    exam: 'Exam vs Engine',
    repair: 'Repair Train',
    repair_eval: 'Repair Eval',
    checkpoint_selection: 'Checkpoint Selection',
    confirm_exam: 'Confirm Exam',
    promotion: 'Promotion Gate',
    arena: 'Arena',
    mcts: 'MCTS',
    self_play: 'Self-Play',
    self_play_gen: 'Self-Play Generation',
    self_play_warmup: 'Replay Warm-Up',
    self_play_train: 'Self-Play Train',
    self_play_exam: 'Self-Play Exam',
    self_play_acceptance: 'Previous Checkpoint Gate',
    training: 'PyTorch Training',
    preparing: 'Preparing',
    encoding: 'Encoding',
    evaluating: 'Evaluating'
  }
  return labels[name] || name
}

function getStageLabel(stage) {
  const labels = {
    generating: 'Генерация партий',
    training: 'Обучение сети',
    engine_eval: 'Матч против engine',
    challenger_eval: 'Матч против champion',
    self_play_challenger: 'Self-play vs champion',
    self_play_previous_checkpoint: 'Self-play vs previous checkpoint',
    replay_warmup: 'Прогрев replay',
    accepted_previous_checkpoint: 'Итерация принята',
    rejected_previous_checkpoint: 'Итерация отклонена',
    best_quick_probe: 'Восстановлен лучший checkpoint',
    decision: 'Финальное решение',
    relabel: 'Переразметка ошибок',
    holdout_gate: 'Ожидание holdout gate',
    deferred: 'Отложено',
    unavailable: 'Недоступно',
  }
  return labels[stage] || stage || ''
}

function getPhaseDetail(progress) {
  if (!progress) return ''
  const phase = getPhaseLabel(progress.phase)
  const stage = getStageLabel(progress.stage)
  const gameProgress = progress.totalGames ? ` · игра ${progress.game || 0}/${progress.totalGames}` : ''
  const iterProgress = progress.totalIterations ? ` · итерация ${progress.iteration || 0}/${progress.totalIterations}` : ''
  const stepProgress = progress.totalSteps ? ` · шаг ${progress.step || 0}/${progress.totalSteps}` : ''
  const acceptance = progress.phase === 'self_play_acceptance' && progress.acceptedVsPreviousCheckpoint != null
    ? ` · ${progress.acceptedVsPreviousCheckpoint ? 'baseline обновлён' : 'baseline не обновлён'}`
    : ''
  const message = progress.message ? ` · ${progress.message}` : ''
  return [phase, stage].filter(Boolean).join(' / ') + iterProgress + gameProgress + stepProgress + acceptance + message
}

function getSelfPlayStat(stats, key) {
  if (!stats) return 0
  if (key === 'wins') return stats.wins ?? stats.wins_p1 ?? 0
  if (key === 'losses') return stats.losses ?? stats.wins_p2 ?? 0
  return stats.draws ?? 0
}

function hasSelfPlayStats(stats) {
  return getSelfPlayStat(stats, 'wins') > 0 || getSelfPlayStat(stats, 'losses') > 0 || getSelfPlayStat(stats, 'draws') > 0
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

function applyGpuTelemetry(payload) {
  const snapshot = extractGpuTelemetry(payload)
  if (!snapshot) return
  gpuTelemetry.value = {
    ...(gpuTelemetry.value || {}),
    ...snapshot,
  }
}

// ===== uPlot Training Charts =====
// Resize uPlot charts when container changes width
function observeChartResize(el, chart) {
  if (!el || !chart) return
  const ro = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const w = Math.round(entry.contentRect.width)
      if (w > 0 && Math.abs(w - chart.width) > 10) {
        chart.setSize({ width: w, height: chart.height })
      }
    }
  })
  ro.observe(el)
}

function createLossChart() {
  if (!lossChartEl.value || lossChart) return
  if (!lossChartEl.value.isConnected) return
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
  observeChartResize(lossChartEl.value, lossChart)
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

function upsertWinratePoint(historyRef, point) {
  const cycle = Number(point.cycle)
  const nextPoint = {
    cycle,
    winrate: Number(point.winrate || 0),
    decisiveWinRate: point.decisiveWinRate != null ? Number(point.decisiveWinRate || 0) : null,
    drawRate: point.drawRate != null ? Number(point.drawRate || 0) : null,
    winrateAsP1: point.winrateAsP1 != null ? Number(point.winrateAsP1 || 0) : null,
    winrateAsP2: point.winrateAsP2 != null ? Number(point.winrateAsP2 || 0) : null,
    balancedSideWinrate: point.balancedSideWinrate != null ? Number(point.balancedSideWinrate || 0) : null,
    tacticalOverrideRate: point.tacticalOverrideRate != null ? Number(point.tacticalOverrideRate || 0) : null,
    valueGuidedRate: point.valueGuidedRate != null ? Number(point.valueGuidedRate || 0) : null,
    modelPolicyRate: point.modelPolicyRate != null ? Number(point.modelPolicyRate || 0) : null,
    wins: Number(point.wins || 0),
    losses: Number(point.losses || 0),
    draws: Number(point.draws || 0),
  }
  const existingIdx = historyRef.value.findIndex(item => Number(item.cycle) === cycle)
  if (existingIdx >= 0) {
    const updated = historyRef.value.slice()
    updated[existingIdx] = { ...updated[existingIdx], ...nextPoint }
    historyRef.value = updated.sort((a, b) => a.cycle - b.cycle)
    return
  }
  historyRef.value = [...historyRef.value, nextPoint].sort((a, b) => a.cycle - b.cycle)
}

function createWinrateChart() {
  if (!winrateChartEl.value || winrateChart) return
  if (!winrateChartEl.value.isConnected) return
  const opts = {
    width: winrateChartEl.value.offsetWidth || 600,
    height: 180,
    title: 'Winrate vs Engine (Quick vs Confirm)',
    cursor: { show: true },
    scales: { x: { time: false }, y: { range: [0, 100] } },
    axes: [
      { stroke: '#666', font: '11px system-ui', size: 28, label: 'Cycle' },
      { stroke: '#1565c0', font: '11px system-ui', size: 45, label: 'Winrate %' },
    ],
    series: [
      { label: 'Cycle' },
      { label: 'Quick probe', stroke: '#1565c0', width: 2, fill: 'rgba(21,101,192,0.08)', points: { show: true, size: 6 } },
      { label: 'Confirm exam', stroke: '#2e7d32', width: 2, dash: [8, 6], points: { show: true, size: 8, stroke: '#2e7d32', fill: '#ffffff' } },
    ],
  }
  winrateChart = new uPlot(opts, [[0], [null], [null]], winrateChartEl.value)
  observeChartResize(winrateChartEl.value, winrateChart)
}

function updateWinrateChart() {
  const quickData = winrateHistory.value || []
  const confirmData = confirmWinrateHistory.value || []
  if (quickData.length < 1 && confirmData.length < 1) return
  const cycles = Array.from(new Set([
    ...quickData.map(d => Number(d.cycle)),
    ...confirmData.map(d => Number(d.cycle)),
  ])).sort((a, b) => a - b)
  const quickByCycle = new Map(quickData.map(d => [Number(d.cycle), (Number(d.winrate) || 0) * 100]))
  const confirmByCycle = new Map(confirmData.map(d => [Number(d.cycle), (Number(d.winrate) || 0) * 100]))
  const quickWinrates = cycles.map(cycle => quickByCycle.has(cycle) ? quickByCycle.get(cycle) : null)
  const confirmWinrates = cycles.map(cycle => confirmByCycle.has(cycle) ? confirmByCycle.get(cycle) : null)
  if (winrateChart) {
    winrateChart.setData([cycles, quickWinrates, confirmWinrates])
  }
}

function destroyCharts() {
  if (lossChart) { lossChart.destroy(); lossChart = null }
  if (accChart) { accChart.destroy(); accChart = null }
  if (winrateChart) { winrateChart.destroy(); winrateChart = null }
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
        requestTrainingStatus('ttt3')
        requestTrainingStatus('ttt5')
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
        if (msg.type === 'background_train.start' || msg.type === 'background_train.started') {
          backgroundProgress.value = {
            epoch: 0,
            epochs: msg.payload.epochs || 1,
            epochPercent: 0,
            newSkills: msg.payload.newSkills || 0,
            totalSkills: msg.payload.totalSkills || 0,
            newSkillsPercent: msg.payload.newSkillsPercent || 0,
            message: msg.payload.message,
            phase: msg.payload.phase,
            stage: msg.payload.stage,
            overallPercent: msg.payload.overallPercent ?? msg.payload.epochPercent ?? 0,
            heartbeatTs: new Date().toISOString(),
            secondsSinceUpdate: 0,
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
            message: msg.payload.message,
            phase: msg.payload.phase,
            stage: msg.payload.stage,
            game: msg.payload.game,
            totalGames: msg.payload.totalGames,
            step: msg.payload.step,
            totalSteps: msg.payload.totalSteps,
            overallPercent: msg.payload.overallPercent ?? msg.payload.epochPercent ?? 0,
            overallEta: msg.payload.overallEta,
            heartbeatTs: new Date().toISOString(),
            secondsSinceUpdate: 0,
          }
          // Используем epochPercent для статуса, если есть batchProgress - показываем его
          const progressText = msg.payload.batchProgress ? 
            `${msg.payload.epochPercent}% (батч ${msg.payload.currentBatch}/${msg.payload.batchesPerEpoch})` : 
            `${msg.payload.epochPercent}%`
          status.value = msg.payload.message || `Фоновое обучение: ${progressText}`
          console.log('[WS] Background training progress:', msg.payload)
        }
        if (msg.type === 'background_train.done') {
          const p = msg.payload || {}
          ttt5Progress.value = { ...(ttt5Progress.value || {}), ...p, phase: 'background_done' }
          trainingMeta.value = { ...(trainingMeta.value || {}), ...p }
          if (p.metricsHistory) metricsHistory.value = p.metricsHistory
          if (Array.isArray(p.winrateHistory)) {
            winrateHistory.value = [...p.winrateHistory]
              .map(item => ({
                cycle: Number(item.cycle || 0),
                winrate: Number(item.winrate || 0),
                decisiveWinRate: item.decisiveWinRate != null ? Number(item.decisiveWinRate || 0) : null,
                drawRate: item.drawRate != null ? Number(item.drawRate || 0) : null,
                winrateAsP1: item.winrateAsP1 != null ? Number(item.winrateAsP1 || 0) : null,
                winrateAsP2: item.winrateAsP2 != null ? Number(item.winrateAsP2 || 0) : null,
                balancedSideWinrate: item.balancedSideWinrate != null ? Number(item.balancedSideWinrate || 0) : null,
                tacticalOverrideRate: item.tacticalOverrideRate != null ? Number(item.tacticalOverrideRate || 0) : null,
                valueGuidedRate: item.valueGuidedRate != null ? Number(item.valueGuidedRate || 0) : null,
                modelPolicyRate: item.modelPolicyRate != null ? Number(item.modelPolicyRate || 0) : null,
                wins: Number(item.wins || 0),
                losses: Number(item.losses || 0),
                draws: Number(item.draws || 0),
              }))
              .sort((a, b) => a.cycle - b.cycle)
          }
          if (p.winrateVsAlgorithm != null) {
            const confirmCycle = Number(p.cycles || winrateHistory.value.length || 0) + 1
            const previousConfirm = confirmWinrateHistory.value.find(item => Number(item.cycle) === confirmCycle)
            upsertWinratePoint(confirmWinrateHistory, {
              cycle: confirmCycle,
              winrate: p.winrateVsAlgorithm,
              decisiveWinRate: p.confirmDecisiveWinRate ?? p.decisiveWinRate ?? previousConfirm?.decisiveWinRate ?? null,
              drawRate: p.confirmDrawRate ?? p.drawRate ?? previousConfirm?.drawRate ?? null,
              winrateAsP1: p.confirmWinrateAsP1 ?? p.winrateAsP1 ?? previousConfirm?.winrateAsP1 ?? null,
              winrateAsP2: p.confirmWinrateAsP2 ?? p.winrateAsP2 ?? previousConfirm?.winrateAsP2 ?? null,
              balancedSideWinrate: p.confirmBalancedSideWinrate ?? p.balancedSideWinrate ?? previousConfirm?.balancedSideWinrate ?? null,
              tacticalOverrideRate: p.confirmTacticalOverrideRate ?? p.tacticalOverrideRate ?? previousConfirm?.tacticalOverrideRate ?? null,
              valueGuidedRate: p.confirmValueGuidedRate ?? p.valueGuidedRate ?? previousConfirm?.valueGuidedRate ?? null,
              modelPolicyRate: p.confirmModelPolicyRate ?? p.modelPolicyRate ?? previousConfirm?.modelPolicyRate ?? null,
              wins: p.confirmWins ?? previousConfirm?.wins ?? 0,
              losses: p.confirmLosses ?? previousConfirm?.losses ?? 0,
              draws: p.confirmDraws ?? previousConfirm?.draws ?? 0,
            })
          }
          let doneStatus = p.message || 'Фоновая оценка завершена'
          if (p.winrateVsChampion != null) doneStatus += ` | vs champion: ${(p.winrateVsChampion * 100).toFixed(1)}%`
          if (p.winrateVsAlgorithm != null) doneStatus += ` | vs engine: ${(p.winrateVsAlgorithm * 100).toFixed(1)}%`
          if (p.confirmBalancedSideWinrate != null) doneStatus += ` | balanced: ${(p.confirmBalancedSideWinrate * 100).toFixed(1)}%`
          if (p.promoted) doneStatus += ' | PROMOTED'
          status.value = doneStatus
          console.log('[WS] Background training done:', p)
          setTimeout(() => {
            backgroundProgress.value = null
          }, 3000)
          if (ws.value && ws.value.readyState === WebSocket.OPEN) {
            ws.value.send(JSON.stringify({ type: 'get_history_stats' }))
          }
        }
        if (msg.type === 'background_train.error') {
          status.value = msg.payload?.message || msg.error || 'Ошибка фонового обучения'
          backgroundProgress.value = null
          console.error('[WS] Background training error:', msg.error)
        }
        if (msg.type === 'training.status') {
          applyTrainingStatus(msg.payload || {})
          console.log('[WS] Training status synced:', msg.payload)
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
              applyGpuTelemetry(msg.payload)
              if (msg.payload.metricsHistory) {
                metricsHistory.value = msg.payload.metricsHistory
              }
              // Collect winrate data from quick exam and confirm exam.
              // Quick exam is a pre-repair probe; confirm exam is the final post-repair truth.
              const _p = msg.payload
              if (_p.winrateVsAlgorithm != null && _p.game != null && _p.totalGames != null) {
                if (_p.phase === 'exam') {
                  const cycle = _p.cycle || _p.iteration || winrateHistory.value.length + 1
                  upsertWinratePoint(winrateHistory, {
                    cycle,
                    winrate: _p.winrateVsAlgorithm,
                    decisiveWinRate: _p.decisiveWinRate,
                    drawRate: _p.drawRate,
                    winrateAsP1: _p.winrateAsP1,
                    winrateAsP2: _p.winrateAsP2,
                    balancedSideWinrate: _p.balancedSideWinrate,
                    tacticalOverrideRate: _p.tacticalOverrideRate,
                    valueGuidedRate: _p.valueGuidedRate,
                    modelPolicyRate: _p.modelPolicyRate,
                    wins: _p.arenaWins || 0,
                    losses: _p.arenaLosses || 0,
                    draws: _p.arenaDraws || 0,
                  })
                } else if (_p.phase === 'confirm_exam') {
                  const cycle = _p.cycle || (_p.totalCycles ? _p.totalCycles + 1 : confirmWinrateHistory.value.length + 1)
                  upsertWinratePoint(confirmWinrateHistory, {
                    cycle,
                    winrate: _p.winrateVsAlgorithm,
                    decisiveWinRate: _p.decisiveWinRate,
                    drawRate: _p.drawRate,
                    winrateAsP1: _p.winrateAsP1,
                    winrateAsP2: _p.winrateAsP2,
                    balancedSideWinrate: _p.balancedSideWinrate,
                    tacticalOverrideRate: _p.tacticalOverrideRate,
                    valueGuidedRate: _p.valueGuidedRate,
                    modelPolicyRate: _p.modelPolicyRate,
                    wins: _p.arenaWins || 0,
                    losses: _p.arenaLosses || 0,
                    draws: _p.arenaDraws || 0,
                  })
                }
              }
              // Build status text from structured data
              const p = msg.payload
              let statusText = ''
              if (p.phase === 'foundation') statusText = `Foundation teacher`
              else if (p.phase === 'turbo_train') statusText = `Turbo train ${p.iteration || p.cycle || ''}/${p.totalIterations || p.totalCycles || ''}`
              else if (p.phase === 'exam') statusText = `Exam vs engine ${p.iteration || p.cycle || ''}/${p.totalIterations || p.totalCycles || ''}`
              else if (p.phase === 'repair') statusText = `Repair train ${p.iteration || p.cycle || ''}/${p.totalIterations || p.totalCycles || ''}`
              else if (p.phase === 'repair_eval') statusText = `Repair eval ${p.iteration || p.cycle || ''}/${p.totalIterations || p.totalCycles || ''}`
              else if (p.phase === 'holdout') statusText = `Holdout & frozen benchmarks ${p.iteration || p.cycle || ''}/${p.totalIterations || p.totalCycles || ''}`
              else if (p.phase === 'tactical') statusText = `Tactical curriculum`
              else if (p.phase === 'bootstrap') statusText = `Bootstrap`
              else if (p.phase === 'self_play') statusText = `Self-play ${p.iteration || ''}/${p.totalIterations || ''}`
              else if (p.phase === 'arena') statusText = `Arena evaluation`
              else if (p.phase === 'confirm_exam') statusText = `Confirm exam (${p.game || 0}/${p.totalGames || 0})`
              else if (p.phase === 'promotion') statusText = `Promotion gate`
              else if (p.phase === 'self_play_gen') statusText = `Self-play ${p.iteration || ''}/${p.totalIterations || ''} (${p.game || 0}/${p.totalGames || 0} игр, ${p.positionsCollected || 0} поз)`
              else if (p.phase === 'self_play_train') statusText = `SP train ${p.iteration || ''}/${p.totalIterations || ''}`
              else if (p.phase === 'self_play_exam') statusText = `SP exam ${p.iteration || ''}/${p.totalIterations || ''}`
              else if (p.phase === 'mcts') statusText = `MCTS ${p.iteration || ''}/${p.totalIterations || ''}`
              if (p.epoch > 0) statusText += ` | Epoch ${p.epoch}/${p.totalEpochs}`
              if (p.winrateVsAlgorithm != null) statusText += ` | WR: ${(p.winrateVsAlgorithm * 100).toFixed(1)}%`
              if (p.winrateAsP1 != null && p.winrateAsP2 != null) statusText += ` | P1/P2: ${(p.winrateAsP1 * 100).toFixed(0)}/${(p.winrateAsP2 * 100).toFixed(0)}`
              if (p.decisiveWinRate != null) statusText += ` | DW: ${(p.decisiveWinRate * 100).toFixed(1)}%`
              if (p.drawRate != null) statusText += ` | DR: ${(p.drawRate * 100).toFixed(1)}%`
              if (p.tacticalOverrideRate != null) statusText += ` | OV: ${(p.tacticalOverrideRate * 100).toFixed(0)}%`
              if (p.valueGuidedRate != null) statusText += ` | VG: ${(p.valueGuidedRate * 100).toFixed(0)}%`
              if (p.deltaWinrate != null) statusText += ` | Δ ${(p.deltaWinrate * 100).toFixed(1)}%`
              if (p.holdoutPolicyAcc != null) statusText += ` | Holdout: ${p.holdoutPolicyAcc}%`
              if (p.frozenBlockAcc != null) statusText += ` | Block: ${p.frozenBlockAcc}%`
              if (p.frozenWinAcc != null) statusText += ` | Win: ${p.frozenWinAcc}%`
              if (p.policyTop1Acc) statusText += ` | Acc: ${p.policyTop1Acc}%`
              else if (p.accuracy) statusText += ` | Acc: ${p.accuracy}%`
              if (p.fixedErrors != null) statusText += ` | Fixed: ${p.fixedErrors}`
              if (p.regressedErrors != null) statusText += ` | Regr: ${p.regressedErrors}`
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
          cancellingTraining.value = false
          destroyCharts()
          trainingMeta.value = msg.payload
          applyGpuTelemetry(msg.payload)
          progress.value = { percent: 0, epoch: 0, epochs: msg.payload.epochs }
          ttt5Progress.value = { ...msg.payload, phase: 'preparing', percent: 0, elapsed: 0, eta: 0 }
          metricsHistory.value = []
          winrateHistory.value = []
          confirmWinrateHistory.value = []
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
      if (msg.type === 'dataset.done') {
        generatingDataset.value = false
        datasetProgress.value = null
        const p = msg.payload
        status.value = `Датасет готов: ${p.count} позиций (${p.variant})`
      }
      if (msg.type === 'train.accepted') {
        status.value = `Обучение ${msg.payload.variant} запущено...`
      }
      if (msg.type === 'train.error') {
        training.value = false
        cancellingTraining.value = false
        if (trainingTimerInterval) { clearInterval(trainingTimerInterval); trainingTimerInterval = null }
        status.value = `Ошибка обучения: ${msg.payload.error}`
        console.error('[WS] Training error:', msg.payload)
      }
      if (msg.type === 'train.cancelled') {
        training.value = false
        cancellingTraining.value = false
        if (trainingTimerInterval) { clearInterval(trainingTimerInterval); trainingTimerInterval = null }
        status.value = `Обучение ${msg.payload.variant} отменено`
      }
      if (msg.type === 'model.promoted') {
        const p = msg.payload
        const wr = p.winrateVsChampion ? ` (wr ${(p.winrateVsChampion * 100).toFixed(1)}%)` : ''
        status.value = `Model promoted: gen ${p.generation}${wr}`
        console.log('[WS] Model promoted:', p)
      }
      if (msg.type === 'train.rejected') {
        // Duplicate launch rejection — show user feedback
        status.value = `Обучение ${msg.payload.variant} уже запущено`
        console.log('[WS] Training rejected (duplicate):', msg.payload)
      }
      if (msg.type === 'promotion.rejected') {
        const p = msg.payload
        status.value = `Promotion rejected: ${p.reason}`
        console.log('[WS] Promotion rejected:', p.reason)
      }
      if (msg.type === 'train.done') {
        training.value = false
        cancellingTraining.value = false
        datasetProgress.value = null
        // Stop timer & show final time
        if (trainingTimerInterval) { clearInterval(trainingTimerInterval); trainingTimerInterval = null }
        const totalTime = formatTime(Math.floor((Date.now() - trainingStartTime.value) / 1000))
        const p = msg.payload
        let doneStatus = `Обучение завершено за ${totalTime}`
        if (p.evaluationQueued) doneStatus += ' | confirm в фоне'
        if (p.winrateVsChampion != null) doneStatus += ` | vs champion: ${(p.winrateVsChampion * 100).toFixed(1)}%`
        if (p.winrateVsAlgorithm != null) doneStatus += ` | vs engine: ${(p.winrateVsAlgorithm * 100).toFixed(1)}%`
        if (p.decisiveWinRate != null) doneStatus += ` | decisive: ${(p.decisiveWinRate * 100).toFixed(1)}%`
        if (p.drawRate != null) doneStatus += ` | draw: ${(p.drawRate * 100).toFixed(1)}%`
        if (p.holdoutPolicyAcc != null) doneStatus += ` | holdout: ${p.holdoutPolicyAcc}%`
        if (p.frozenBlockAcc != null) doneStatus += ` | block: ${p.frozenBlockAcc}%`
        if (p.frozenWinAcc != null) doneStatus += ` | win: ${p.frozenWinAcc}%`
        if (p.failureBankSize != null) doneStatus += ` | failures: ${p.failureBankSize}`
        if (p.promoted) doneStatus += ' | PROMOTED'
        ttt5Progress.value = { ...(ttt5Progress.value || {}), ...p, phase: 'done' }
        trainingMeta.value = { ...(trainingMeta.value || {}), ...p }
        if (p.metricsHistory) metricsHistory.value = p.metricsHistory
        if (Array.isArray(p.winrateHistory)) {
          winrateHistory.value = [...p.winrateHistory]
            .map(item => ({
              cycle: Number(item.cycle || 0),
              winrate: Number(item.winrate || 0),
              decisiveWinRate: item.decisiveWinRate != null ? Number(item.decisiveWinRate || 0) : null,
              drawRate: item.drawRate != null ? Number(item.drawRate || 0) : null,
              winrateAsP1: item.winrateAsP1 != null ? Number(item.winrateAsP1 || 0) : null,
              winrateAsP2: item.winrateAsP2 != null ? Number(item.winrateAsP2 || 0) : null,
              balancedSideWinrate: item.balancedSideWinrate != null ? Number(item.balancedSideWinrate || 0) : null,
              tacticalOverrideRate: item.tacticalOverrideRate != null ? Number(item.tacticalOverrideRate || 0) : null,
              valueGuidedRate: item.valueGuidedRate != null ? Number(item.valueGuidedRate || 0) : null,
              modelPolicyRate: item.modelPolicyRate != null ? Number(item.modelPolicyRate || 0) : null,
              wins: Number(item.wins || 0),
              losses: Number(item.losses || 0),
              draws: Number(item.draws || 0),
            }))
            .sort((a, b) => a.cycle - b.cycle)
        }
        if (p.winrateVsAlgorithm != null) {
          const confirmCycle = Number(p.cycles || winrateHistory.value.length || 0) + 1
          const previousConfirm = confirmWinrateHistory.value.find(item => Number(item.cycle) === confirmCycle)
          upsertWinratePoint(confirmWinrateHistory, {
            cycle: confirmCycle,
            winrate: p.winrateVsAlgorithm,
            decisiveWinRate: p.confirmDecisiveWinRate ?? p.decisiveWinRate ?? previousConfirm?.decisiveWinRate ?? null,
            drawRate: p.confirmDrawRate ?? p.drawRate ?? previousConfirm?.drawRate ?? null,
            winrateAsP1: p.confirmWinrateAsP1 ?? p.winrateAsP1 ?? previousConfirm?.winrateAsP1 ?? null,
            winrateAsP2: p.confirmWinrateAsP2 ?? p.winrateAsP2 ?? previousConfirm?.winrateAsP2 ?? null,
            balancedSideWinrate: p.confirmBalancedSideWinrate ?? p.balancedSideWinrate ?? previousConfirm?.balancedSideWinrate ?? null,
            tacticalOverrideRate: p.confirmTacticalOverrideRate ?? p.tacticalOverrideRate ?? previousConfirm?.tacticalOverrideRate ?? null,
            valueGuidedRate: p.confirmValueGuidedRate ?? p.valueGuidedRate ?? previousConfirm?.valueGuidedRate ?? null,
            modelPolicyRate: p.confirmModelPolicyRate ?? p.modelPolicyRate ?? previousConfirm?.modelPolicyRate ?? null,
            wins: p.confirmWins ?? previousConfirm?.wins ?? 0,
            losses: p.confirmLosses ?? previousConfirm?.losses ?? 0,
            draws: p.confirmDraws ?? previousConfirm?.draws ?? 0,
          })
        }
        status.value = doneStatus
        console.log('[WS] Training completed:', p)
        // Обновляем статистику истории после обучения
        if (ws.value && ws.value.readyState === WebSocket.OPEN) {
          ws.value.send(JSON.stringify({ type: 'get_history_stats' }))
        }
      }
        // Corpus events
        if (msg.type === 'corpus.analysis.queued') {
          corpusAnalyzing.value = true
          corpusLastEvent.value = `Анализ партии ${msg.payload.gameId?.slice(0, 8) || ''}...`
          console.log('[WS] Corpus analysis queued:', msg.payload)
        }
        if (msg.type === 'corpus.updated') {
          const p = msg.payload
          corpusStatus.value = {
            recentCount: p.recentCount || 0,
            hardMistakeCount: p.hardMistakeCount || 0,
            conversionCount: p.conversionCount || 0,
            weakSideCount: p.weakSideCount || 0,
            analyzedGames: p.analyzedGames || 0,
            analyzedPositions: p.analyzedPositions || 0,
            variant: p.variant,
          }
          corpusLastEvent.value = `+${p.analyzedPositions || 0} позиций из ${p.analyzedGames || 0} игр`
          console.log('[WS] Corpus updated:', p)
        }
        if (msg.type === 'corpus.analysis.done') {
          corpusAnalyzing.value = false
          corpusLastEvent.value = `Готово: ${msg.payload.analyzedPositions || 0} позиций`
          console.log('[WS] Corpus analysis done:', msg.payload)
        }

        if (msg.type === 'predict.result') {
          console.log('[WS] Received predict.result, resetting waiting')
          waiting.value = false
          const move = msg.payload.move
          const boardBeforeBotMove = [...board.value]

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
              lastMoveIdx.value = move
              triggerProbFade()
              requestMoveCommentary(boardBeforeBotMove, move, 2, 'bot')
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
          destroyCharts()
          metricsHistory.value = []
          winrateHistory.value = []
          confirmWinrateHistory.value = []
          modelConfidence.value = null
          commentaryEntries.value = []
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
          applyGpuTelemetry(msg.payload)
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
        if (msg.type === 'commentary.result') {
          pushCommentaryEntry(msg.payload)
        }
        if (msg.type === 'error') {
          console.error('[WS] Server error:', msg.error)
          training.value = false
          cancellingTraining.value = false
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
    const pct = backgroundProgress.value
      ? getOverallPercent(backgroundProgress.value)
      : (ttt5Progress.value ? getOverallPercent(ttt5Progress.value) : (progress.value?.percent || 0))
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
      // Self-play params
      if (preset.selfPlay) {
        payload.selfPlay = true
        payload.selfPlayIterations = preset.selfPlayIterations || 10
        payload.selfPlayGames = preset.selfPlayGames || 100
        payload.selfPlaySims = preset.selfPlaySims || 100
        payload.selfPlayTrainSteps = preset.selfPlayTrainSteps || 80
      }
    }
    ws.value.send(JSON.stringify({ type: trainType, payload }))
    console.log(`[Train] ${trainType} sent:`, payload)
  } catch (e) {
    console.error('[Train] Send error:', e)
    training.value = false
    status.value = 'Ошибка отправки запроса: ' + e.message
  }
}

function cancelTraining() {
  if (!training.value) return
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
    status.value = 'Ошибка: WebSocket не подключен. Не удалось остановить обучение.'
    return
  }
  cancellingTraining.value = true
  status.value = `Останавливаем обучение ${variant.value}...`
  try {
    ws.value.send(JSON.stringify({
      type: 'cancel_training',
      payload: { variant: variant.value }
    }))
  } catch (e) {
    cancellingTraining.value = false
    status.value = 'Ошибка отправки запроса на остановку: ' + e.message
    console.error('[Train] Cancel send error:', e)
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

function generateDataset() {
  if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
    status.value = 'WebSocket не подключен'
    return
  }
  generatingDataset.value = true
  status.value = 'Генерация minimax датасета...'
  ws.value.send(JSON.stringify({ type: 'generate_dataset', payload: { variant: variant.value, count: 5000 } }))
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
    winLine.value = getWinningLine(board.value);
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
    winLine.value = getWinningLine(board.value);
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
  winLine.value = null
  lastMoveIdx.value = -1
  probFading.value = false
  waiting.value = false
  modelConfidence.value = null
  commentaryEntries.value = []
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
          modelDecisionMode: modelDecisionMode.value,
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
  const boardBeforeMove = [...board.value]
  
  // Сохраняем ход человека для обучения
  saveMove(board.value, idx, 1)
  
  // Делаем ход
  board.value[idx] = 1
  lastMoveIdx.value = idx
  requestMoveCommentary(boardBeforeMove, idx, 1, 'player')
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
          modelDecisionMode: modelDecisionMode.value,
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
      if (lossChart && lossChartEl.value && !lossChartEl.value.querySelector('canvas')) {
        try { lossChart.destroy() } catch (_) {}
        lossChart = null
      }
      if (!lossChart && lossChartEl.value) createLossChart()
      updateCharts()
    })
  }
}, { deep: true })

watch([winrateHistory, confirmWinrateHistory], () => {
  if (hasWinrateChartData.value) {
    nextTick(() => {
      if (winrateChart && winrateChartEl.value && !winrateChartEl.value.querySelector('canvas')) {
        try { winrateChart.destroy() } catch (_) {}
        winrateChart = null
      }
      if (!winrateChart && winrateChartEl.value) createWinrateChart()
      updateWinrateChart()
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
  if (!val && metricsHistory.value.length === 0 && !hasWinrateChartData.value) {
    setTimeout(destroyCharts, 500)
  }
})

onUnmounted(() => {
  destroyCharts()
  if (gpuPollInterval) clearInterval(gpuPollInterval)
  if (trainingStatusPollInterval) clearInterval(trainingStatusPollInterval)
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
  applyTheme()
  // Даем небольшую задержку перед первым подключением
  setTimeout(() => {
    connectWS()
  }, 100)
  gpuPollInterval = setInterval(() => {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify({ type: 'get_gpu_info' }))
    }
  }, 3000)
  trainingStatusPollInterval = setInterval(() => {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      requestTrainingStatus(variant.value)
    }
  }, 1000)
})
</script>

<style>
/* ===== Theme System: CSS Custom Properties ===== */
:root {
  --bg-primary: #f8fafc;
  --bg-secondary: #f1f5f9;
  --bg-card: #ffffff;
  --text-primary: #1e293b;
  --text-secondary: #64748b;
  --border-color: #e2e8f0;
  --accent: #2563eb;
  --accent-green: #16a34a;
  --accent-red: #dc2626;
  --accent-orange: #ea580c;
  --shadow-card: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --board-bg: #DCB35C;
  --board-line: #8B6914;
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  color: var(--text-primary);
  background: var(--bg-primary);
}

[data-theme="dark"] {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-card: #1e293b;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --border-color: #334155;
  --accent: #3b82f6;
  --accent-green: #22c55e;
  --accent-red: #ef4444;
  --accent-orange: #f97316;
  --shadow-card: 0 1px 3px rgba(0,0,0,0.3);
  --board-bg: #a3882e;
  --board-line: #6b5210;
  --color-x: #ff6b6b;
  --color-o: #74b9ff;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: #334155;
    --accent: #3b82f6;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-orange: #f97316;
    --shadow-card: 0 1px 3px rgba(0,0,0,0.3);
    --board-bg: #a3882e;
    --board-line: #6b5210;
  }
}

*, *::before, *::after {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg-primary);
  color: var(--text-primary);
}

/* ===== App Shell ===== */
.app-shell {
  max-width: 100%;
  margin: 0 auto;
  padding: 0 16px 16px;
  min-height: 100vh;
}

/* ===== Header ===== */
.app-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 0;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border-color);
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg-primary);
  backdrop-filter: blur(8px);
}

.logo {
  font-size: 1.4em;
  font-weight: 800;
  color: var(--accent);
  letter-spacing: -0.02em;
  white-space: nowrap;
}

.variant-tabs {
  display: flex;
  gap: 4px;
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 3px;
}

.variant-tabs button {
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--text-secondary);
  font-weight: 600;
  font-size: 0.9em;
  cursor: pointer;
  transition: all 0.2s ease;
}

.variant-tabs button.active {
  background: var(--accent);
  color: #fff;
  box-shadow: 0 1px 3px rgba(37,99,235,0.3);
}

.variant-tabs button:hover:not(.active) {
  color: var(--text-primary);
  background: var(--bg-card);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-left: auto;
}

.gpu-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.8em;
  font-weight: 600;
  white-space: nowrap;
}

.gpu-pill.gpu-active {
  background: color-mix(in srgb, var(--accent-green) 15%, transparent);
  color: var(--accent-green);
  border: 1px solid color-mix(in srgb, var(--accent-green) 30%, transparent);
}

.gpu-pill.gpu-inactive {
  background: color-mix(in srgb, var(--accent-orange) 15%, transparent);
  color: var(--accent-orange);
  border: 1px solid color-mix(in srgb, var(--accent-orange) 30%, transparent);
}

.gpu-pill .gpu-icon { font-size: 1em; }
.gpu-pill .gpu-backend { font-size: 0.85em; opacity: 0.7; font-family: monospace; }

.theme-toggle {
  display: flex;
  gap: 2px;
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 3px;
}

.theme-toggle button {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: transparent;
  cursor: pointer;
  font-size: 1em;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}

.theme-toggle button.active {
  background: var(--bg-card);
  box-shadow: var(--shadow-card);
}

.theme-toggle button:hover:not(.active) {
  background: var(--bg-card);
}

/* ===== Main Layout ===== */
.main-layout {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: flex-start;
  max-height: calc(100vh - 70px);
  overflow: hidden;
}

.board-area {
  flex: 0 0 auto;
  width: min(420px, 40vw);
  min-width: 280px;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sidebar-area {
  flex: 1 1 0;
  min-width: 320px;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ===== Cards ===== */
.card {
  background: var(--bg-card);
  border-radius: 10px;
  box-shadow: var(--shadow-card);
  padding: 0.75rem;
  border: 1px solid var(--border-color);
}

.card-title {
  margin: 0 0 8px 0;
  font-size: 0.85em;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.card-title-sm {
  font-size: 0.8em;
  font-weight: 600;
  color: var(--accent);
  text-transform: uppercase;
  cursor: pointer;
  padding: 4px 0;
}

.details-card {
  padding: 0.5rem 0.75rem;
}

.details-card[open] summary {
  margin-bottom: 8px;
}

.compact-train-header {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
}

.compact-train-header .button-group {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.btn-sm {
  padding: 5px 10px !important;
  font-size: 0.8em !important;
  min-height: 32px !important;
}

.preset-row-compact {
  display: flex;
  gap: 4px;
}

.preset-pill {
  padding: 4px 10px;
  border-radius: 20px;
  border: 1px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 0.75em;
  cursor: pointer;
  transition: all 0.15s ease;
  white-space: nowrap;
}

.preset-pill.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}

.preset-selfplay.active {
  background: var(--accent-orange);
  border-color: var(--accent-orange);
}

.preset-pill:hover:not(:disabled) {
  border-color: var(--accent);
}

.compact-train-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  color: var(--text-secondary);
  font-size: 0.8em;
}

.history-badge {
  background: var(--bg-secondary);
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.85em;
  white-space: nowrap;
}

.history-badge a {
  color: var(--accent-red);
  text-decoration: none;
  margin-left: 4px;
}

.auto-train-check {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  color: var(--text-secondary);
  font-size: 0.8em;
  cursor: pointer;
}

.auto-train-check input {
  accent-color: var(--accent);
}

/* ===== Corpus Status Panel ===== */
.corpus-card {
  background: color-mix(in srgb, var(--accent) 6%, var(--bg-card));
  border-color: color-mix(in srgb, var(--accent) 25%, var(--border-color));
}

.corpus-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.corpus-title {
  font-size: 0.8em;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.corpus-spinner {
  font-size: 0.85em;
  animation: spin 1.2s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.corpus-ready {
  font-size: 0.75em;
  color: var(--accent-green);
}

.corpus-buckets {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 4px;
}

.corpus-bucket {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 4px 2px;
  border-radius: 6px;
  background: var(--bg-secondary);
}

.bucket-count {
  font-size: 1.1em;
  font-weight: 700;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}

.bucket-label {
  font-size: 0.65em;
  color: var(--text-secondary);
  text-align: center;
  line-height: 1.1;
}

.bucket-mistakes .bucket-count { color: var(--accent-red); }
.bucket-conversion .bucket-count { color: var(--accent-orange); }
.bucket-weak .bucket-count { color: var(--accent); }

.corpus-event {
  margin-top: 4px;
  color: var(--text-secondary);
  font-size: 0.75em;
}

.custom-settings-compact {
  display: flex;
  gap: 12px;
  margin-top: 6px;
  font-size: 0.8em;
}

.custom-settings-compact label {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--text-secondary);
}

.custom-settings-compact input {
  width: 70px;
  padding: 3px 6px;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 0.9em;
}

/* ===== Buttons ===== */
.btn {
  padding: 8px 14px;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.88em;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.btn:hover:not(:disabled) {
  transform: scale(1.02);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.btn-primary {
  background: var(--accent);
  color: #fff;
}

.btn-primary:hover:not(:disabled) {
  filter: brightness(1.1);
}

.btn-secondary {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--border-color);
}

.btn-danger {
  background: var(--accent-red);
  color: #fff;
}

.btn-danger:hover:not(:disabled) {
  filter: brightness(1.1);
}

.btn-small {
  padding: 4px 10px;
  font-size: 0.82em;
  background: var(--bg-secondary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

/* ===== Game controls ===== */
.game-type-selector,
.mode-selector {
  display: flex;
  gap: 12px;
  margin-bottom: 10px;
  padding: 8px 10px;
  background: var(--bg-secondary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.mode-label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  user-select: none;
  font-size: 0.9em;
  color: var(--text-primary);
}

.mode-label input[type="radio"],
.mode-label input[type="checkbox"] { cursor: pointer; }
.mode-label:has(input:disabled) { opacity: 0.5; cursor: not-allowed; }

.pause-control {
  margin-bottom: 10px;
  padding: 8px 10px;
  background: var(--bg-secondary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.pause-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9em;
  color: var(--text-primary);
}

.pause-label input[type="number"],
.pause-label select {
  width: 80px;
  padding: 4px 8px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-card);
  color: var(--text-primary);
}

/* ===== Board ===== */
.board-wrapper {
  display: flex;
  justify-content: center;
  padding: 8px 0;
}

.board {
  display: grid;
  gap: 3px;
  width: 100%;
  max-width: 560px;
  aspect-ratio: 1 / 1;
  background: var(--board-line);
  border-radius: 8px;
  padding: 6px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.cell {
  position: relative;
  aspect-ratio: 1;
  width: 100%;
  height: auto;
  border: none;
  border-radius: 4px;
  background: var(--board-bg);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s ease, box-shadow 0.2s ease;
  transition: background 0.15s ease;
  position: relative;
  font-size: 0;
}

.cell:disabled {
  cursor: not-allowed;
}

.cell:not(:disabled):hover {
  background: color-mix(in srgb, var(--board-bg) 85%, #fff);
}

/* Stones */
/* ===== X and O Marks (SVG) ===== */
.mark {
  width: 65%;
  height: 65%;
  transition: transform 0.15s ease;
}

.mark-x {
  color: var(--color-x, #e74c3c);
  filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.3));
}

.mark-o {
  color: var(--color-o, #3498db);
  filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.3));
}

.mark-ghost {
  opacity: 0;
  color: var(--text-secondary);
  transition: opacity 0.15s ease;
}

.cell:not(:disabled):hover .mark-ghost.ghost-visible {
  opacity: 0.2;
}

/* Last move highlight */
.cell-last {
  box-shadow: inset 0 0 0 3px rgba(255, 193, 7, 0.7) !important;
}

/* Winning cells pulse */
.cell-win {
  animation: winPulse 0.8s ease-in-out infinite alternate;
}

@keyframes winPulse {
  from { box-shadow: inset 0 0 0 3px rgba(231, 76, 60, 0.5); }
  to { box-shadow: inset 0 0 0 3px rgba(231, 76, 60, 1); }
}

/* Win strike-through SVG overlay */
.board-wrapper {
  position: relative;
}

.win-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10;
}

/* ===== Inline Heatmap ===== */
.cell-prob {
  position: absolute;
  bottom: 2px;
  right: 3px;
  font-size: 0.6em;
  font-weight: 700;
  color: var(--accent-green);
  opacity: 1;
  transition: opacity 1.5s ease;
  pointer-events: none;
  text-shadow: 0 0 3px var(--bg-card);
}

.cell-prob.prob-fade {
  opacity: 0;
}

.heatmap-toggle {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  cursor: pointer;
  margin-top: 0.5rem;
}

.heatmap-toggle input {
  accent-color: var(--accent);
}

/* ===== Game Controls Row ===== */
.game-controls-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  padding-top: 8px;
}

.status {
  color: var(--text-secondary);
  font-size: 0.9em;
}

.status.game-over {
  font-weight: 700;
  font-size: 1.05em;
  color: var(--text-primary);
}

/* Confidence indicator */
.confidence-indicator {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.82em;
  font-weight: 600;
  transition: all 0.2s ease;
}

.confidence-high { background: var(--accent-green); color: #fff; }
.confidence-medium { background: var(--accent-orange); color: #fff; }
.confidence-low { background: var(--accent-red); color: #fff; }
.confidence-unknown { background: var(--text-secondary); color: #fff; }

/* ===== Training Controls ===== */
.controls {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.button-group {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.disabled-btn { opacity: 0.5; cursor: not-allowed; }

.history-info {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  background: var(--bg-secondary);
  border-radius: 8px;
  font-size: 0.88em;
  color: var(--text-secondary);
}

.checkbox-group {
  padding: 8px 10px;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.checkbox-group label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 0.88em;
  color: var(--text-primary);
}

.checkbox-group input[type="checkbox"] { cursor: pointer; }

.hint-text {
  display: block;
  margin-top: 4px;
  color: var(--text-secondary);
  font-size: 0.82em;
}

/* ===== Settings ===== */
.setting-row {
  margin-bottom: 10px;
}

.setting-row label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 0.9em;
  color: var(--text-primary);
}

.setting-row label span { flex: 1; }

.setting-row input[type="number"] {
  width: 100px;
  padding: 6px 8px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  text-align: center;
  background: var(--bg-card);
  color: var(--text-primary);
}

.setting-row input[type="number"]:invalid { border-color: var(--accent-red); }
.setting-row input[type="number"]:disabled { background: var(--bg-secondary); cursor: not-allowed; opacity: 0.6; }

.validation-error {
  display: block;
  color: var(--accent-red);
  font-size: 0.82em;
  margin-top: 4px;
}

.setting-hint {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border-color);
}

.setting-hint small,
.game-hint small {
  color: var(--text-secondary);
  font-size: 0.82em;
}

/* ===== Presets ===== */
.preset-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-bottom: 10px;
}

.preset-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 10px 4px;
  border: 2px solid var(--border-color);
  border-radius: 10px;
  background: var(--bg-card);
  cursor: pointer;
  transition: all 0.2s ease;
  color: var(--text-primary);
}

.preset-btn:hover:not(:disabled) {
  border-color: var(--accent-green);
  transform: scale(1.02);
}

.preset-btn.active {
  border-color: var(--accent-green);
  background: color-mix(in srgb, var(--accent-green) 10%, var(--bg-card));
  box-shadow: 0 0 0 1px var(--accent-green);
}

.preset-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.preset-icon { font-size: 18px; }
.preset-name { font-size: 0.78em; font-weight: 600; }
.preset-time { font-size: 0.68em; color: var(--text-secondary); }
.preset-custom .preset-name { color: var(--text-secondary); }

.preset-summary {
  padding: 6px 10px;
  background: var(--bg-secondary);
  border-radius: 8px;
  margin-bottom: 8px;
}

.preset-summary small { color: var(--text-secondary); font-size: 0.82em; }
.custom-settings { padding: 8px 0; }

/* ===== Progress Bars ===== */
.progress {
  width: 100%;
  height: 8px;
  background: var(--bg-secondary);
  border-radius: 6px;
  margin-top: 8px;
  overflow: hidden;
}

.progress .bar {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-green), color-mix(in srgb, var(--accent-green) 70%, #7ff));
  transition: width 0.3s ease;
  border-radius: 6px;
}

.dataset-bar {
  background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 70%, #7bf)) !important;
}

.ttt5-bar {
  background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 60%, #7df)) !important;
  transition: width 0.4s ease;
}

.bg-train-bar {
  background: linear-gradient(90deg, var(--accent), #60a5fa) !important;
}

.overall-progress-panel {
  margin-bottom: 12px;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: color-mix(in srgb, var(--bg-secondary) 72%, transparent);
}

.overall-progress-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}

.overall-progress {
  height: 10px;
}

.overall-bar {
  transition: width 0.25s ease;
}

.overall-progress-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 8px;
  color: var(--text-secondary);
  font-size: 0.92rem;
}

.overall-progress-value {
  color: var(--text-primary);
  font-weight: 700;
  white-space: nowrap;
}

.overall-progress-text {
  flex: 1;
  min-width: 0;
}

.heartbeat-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.8rem;
  border: 1px solid var(--border-color);
  white-space: nowrap;
}

.heartbeat-fresh {
  color: var(--accent-green);
  border-color: color-mix(in srgb, var(--accent-green) 35%, var(--border-color));
  background: color-mix(in srgb, var(--accent-green) 10%, transparent);
}

.heartbeat-stale {
  color: var(--accent-red);
  border-color: color-mix(in srgb, var(--accent-red) 35%, var(--border-color));
  background: color-mix(in srgb, var(--accent-red) 10%, transparent);
}

/* ===== GPU Telemetry ===== */
.gpu-live-stats {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 0.82em;
  color: var(--text-secondary);
}

.gpu-live-stats span {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 3px 8px;
}

/* ===== Training Meta Strip ===== */
.training-meta-strip {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 0.78em;
  color: var(--text-secondary);
}

.training-meta-strip span {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 3px 8px;
}

/* ===== Runtime Pills ===== */
.runtime-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.runtime-pill {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 0.78em;
  color: var(--text-secondary);
}

/* ===== Dataset Progress ===== */
.dataset-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: var(--text-primary);
}

.workers-badge {
  background: var(--accent);
  color: #fff;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.75em;
  font-weight: 600;
}

.dataset-info {
  margin-top: 6px;
  font-size: 0.82em;
  color: var(--text-secondary);
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.rate-info { color: var(--accent-green); font-weight: 500; }
.time-info { color: var(--text-secondary); }

.dataset-stage {
  margin-top: 8px;
  padding: 6px 8px;
  background: var(--bg-secondary);
  border-radius: 6px;
  font-size: 0.78em;
  color: var(--text-secondary);
}

/* ===== TTT5 Training Panel ===== */
.ttt5-training-panel {
  border-color: color-mix(in srgb, var(--accent) 30%, var(--border-color));
}

.phase-badges {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.phase-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.78em;
  font-weight: 600;
  letter-spacing: 0.3px;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.phase-badge::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.phase-pending {
  background: var(--bg-secondary);
  color: var(--text-secondary);
}

.phase-pending::before { background: var(--text-secondary); opacity: 0.4; }

.phase-active {
  background: var(--accent);
  color: #fff;
  animation: pulseBadge 1.5s ease-in-out infinite;
  box-shadow: 0 0 8px color-mix(in srgb, var(--accent) 40%, transparent);
}

.phase-active::before { background: #fff; }

.phase-completed {
  background: var(--accent-green);
  color: #fff;
}

.phase-completed::before { background: #fff; }

@keyframes pulseBadge {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.85; transform: scale(1.03); }
}

/* ===== Training Counters ===== */
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
  min-width: 52px;
}

.counter-label {
  font-size: 0.68em;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
  margin-bottom: 2px;
}

.counter-value {
  font-size: 1.05em;
  font-weight: 700;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}

.counter-total {
  font-weight: 400;
  color: var(--text-secondary);
  font-size: 0.9em;
}

.counter-unit {
  font-weight: 400;
  font-size: 0.8em;
  color: var(--text-secondary);
}

/* ===== Epoch / Metrics ===== */
.epoch-detail {
  margin-top: 8px;
  padding: 6px 10px;
  background: var(--bg-secondary);
  border-radius: 8px;
  font-size: 0.85em;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
}

.epoch-text { font-weight: 600; color: var(--text-primary); }
.batch-text { color: var(--text-secondary); }

.metric-text {
  color: var(--text-secondary);
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 0.92em;
}

.acc-text {
  color: var(--accent-green);
  font-weight: 600;
}

/* ===== Self-play Stats ===== */
.selfplay-stats {
  margin-top: 8px;
  padding: 5px 10px;
  background: var(--bg-secondary);
  border-radius: 8px;
  font-size: 0.85em;
  color: var(--text-secondary);
}

.sp-win { color: var(--accent-green); font-weight: 600; }
.sp-loss { color: var(--accent-red); font-weight: 600; }
.sp-draw { color: var(--accent-orange); font-weight: 600; }

/* ===== Strength Block ===== */
.strength-block {
  margin-top: 10px;
  padding: 8px 10px;
  background: color-mix(in srgb, var(--accent) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
  border-radius: 8px;
}

.strength-header {
  font-size: 0.75em;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}

.strength-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.strength-subblock {
  margin-top: 8px;
}

.strength-subheader {
  font-size: 0.72em;
  font-weight: 700;
  color: color-mix(in srgb, var(--text-secondary, #c8d0dd) 82%, var(--accent) 18%);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 4px;
}

.strength-pill {
  padding: 2px 8px;
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  border-radius: 6px;
  font-size: 0.8em;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}

.strength-pill.promoted {
  background: color-mix(in srgb, var(--accent-green) 15%, transparent);
  color: var(--accent-green);
  font-weight: 700;
}

.strength-pill.rejected {
  background: color-mix(in srgb, var(--accent-red) 12%, transparent);
  color: var(--accent-red);
}

/* ===== Commentary ===== */
.commentary-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.commentary-panel {
  margin-top: 10px;
}

.commentary-header {
  font-size: 0.75em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.commentary-card {
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--bg-secondary);
  border-left: 4px solid var(--text-secondary);
}

.commentary-card.commentary-positive { border-left-color: var(--accent-green); }
.commentary-card.commentary-warning { border-left-color: var(--accent-orange); }
.commentary-card.commentary-danger { border-left-color: var(--accent-red); }

.commentary-title {
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 4px;
  font-size: 0.92em;
}

.commentary-text {
  color: var(--text-primary);
  line-height: 1.45;
  font-size: 0.9em;
}

.commentary-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 8px;
  font-size: 0.8em;
  color: var(--text-secondary);
}

.commentary-empty {
  font-size: 0.88em;
  color: var(--text-secondary);
}

/* ===== Logs ===== */
.logs {
  margin-top: 8px;
  font-size: 0.85em;
  color: var(--text-secondary);
}

.training-timer {
  color: var(--accent);
  font-weight: 500;
}

/* ===== Training Charts ===== */
.training-charts {
  margin-top: 10px;
  padding: 8px;
  background: var(--bg-secondary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.winrate-chart-block {
  border-color: color-mix(in srgb, var(--accent) 30%, var(--border-color));
}

.winrate-chart-meta,
.winrate-chart-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.winrate-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.82em;
  color: var(--text-secondary);
}

.winrate-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.winrate-dot.quick {
  background: var(--accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 20%, transparent);
}

.winrate-dot.confirm {
  background: var(--bg-card);
  border: 2px dashed var(--accent-green);
  box-sizing: border-box;
}

.uplot-chart {
  width: 100%;
  max-width: 100%;
}

/* Make uPlot canvas fill container */
.uplot-chart :deep(.u-wrap),
.uplot-chart :deep(.u-under),
.uplot-chart :deep(.u-over),
.uplot-chart :deep(canvas) {
  width: 100% !important;
}

/* ===== Policy Heatmap ===== */
.policy-heatmap-container {
  margin-top: 8px;
  text-align: center;
}

.heatmap-label {
  font-size: 0.7em;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.policy-heatmap {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  display: block;
  margin: 0 auto;
}

/* ===== Background Training ===== */
.background-training {
  border-color: color-mix(in srgb, var(--accent) 40%, var(--border-color));
}

.bg-train-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: var(--text-primary);
}

.bg-train-epochs {
  font-size: 0.82em;
  color: var(--text-secondary);
}

.bg-train-details {
  margin-top: 8px;
  font-size: 0.88em;
  color: var(--text-primary);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.accent-text { color: var(--accent); font-weight: 600; }
.secondary-text { color: var(--text-secondary); margin-left: 6px; }

/* ===== Responsive: Mobile ===== */
@media (max-width: 640px) {
  .app-header {
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 0;
  }

  .header-right {
    margin-left: 0;
    width: 100%;
    justify-content: flex-end;
  }

  .main-layout {
    flex-direction: column;
    max-height: none;
    overflow: visible;
  }

  .board-area,
  .sidebar-area {
    max-width: 100%;
    max-height: none;
    overflow: visible;
    flex-basis: 100%;
  }

  .preset-row {
    grid-template-columns: repeat(2, 1fr);
  }

  .button-group {
    flex-direction: column;
  }

  .button-group .btn {
    width: 100%;
  }

  .game-controls-row {
    flex-direction: column;
    align-items: flex-start;
  }

  .training-counters {
    gap: 8px;
  }

  .strength-metrics {
    gap: 3px;
  }

  .gpu-pill .gpu-backend {
    display: none;
  }
}

/* ===== Sparkline (legacy compat) ===== */
.sparkline-container { margin-top: 10px; }
.sparkline-legend { display: flex; gap: 14px; font-size: 0.72em; color: var(--text-secondary); margin-bottom: 4px; }
.legend-acc { color: var(--accent-green); }
.legend-loss { color: var(--accent-red); }
.sparkline-canvas { display: block; width: 320px; height: 50px; background: var(--bg-secondary); border-radius: 4px; border: 1px solid var(--border-color); }

/* ===== Select elements ===== */
select {
  padding: 4px 8px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-card);
  color: var(--text-primary);
  font-size: 0.88em;
}
</style>
