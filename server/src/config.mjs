// Общая конфигурация игры и модели
export const BOARD_N = 3;               // для масштабирования поменяйте на 10
export const SEED = 42;                 // Seed для воспроизводимости
export const VOCAB_SIZE = 3;            // значения клетки: 0=пусто, 1=мой, 2=оппонент (или 3-категории)
export const TRANSFORMER_CFG = {
  dModel: 64,                           // Размерность модели (оптимизировано: 64 достаточно для ~5K позиций)
  numLayers: 3,                         // 3 слоя transformer — достаточно для крестиков-ноликов
  heads: 4,                             // 4 головы × 16 keyDim = 64
  dropout: 0.0                          // Dropout не нужен — датасет покрывает все позиции
};
export const TRAIN = {
  batchSize: 64,                        // Маленький батч для частых обновлений (мало данных)
  epochs: 50,                           // 50 эпох для полной конвергенции на ~600 позициях
  lr: 2e-3,                             // Начальный LR (cosine decay снизит к концу)
  weightValue: 0.25                      // Фокус на policy (главное — правильные ходы)
};
export const PV_RESNET = {
  filters: 64,                          // Количество фильтров в ResNet
  blocks: 8                              // Количество ResNet блоков
};

// ===== TTT5 (5x5, 4-in-a-row) =====
export const TTT5_BOARD_N = 5;
export const TTT5_WIN_LEN = 4;           // 4 в ряд для интересной игры на 5x5

export const TTT5_TRANSFORMER_CFG = {
  dModel: 128,                           // Больше для 25 токенов
  numLayers: 4,                          // 4 слоя — вернули для стабильности на 5 эпохах
  heads: 4,                             // 4 головы × 32 keyDim = 128
  dropout: 0.05,                        // Легкий dropout — данных меньше чем позиций
};

export const TTT5_TRAIN = {
  batchSize: 64,
  epochs: 30,
  lr: 1e-3,                             // Ниже чем 3x3 — более стабильное обучение
  weightValue: 0.5,                      // Value важнее для MCTS
};

export const TTT5_MCTS = {
  inferenceSimulations: 80,              // Для игры в реальном времени
  trainingSimulations: 200,              // Глубокий поиск для качественного обучения
  cpuct: 1.5,
  temperature: 1.0,                      // Для обучения (исследование)
  inferenceTemperature: 0.1,             // Для игры (эксплуатация)
  dirichletAlpha: 0.3,                   // Dirichlet noise alpha
  dirichletEpsilon: 0.10,                // Включаем умеренный root-noise только в early-game self-play
  explorationMoves: 5,                   // Ходов с temperature=1.0 (снижено с 8)
  exploitationTemp: 0.3,                 // Temperature после exploration phase
};

export const TTT5_CURRICULUM = {
  replayMax: 5000,                       // Общий replay буфер
  hardMax: 1400,                         // Буфер сложных/ошибочных позиций
  trainingSampleSize: 2400,              // Размер смешанного train batch по позициям
  freshRatio: 0.4,                       // Свежий self-play текущей итерации
  hardRatio: 0.35,                       // Переобучение на ошибках
  replayRatio: 0.25,                     // Стабилизация общим replay
  seedGameRatio: 0.35,                   // Доля игр, стартующих из hard-позиций
  maxHardPerGame: 10,                    // Сколько трудных кейсов брать из одной партии
  lateGameWindow: 8,                     // Последние plies считаются критическими
};

// ===== Gomoku Engine V2 (7x7 — 16x16, 5-in-a-row) =====
// Detailed engine config lives in ./engine/config.mjs
export { GOMOKU_VARIANTS, GOMOKU_TRANSFORMER_CFG, GOMOKU_TRAIN_CFG } from './engine/config.mjs';
