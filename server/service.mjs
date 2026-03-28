import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import fs from 'fs/promises';
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import { TRAIN, TRANSFORMER_CFG, TTT5_TRANSFORMER_CFG, TTT5_MCTS, TTT5_WIN_LEN } from './src/config.mjs';
import { SYMMETRY_MAPS, transformBoard, inverseTransformPolicy } from './src/ttt3_symmetry.mjs';
import { SYMMETRY_MAPS_5, transformBoard as transformBoard5, inverseTransformPolicy as inverseTransformPolicy5 } from './src/ttt5_symmetry.mjs';
import { createGameAdapter } from './src/game_adapter.mjs';
import { inferencePolicy } from './src/inference_policy.mjs';
import { createGomokuEngine, GOMOKU_VARIANTS } from './src/engine/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const _MODEL_DIR = path.resolve(__dirname, 'saved');

// ===== Хранилище игровой истории для дообучения =====
// TTT3: { planes: Float32Array(27), policy: Float32Array(9), value: number }
let gameHistory = [];
const MAX_HISTORY_SIZE = 10000;

// TTT5: { planes: Float32Array(75), policy: Float32Array(25), value: number }
let ttt5GameHistory = [];
const MAX_TTT5_HISTORY = 10000;

// Хранилище полных игр для анализа ошибок
let gameSequences = [];
const MAX_GAME_SEQUENCES = 100;

async function ensureDir(p) { try { await fs.mkdir(p, { recursive: true }); } catch {} }

// ===== TTT3 Transformer Model =====
let ttt3Model = null;
let ttt3ModelLoading = false;

// Перезагрузка модели (сброс кеша)
export function reloadTTT3Model() {
  console.log('[PredictTTT3] Reloading model: clearing cache...');
  if (ttt3Model) {
    try {
      ttt3Model.dispose();
    } catch (e) {
      console.warn('[PredictTTT3] Error disposing old model:', e.message);
    }
  }
  ttt3Model = null;
  ttt3ModelLoading = false;
}

// Загрузка TTT3 Transformer модели
async function ensureTTT3Model() {
  if (ttt3ModelLoading) {
    while (ttt3ModelLoading) await new Promise(r => setTimeout(r, 25));
    return ttt3Model;
  }

  if (ttt3Model) {
    return ttt3Model;
  }

  ttt3ModelLoading = true;
  try {
    const ttt3ModelPath = path.join(_MODEL_DIR, 'ttt3_transformer', 'model.json');
    const exists = await fs.stat(ttt3ModelPath).then(() => true).catch(() => false);

    if (exists) {
      try {
        ttt3Model = await tf.loadLayersModel(`file://${ttt3ModelPath}`);
        console.log('[PredictTTT3] ✓ Loaded TTT3 Transformer model');
        console.log('[PredictTTT3] Model inputs:', ttt3Model.inputs.map(i => i.shape));
        console.log('[PredictTTT3] Model outputs:', ttt3Model.outputs.map(o => o.shape));
      } catch (loadError) {
        console.error('[PredictTTT3] ❌ Failed to load model:', loadError.message);
        console.error('[PredictTTT3] Solution: Delete the model (click "Clear Model") and retrain.');
        ttt3Model = null;
      }
    } else {
      console.log('[PredictTTT3] TTT3 model not found. Use "Train" button to train from scratch.');
    }
  } catch (e) {
    console.error('[PredictTTT3] Error loading model:', e);
    ttt3Model = null;
  } finally {
    ttt3ModelLoading = false;
  }

  return ttt3Model;
}

// Проверяет, существует ли обученная модель
export async function hasTrainedModel() {
  try {
    const modelPath = path.join(_MODEL_DIR, 'ttt3_transformer', 'model.json');
    await fs.access(modelPath);
    return true;
  } catch {
    return false;
  }
}

// ===== TTT5 Transformer Model =====
let ttt5Model = null;
let ttt5ModelLoading = false;
const ttt5Adapter = createGameAdapter({ variant: 'ttt5', winLen: TTT5_WIN_LEN });

export function reloadTTT5Model() {
  console.log('[PredictTTT5] Reloading model: clearing cache...');
  if (ttt5Model) {
    try { ttt5Model.dispose(); } catch (e) { console.warn('[PredictTTT5] Error disposing:', e.message); }
  }
  ttt5Model = null;
  ttt5ModelLoading = false;
}

async function ensureTTT5Model() {
  if (ttt5ModelLoading) {
    while (ttt5ModelLoading) await new Promise(r => setTimeout(r, 25));
    return ttt5Model;
  }
  if (ttt5Model) return ttt5Model;

  ttt5ModelLoading = true;
  try {
    const ttt5ModelPath = path.join(_MODEL_DIR, 'ttt5_transformer', 'model.json');
    const exists = await fs.stat(ttt5ModelPath).then(() => true).catch(() => false);
    if (exists) {
      try {
        ttt5Model = await tf.loadLayersModel(`file://${ttt5ModelPath}`);
        console.log('[PredictTTT5] Loaded TTT5 Transformer model');
      } catch (loadError) {
        console.error('[PredictTTT5] Failed to load model:', loadError.message);
        ttt5Model = null;
      }
    } else {
      console.log('[PredictTTT5] TTT5 model not found. Train first.');
    }
  } catch (e) {
    console.error('[PredictTTT5] Error loading model:', e);
    ttt5Model = null;
  } finally {
    ttt5ModelLoading = false;
  }
  return ttt5Model;
}

// ===== Конвертация доски из WebSocket {0,1,2} в игровой формат {0,+1,-1} =====
// WebSocket: 0=empty, 1=player1(X), 2=player2(O)
// Игровой движок: 0=empty, +1=X, -1=O
function convertBoard(board) {
  return new Int8Array(board.map(v => v === 2 ? -1 : v));
}

function countFilledCells(board) {
  let filled = 0;
  for (let i = 0; i < board.length; i++) {
    if (board[i] !== 0) filled++;
  }
  return filled;
}

function getDynamicTTT5Simulations(board) {
  const base = TTT5_MCTS.inferenceSimulations;
  const filled = countFilledCells(board);
  if (filled < 4) return Math.max(base, 80);
  if (filled < 10) return Math.max(base, 120);
  return Math.max(base, 180);
}

function chooseTTT5FallbackMove(board, player) {
  // Без обученной модели бот играет случайно (только базовая безопасность)
  const legal = ttt5Adapter.legalMoves(board);
  if (legal.length === 0) return -1;

  // Только самые базовые проверки: не проиграть в 1 ход
  const winMove = ttt5Adapter.findImmediateWin(board, player);
  if (winMove >= 0) return winMove;
  const blockMove = ttt5Adapter.findImmediateBlock(board, player);
  if (blockMove >= 0) return blockMove;

  // Без модели — случайный ход из легальных
  return legal[Math.floor(Math.random() * legal.length)];
}

// ===== Предсказание хода =====
async function predictTTT3Move({ board, current = 1 }) {
  const { encodePlanes, maskLegalMoves, legalMoves } = await import('./src/game_ttt3.mjs');
  const { maskLogits } = await import('./src/model_pv_transformer_seq.mjs');
  const { safePick } = await import('./src/safety.mjs');

  if (board.length !== 9) {
    throw new Error(`Invalid board size: expected 9, got ${board.length}`);
  }

  // КРИТИЧНО: конвертируем из {0,1,2} → {0,+1,-1}
  // Без этого encodePlanes не видит фигуры оппонента (2 ≠ -1),
  // blockingMove не может симулировать ходы оппонента
  const boardInt = convertBoard(board);

  const model = await ensureTTT3Model();

  if (!model) {
    // Нет обученной модели — случайные ходы
    const moves = legalMoves(boardInt);
    if (moves.length === 0) {
      return { move: -1, probs: Array(9).fill(0), isRandom: true, mode: 'model', fallback: 'random' };
    }
    const randomMove = moves[Math.floor(Math.random() * moves.length)];
    const probs = Array(9).fill(0);
    probs[randomMove] = 1.0;
    console.log('[PredictTTT3] No trained model, using RANDOM move.');
    return { move: randomMove, probs, value: 0, isRandom: true, mode: 'model', fallback: 'random' };
  }

  // Используем обученную модель
  const player = current === 1 ? 1 : -1;
  const symmetryBoards = SYMMETRY_MAPS.map(({ map }) => transformBoard(boardInt, map));
  const xInput = new Float32Array(symmetryBoards.length * 27);
  const maskInput = new Float32Array(symmetryBoards.length * 9);
  const posInput = new Int32Array(symmetryBoards.length * 9);

  for (let i = 0; i < symmetryBoards.length; i++) {
    const planes = encodePlanes(symmetryBoards[i], player);
    const mask = maskLegalMoves(symmetryBoards[i]);
    xInput.set(planes, i * 27);
    maskInput.set(mask, i * 9);
    for (let j = 0; j < 9; j++) {
      posInput[i * 9 + j] = j;
    }
  }

  const x = tf.tensor3d(xInput, [symmetryBoards.length, 9, 3]);
  const pos = tf.tensor2d(posInput, [symmetryBoards.length, 9], 'int32');
  const maskTensor = tf.tensor2d(maskInput, [symmetryBoards.length, 9]);

  const { probsData, valuesData } = tf.tidy(() => {
    const [logits, valueTensor] = model.predict([x, pos]);
    const maskedLogits = maskLogits(logits, maskTensor);
    const probs = tf.softmax(maskedLogits);
    return {
      probsData: probs.dataSync(),
      valuesData: valueTensor.dataSync()
    };
  });

  x.dispose();
  pos.dispose();
  maskTensor.dispose();

  const averagedPolicy = new Float32Array(9);
  let valueSum = 0;

  for (let i = 0; i < SYMMETRY_MAPS.length; i++) {
    const start = i * 9;
    const transformedPolicy = probsData.slice(start, start + 9);
    const restoredPolicy = inverseTransformPolicy(transformedPolicy, SYMMETRY_MAPS[i].map);

    for (let j = 0; j < 9; j++) {
      averagedPolicy[j] += restoredPolicy[j];
    }
    valueSum += valuesData[i];
  }

  const policyArray = Array.from(averagedPolicy, v => v / SYMMETRY_MAPS.length);
  const value = valueSum / SYMMETRY_MAPS.length;

  // Safety rules — используем boardInt (формат {-1,0,+1})
  const moves = legalMoves(boardInt);
  if (moves.length === 0) {
    return { move: -1, probs: Array(9).fill(0), value, isRandom: false, mode: 'model' };
  }

  const policyMax = Math.max(...policyArray);
  const policyMaxIdx = policyArray.indexOf(policyMax);

  const { winningMove, blockingMove } = await import('./src/safety.mjs');
  const winMove = winningMove(boardInt, player);
  const blockMove = blockingMove(boardInt, player);

  const move = safePick(boardInt, player, policyArray);

  const wasChangedBySafety = move !== policyMaxIdx;
  if (wasChangedBySafety) {
    console.log('[PredictTTT3] ⚠️ Safety override:', {
      model: policyMaxIdx,
      safety: move,
      reason: winMove >= 0 ? 'winning' : (blockMove >= 0 ? 'blocking' : 'unknown')
    });
  }

  return {
    move,
    probs: policyArray,
    value,
    isRandom: false,
    mode: 'model',
    confidence: policyMax
  };
}

// ===== Предсказание хода TTT5 (5x5, 4-in-a-row) =====
async function predictTTT5Move({ board, current = 1 }) {
  const { maskLogits } = await import('./src/model_pv_transformer_seq.mjs');

  if (board.length !== 25) {
    throw new Error(`Invalid board size for TTT5: expected 25, got ${board.length}`);
  }

  const boardInt = convertBoard(board);
  const model = await ensureTTT5Model();
  const player = current === 1 ? 1 : -1;

  if (!model) {
    const move = chooseTTT5FallbackMove(boardInt, player);
    if (move < 0) {
      return { move: -1, probs: Array(25).fill(0), isRandom: true, mode: 'model', fallback: 'random' };
    }
    const probs = Array(25).fill(0);
    probs[move] = 1;
    return { move, probs, value: 0, isRandom: false, mode: 'model', fallback: 'heuristic' };
  }

  // Create nnEval callback for inference policy
  const nnEval = async (b, p) => {
    const symmetryBoards = SYMMETRY_MAPS_5.map(({ map }) => transformBoard5(b, map));
    const xInput = new Float32Array(symmetryBoards.length * 75);
    const posInput = new Int32Array(symmetryBoards.length * 25);
    const maskInput = new Float32Array(symmetryBoards.length * 25);

    for (let i = 0; i < symmetryBoards.length; i++) {
      const planes = ttt5Adapter.encodePlanes(symmetryBoards[i], p);
      const mask = ttt5Adapter.maskLegalMoves(symmetryBoards[i]);
      xInput.set(planes, i * 75);
      maskInput.set(mask, i * 25);
      for (let j = 0; j < 25; j++) posInput[i * 25 + j] = j;
    }

    const x = tf.tensor3d(xInput, [symmetryBoards.length, 25, 3]);
    const posIndices = tf.tensor2d(posInput, [symmetryBoards.length, 25], 'int32');
    const maskTensor = tf.tensor2d(maskInput, [symmetryBoards.length, 25]);

    const { probsData, valueData } = tf.tidy(() => {
      const [logits, valueTensor] = model.predict([x, posIndices]);
      const masked = maskLogits(logits, maskTensor);
      const probs = tf.softmax(masked, -1);
      return {
        probsData: probs.dataSync(),
        valueData: valueTensor.dataSync(),
      };
    });

    x.dispose();
    posIndices.dispose();
    maskTensor.dispose();

    const averagedPolicy = new Float32Array(25);
    let valueSum = 0;
    for (let i = 0; i < SYMMETRY_MAPS_5.length; i++) {
      const start = i * 25;
      const transformedPolicy = probsData.slice(start, start + 25);
      const restoredPolicy = inverseTransformPolicy5(transformedPolicy, SYMMETRY_MAPS_5[i].map);
      for (let j = 0; j < 25; j++) averagedPolicy[j] += restoredPolicy[j];
      valueSum += valueData[i];
    }

    for (let i = 0; i < 25; i++) averagedPolicy[i] /= SYMMETRY_MAPS_5.length;

    return {
      policy: averagedPolicy,
      value: valueSum / SYMMETRY_MAPS_5.length,
    };
  };

  // Use InferencePolicy with MCTS for stable play
  const result = await inferencePolicy({
    adapter: ttt5Adapter,
    nnEval,
    board: boardInt,
    player,
    useMCTS: true,
    mctsSimulations: getDynamicTTT5Simulations(boardInt),
    mctsCpuct: TTT5_MCTS.cpuct,
    temperature: TTT5_MCTS.inferenceTemperature,
  });

  return {
    move: result.move,
    probs: result.policy,
    value: result.value,
    isRandom: false,
    mode: 'model',
    source: result.source,
    confidence: Math.max(...result.policy),
  };
}

// ===== Gomoku Engine V2 (7x7 — 16x16, 5-in-a-row) =====
const gomokuEngines = new Map(); // cache engines by variant

function getGomokuEngine(variant) {
  if (gomokuEngines.has(variant)) return gomokuEngines.get(variant);

  const config = GOMOKU_VARIANTS[variant];
  if (!config) {
    // Parse from variant name: gomokuN -> N
    const N = parseInt(variant.replace('gomoku', ''), 10);
    if (!N || N < 7 || N > 16) throw new Error(`Unknown gomoku variant: ${variant}`);
    const engine = createGomokuEngine({ N, winLen: 5 });
    gomokuEngines.set(variant, engine);
    return engine;
  }

  const engine = createGomokuEngine({ N: config.N, winLen: config.winLen });
  gomokuEngines.set(variant, engine);
  return engine;
}

function inferGomokuVariant(boardLen) {
  const N = Math.round(Math.sqrt(boardLen));
  if (N * N !== boardLen || N < 7 || N > 16) return null;
  return `gomoku${N}`;
}

async function predictGomokuMove({ board, current = 1, variant }) {
  const boardInt = convertBoard(board);
  const player = current === 1 ? 1 : -1;
  const engine = getGomokuEngine(variant);

  console.log(`[PredictGomoku] variant=${variant}, N=${engine.N}, player=${player}, board.length=${board.length}`);

  const result = engine.bestMove(boardInt, player, {
    timeLimitMs: 3000,
  });

  console.log(`[PredictGomoku] move=${result.move}, value=${result.value?.toFixed(3)}, source=${result.source}, depth=${result.depth}, nodes=${result.nodesSearched}`);

  return {
    move: result.move,
    probs: result.policy ? Array.from(result.policy) : Array(board.length).fill(0),
    value: result.value,
    isRandom: false,
    mode: 'engine',
    source: result.source,
    confidence: result.policy ? Math.max(...result.policy) : 1.0,
    depth: result.depth,
    nodesSearched: result.nodesSearched,
  };
}

export async function predictMove({ board, current = 1, mode = 'model', variant = 'ttt3' }) {
  // Gomoku engine routing (7x7 — 16x16)
  if (variant.startsWith('gomoku') || (board.length > 25 && inferGomokuVariant(board.length))) {
    const gomokuVariant = variant.startsWith('gomoku') ? variant : inferGomokuVariant(board.length);
    try {
      return await predictGomokuMove({ board, current, variant: gomokuVariant });
    } catch (e) {
      console.error('[Predict] Error in Gomoku prediction:', e);
      // Fallback: center move
      const N = Math.round(Math.sqrt(board.length));
      const mid = Math.floor(N / 2) * N + Math.floor(N / 2);
      return { move: mid, probs: Array(board.length).fill(0), isRandom: true, mode: 'engine' };
    }
  }

  // TTT5 routing
  if (variant === 'ttt5' || board.length === 25) {
    if (mode === 'algorithm') {
      // No minimax for 5x5 — use model+MCTS instead
      console.log('[Predict] TTT5 algorithm mode → using model+MCTS');
    }
    try {
      return await predictTTT5Move({ board, current });
    } catch (e) {
      console.error('[Predict] Error in TTT5 prediction:', e);
      const moves = ttt5Adapter.legalMoves(convertBoard(board));
      const randomMove = moves.length > 0 ? moves[Math.floor(Math.random() * moves.length)] : -1;
      return { move: randomMove, probs: Array(25).fill(0), isRandom: true, mode: 'model' };
    }
  }

  // ===== TTT3 routing below =====

  // Режим алгоритма — minimax (только для 3x3)
  if (mode === 'algorithm') {
    const { getTeacherValueAndPolicy } = await import('./src/ttt3_minimax.mjs');
    const { legalMoves } = await import('./src/game_ttt3.mjs');

    const player = current === 1 ? 1 : -1;
    const boardInt8 = convertBoard(board); // {0,1,2} → {0,+1,-1}
    const moves = legalMoves(boardInt8);
    if (moves.length === 0) {
      return { move: -1, probs: Array(9).fill(0), mode: 'algorithm' };
    }

    const { policy } = getTeacherValueAndPolicy(boardInt8, player);
    let bestMove = -1;
    let bestProb = -1;
    for (const m of moves) {
      if (policy[m] > bestProb) {
        bestProb = policy[m];
        bestMove = m;
      }
    }

    console.log('[Predict] Using algorithm (minimax), move:', bestMove);
    return { move: bestMove, probs: Array.from(policy), mode: 'algorithm' };
  }

  // Режим модели (TTT3)
  if (board.length === 9) {
    try {
      return await predictTTT3Move({ board, current });
    } catch (e) {
      console.error('[Predict] Error in TTT3 prediction:', e);
    }
  }

  // Fallback — случайный ход
  const { legalMoves: getLegalMoves } = await import('./src/game_ttt3.mjs');
  const boardInt8 = convertBoard(board); // {0,1,2} → {0,+1,-1}
  const moves = getLegalMoves(boardInt8);
  if (moves.length === 0) {
    return { move: -1, probs: Array(9).fill(0), isRandom: true, mode: 'model' };
  }
  const randomMove = moves[Math.floor(Math.random() * moves.length)];
  return { move: randomMove, probs: Array(9).fill(1 / 9), isRandom: true, mode: 'model' };
}

// ===== Игровая история =====

// Сохраняет ход для TTT3 или TTT5.
export async function saveGameMove({ board, move, current, gameId, variant }) {
  try {
    const inferredVariant = variant || (board?.length === 25 ? 'ttt5' : 'ttt3');

    const player = current === 1 ? 1 : -1;
    const boardInt8 = convertBoard(board);

    // Сохраняем в последовательность игры (для обоих вариантов)
    if (gameId !== undefined) {
      const gameSeq = gameSequences.find(g => g.id === gameId);
      if (gameSeq) {
        gameSeq.moves.push({ board: [...board], move, current });
      }
    }

    if (inferredVariant === 'ttt5' && board.length === 25) {
      // TTT5: сохраняем с uniform policy (без minimax — учитель будет MCTS при анализе)
      const planes = ttt5Adapter.encodePlanes(boardInt8, player);
      const uniformPolicy = new Float32Array(25).fill(1 / 25);

      ttt5GameHistory.push({
        planes: new Float32Array(planes),
        policy: uniformPolicy,
        value: 0
      });

      if (ttt5GameHistory.length > MAX_TTT5_HISTORY) {
        ttt5GameHistory = ttt5GameHistory.slice(-MAX_TTT5_HISTORY);
      }

      console.log(`[TTT5 GameHistory] Saved move, total: ${ttt5GameHistory.length}`);
    } else if (inferredVariant === 'ttt3' && board.length === 9) {
      // TTT3: сохраняем с minimax teacher
      const { encodePlanes } = await import('./src/game_ttt3.mjs');
      const { getTeacherValueAndPolicy } = await import('./src/ttt3_minimax.mjs');

      const planes = encodePlanes(boardInt8, player);
      const { value, policy } = getTeacherValueAndPolicy(boardInt8, player);

      gameHistory.push({
        planes: new Float32Array(planes),
        policy: new Float32Array(policy),
        value
      });

      if (gameHistory.length > MAX_HISTORY_SIZE) {
        gameHistory = gameHistory.slice(-MAX_HISTORY_SIZE);
      }

      console.log(`[GameHistory] Saved move (TTT3), total: ${gameHistory.length}`);
    }
  } catch (e) {
    console.error('[GameHistory] Error saving move:', e);
  }
}

// Начинает новую игру
export function startNewGame({ playerRole = 1, variant = 'ttt3' } = {}) {
  const gameId = Date.now() + Math.random();
  gameSequences.push({
    id: gameId,
    moves: [],
    winner: null,
    playerRole,
    variant,
  });

  if (gameSequences.length > MAX_GAME_SEQUENCES) {
    gameSequences = gameSequences.slice(-MAX_GAME_SEQUENCES);
  }

  return gameId;
}

// ===== Анализ ошибок и генерация коррекций =====
let backgroundTraining = false;
let backgroundTrainingPromise = null;

export async function finishGame({ gameId, winner, patternsPerError = 100, autoTrain = false, progressCb = null, incrementalBatchSize = 256 }) {
  const gameSeq = gameSequences.find(g => g.id === gameId);
  if (!gameSeq) return;

  gameSeq.winner = winner;
  const isTTT5 = gameSeq.variant === 'ttt5';

  // Модель проиграла?
  const modelLost = (gameSeq.playerRole === 1 && winner === 2) || (gameSeq.playerRole === 2 && winner === 1);

  if (isTTT5) {
    // ===== TTT5 Online Learning =====
    const patternsBefore = ttt5GameHistory.length;

    if (modelLost) {
      console.log(`[TTT5 ErrorDetection] Model lost game ${gameId}, analyzing mistakes...`);
      await analyzeTTT5Errors(gameSeq, patternsPerError);
    }

    const newSkillsCount = ttt5GameHistory.length - patternsBefore;

    if (autoTrain && !backgroundTraining) {
      console.log('[TTT5 FinishGame] Auto-train enabled, starting TTT5 background training...');
      startTTT5BackgroundTraining(progressCb, newSkillsCount, incrementalBatchSize);
    }
  } else if (!gameSeq.variant || gameSeq.variant === 'ttt3') {
    // ===== TTT3 Online Learning (original) =====
    const patternsBefore = gameHistory.length;

    if (modelLost) {
      console.log(`[ErrorDetection] Model lost game ${gameId}, analyzing mistakes...`);
      await analyzeAndGenerateCorrections(gameSeq, patternsPerError);
    }

    const newSkillsCount = gameHistory.length - patternsBefore;

    if (autoTrain && !backgroundTraining) {
      console.log('[FinishGame] Auto-train enabled, starting background training...');
      startBackgroundTraining(progressCb, newSkillsCount, incrementalBatchSize);
    }
  }
}

// Анализирует ошибки и генерирует ЛЕГАЛЬНЫЕ обучающие паттерны
async function analyzeAndGenerateCorrections(gameSeq, patternsPerError = 100) {
  try {
    const { encodePlanes, legalMoves, winner: getWinner, emptyBoard, applyMove } = await import('./src/game_ttt3.mjs');
    const { getTeacherValueAndPolicy } = await import('./src/ttt3_minimax.mjs');
    const corrections = [];

    for (let i = 0; i < gameSeq.moves.length; i++) {
      const moveData = gameSeq.moves[i];

      // Проверяем, это ход модели?
      const isModelMove = (gameSeq.playerRole === 1 && moveData.current === 1) ||
                          (gameSeq.playerRole === 2 && moveData.current === 2);
      if (!isModelMove) continue;

      const player = moveData.current === 1 ? 1 : -1;
      const boardInt8 = convertBoard(moveData.board); // {0,1,2} → {0,+1,-1}

      // Получаем оптимальный ход от minimax
      const { policy: teacherPolicy } = getTeacherValueAndPolicy(boardInt8, player);

      // Проверяем, совпал ли ход модели с оптимальным
      if (teacherPolicy[moveData.move] < 0.01) {
        // Модель сделала ошибку!
        console.log(`[ErrorDetection] Mistake at move ${i}: model chose ${moveData.move}, optimal: ${teacherPolicy.map((p, j) => p > 0.01 ? j : '').filter(Boolean).join(',')}`);

        // Генерируем легальные вариации из дерева игры
        const variations = await generateLegalVariations(boardInt8, player, patternsPerError);
        corrections.push(...variations);
      }
    }

    if (corrections.length > 0) {
      console.log(`[ErrorDetection] Generated ${corrections.length} correction patterns`);
      gameHistory.push(...corrections);

      if (gameHistory.length > MAX_HISTORY_SIZE) {
        gameHistory = gameHistory.slice(-MAX_HISTORY_SIZE);
      }
    }
  } catch (e) {
    console.error('[ErrorDetection] Error analyzing mistakes:', e);
  }
}

// Генерирует ЛЕГАЛЬНЫЕ вариации позиции из дерева игры
async function generateLegalVariations(targetBoard, targetPlayer, count = 100) {
  const { encodePlanes, legalMoves, winner: getWinner, emptyBoard, applyMove } = await import('./src/game_ttt3.mjs');
  const { getTeacherValueAndPolicy } = await import('./src/ttt3_minimax.mjs');

  const variations = [];
  const visited = new Set();

  // Добавляем саму ошибочную позицию
  const { value, policy } = getTeacherValueAndPolicy(targetBoard, targetPlayer);
  variations.push({
    planes: new Float32Array(encodePlanes(targetBoard, targetPlayer)),
    policy: new Float32Array(policy),
    value
  });
  visited.add(Array.from(targetBoard).join(','));

  // Подсчитываем глубину целевой позиции (количество заполненных клеток)
  const targetDepth = Array.from(targetBoard).filter(v => v !== 0).length;

  // Обходим дерево игры, собираем позиции на похожей глубине
  function collect(board, player, depth) {
    if (variations.length >= count) return;

    const key = Array.from(board).join(',');
    if (visited.has(key)) return;
    visited.add(key);

    const w = getWinner(board);
    if (w !== null) return; // Терминальное состояние — пропускаем

    const filled = Array.from(board).filter(v => v !== 0).length;

    // Берём позиции на похожей глубине (±1 ход)
    if (Math.abs(filled - targetDepth) <= 1 && filled > 0) {
      const { value: v, policy: pol } = getTeacherValueAndPolicy(board, player);
      variations.push({
        planes: new Float32Array(encodePlanes(board, player)),
        policy: new Float32Array(pol),
        value: v
      });
    }

    // Продолжаем обход только если ещё нужны вариации
    if (variations.length < count) {
      const moves = legalMoves(board);
      // Перемешиваем для разнообразия
      for (let i = moves.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [moves[i], moves[j]] = [moves[j], moves[i]];
      }
      for (const move of moves) {
        if (variations.length >= count) return;
        collect(applyMove(board, move, player), -player, depth + 1);
      }
    }
  }

  collect(emptyBoard(), 1, 0);
  return variations.slice(0, count);
}

// ===== TTT5 Error Analysis (uses MCTS as teacher) =====
async function analyzeTTT5Errors(gameSeq, patternsPerError = 20) {
  try {
    const { maskLogits } = await import('./src/model_pv_transformer_seq.mjs');
    const model = await ensureTTT5Model();
    const corrections = [];

    // Создаём nnEval для MCTS (если модель загружена)
    let nnEval = null;
    if (model) {
      nnEval = async (b, p) => {
        const symmetryBoards = SYMMETRY_MAPS_5.map(({ map }) => transformBoard5(b, map));
        const xInput = new Float32Array(symmetryBoards.length * 75);
        const posInput = new Int32Array(symmetryBoards.length * 25);
        const maskInput = new Float32Array(symmetryBoards.length * 25);

        for (let i = 0; i < symmetryBoards.length; i++) {
          const planes = ttt5Adapter.encodePlanes(symmetryBoards[i], p);
          const mask = ttt5Adapter.maskLegalMoves(symmetryBoards[i]);
          xInput.set(planes, i * 75);
          maskInput.set(mask, i * 25);
          for (let j = 0; j < 25; j++) posInput[i * 25 + j] = j;
        }

        const x = tf.tensor3d(xInput, [symmetryBoards.length, 25, 3]);
        const posIndices = tf.tensor2d(posInput, [symmetryBoards.length, 25], 'int32');
        const maskTensor = tf.tensor2d(maskInput, [symmetryBoards.length, 25]);

        const { probsData, valueData } = tf.tidy(() => {
          const [logits, valueTensor] = model.predict([x, posIndices]);
          const masked = maskLogits(logits, maskTensor);
          const probs = tf.softmax(masked, -1);
          return { probsData: probs.dataSync(), valueData: valueTensor.dataSync() };
        });

        x.dispose(); posIndices.dispose(); maskTensor.dispose();

        const averagedPolicy = new Float32Array(25);
        let valueSum = 0;
        for (let i = 0; i < SYMMETRY_MAPS_5.length; i++) {
          const start = i * 25;
          const transformedPolicy = probsData.slice(start, start + 25);
          const restoredPolicy = inverseTransformPolicy5(transformedPolicy, SYMMETRY_MAPS_5[i].map);
          for (let j = 0; j < 25; j++) averagedPolicy[j] += restoredPolicy[j];
          valueSum += valueData[i];
        }
        for (let i = 0; i < 25; i++) averagedPolicy[i] /= SYMMETRY_MAPS_5.length;

        return { policy: averagedPolicy, value: valueSum / SYMMETRY_MAPS_5.length };
      };
    }

    for (let i = 0; i < gameSeq.moves.length; i++) {
      const moveData = gameSeq.moves[i];

      // Проверяем, это ход модели?
      const isModelMove = (gameSeq.playerRole === 1 && moveData.current === 1) ||
                          (gameSeq.playerRole === 2 && moveData.current === 2);
      if (!isModelMove) continue;

      const player = moveData.current === 1 ? 1 : -1;
      const boardInt8 = convertBoard(moveData.board);

      if (nnEval) {
        // Используем MCTS для получения "учительской" политики
        try {
          const result = await inferencePolicy({
            adapter: ttt5Adapter,
            nnEval,
            board: boardInt8,
            player,
            useMCTS: true,
            mctsSimulations: 200, // Глубокий поиск для учителя
            mctsCpuct: TTT5_MCTS.cpuct,
            temperature: 0.3,
          });

          const teacherPolicy = result.policy;

          // Если MCTS сильно не согласен с ходом модели
          if (teacherPolicy[moveData.move] < 0.05) {
            console.log(`[TTT5 ErrorDetection] Mistake at move ${i}: model chose ${moveData.move}, MCTS best: ${teacherPolicy.indexOf(Math.max(...teacherPolicy))}`);

            // Добавляем коррекцию — позицию с MCTS policy
            const planes = ttt5Adapter.encodePlanes(boardInt8, player);
            corrections.push({
              planes: new Float32Array(planes),
              policy: new Float32Array(teacherPolicy),
              value: result.value,
            });
          }
        } catch (e) {
          console.warn(`[TTT5 ErrorDetection] MCTS failed for move ${i}:`, e.message);
        }
      } else {
        // Без модели — сохраняем позицию с heuristic safety-based policy
        const planes = ttt5Adapter.encodePlanes(boardInt8, player);
        const winMove = ttt5Adapter.findImmediateWin(boardInt8, player);
        const blockMove = ttt5Adapter.findImmediateBlock(boardInt8, player);

        if (winMove >= 0 && moveData.move !== winMove) {
          const policy = new Float32Array(25);
          policy[winMove] = 1.0;
          corrections.push({ planes: new Float32Array(planes), policy, value: 1.0 });
          console.log(`[TTT5 ErrorDetection] Missed win at move ${i}`);
        } else if (blockMove >= 0 && moveData.move !== blockMove) {
          const policy = new Float32Array(25);
          policy[blockMove] = 1.0;
          corrections.push({ planes: new Float32Array(planes), policy, value: -0.3 });
          console.log(`[TTT5 ErrorDetection] Missed block at move ${i}`);
        }
      }
    }

    if (corrections.length > 0) {
      console.log(`[TTT5 ErrorDetection] Generated ${corrections.length} correction patterns`);
      ttt5GameHistory.push(...corrections);

      if (ttt5GameHistory.length > MAX_TTT5_HISTORY) {
        ttt5GameHistory = ttt5GameHistory.slice(-MAX_TTT5_HISTORY);
      }
    }
  } catch (e) {
    console.error('[TTT5 ErrorDetection] Error:', e);
  }
}

// Fisher-Yates shuffle
function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function getEffectiveBatchSize(requestedBatchSize, totalSamples) {
  let maxUsefulBatch = 256;

  if (totalSamples <= 8192) {
    maxUsefulBatch = 64;
  } else if (totalSamples <= 32768) {
    maxUsefulBatch = 128;
  }

  return Math.max(32, Math.min(requestedBatchSize, maxUsefulBatch, totalSamples));
}

// ===== Дообучение на играх (с experience replay) =====
export async function trainOnGames(progressCb, { epochs = 1, batchSize = 256, variant = 'ttt3' } = {}) {
  try {
    // TTT5 routing
    if (variant === 'ttt5') {
      return await trainTTT5OnGames(progressCb, { epochs, batchSize });
    }

    if (variant !== 'ttt3') {
      throw new Error(`Incremental training is supported only for ttt3/ttt5 (got ${variant})`);
    }

    if (gameHistory.length < 10) {
      throw new Error(`Недостаточно данных для обучения. Нужно минимум 10 ходов, есть ${gameHistory.length}`);
    }

    // Загружаем TTT3 модель (ту же, которая делает предсказания!)
    const model = await ensureTTT3Model();
    if (!model) {
      throw new Error('TTT3 модель не найдена. Сначала обучите модель с нуля.');
    }

    // Кастомная loss: softmax cross-entropy из логитов
    // Модель выдаёт raw logits, а categoricalCrossentropy ожидает probabilities!
    function policyLossFromLogits(yTrue, yPred) {
      const logProbs = tf.logSoftmax(yPred, -1);
      return tf.neg(tf.sum(tf.mul(yTrue, logProbs), -1)).mean();
    }

    // Компилируем для обучения
    model.compile({
      optimizer: tf.train.adam(TRAIN.lr * 0.5), // Пониженный LR для дообучения
      loss: [policyLossFromLogits, 'meanSquaredError'],
      lossWeights: [1.0, TRAIN.weightValue]
    });

    // ===== Experience Replay =====
    // Смешиваем новые паттерны (ошибки) с данными из minimax (3:1 ratio)
    const { teacherBatches } = await import('./src/ttt3_minimax.mjs');
    const replayTarget = Math.max(gameHistory.length * 3, 1000);
    const replayData = [];

    for (const batch of teacherBatches({ batchSize: replayTarget })) {
      for (let i = 0; i < batch.count && replayData.length < replayTarget; i++) {
        replayData.push({
          planes: new Float32Array(batch.x.slice(i * 27, (i + 1) * 27)),
          policy: new Float32Array(batch.yPolicy.slice(i * 9, (i + 1) * 9)),
          value: batch.yValue[i]
        });
      }
      if (replayData.length >= replayTarget) break;
    }

    // Объединяем и перемешиваем
    const combined = [...gameHistory, ...replayData];
    shuffleArray(combined);

    const N = combined.length;
    const actualEpochs = epochs;
    const effectiveBatchSize = getEffectiveBatchSize(batchSize, N);

    progressCb?.({ type: 'train.start', payload: { epochs: actualEpochs, batchSize, nTrain: N, nVal: 0 } });
    progressCb?.({
      type: 'train.status',
      payload: {
        message: `Подготовка данных (${gameHistory.length} ошибок + ${replayData.length} replay = ${N}, batch ${effectiveBatchSize})...`
      }
    });

    console.log(`[TrainOnGames] Training on ${N} samples (${gameHistory.length} errors + ${replayData.length} replay)`);

    // Создаём тензоры
    const planesArr = new Float32Array(N * 27);
    const posArr = new Int32Array(N * 9);
    const policyArr = new Float32Array(N * 9);
    const valueArr = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      planesArr.set(combined[i].planes, i * 27);
      policyArr.set(combined[i].policy, i * 9);
      valueArr[i] = combined[i].value;
      for (let j = 0; j < 9; j++) {
        posArr[i * 9 + j] = j;
      }
    }

    const xTensor = tf.tensor3d(Array.from(planesArr), [N, 9, 3]);
    const posTensor = tf.tensor2d(Array.from(posArr), [N, 9], 'int32');
    const yPolicyTensor = tf.tensor2d(Array.from(policyArr), [N, 9]);
    const yValueTensor = tf.tensor2d(Array.from(valueArr), [N, 1]);

    progressCb?.({ type: 'train.status', payload: { message: 'Дообучение с experience replay...' } });

    await model.fit([xTensor, posTensor], [yPolicyTensor, yValueTensor], {
      epochs: actualEpochs,
      batchSize: effectiveBatchSize,
      shuffle: true,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`[TrainOnGames] Epoch ${epoch + 1}/${actualEpochs}, loss: ${logs.loss?.toFixed(4)}`);
          progressCb?.({
            type: 'train.progress',
            payload: {
              epoch: epoch + 1,
              epochs: actualEpochs,
              loss: Number(logs.loss ?? 0).toFixed(4),
              acc: '0',
              val_loss: '0',
              val_acc: '0',
              percent: Math.round(((epoch + 1) / actualEpochs) * 100),
            }
          });
        },
        onTrainEnd: () => {
          progressCb?.({ type: 'train.done', payload: { saved: true } });
        },
      }
    });

    // Сохраняем в TTT3 директорию
    const saveDir = path.join(_MODEL_DIR, 'ttt3_transformer');
    await ensureDir(saveDir);
    await model.save(`file://${saveDir}`);

    // Перезагружаем модель
    reloadTTT3Model();
    console.log('[TrainOnGames] Model saved and reloaded');

    // Cleanup
    xTensor.dispose();
    posTensor.dispose();
    yPolicyTensor.dispose();
    yValueTensor.dispose();

  } catch (e) {
    console.error('[TrainOnGames] Error:', e);
    progressCb?.({ type: 'error', error: String(e) });
    throw e;
  }
}

// ===== Фоновое мини-обучение после игры =====
async function startBackgroundTraining(progressCb = null, newSkillsCount = 0, incrementalBatchSize = 256) {
  if (backgroundTraining) {
    console.log('[BackgroundTrain] Training already in progress, skipping...');
    return;
  }

  const stats = getGameHistoryStats();
  if (stats.count < 10) {
    console.log(`[BackgroundTrain] Not enough data (${stats.count} moves), skipping...`);
    return;
  }

  backgroundTraining = true;
  console.log('[BackgroundTrain] Starting background training...');

  const totalSkills = stats.count;
  progressCb?.({
    type: 'background_train.start',
    payload: {
      newSkills: newSkillsCount,
      totalSkills: totalSkills,
      epochs: 1,
      message: `Новые навыки: ${newSkillsCount} из ${totalSkills}`
    }
  });

  backgroundTrainingPromise = (async () => {
    try {
      await trainOnGames(
        (ev) => {
          if (ev.type === 'train.progress') {
            progressCb?.({
              type: 'background_train.progress',
              payload: {
                epoch: ev.payload?.epoch || 0,
                epochs: ev.payload?.epochs || 1,
                epochPercent: ev.payload?.percent || 0,
                newSkills: newSkillsCount,
                totalSkills: totalSkills,
                loss: ev.payload?.loss,
                message: `Эпоха ${ev.payload?.epoch}/${ev.payload?.epochs} · Навыки: ${newSkillsCount}`
              }
            });
          } else if (ev.type === 'train.done') {
            progressCb?.({
              type: 'background_train.done',
              payload: {
                newSkills: newSkillsCount,
                totalSkills: totalSkills,
                message: `Обучение завершено: ${newSkillsCount} навыков усвоено`
              }
            });
          } else if (ev.type === 'error') {
            progressCb?.({
              type: 'background_train.error',
              error: ev.error,
              payload: { message: `Ошибка: ${ev.error}` }
            });
          }
        },
        {
          epochs: 1,
          batchSize: incrementalBatchSize,
        }
      );
      console.log('[BackgroundTrain] Background training completed');
    } catch (e) {
      console.error('[BackgroundTrain] Error:', e);
      progressCb?.({
        type: 'background_train.error',
        error: String(e),
        payload: { message: `Ошибка: ${e.message}` }
      });
    } finally {
      backgroundTraining = false;
      backgroundTrainingPromise = null;
    }
  })();
}

export function isBackgroundTraining() {
  return backgroundTraining;
}

// ===== TTT5 Incremental Training on Games =====
async function trainTTT5OnGames(progressCb, { epochs = 1, batchSize = 256 } = {}) {
  if (ttt5GameHistory.length < 5) {
    throw new Error(`Недостаточно данных TTT5. Нужно минимум 5 ходов, есть ${ttt5GameHistory.length}`);
  }

  const model = await ensureTTT5Model();
  if (!model) {
    throw new Error('TTT5 модель не найдена. Сначала обучите модель с нуля.');
  }

  function policyLossFromLogits(yTrue, yPred) {
    const logProbs = tf.logSoftmax(yPred, -1);
    return tf.neg(tf.sum(tf.mul(yTrue, logProbs), -1)).mean();
  }

  const { TTT5_TRAIN } = await import('./src/config.mjs');

  model.compile({
    optimizer: tf.train.adam((TTT5_TRAIN?.lr || 1e-3) * 0.3),
    loss: [policyLossFromLogits, 'meanSquaredError'],
    lossWeights: [1.0, TTT5_TRAIN?.weightValue || 0.5]
  });

  // Self-distillation replay: old entries serve as replay buffer
  const combined = shuffleArray([...ttt5GameHistory]);
  const N = combined.length;
  const effectiveBatchSize = getEffectiveBatchSize(batchSize, N);

  progressCb?.({ type: 'train.start', payload: { epochs, batchSize, nTrain: N, nVal: 0, variant: 'ttt5' } });
  progressCb?.({
    type: 'train.status',
    payload: { message: `TTT5: ${N} позиций, batch ${effectiveBatchSize}...` }
  });

  console.log(`[TTT5 TrainOnGames] Training on ${N} samples`);

  const planesArr = new Float32Array(N * 75);
  const posArr = new Int32Array(N * 25);
  const policyArr = new Float32Array(N * 25);
  const valueArr = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    planesArr.set(combined[i].planes, i * 75);
    policyArr.set(combined[i].policy, i * 25);
    valueArr[i] = combined[i].value;
    for (let j = 0; j < 25; j++) posArr[i * 25 + j] = j;
  }

  const xTensor = tf.tensor3d(Array.from(planesArr), [N, 25, 3]);
  const posTensor = tf.tensor2d(Array.from(posArr), [N, 25], 'int32');
  const yPolicyTensor = tf.tensor2d(Array.from(policyArr), [N, 25]);
  const yValueTensor = tf.tensor2d(Array.from(valueArr), [N, 1]);

  await model.fit([xTensor, posTensor], [yPolicyTensor, yValueTensor], {
    epochs,
    batchSize: effectiveBatchSize,
    shuffle: true,
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`[TTT5 TrainOnGames] Epoch ${epoch + 1}/${epochs}, loss: ${logs.loss?.toFixed(4)}`);
        progressCb?.({
          type: 'train.progress',
          payload: {
            epoch: epoch + 1, epochs,
            loss: Number(logs.loss ?? 0).toFixed(4),
            percent: Math.round(((epoch + 1) / epochs) * 100),
          }
        });
      },
      onTrainEnd: () => {
        progressCb?.({ type: 'train.done', payload: { saved: true, variant: 'ttt5' } });
      },
    }
  });

  const saveDir = path.join(_MODEL_DIR, 'ttt5_transformer');
  await ensureDir(saveDir);
  await model.save(`file://${saveDir}`);
  reloadTTT5Model();

  xTensor.dispose(); posTensor.dispose(); yPolicyTensor.dispose(); yValueTensor.dispose();
  console.log('[TTT5 TrainOnGames] Model saved and reloaded');
}

// ===== TTT5 Background Training =====
async function startTTT5BackgroundTraining(progressCb = null, newSkillsCount = 0, incrementalBatchSize = 256) {
  if (backgroundTraining) {
    console.log('[TTT5 BackgroundTrain] Training already in progress, skipping...');
    return;
  }

  if (ttt5GameHistory.length < 5) {
    console.log(`[TTT5 BackgroundTrain] Not enough data (${ttt5GameHistory.length}), skipping...`);
    return;
  }

  backgroundTraining = true;
  console.log('[TTT5 BackgroundTrain] Starting...');

  const totalSkills = ttt5GameHistory.length;
  progressCb?.({
    type: 'background_train.start',
    payload: { newSkills: newSkillsCount, totalSkills, epochs: 1, message: `TTT5: ${newSkillsCount} новых навыков` }
  });

  backgroundTrainingPromise = (async () => {
    try {
      await trainTTT5OnGames(
        (ev) => {
          if (ev.type === 'train.progress') {
            progressCb?.({
              type: 'background_train.progress',
              payload: {
                epoch: ev.payload?.epoch || 0, epochs: ev.payload?.epochs || 1,
                epochPercent: ev.payload?.percent || 0,
                newSkills: newSkillsCount, totalSkills, loss: ev.payload?.loss,
                message: `TTT5 эпоха ${ev.payload?.epoch}/${ev.payload?.epochs}`
              }
            });
          } else if (ev.type === 'train.done') {
            progressCb?.({
              type: 'background_train.done',
              payload: { newSkills: newSkillsCount, totalSkills, message: `TTT5: ${newSkillsCount} навыков усвоено` }
            });
          }
        },
        { epochs: 1, batchSize: incrementalBatchSize }
      );
      console.log('[TTT5 BackgroundTrain] Completed');
    } catch (e) {
      console.error('[TTT5 BackgroundTrain] Error:', e);
      progressCb?.({ type: 'background_train.error', error: String(e), payload: { message: `Ошибка: ${e.message}` } });
    } finally {
      backgroundTraining = false;
      backgroundTrainingPromise = null;
    }
  })();
}

// ===== Очистка =====
export function clearGameHistory() {
  gameHistory = [];
  ttt5GameHistory = [];
  console.log('[GameHistory] Cleared (TTT3 + TTT5)');
  return { success: true, count: 0, ttt5Count: 0 };
}

export function getGameHistoryStats() {
  return {
    count: gameHistory.length,
    maxSize: MAX_HISTORY_SIZE,
    ttt5Count: ttt5GameHistory.length,
    ttt5MaxSize: MAX_TTT5_HISTORY,
  };
}

// Обучение TTT3 Transformer с нуля
export async function trainTTT3WithProgress(progressCb, { epochs, batchSize, earlyStop = true } = {}) {
  const { trainTTT3WithProgress: trainTTT3 } = await import('./src/train_ttt3_transformer_service.mjs');
  return await trainTTT3(progressCb, { epochs, batchSize, earlyStop });
}

// Обучение TTT5 Transformer (bootstrap + MCTS self-play)
export async function trainTTT5WithProgress(progressCb, opts = {}) {
  const { trainTTT5WithProgress: trainTTT5 } = await import('./src/train_ttt5_service.mjs');
  return await trainTTT5(progressCb, opts);
}

// Обучение Gomoku Engine V2 (7x7 — 16x16)
export async function trainGomokuWithProgress(progressCb, opts = {}) {
  const { trainGomokuWithProgress: trainGomoku } = await import('./src/train_gomoku_service.mjs');
  return await trainGomoku(progressCb, opts);
}

// Очистка модели
export async function clearModel(variant = 'all') {
  try {
    console.log(`[Clear] Clearing models (variant=${variant})...`);
    let deletedFiles = [];

    // Очистка TTT3
    if (variant === 'all' || variant === 'ttt3') {
      if (ttt3Model) { ttt3Model.dispose?.(); ttt3Model = null; }
      ttt3ModelLoading = false;

      const ttt3ModelDir = path.join(_MODEL_DIR, 'ttt3_transformer');
      try {
        const files = await fs.readdir(ttt3ModelDir);
        for (const file of files) {
          if (file.endsWith('.json') || file.endsWith('.bin')) {
            await fs.unlink(path.join(ttt3ModelDir, file));
            deletedFiles.push(`ttt3_transformer/${file}`);
          }
        }
      } catch (e) { /* dir may not exist */ }

      // Legacy cleanup
      try {
        const oldModelPath = path.join(_MODEL_DIR, 'model.json');
        await fs.unlink(oldModelPath);
        deletedFiles.push('model.json');
      } catch (e) {
        if (e.code !== 'ENOENT') console.warn('[Clear] Error:', e.message);
      }

      try {
        const files = await fs.readdir(_MODEL_DIR);
        for (const file of files) {
          if (file.startsWith('weights') || file.endsWith('.bin')) {
            await fs.unlink(path.join(_MODEL_DIR, file));
            deletedFiles.push(file);
          }
        }
      } catch (e) { /* ignore */ }
    }

    // Очистка TTT5
    if (variant === 'all' || variant === 'ttt5') {
      if (ttt5Model) { ttt5Model.dispose?.(); ttt5Model = null; }
      ttt5ModelLoading = false;

      const ttt5ModelDir = path.join(_MODEL_DIR, 'ttt5_transformer');
      try {
        const files = await fs.readdir(ttt5ModelDir);
        for (const file of files) {
          if (file.endsWith('.json') || file.endsWith('.bin')) {
            await fs.unlink(path.join(ttt5ModelDir, file));
            deletedFiles.push(`ttt5_transformer/${file}`);
          }
        }
      } catch (e) { /* dir may not exist */ }
    }

    console.log(`[Clear] Cleared. Deleted: ${deletedFiles.join(', ') || 'none'}`);
    return { success: true, deletedFiles };
  } catch (e) {
    console.error('[Clear] Error:', e);
    throw e;
  }
}
