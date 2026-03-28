// Training pipeline for 5x5 Tic-Tac-Toe (4-in-a-row) Transformer.
// Unlike TTT3 (which uses minimax as perfect teacher), TTT5 uses:
// Phase 1: Tactical curriculum — win/block/double-threat positions
// Phase 2: Bootstrap — self-play with heuristic policy
// Phase 3: MCTS Self-Play — AlphaZero-style iterative improvement with replay buffer

import tfpkg from './tf.mjs';
const tf = tfpkg;
import { buildPVTransformerSeq, maskLogits } from './model_pv_transformer_seq.mjs';
import { createGameAdapter } from './game_adapter.mjs';
import { mctsPUCT } from './mcts_puct.mjs';
import {
  SYMMETRY_MAPS_5, transformPlanes, transformPolicy,
} from './ttt5_symmetry.mjs';
import {
  TTT5_TRAIN, TTT5_TRANSFORMER_CFG, TTT5_BOARD_N, TTT5_WIN_LEN, TTT5_MCTS, TTT5_CURRICULUM,
} from './config.mjs';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const SEQ_LEN = TTT5_BOARD_N * TTT5_BOARD_N; // 25
const PLANES_LEN = SEQ_LEN * 3; // 75
const adapter = createGameAdapter({ variant: 'ttt5', winLen: TTT5_WIN_LEN });

const SAVE_DIR = path.join(__dirname, '..', 'saved', 'ttt5_transformer');
const GPU_PERF_MODE = process.env.USE_GPU_BIG === '1';

// ===== Training State for structured progress =====
function createTrainingState(totalPhases) {
  return {
    startTime: Date.now(),
    phase: 'init',
    phaseName: '',
    phaseStep: 0,
    totalPhases,
    iteration: 0,
    totalIterations: 0,
    game: 0,
    totalGames: 0,
    totalPositions: 0,
    epoch: 0,
    totalEpochs: 0,
    batch: 0,
    totalBatches: 0,
    loss: 0,
    accuracy: 0,
    mae: 0,
    selfPlayStats: { wins: 0, losses: 0, draws: 0 },
    hardPositions: 0,
    seededGames: 0,
    metricsHistory: [],
    gameStartTime: 0,
    effectivePositions: 0,
    completedPhases: [],
  };
}

function computeOverallProgress(state) {
  if (state.totalPhases === 0) return 0;
  const phaseWeight = 1 / state.totalPhases;
  const completedPhases = state.completedPhases.length;

  let currentPhaseProgress = 0;
  if (state.phase === 'generating') {
    if (state.totalGames > 0) {
      currentPhaseProgress = state.game / state.totalGames * 0.5;
    }
  } else if (state.phase === 'training') {
    currentPhaseProgress = 0.5;
    if (state.totalEpochs > 0) {
      const epochProgress = (state.epoch - 1 + (state.totalBatches > 0 ? state.batch / state.totalBatches : 0)) / state.totalEpochs;
      currentPhaseProgress += epochProgress * 0.5;
    }
  } else if (state.phase === 'mcts_game') {
    // MCTS phase spans multiple iterations
    if (state.totalIterations > 0) {
      const iterBase = (state.iteration - 1) / state.totalIterations;
      const withinIter = state.totalGames > 0 ? (state.game / state.totalGames) * 0.5 / state.totalIterations : 0;
      currentPhaseProgress = iterBase + withinIter;
    }
  } else if (state.phase === 'mcts_train') {
    if (state.totalIterations > 0) {
      const iterBase = (state.iteration - 1) / state.totalIterations;
      const trainPart = state.totalEpochs > 0
        ? (0.5 + (state.epoch - 1 + (state.totalBatches > 0 ? state.batch / state.totalBatches : 0)) / state.totalEpochs * 0.5) / state.totalIterations
        : 0.5 / state.totalIterations;
      currentPhaseProgress = iterBase + trainPart;
    }
  }

  return Math.min(1, (completedPhases + currentPhaseProgress) * phaseWeight);
}

function emitProgress(state, progressCb) {
  const elapsed = (Date.now() - state.startTime) / 1000;
  const stageElapsed = state.gameStartTime
    ? Math.max(1e-6, (Date.now() - state.gameStartTime) / 1000)
    : 0;
  let speed = 0;
  let speedUnit = 'g/s';

  if (stageElapsed > 0) {
    if (state.phase === 'generating' || state.phase === 'mcts_game') {
      speed = state.game > 0 ? state.game / stageElapsed : 0;
      speedUnit = 'g/s';
    } else if (state.phase === 'training' || state.phase === 'mcts_train') {
      const completedBatches = state.totalBatches > 0
        ? Math.max(0, (Math.max(0, state.epoch - 1) * state.totalBatches) + state.batch)
        : 0;
      speed = completedBatches > 0 ? completedBatches / stageElapsed : 0;
      speedUnit = 'b/s';
    }
  }

  const totalProgress = computeOverallProgress(state);
  const eta = totalProgress > 0.01 ? (elapsed / totalProgress) * (1 - totalProgress) : 0;

  progressCb?.({
    type: 'train.progress',
    payload: {
      phase: state.phaseName,
      stage: state.phase,
      phaseStep: state.phaseStep,
      totalPhases: state.totalPhases,
      completedPhases: state.completedPhases,
      iteration: state.iteration,
      totalIterations: state.totalIterations,
      game: state.game,
      totalGames: state.totalGames,
      positions: state.totalPositions,
      epoch: state.epoch,
      totalEpochs: state.totalEpochs,
      batch: state.batch,
      totalBatches: state.totalBatches,
      loss: state.loss,
      accuracy: state.accuracy,
      mae: state.mae,
      elapsed: Math.round(elapsed),
      eta: Math.max(0, Math.round(eta)),
      speed: parseFloat(speed.toFixed(2)),
      speedUnit,
      selfPlayStats: { ...state.selfPlayStats },
      hardPositions: state.hardPositions,
      seededGames: state.seededGames,
      effectivePositions: state.effectivePositions,
      metricsHistory: state.metricsHistory.slice(-50),
      percent: Math.round(totalProgress * 100),
    }
  });
}

// ===== Custom loss: softmax cross-entropy from logits =====
function policyLossFromLogits(yTrue, yPred) {
  const legalMask = yTrue.greater(0).toFloat();
  const maskedLogits = maskLogits(yPred, legalMask);
  const logProbs = tf.logSoftmax(maskedLogits, -1);
  return tf.neg(tf.sum(tf.mul(yTrue, logProbs), -1)).mean();
}

// Fisher-Yates shuffle
function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function augmentSampleBySymmetry(planes, policy, value) {
  return SYMMETRY_MAPS_5.map(({ map }) => ({
    planes: transformPlanes(planes, map),
    policy: transformPolicy(policy, map),
    value,
  }));
}

// Multi-phase LR schedule (same as TTT3)
function buildLrPhases(baseLR, epochs) {
  if (epochs <= 8) {
    return [
      { fraction: 0.8, lr: baseLR },
      { fraction: 0.2, lr: baseLR / 4 },
    ];
  }
  if (epochs <= 20) {
    return [
      { fraction: 0.6, lr: baseLR },
      { fraction: 0.3, lr: baseLR / 4 },
      { fraction: 0.1, lr: baseLR / 10 },
    ];
  }
  return [
    { fraction: 0.4, lr: baseLR },
    { fraction: 0.3, lr: baseLR / 4 },
    { fraction: 0.3, lr: baseLR / 20 },
  ];
}

function shouldRunEpochEval(epoch, epochs) {
  if (!GPU_PERF_MODE || epochs <= 2) return true;
  return epoch === 0 || epoch === epochs - 1 || ((epoch + 1) % 3 === 0);
}

function isResourceExhaustedError(error) {
  const message = String(error?.message || error || '');
  return (
    message.includes('RESOURCE_EXHAUSTED') ||
    message.includes('failed to allocate memory') ||
    message.includes('OOM when allocating tensor') ||
    message.includes('CUDA_ERROR_OUT_OF_MEMORY') ||
    message.toLowerCase().includes('out of memory')
  );
}

function nextLowerBatchSize(currentBatchSize, totalSamples) {
  const candidates = [768, 640, 576, 544, 512, 448, 384, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32];
  for (const candidate of candidates) {
    if (candidate < currentBatchSize && candidate <= totalSamples) return candidate;
  }
  return 0;
}

function normalizePolicyFromScores(entries, length, power = 1.0) {
  const out = new Float32Array(length);
  let sum = 0;
  for (const { move, score } of entries) {
    const v = Math.pow(Math.max(0, score), power);
    out[move] = v;
    sum += v;
  }
  if (sum <= 1e-8) {
    const uniform = entries.length > 0 ? 1 / entries.length : 0;
    for (const { move } of entries) out[move] = uniform;
    return out;
  }
  for (const { move } of entries) out[move] /= sum;
  return out;
}

function sharpenPolicy(policy, moves, power = 1.35) {
  const out = new Float32Array(policy.length);
  let sum = 0;
  for (const mv of moves) {
    const v = Math.pow(Math.max(0, policy[mv]), power);
    out[mv] = v;
    sum += v;
  }
  if (sum <= 1e-8) {
    const uniform = moves.length > 0 ? 1 / moves.length : 0;
    for (const mv of moves) out[mv] = uniform;
    return out;
  }
  for (const mv of moves) out[mv] /= sum;
  return out;
}

function argmaxOnMoves(policy, moves) {
  let bestMove = -1;
  let bestScore = -Infinity;
  for (const mv of moves) {
    if (policy[mv] > bestScore) {
      bestScore = policy[mv];
      bestMove = mv;
    }
  }
  return bestMove;
}

function distillPolicyTarget(policy, moves, { power = 1.65, primaryWeight = 0.6 } = {}) {
  const sharpened = sharpenPolicy(policy, moves, power);
  const out = new Float32Array(policy.length);
  const bestMove = argmaxOnMoves(sharpened, moves);
  const residualWeight = Math.max(0, 1 - primaryWeight);

  for (const mv of moves) out[mv] = sharpened[mv] * residualWeight;
  if (bestMove >= 0) out[bestMove] += primaryWeight;

  return out;
}

function oneHotPolicy(move, length = SEQ_LEN) {
  const out = new Float32Array(length);
  if (move >= 0) out[move] = 1;
  return out;
}

function getForcedTrainingTarget(board, player, policyProbs = null) {
  const win = adapter.findImmediateWin(board, player);
  if (win >= 0) return { move: win, policy: oneHotPolicy(win), source: 'forced_win' };

  const block = adapter.findImmediateBlock(board, player);
  if (block >= 0) return { move: block, policy: oneHotPolicy(block), source: 'forced_block' };

  const candidates = adapter.candidateMoves(board, {
    radius: 2,
    maxMoves: 12,
    policyProbs,
    includePolicyTopK: 6,
  });

  const oppWins = adapter.collectImmediateWins(board, -player, candidates);
  if (oppWins.length > 1) {
    const multiBlock = adapter.findBestDefensiveMove(board, player, candidates, policyProbs);
    if (multiBlock >= 0) {
      return { move: multiBlock, policy: oneHotPolicy(multiBlock), source: 'forced_multi_block' };
    }
  }

  const doubleThreat = adapter.findDoubleThreatMove(board, player, candidates);
  if (doubleThreat >= 0) {
    return { move: doubleThreat, policy: oneHotPolicy(doubleThreat), source: 'forced_double_threat' };
  }

  return null;
}

function computeReplayPriority(board, player, policy, source = 'mcts') {
  const legal = adapter.legalMoves(board);
  let priority = 1;

  if (source.startsWith('forced_')) priority += 4;

  const bestMove = argmaxOnMoves(policy, legal);
  const bestProb = bestMove >= 0 ? policy[bestMove] : 0;
  priority += (1 - bestProb) * 0.75;

  const ownThreats = adapter.countImmediateWins(board, player, legal);
  const oppThreats = adapter.countImmediateWins(board, -player, legal);
  priority += Math.min(3, ownThreats + oppThreats) * 0.8;

  const filled = SEQ_LEN - legal.length;
  if (filled >= 12) priority += 0.5;

  return priority;
}

function buildReplayTrainingSet(replayBuffer, sampleSize = 2200) {
  if (replayBuffer.length <= sampleSize) return replayBuffer.slice();

  const recentCount = Math.max(1, Math.floor(sampleSize * 0.35));
  const priorityCount = Math.max(1, Math.floor(sampleSize * 0.45));

  const recent = replayBuffer.slice(-recentCount);
  const recentSet = new Set(recent);
  const remaining = replayBuffer.filter((entry) => !recentSet.has(entry));

  const prioritySlice = remaining
    .slice()
    .sort((a, b) => (b.priority ?? 1) - (a.priority ?? 1))
    .slice(0, Math.min(priorityCount, remaining.length));
  const prioritySet = new Set(prioritySlice);

  const randomPool = remaining.filter((entry) => !prioritySet.has(entry));
  shuffleArray(randomPool);

  const needed = Math.max(0, sampleSize - recent.length - prioritySlice.length);
  return shuffleArray([
    ...recent,
    ...prioritySlice,
    ...randomPool.slice(0, needed),
  ]);
}

function trimPriorityBuffer(buffer, maxSize) {
  if (buffer.length <= maxSize) return;
  buffer.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  buffer.length = maxSize;
}

function pickWeightedSeed(hardBuffer) {
  if (!hardBuffer.length) return null;

  let total = 0;
  for (const item of hardBuffer) total += Math.max(0.1, item.priority ?? 1);
  let r = Math.random() * total;
  for (const item of hardBuffer) {
    r -= Math.max(0.1, item.priority ?? 1);
    if (r <= 0) return item;
  }
  return hardBuffer[hardBuffer.length - 1];
}

function invertBoardColors(board) {
  const flipped = adapter.cloneBoard(board);
  for (let i = 0; i < flipped.length; i++) {
    flipped[i] = -flipped[i];
  }
  return flipped;
}

function createSelfPlayStartWithFallback(hardBuffer, fallbackPlayer = 1) {
  if (hardBuffer.length > 0 && Math.random() < TTT5_CURRICULUM.seedGameRatio) {
    const seed = pickWeightedSeed(hardBuffer);
    if (seed?.board && !adapter.isTerminal(seed.board)) {
      const invertPerspective = Math.random() < 0.5;
      return {
        board: invertPerspective ? invertBoardColors(seed.board) : adapter.cloneBoard(seed.board),
        player: invertPerspective ? -seed.player : seed.player,
        seeded: true,
      };
    }
  }

  return {
    board: adapter.emptyBoard(),
    player: fallbackPlayer,
    seeded: false,
  };
}

function extractHardExamples(gameRecord, winner) {
  const hardExamples = [];
  const totalPlies = gameRecord.length;

  for (let i = 0; i < gameRecord.length; i++) {
    const rec = gameRecord[i];
    const value = winner === rec.player ? 1 : (winner === -rec.player ? -1 : 0);
    const losingSide = value < 0;
    const lateGame = totalPlies - i <= TTT5_CURRICULUM.lateGameWindow;
    const forced = rec.source?.startsWith('forced_');
    const unstableTarget = (rec.targetPeak ?? 1) < 0.5;
    const topMove = argmaxOnMoves(rec.policy, adapter.legalMoves(rec.board));
    const sampledOffTarget = rec.move !== topMove;

    if (!(forced || (losingSide && (lateGame || unstableTarget || sampledOffTarget)))) {
      continue;
    }

    let priority = (rec.priority ?? 1);
    if (forced) priority += 4;
    if (losingSide) priority += 2.5;
    if (lateGame) priority += 1.25;
    if (unstableTarget) priority += 0.75;
    if (sampledOffTarget) priority += 0.5;

    hardExamples.push({
      board: adapter.cloneBoard(rec.board),
      player: rec.player,
      planes: rec.planes,
      policy: rec.policy,
      value,
      priority,
      source: rec.source,
    });
  }

  hardExamples.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  return hardExamples.slice(0, TTT5_CURRICULUM.maxHardPerGame);
}

function buildCurriculumBatch(replayBuffer, hardBuffer, recentSamples) {
  const sampleSize = TTT5_CURRICULUM.trainingSampleSize;
  const freshCount = Math.max(1, Math.floor(sampleSize * TTT5_CURRICULUM.freshRatio));
  const hardCount = Math.max(1, Math.floor(sampleSize * TTT5_CURRICULUM.hardRatio));
  const replayCount = Math.max(1, sampleSize - freshCount - hardCount);

  const fresh = recentSamples.slice(-freshCount);
  const freshSet = new Set(fresh);

  const hard = hardBuffer
    .slice()
    .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0))
    .slice(0, Math.min(hardCount, hardBuffer.length));

  const replayPool = replayBuffer.filter((entry) => !freshSet.has(entry));
  const replaySlice = buildReplayTrainingSet(replayPool, replayCount);

  return shuffleArray([
    ...fresh,
    ...hard,
    ...replaySlice,
  ]);
}

function randomGuidedMove(board, player) {
  const legal = adapter.legalMoves(board);
  if (legal.length === 0) return -1;
  const candidates = adapter.candidateMoves(board, { radius: 2, maxMoves: 10 });
  const moves = candidates.length ? candidates : legal;
  return moves[Math.floor(Math.random() * moves.length)];
}

function heuristicPolicy(board, player) {
  const legal = adapter.legalMoves(board);
  const policy = new Float32Array(SEQ_LEN);
  if (legal.length === 0) return policy;

  const win = adapter.findImmediateWin(board, player);
  if (win >= 0) {
    policy[win] = 1;
    return policy;
  }

  const block = adapter.findImmediateBlock(board, player);
  if (block >= 0) {
    policy[block] = 1;
    return policy;
  }

  const candidates = adapter.candidateMoves(board, { radius: 2, maxMoves: 12 });
  const movePool = candidates.length ? candidates : legal;

  const multiBlock = adapter.findBestDefensiveMove(board, player, movePool);
  const oppImmediateWins = adapter.collectImmediateWins(board, -player, movePool);
  if (oppImmediateWins.length > 1 && multiBlock >= 0) {
    policy[multiBlock] = 1;
    return policy;
  }

  const doubleThreat = adapter.findDoubleThreatMove(board, player, movePool);
  if (doubleThreat >= 0) {
    policy[doubleThreat] = 1;
    return policy;
  }

  const entries = movePool.map((move) => {
    const prev = board[move];

    board[move] = player;
    const ownImmediate = adapter.countImmediateWins(board, player);
    const oppImmediateAfter = adapter.countImmediateWins(board, -player);
    board[move] = prev;

    board[move] = -player;
    const oppImmediateIfIgnored = adapter.countImmediateWins(board, -player);
    board[move] = prev;

    const centerBias = 1 + Math.max(0, 4 - (Math.abs((move / 5 | 0) - 2) + Math.abs((move % 5) - 2)));
    const tacticalScore =
      centerBias +
      ownImmediate * 24 +
      Math.max(0, oppImmediateIfIgnored - oppImmediateAfter) * 18 +
      (oppImmediateAfter === 0 ? 3 : 0);

    return { move, score: tacticalScore };
  });

  return normalizePolicyFromScores(entries, SEQ_LEN, 1.3);
}

function sampleMoveFromPolicy(policy, moves) {
  let sum = 0;
  for (const mv of moves) sum += policy[mv];
  if (sum <= 1e-8) return moves[Math.floor(Math.random() * moves.length)];

  let r = Math.random() * sum;
  for (const mv of moves) {
    r -= policy[mv];
    if (r <= 0) return mv;
  }
  return moves[moves.length - 1];
}

function getPhaseEpochs(totalEpochs, isNewModel) {
  const epochs = Math.max(3, totalEpochs);
  if (!isNewModel) {
    return { tactical: 0, bootstrap: 0, mcts: epochs };
  }

  const tactical = Math.max(1, Math.round(epochs * 0.2));
  const bootstrap = Math.max(1, Math.round(epochs * 0.2));
  const mcts = Math.max(1, epochs - tactical - bootstrap);
  return { tactical, bootstrap, mcts };
}

function generateTacticalCurriculum(numSamples = 240) {
  const samples = [];
  let attempts = 0;

  while (samples.length < numSamples && attempts < numSamples * 20) {
    attempts++;
    let board = adapter.emptyBoard();
    let player = Math.random() < 0.5 ? 1 : -1;
    const depth = 3 + Math.floor(Math.random() * 8);

    for (let t = 0; t < depth && !adapter.isTerminal(board); t++) {
      const mv = randomGuidedMove(board, player);
      if (mv < 0) break;
      board = adapter.applyMove(board, mv, player);
      player = -player;
    }

    if (adapter.isTerminal(board)) continue;

    const win = adapter.findImmediateWin(board, player);
    if (win >= 0) {
      const policy = new Float32Array(SEQ_LEN);
      policy[win] = 1;
      samples.push({ planes: adapter.encodePlanes(board, player), policy, value: 1 });
      continue;
    }

    const block = adapter.findImmediateBlock(board, player);
    if (block >= 0) {
      const policy = new Float32Array(SEQ_LEN);
      policy[block] = 1;
      samples.push({ planes: adapter.encodePlanes(board, player), policy, value: 0.35 });
      continue;
    }

    const candidates = adapter.candidateMoves(board, { radius: 2, maxMoves: 12 });
    const multiBlock = adapter.findBestDefensiveMove(board, player, candidates);
    const oppWins = adapter.collectImmediateWins(board, -player, candidates);
    if (oppWins.length > 1 && multiBlock >= 0) {
      const policy = new Float32Array(SEQ_LEN);
      policy[multiBlock] = 1;
      samples.push({ planes: adapter.encodePlanes(board, player), policy, value: 0.2 });
      continue;
    }

    const doubleThreat = adapter.findDoubleThreatMove(board, player, candidates);
    if (doubleThreat >= 0) {
      const policy = new Float32Array(SEQ_LEN);
      policy[doubleThreat] = 1;
      samples.push({ planes: adapter.encodePlanes(board, player), policy, value: 0.8 });
    }
  }

  return samples;
}

// ===== Neural Network forward pass =====
function forwardPass(model, board, player) {
  return tf.tidy(() => {
    const planes = adapter.encodePlanes(board, player);
    const x = tf.tensor3d(Array.from(planes), [1, SEQ_LEN, 3]);
    const posIndices = tf.tensor2d(
      Array.from({ length: SEQ_LEN }, (_, i) => i),
      [1, SEQ_LEN],
      'int32'
    );

    const [logits, valueTensor] = model.predict([x, posIndices]);

    // Mask illegal moves
    const mask = adapter.maskLegalMoves(board);
    const maskTensor = tf.tensor2d(Array.from(mask), [1, SEQ_LEN]);
    const masked = maskLogits(logits, maskTensor);
    const probs = tf.softmax(masked, -1);

    const policy = probs.dataSync();
    const value = valueTensor.dataSync()[0];

    return { policy: new Float32Array(policy), value };
  });
}

/**
 * Batched forward pass — evaluates multiple positions in a single model.predict call.
 * Optimized: pre-allocated posRow, flat mask buffer, minimal tensor creation.
 */
const _posRow = Int32Array.from({ length: SEQ_LEN }, (_, i) => i);

function forwardPassBatched(model, requests) {
  // requests: Array<{ board, player }>
  const batchSize = requests.length;
  if (batchSize === 0) return [];

  return tf.tidy(() => {
    // Pre-allocate flat buffers to avoid per-item array creation
    const planesFlat = new Float32Array(batchSize * SEQ_LEN * 3);
    const masksFlat = new Float32Array(batchSize * SEQ_LEN);
    const posFlat = new Int32Array(batchSize * SEQ_LEN);

    for (let i = 0; i < batchSize; i++) {
      const { board, player } = requests[i];
      const planes = adapter.encodePlanes(board, player);
      planesFlat.set(planes, i * SEQ_LEN * 3);
      const mask = adapter.maskLegalMoves(board);
      masksFlat.set(mask, i * SEQ_LEN);
      posFlat.set(_posRow, i * SEQ_LEN);
    }

    const x = tf.tensor(planesFlat, [batchSize, SEQ_LEN, 3]);
    const posIndices = tf.tensor(posFlat, [batchSize, SEQ_LEN], 'int32');

    const [logits, valueTensor] = model.predict([x, posIndices]);

    // Mask & softmax
    const maskTensor = tf.tensor(masksFlat, [batchSize, SEQ_LEN]);
    const masked = maskLogits(logits, maskTensor);
    const probs = tf.softmax(masked, -1);

    const policiesFlat = probs.dataSync();
    const valuesFlat = valueTensor.dataSync();

    const results = [];
    for (let i = 0; i < batchSize; i++) {
      const start = i * SEQ_LEN;
      const end = start + SEQ_LEN;
      results.push({
        policy: policiesFlat.subarray(start, end),
        value: valuesFlat[i],
      });
    }
    return results;
  });
}

// ===== Heuristic policy for bootstrap phase =====
function heuristicMove(board, player) {
  const win = adapter.findImmediateWin(board, player);
  if (win >= 0) return win;
  const block = adapter.findImmediateBlock(board, player);
  if (block >= 0) return block;

  const candidates = adapter.candidateMoves(board, { radius: 2, maxMoves: 12 });
  const doubleThreat = adapter.findDoubleThreatMove(board, player, candidates);
  if (doubleThreat >= 0) return doubleThreat;

  const center = adapter.centerMove();
  if (board[center] === 0) return center;

  const defensiveMove = adapter.findBestDefensiveMove(board, player, candidates);
  if (defensiveMove >= 0) return defensiveMove;

  const moves = candidates.length ? candidates : adapter.legalMoves(board);
  const scored = moves.map(mv => {
    let score = 0;
    const r = Math.floor(mv / 5), c = mv % 5;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5) {
          if (board[nr * 5 + nc] === player) score += 2;
          if (board[nr * 5 + nc] === -player) score += 1;
        }
      }
    }
    score += (2 - Math.abs(r - 2)) + (2 - Math.abs(c - 2));
    return { mv, score };
  });
  scored.sort((a, b) => b.score - a.score);

  const topK = Math.min(3, scored.length);
  return scored[Math.floor(Math.random() * topK)].mv;
}

// Sample from policy (for exploration during training)
function sampleFromPolicy(pi, moves) {
  const total = moves.reduce((s, m) => s + pi[m], 0);
  if (total < 1e-8) return moves[Math.floor(Math.random() * moves.length)];

  let r = Math.random() * total;
  for (const m of moves) {
    r -= pi[m];
    if (r <= 0) return m;
  }
  return moves[moves.length - 1];
}

// ===== Train model on collected data =====
async function trainOnData(model, trainingData, progressCb, { epochs = 5, phase = '', batchSize: requestedBatchSize = TTT5_TRAIN.batchSize, trainingState = null } = {}) {
  if (trainingData.length === 0) return;

  // Apply symmetry augmentation
  const augmented = [];
  for (const sample of trainingData) {
    augmented.push(...augmentSampleBySymmetry(sample.planes, sample.policy, sample.value));
  }

  shuffleArray(augmented);
  const N = augmented.length;
  console.log(`[TrainTTT5] ${phase}: Training on ${N} augmented samples (${trainingData.length} base)`);

  // Build tensors
  const xArr = new Float32Array(N * PLANES_LEN);
  const posArr = new Int32Array(N * SEQ_LEN);
  const policyArr = new Float32Array(N * SEQ_LEN);
  const valueArr = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    xArr.set(augmented[i].planes, i * PLANES_LEN);
    for (let j = 0; j < SEQ_LEN; j++) posArr[i * SEQ_LEN + j] = j;
    const policyOffset = i * SEQ_LEN;
    let policySum = 0;
    for (let j = 0; j < SEQ_LEN; j++) {
      const legalFloor = augmented[i].planes[j * 3 + 2] > 0 ? 5e-4 : 0;
      const value = (augmented[i].policy[j] ?? 0) + legalFloor;
      policyArr[policyOffset + j] = value;
      policySum += value;
    }
    if (policySum > 1e-8) {
      for (let j = 0; j < SEQ_LEN; j++) {
        policyArr[policyOffset + j] /= policySum;
      }
    }
    valueArr[i] = augmented[i].value;
  }

  const xTensor = tf.tensor3d(xArr, [N, SEQ_LEN, 3]);
  const posTensor = tf.tensor2d(posArr, [N, SEQ_LEN], 'int32');
  const yPolicyTensor = tf.tensor2d(policyArr, [N, SEQ_LEN]);
  const yValueTensor = tf.tensor2d(valueArr, [N, 1]);

  let effectiveBatchSize = Math.max(16, Math.min(requestedBatchSize, N));
  let totalBatches = Math.ceil(N / effectiveBatchSize);

  // LR schedule
  const baseLR = TTT5_TRAIN.lr;
  const lrPhases = buildLrPhases(baseLR, epochs);
  let currentLR = baseLR;

  function getLRForEpoch(epoch) {
    let accumulated = 0;
    for (const p of lrPhases) {
      accumulated += Math.round(p.fraction * epochs);
      if (epoch < accumulated) return p.lr;
    }
    return lrPhases[lrPhases.length - 1].lr;
  }

  function compileWithLR(lr) {
    model.compile({
      optimizer: tf.train.adam(lr),
      loss: [policyLossFromLogits, 'meanSquaredError'],
      lossWeights: [1.0, TTT5_TRAIN.weightValue],
    });
  }

  compileWithLR(currentLR);

  if (trainingState) {
    trainingState.totalEpochs = epochs;
    trainingState.totalPositions = trainingData.length;
    trainingState.effectivePositions = N;
  }

  for (let epoch = 0; epoch < epochs; epoch++) {
    const epochLR = getLRForEpoch(epoch);
    if (Math.abs(epochLR - currentLR) > 1e-8) {
      currentLR = epochLR;
      compileWithLR(currentLR);
    }

    if (trainingState) {
      trainingState.epoch = epoch + 1;
      trainingState.batch = 0;
      trainingState.totalBatches = totalBatches;
    }

    let history;
    for (;;) {
      totalBatches = Math.ceil(N / effectiveBatchSize);
      if (trainingState) {
        trainingState.totalBatches = totalBatches;
      }

      try {
        history = await model.fit(
          [xTensor, posTensor],
          [yPolicyTensor, yValueTensor],
          {
            epochs: 1,
            batchSize: effectiveBatchSize,
            shuffle: true,
            verbose: 0,
            callbacks: {
              onBatchEnd: (batch) => {
                if (trainingState) {
                  trainingState.batch = batch + 1;
                  // Emit every 3 batches to avoid WS flooding
                  if ((batch + 1) % 3 === 0 || batch === totalBatches - 1) {
                    emitProgress(trainingState, progressCb);
                  }
                }
              }
            }
          }
        );
        break;
      } catch (error) {
        if (!isResourceExhaustedError(error)) throw error;

        const loweredBatch = nextLowerBatchSize(effectiveBatchSize, N);
        if (loweredBatch < 32) throw error;

        console.warn(`[TrainTTT5] ${phase}: OOM at batchSize=${effectiveBatchSize}, retrying with batchSize=${loweredBatch}`);
        progressCb?.({
          type: 'train.status',
          payload: {
            message: `GPU память переполнена на batch ${effectiveBatchSize}, пробую batch ${loweredBatch}...`
          }
        });
        effectiveBatchSize = loweredBatch;
        if (trainingState) {
          trainingState.batch = 0;
        }
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
    }

    const loss = history.history.loss[0];

    // Metric evaluation adds synchronous GPU→CPU barriers, so in perf mode we do it sparsely.
    let accuracy = trainingState?.accuracy ? Number(trainingState.accuracy) / 100 : 0;
    let mae = trainingState?.mae ? Number(trainingState.mae) : 0;
    if (shouldRunEpochEval(epoch, epochs)) {
      const evalSize = Math.min(N, GPU_PERF_MODE ? 256 : 512);
      const evalResult = tf.tidy(() => {
        const evalX = xTensor.slice([0, 0, 0], [evalSize, SEQ_LEN, 3]);
        const evalPos = posTensor.slice([0, 0], [evalSize, SEQ_LEN]);
        const [pLogits, vPred] = model.predict([evalX, evalPos]);
        const legalMask = evalX.slice([0, 0, 2], [evalSize, SEQ_LEN, 1]).reshape([evalSize, SEQ_LEN]);
        const maskedLogits = maskLogits(pLogits, legalMask);

        const modelArgmax = maskedLogits.argMax(-1);
        const targetArgmax = yPolicyTensor.slice([0, 0], [evalSize, SEQ_LEN]).argMax(-1);
        const matches = modelArgmax.equal(targetArgmax).sum().dataSync()[0];

        const targetVal = yValueTensor.slice([0, 0], [evalSize, 1]);
        const valMae = vPred.sub(targetVal).abs().mean().dataSync()[0];

        return { matches, valMae };
      });
      accuracy = evalResult.matches / evalSize;
      mae = evalResult.valMae;
    }

    if (trainingState) {
      trainingState.loss = Number(loss).toFixed(4);
      trainingState.accuracy = (accuracy * 100).toFixed(2);
      trainingState.mae = mae.toFixed(4);
      trainingState.metricsHistory.push({
        loss: parseFloat(Number(loss).toFixed(4)),
        acc: parseFloat((accuracy * 100).toFixed(2)),
        phase: trainingState.phaseName,
      });
      emitProgress(trainingState, progressCb);
    } else {
      progressCb?.({
        type: 'train.progress',
        payload: {
          epoch: epoch + 1,
          epochs,
          loss: Number(loss).toFixed(4),
          percent: Math.round(((epoch + 1) / epochs) * 100),
          accuracy: (accuracy * 100).toFixed(2),
          mae: mae.toFixed(4),
        }
      });
    }

    const metricSuffix = shouldRunEpochEval(epoch, epochs)
      ? `, Acc: ${(accuracy * 100).toFixed(1)}%, MAE: ${mae.toFixed(4)}`
      : ', Acc: cached, MAE: cached';
    console.log(`[TrainTTT5] ${phase} Epoch ${epoch + 1}/${epochs} - Loss: ${Number(loss).toFixed(4)}${metricSuffix} (LR: ${currentLR.toExponential(1)})`);
  }

  xTensor.dispose();
  posTensor.dispose();
  yPolicyTensor.dispose();
  yValueTensor.dispose();

  return effectiveBatchSize;
}

// ===== Phase 1: Bootstrap with heuristic self-play =====
async function bootstrapPhase(model, progressCb, numGames = 200, epochs = 5, batchSize = TTT5_TRAIN.batchSize, trainingState = null) {
  console.log(`[TrainTTT5] Bootstrap phase: ${numGames} games with heuristic policy`);

  if (trainingState) {
    trainingState.phaseName = 'bootstrap';
    trainingState.phase = 'generating';
    trainingState.game = 0;
    trainingState.totalGames = numGames;
    trainingState.gameStartTime = Date.now();
    trainingState.effectivePositions = 0;
    emitProgress(trainingState, progressCb);
  }

  const trainingData = [];

  for (let g = 0; g < numGames; g++) {
    const gameRecord = [];
    let board = adapter.emptyBoard();
    let player = 1;

    while (!adapter.isTerminal(board)) {
      const legal = adapter.legalMoves(board);
      const forcedTarget = getForcedTrainingTarget(board, player);
      const sampledPolicy = forcedTarget?.policy || heuristicPolicy(board, player);
      const targetPolicy = forcedTarget?.policy || distillPolicyTarget(sampledPolicy, legal, { power: 1.8, primaryWeight: 0.72 });
      const move = forcedTarget?.move ?? sampleMoveFromPolicy(sampledPolicy, legal);
      gameRecord.push({
        planes: adapter.encodePlanes(board, player),
        policy: targetPolicy,
        player,
        priority: computeReplayPriority(board, player, targetPolicy, forcedTarget?.source || 'bootstrap'),
      });
      board = adapter.applyMove(board, move, player);
      player = -player;
    }

    const w = adapter.winner(board);
    for (const rec of gameRecord) {
      const value = w === rec.player ? 1 : (w === -rec.player ? -1 : 0);
      trainingData.push({
        planes: rec.planes,
        policy: rec.policy,
        value,
        priority: rec.priority + Math.abs(value) * 1.25,
      });
    }

    if (trainingState) {
      trainingState.game = g + 1;
      trainingState.totalPositions = trainingData.length;
      if (w === 1) trainingState.selfPlayStats.wins++;
      else if (w === -1) trainingState.selfPlayStats.losses++;
      else trainingState.selfPlayStats.draws++;
      if ((g + 1) % 10 === 0 || g === numGames - 1) {
        emitProgress(trainingState, progressCb);
      }
    } else if ((g + 1) % 50 === 0) {
      progressCb?.({ type: 'train.status', payload: { message: `Bootstrap: игра ${g + 1}/${numGames}` } });
    }
  }

  console.log(`[TrainTTT5] Bootstrap: collected ${trainingData.length} positions from ${numGames} games`);

  if (trainingState) {
    trainingState.phase = 'training';
    trainingState.epoch = 0;
    trainingState.batch = 0;
    trainingState.gameStartTime = Date.now();
    emitProgress(trainingState, progressCb);
  }

  return await trainOnData(model, trainingData, progressCb, { epochs, phase: 'bootstrap', batchSize, trainingState });
}

// ===== Phase 2: MCTS Self-Play with Replay Buffer =====
async function mctsPhase(model, progressCb, iterations = 5, gamesPerIter = 50, epochsPerIter = 5, batchSize = TTT5_TRAIN.batchSize, trainingState = null, sims = TTT5_MCTS.trainingSimulations) {
  console.log(`[TrainTTT5] MCTS phase: ${iterations} iterations, ${gamesPerIter} games each, ${sims} sims/move`);

  const replayBuffer = [];
  const hardBuffer = [];

  for (let iter = 0; iter < iterations; iter++) {
    if (trainingState) {
      trainingState.phaseName = 'mcts';
      trainingState.phase = 'mcts_game';
      trainingState.iteration = iter + 1;
      trainingState.totalIterations = iterations;
      trainingState.game = 0;
      trainingState.totalGames = gamesPerIter;
      trainingState.gameStartTime = Date.now();
      trainingState.selfPlayStats = { wins: 0, losses: 0, draws: 0 };
      trainingState.hardPositions = hardBuffer.length;
      trainingState.seededGames = 0;
      trainingState.effectivePositions = 0;
      emitProgress(trainingState, progressCb);
    } else {
      progressCb?.({
        type: 'train.status',
        payload: { message: `MCTS итерация ${iter + 1}/${iterations}: генерация игр...` }
      });
    }

    // === Batched NN-eval queue ===
    // Accumulate pending evaluations and flush in large batches for GPU saturation
    let evalQueue = [];
    let flushScheduled = false;
    const concurrentGames = Math.max(1, TTT5_MCTS.concurrentGames || 1);
    const selfPlayBatchParallel = Math.max(32, TTT5_MCTS.batchParallel || 32);
    const BATCH_SIZE = Math.max(selfPlayBatchParallel, concurrentGames * 32);

    function flushEvalQueue() {
      flushScheduled = false;
      if (evalQueue.length === 0) return;
      const batch = evalQueue.splice(0);
      const requests = batch.map(b => ({ board: b.board, player: b.player }));
      const results = forwardPassBatched(model, requests);
      for (let i = 0; i < batch.length; i++) {
        batch[i].resolve(results[i]);
      }
    }

    const nnEval = (board, player) => {
      return new Promise((resolve) => {
        evalQueue.push({ board: board.slice(), player, resolve });
        if (evalQueue.length >= BATCH_SIZE) {
          flushEvalQueue();
        } else if (!flushScheduled) {
          flushScheduled = true;
          // Collect all leaf expansions produced in the current tick without adding a timer stall.
          queueMicrotask(flushEvalQueue);
        }
      });
    };

    const recentSamples = [];

    async function playSelfPlayGame(gameIndex) {
      const defaultPlayer = (iter + gameIndex) % 2 === 0 ? 1 : -1;
      const start = createSelfPlayStartWithFallback(hardBuffer, defaultPlayer);
      let board = start.board;
      let player = start.player;
      const gameRecord = [];

      while (!adapter.isTerminal(board)) {
        const legal = adapter.legalMoves(board);
        const forcedTarget = getForcedTrainingTarget(board, player);
        let policyTarget;
        let move;
        let source;
        let targetPeak = 1;

        if (forcedTarget) {
          policyTarget = forcedTarget.policy;
          move = forcedTarget.move;
          source = forcedTarget.source;
        } else {
          const moveNumber = SEQ_LEN - legal.length;
          const explorationMoves = TTT5_MCTS.explorationMoves || 8;
          const inExploration = moveNumber < explorationMoves;
          const currentTemp = inExploration
            ? TTT5_MCTS.temperature
            : (TTT5_MCTS.exploitationTemp || 0.3);
          const currentDirichletAlpha = inExploration ? (TTT5_MCTS.dirichletAlpha || 0) : 0;
          const currentDirichletEpsilon = inExploration ? (TTT5_MCTS.dirichletEpsilon || 0) : 0;

          const mcts = mctsPUCT({
            N: adapter.N,
            nnEval,
            C_puct: TTT5_MCTS.cpuct,
            sims,
            temperature: currentTemp,
            dirichletAlpha: currentDirichletAlpha,
            dirichletEpsilon: currentDirichletEpsilon,
            batchParallel: selfPlayBatchParallel,
            winnerFn: (b) => adapter.winner(b),
            legalMovesFn: (b) => adapter.legalMoves(b),
            candidateMovesFn: (b, p, legalMoves, policyProbs) => adapter.candidateMoves(b, {
              radius: 2,
              maxMoves: 14,
              policyProbs,
              includePolicyTopK: 6,
            }),
          });

          const { pi } = await mcts.run(adapter.cloneBoard(board), player);
          targetPeak = Math.max(...legal.map((mv) => pi[mv] || 0));
          const primaryWeight = targetPeak >= 0.6 ? 0.74 : (targetPeak >= 0.4 ? 0.66 : 0.58);
          policyTarget = distillPolicyTarget(pi, legal, { power: 1.75, primaryWeight });
          move = sampleFromPolicy(pi, legal);
          source = 'mcts';
        }

        gameRecord.push({
          board: adapter.cloneBoard(board),
          planes: adapter.encodePlanes(board, player),
          policy: policyTarget,
          player,
          move,
          source,
          targetPeak,
          priority: computeReplayPriority(board, player, policyTarget, source),
        });

        board = adapter.applyMove(board, move, player);
        player = -player;
      }

      const winner = adapter.winner(board);
      const samples = [];
      for (const rec of gameRecord) {
        const value = winner === rec.player ? 1 : (winner === -rec.player ? -1 : 0);
        samples.push({
          board: rec.board,
          player: rec.player,
          planes: rec.planes,
          policy: rec.policy,
          value,
          priority: rec.priority + Math.abs(value) * 1.25,
          source: rec.source,
          move: rec.move,
          targetPeak: rec.targetPeak,
        });
      }

      return {
        winner,
        seeded: start.seeded,
        samples,
        hardExamples: extractHardExamples(gameRecord, winner),
      };
    }

    for (let batchStart = 0; batchStart < gamesPerIter; batchStart += concurrentGames) {
      const batchEnd = Math.min(gamesPerIter, batchStart + concurrentGames);
      const batchResults = await Promise.all(
        Array.from({ length: batchEnd - batchStart }, (_, idx) => playSelfPlayGame(batchStart + idx))
      );

      for (let idx = 0; idx < batchResults.length; idx++) {
        const g = batchStart + idx;
        const result = batchResults[idx];

        replayBuffer.push(...result.samples);
        recentSamples.push(...result.samples);
        hardBuffer.push(...result.hardExamples);

        if (trainingState) {
          trainingState.game = g + 1;
          trainingState.totalPositions = replayBuffer.length;
          trainingState.hardPositions = hardBuffer.length;
          if (result.seeded) trainingState.seededGames += 1;
          if (result.winner === 1) trainingState.selfPlayStats.wins++;
          else if (result.winner === -1) trainingState.selfPlayStats.losses++;
          else trainingState.selfPlayStats.draws++;
          if ((g + 1) % 5 === 0 || g === gamesPerIter - 1) {
            emitProgress(trainingState, progressCb);
          }
        } else if ((g + 1) % 10 === 0) {
          progressCb?.({
            type: 'train.status',
            payload: { message: `MCTS ${iter + 1}/${iterations}: игра ${g + 1}/${gamesPerIter}` }
          });
        }
      }
    }

    // ===== TRIM REPLAY BUFFER if too large =====
    if (replayBuffer.length > TTT5_CURRICULUM.replayMax) {
      replayBuffer.splice(0, replayBuffer.length - TTT5_CURRICULUM.replayMax);
    }
    trimPriorityBuffer(hardBuffer, TTT5_CURRICULUM.hardMax);

    console.log(`[TrainTTT5] MCTS iter ${iter + 1}: replay=${replayBuffer.length}, hard=${hardBuffer.length}, recent=${recentSamples.length}`);

    if (trainingState) {
      trainingState.phase = 'mcts_train';
      trainingState.epoch = 0;
      trainingState.batch = 0;
      trainingState.hardPositions = hardBuffer.length;
      trainingState.gameStartTime = Date.now();
      emitProgress(trainingState, progressCb);
    } else {
      progressCb?.({
        type: 'train.status',
        payload: { message: `MCTS ${iter + 1}/${iterations}: анализ ошибок и обучение на curriculum batch...` }
      });
    }

    const replayBatch = buildCurriculumBatch(replayBuffer, hardBuffer, recentSamples);
    if (trainingState) {
      trainingState.totalPositions = replayBatch.length;
    }

    batchSize = await trainOnData(model, replayBatch, progressCb, {
      epochs: epochsPerIter,
      phase: `mcts_iter_${iter + 1}`,
      batchSize,
      trainingState,
    });

    // Save checkpoint after each iteration
    await fs.mkdir(SAVE_DIR, { recursive: true });
    await model.save(`file://${SAVE_DIR}`);
    console.log(`[TrainTTT5] Checkpoint saved after MCTS iteration ${iter + 1}`);
  }

  return batchSize;
}

// ===== Try to load existing model =====
async function tryLoadModel() {
  try {
    const modelPath = path.join(SAVE_DIR, 'model.json');
    await fs.access(modelPath);
    const model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('[TrainTTT5] Loaded existing checkpoint');
    return model;
  } catch {
    return null;
  }
}

// ===== Main entry point =====
export async function trainTTT5WithProgress(progressCb, {
  epochs = TTT5_TRAIN.epochs,
  batchSize: _batchSize = TTT5_TRAIN.batchSize,
  bootstrapGames = 200,
  mctsIterations = 10,
  mctsGamesPerIter = 100,
  mctsTrainingSims: _mctsTrainingSims = TTT5_MCTS.trainingSimulations,
} = {}) {
  const MAX_SAFE_BATCH = 256; // RTX 3060 6GB safe limit (768 → OOM)
  let batchSize = Number.isFinite(_batchSize)
    ? Math.min(MAX_SAFE_BATCH, Math.max(GPU_PERF_MODE ? 256 : 16, Math.round(_batchSize / 32) * 32))
    : Math.min(MAX_SAFE_BATCH, TTT5_TRAIN.batchSize);
  const mctsTrainingSims = Number.isFinite(_mctsTrainingSims)
    ? Math.max(16, Math.floor(_mctsTrainingSims))
    : TTT5_MCTS.trainingSimulations;
  try {
    console.log('[TrainTTT5] Starting TTT5 training...');
    console.log('[TrainTTT5] Config:', {
      boardSize: `${TTT5_BOARD_N}x${TTT5_BOARD_N}`,
      winLen: TTT5_WIN_LEN,
      ...TTT5_TRANSFORMER_CFG,
      mctsTrainingSims,
      dirichletAlpha: TTT5_MCTS.dirichletAlpha,
      dirichletEpsilon: TTT5_MCTS.dirichletEpsilon,
    });

    // Check GPU
    const { getGpuInfo } = await import('./tf.mjs');
    const gpuInfo = getGpuInfo();
    const backend = tf.getBackend();
    const isGPU = backend === 'tensorflow' && gpuInfo.available;

    console.log('[TrainTTT5] Backend:', backend, isGPU ? '(GPU)' : '(CPU)');

    // Try loading existing model
    let model = await tryLoadModel();
    const isNewModel = !model;
    const phaseEpochs = getPhaseEpochs(epochs, isNewModel);
    const effectiveMctsIterations = Math.max(1, Math.min(mctsIterations, Math.max(1, Math.ceil(epochs / 2))));
    const mctsEpochsPerIter = Math.max(1, Math.round(phaseEpochs.mcts / effectiveMctsIterations));

    // Compute total phases for progress tracking
    let totalPhases = effectiveMctsIterations; // MCTS iterations
    if (isNewModel) totalPhases += 2; // tactical + bootstrap

    const trainingState = createTrainingState(totalPhases);

    progressCb?.({
      type: 'train.start',
      payload: {
        epochs,
        batchSize,
        modelType: 'ttt5_transformer',
        gpu: gpuInfo.available,
        variant: 'ttt5',
      }
    });

    if (!model) {
      progressCb?.({ type: 'train.status', payload: { message: 'Создание новой модели для 5x5...' } });
      model = buildPVTransformerSeq({
        dModel: TTT5_TRANSFORMER_CFG.dModel,
        numLayers: TTT5_TRANSFORMER_CFG.numLayers,
        heads: TTT5_TRANSFORMER_CFG.heads,
        dropout: TTT5_TRANSFORMER_CFG.dropout,
        seqLen: SEQ_LEN,
        inDim: 3,
      });
    } else {
      progressCb?.({ type: 'train.status', payload: { message: 'Загружена существующая модель 5x5' } });
    }

    // Phase 1: Tactical curriculum (only for new models)
    if (isNewModel) {
      trainingState.phaseStep = 1;
      trainingState.phaseName = 'tactical';
      trainingState.phase = 'training';
      emitProgress(trainingState, progressCb);

      const tacticalData = generateTacticalCurriculum(Math.max(240, Math.floor(bootstrapGames * 1.5)));
      if (tacticalData.length > 0) {
        trainingState.totalPositions = tacticalData.length;
        batchSize = await trainOnData(model, tacticalData, progressCb, {
          epochs: phaseEpochs.tactical,
          phase: 'tactical_curriculum',
          batchSize,
          trainingState,
        });
      }
      trainingState.completedPhases.push('tactical');

      // Phase 2: Bootstrap
      trainingState.phaseStep = 2;
      batchSize = await bootstrapPhase(model, progressCb, bootstrapGames, phaseEpochs.bootstrap, batchSize, trainingState);
      trainingState.completedPhases.push('bootstrap');

      await fs.mkdir(SAVE_DIR, { recursive: true });
      await model.save(`file://${SAVE_DIR}`);
      console.log('[TrainTTT5] Bootstrap checkpoint saved');
    }

    // Phase 3: MCTS Self-Play
    trainingState.phaseStep = isNewModel ? 3 : 1;
    batchSize = await mctsPhase(model, progressCb, effectiveMctsIterations, mctsGamesPerIter, mctsEpochsPerIter, batchSize, trainingState, mctsTrainingSims);

    // Final save
    await model.save(`file://${SAVE_DIR}`);
    progressCb?.({ type: 'train.status', payload: { message: 'Обучение TTT5 завершено!' } });

    // Reload model in service
    try {
      const { reloadTTT5Model } = await import('../service.mjs');
      reloadTTT5Model();
    } catch (e) {
      console.warn('[TrainTTT5] Could not reload model in service:', e.message);
    }

    progressCb?.({ type: 'train.done', payload: { saved: true, variant: 'ttt5' } });
    console.log('[TrainTTT5] Training completed!');
  } catch (e) {
    console.error('[TrainTTT5] Error during training:', e);
    progressCb?.({ type: 'error', error: String(e) });
    throw e;
  }
}
