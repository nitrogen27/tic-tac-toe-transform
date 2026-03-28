// Gomoku Engine V2 Training Pipeline
// 3-phase training: tactical curriculum → alpha-beta bootstrap → NN-guided self-play

import tfpkg from './tf.mjs';
const tf = tfpkg;
import fs from 'fs/promises';
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import { createGomokuEngine, GomokuBoard } from './engine/index.mjs';
import { GOMOKU_TRANSFORMER_CFG, GOMOKU_TRAIN_CFG, GOMOKU_VARIANTS } from './engine/config.mjs';
import { buildPVTransformerSeq, maskLogits } from './model_pv_transformer_seq.mjs';
import { encodePlanes, maskFromBoard, legalMoves } from './game_nxn.mjs';
import { findWinningMoves, evaluateMovePatterns } from './engine/patterns.mjs';
import { getSymmetryMaps, transformPlanes, transformPolicy, inverseTransformPolicy } from './engine/symmetry_nxn.mjs';
import { generateCandidates } from './engine/candidates.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const MODEL_DIR = path.resolve(__dirname, '..', 'saved');

async function ensureDir(p) { try { await fs.mkdir(p, { recursive: true }); } catch {} }

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Main training entry point for Gomoku Engine V2.
 *
 * @param {Function} progressCb - WebSocket progress callback
 * @param {object} opts
 * @param {string} [opts.variant='gomoku9'] - board variant
 * @param {number} [opts.epochs] - override epochs
 * @param {number} [opts.selfPlayGames] - override self-play games
 * @param {number} [opts.iterations=3] - training iterations
 */
export async function trainGomokuWithProgress(progressCb, opts = {}) {
  const variant = opts.variant || 'gomoku9';
  const config = GOMOKU_VARIANTS[variant] || { N: 9, winLen: 5, seqLen: 81 };
  const N = config.N;
  const winLen = config.winLen;
  const seqLen = config.seqLen;
  const iterations = opts.iterations || 3;

  progressCb?.({ type: 'train.start', payload: { variant, N, winLen, iterations } });
  progressCb?.({ type: 'train.status', payload: { message: `Starting Gomoku ${N}x${N} training (${iterations} iterations)...` } });

  // Create engine (no NN initially)
  const engine = createGomokuEngine({ N, winLen });

  // Build or load model
  const modelDir = path.join(MODEL_DIR, `gomoku${N}_transformer`);
  await ensureDir(modelDir);
  let model = await loadOrCreateModel(modelDir, seqLen, progressCb);

  const replayBuffer = [];
  const maxReplaySize = GOMOKU_TRAIN_CFG.replayBufferMax;

  // Phase 1: Tactical Curriculum
  progressCb?.({ type: 'train.status', payload: { phase: 'tactical', message: 'Phase 1: Tactical Curriculum...' } });
  const tacticalData = generateTacticalCurriculum(N, winLen, 2000);
  progressCb?.({ type: 'train.status', payload: { message: `Generated ${tacticalData.length} tactical positions` } });

  if (tacticalData.length > 0) {
    await trainOnBatch(model, tacticalData, seqLen, {
      epochs: 3,
      batchSize: GOMOKU_TRAIN_CFG.batchSize,
      lr: GOMOKU_TRAIN_CFG.lr * 2,
      progressCb,
      phaseLabel: 'Tactical',
    });
  }

  // Phase 2: Alpha-Beta Bootstrap
  progressCb?.({ type: 'train.status', payload: { phase: 'bootstrap', message: 'Phase 2: Alpha-Beta Bootstrap...' } });
  const bootstrapGames = opts.selfPlayGames || 100;
  const bootstrapData = await generateBootstrapGames(engine, N, winLen, bootstrapGames, progressCb);
  replayBuffer.push(...bootstrapData);

  progressCb?.({ type: 'train.status', payload: { message: `Bootstrap: ${bootstrapData.length} positions from ${bootstrapGames} games` } });

  await trainOnBatch(model, [...tacticalData, ...bootstrapData], seqLen, {
    epochs: opts.epochs || GOMOKU_TRAIN_CFG.epochs,
    batchSize: GOMOKU_TRAIN_CFG.batchSize,
    lr: GOMOKU_TRAIN_CFG.lr,
    progressCb,
    phaseLabel: 'Bootstrap',
  });

  // Phase 3: Self-Play with NN Guidance
  for (let iter = 0; iter < iterations; iter++) {
    progressCb?.({ type: 'train.status', payload: {
      phase: 'selfplay',
      iteration: iter + 1,
      iterations,
      message: `Phase 3: Self-Play iteration ${iter + 1}/${iterations}...`
    }});

    const selfPlayGames = Math.max(30, (opts.selfPlayGames || 50));
    const selfPlayData = await generateSelfPlayGames(engine, model, N, winLen, seqLen, selfPlayGames, progressCb);

    replayBuffer.push(...selfPlayData);
    if (replayBuffer.length > maxReplaySize) {
      replayBuffer.splice(0, replayBuffer.length - maxReplaySize);
    }

    // Mix fresh + replay
    const trainBatch = buildTrainingBatch(selfPlayData, replayBuffer, tacticalData);

    await trainOnBatch(model, trainBatch, seqLen, {
      epochs: Math.max(3, (opts.epochs || 5)),
      batchSize: GOMOKU_TRAIN_CFG.batchSize,
      lr: GOMOKU_TRAIN_CFG.lr / (1 + iter * 0.5),
      progressCb,
      phaseLabel: `SelfPlay-${iter + 1}`,
    });

    // Save checkpoint
    await model.save(`file://${modelDir}`);
    progressCb?.({ type: 'train.status', payload: { message: `Checkpoint saved after iteration ${iter + 1}` } });
  }

  // Final save
  await model.save(`file://${modelDir}`);
  progressCb?.({ type: 'train.done', payload: { saved: true, modelDir, variant, positions: replayBuffer.length } });

  return { success: true, modelDir, variant };
}

/**
 * Generate tactical curriculum: positions with clear tactical answers.
 */
function generateTacticalCurriculum(N, winLen, targetCount) {
  const data = [];
  const board = new GomokuBoard(N, winLen);

  // Generate random positions and extract tactical situations
  for (let g = 0; g < targetCount * 2 && data.length < targetCount; g++) {
    // Reset board
    board.cells.fill(0);
    board.moveCount = 0;
    board.lastMove = -1;
    board.history = [];
    board.hashHi = 0;
    board.hashLo = 0;

    // Play random moves to a random depth (3-15)
    const depth = 3 + Math.floor(Math.random() * Math.min(13, N * N - 3));
    const players = [1, -1];

    for (let d = 0; d < depth; d++) {
      const player = players[d % 2];
      const legal = board.legalMoves();
      if (legal.length === 0) break;

      // Bias toward center region
      const mid = Math.floor(N / 2);
      const scored = legal.map(m => {
        const r = (m / N) | 0, c = m % N;
        return { m, score: Math.max(0, N - (Math.abs(r - mid) + Math.abs(c - mid))) + Math.random() * 3 };
      });
      scored.sort((a, b) => b.score - a.score);
      const move = scored[0].m;

      board.makeMove(move, player);
      if (board.winner() !== null) break;
    }

    if (board.winner() !== null) continue;

    // Check for tactical situations
    const currentPlayer = players[board.moveCount % 2];
    const cells = board.cells;

    // Win-in-one
    const winMoves = findWinningMoves(board, currentPlayer);
    if (winMoves.length > 0) {
      const policy = new Float32Array(N * N);
      policy[winMoves[0]] = 1.0;
      data.push({
        planes: new Float32Array(encodePlanes(cells, currentPlayer)),
        policy,
        value: 1.0,
      });
      continue;
    }

    // Block-in-one
    const blockMoves = findWinningMoves(board, -currentPlayer);
    if (blockMoves.length > 0) {
      const policy = new Float32Array(N * N);
      // Spread probability among all blocking moves
      for (const m of blockMoves) policy[m] = 1.0 / blockMoves.length;
      data.push({
        planes: new Float32Array(encodePlanes(cells, currentPlayer)),
        policy,
        value: blockMoves.length > 1 ? -0.5 : 0.1,
      });
      continue;
    }

    // Evaluate moves by pattern scores
    const candidates = generateCandidates(board, { radius: 2, maxCandidates: 20 });
    if (candidates.length > 0) {
      const scores = candidates.map(m => ({
        m,
        score: evaluateMovePatterns(board, m, currentPlayer),
      }));
      scores.sort((a, b) => b.score - a.score);

      if (scores[0].score > 1000) { // non-trivial pattern
        const total = scores.reduce((s, x) => s + Math.max(0, x.score), 0);
        if (total > 0) {
          const policy = new Float32Array(N * N);
          for (const { m, score } of scores) {
            if (score > 0) policy[m] = score / total;
          }
          data.push({
            planes: new Float32Array(encodePlanes(cells, currentPlayer)),
            policy,
            value: 0.0,
          });
        }
      }
    }
  }

  return data;
}

/**
 * Generate games using alpha-beta engine (no NN) for bootstrap.
 */
async function generateBootstrapGames(engine, N, winLen, numGames, progressCb) {
  const data = [];
  const players = [1, -1];

  for (let g = 0; g < numGames; g++) {
    if (g % 10 === 0) {
      progressCb?.({ type: 'train.status', payload: {
        message: `Bootstrap game ${g + 1}/${numGames} (${data.length} positions)...`
      }});
    }

    const board = new GomokuBoard(N, winLen);
    const gamePositions = [];

    while (true) {
      const winner = board.winner();
      if (winner !== null) {
        // Assign values: +1 for winner's positions, -1 for loser's
        for (const pos of gamePositions) {
          if (winner === 0) pos.value = 0;
          else pos.value = pos.player === winner ? 1.0 : -1.0;
        }
        break;
      }

      const currentPlayer = players[board.moveCount % 2];

      // Save position before move
      const planes = new Float32Array(encodePlanes(board.cells, currentPlayer));

      // Get engine move
      const result = engine.bestMove(board.toArray(), currentPlayer, {
        maxDepth: 6,
        timeLimitMs: 500,
      });

      if (result.move < 0) break;

      // Build policy from engine
      const policy = result.policy || new Float32Array(N * N);

      gamePositions.push({
        planes,
        policy: new Float32Array(policy),
        value: 0, // will be set after game ends
        player: currentPlayer,
      });

      board.makeMove(result.move, currentPlayer);
    }

    // Add to dataset (remove player field)
    for (const pos of gamePositions) {
      data.push({
        planes: pos.planes,
        policy: pos.policy,
        value: pos.value,
      });
    }
  }

  return data;
}

/**
 * Generate self-play games with NN-guided engine.
 */
async function generateSelfPlayGames(engine, model, N, winLen, seqLen, numGames, progressCb) {
  const data = [];
  const symmetryMaps = getSymmetryMaps(N);
  const players = [1, -1];

  for (let g = 0; g < numGames; g++) {
    if (g % 5 === 0) {
      progressCb?.({ type: 'train.status', payload: {
        message: `Self-play game ${g + 1}/${numGames} (${data.length} positions)...`
      }});
    }

    const board = new GomokuBoard(N, winLen);
    const gamePositions = [];

    while (true) {
      const winner = board.winner();
      if (winner !== null) {
        for (const pos of gamePositions) {
          if (winner === 0) pos.value = 0;
          else pos.value = pos.player === winner ? 1.0 : -1.0;
        }
        break;
      }

      const currentPlayer = players[board.moveCount % 2];

      // Get NN policy for move ordering
      let nnPolicy = null;
      try {
        nnPolicy = await getSymmetryEnsemblePolicy(model, board.cells, currentPlayer, N, seqLen, symmetryMaps);
      } catch (e) {
        // NN failed — continue without
      }

      const planes = new Float32Array(encodePlanes(board.cells, currentPlayer));

      // Use engine with NN guidance
      const result = engine.bestMove(board.toArray(), currentPlayer, {
        maxDepth: 8,
        timeLimitMs: 1000,
        nnPolicy,
      });

      if (result.move < 0) break;

      const policy = result.policy || new Float32Array(N * N);

      // Add temperature noise for exploration in early moves
      if (board.moveCount < 6) {
        const temp = 1.2;
        const legal = board.legalMoves();
        const candidates = generateCandidates(board, { radius: 2, maxCandidates: 20 });
        for (const m of candidates) {
          policy[m] = Math.pow(Math.max(policy[m], 0.01), 1 / temp);
        }
        // Normalize
        const sum = policy.reduce((s, v) => s + v, 0);
        if (sum > 0) for (let i = 0; i < policy.length; i++) policy[i] /= sum;
      }

      gamePositions.push({ planes, policy: new Float32Array(policy), value: 0, player: currentPlayer });

      board.makeMove(result.move, currentPlayer);
    }

    for (const pos of gamePositions) {
      data.push({ planes: pos.planes, policy: pos.policy, value: pos.value });
    }
  }

  return data;
}

/**
 * Get NN policy through symmetry ensemble (8 forward passes).
 */
async function getSymmetryEnsemblePolicy(model, cells, player, N, seqLen, symmetryMaps) {
  const numSymmetries = symmetryMaps.length;
  const planesPerCell = 3;
  const totalPlanes = seqLen * planesPerCell;

  const xBatch = new Float32Array(numSymmetries * totalPlanes);
  const posBatch = new Int32Array(numSymmetries * seqLen);

  for (let s = 0; s < numSymmetries; s++) {
    const { map } = symmetryMaps[s];
    const transformedCells = new Int8Array(seqLen);
    for (let i = 0; i < seqLen; i++) transformedCells[i] = cells[map[i]];

    const planes = encodePlanes(transformedCells, player);
    xBatch.set(planes, s * totalPlanes);

    for (let j = 0; j < seqLen; j++) posBatch[s * seqLen + j] = j;
  }

  const xTensor = tf.tensor3d(xBatch, [numSymmetries, seqLen, planesPerCell]);
  const posTensor = tf.tensor2d(posBatch, [numSymmetries, seqLen], 'int32');

  const { probsData } = tf.tidy(() => {
    const [logits] = model.predict([xTensor, posTensor]);
    // Mask illegal moves
    const maskArr = new Float32Array(numSymmetries * seqLen);
    for (let s = 0; s < numSymmetries; s++) {
      const { map } = symmetryMaps[s];
      for (let i = 0; i < seqLen; i++) {
        maskArr[s * seqLen + i] = cells[map[i]] === 0 ? 1.0 : 0.0;
      }
    }
    const maskTensor = tf.tensor2d(maskArr, [numSymmetries, seqLen]);
    const masked = maskLogits(logits, maskTensor);
    const probs = tf.softmax(masked, -1);
    return { probsData: probs.dataSync() };
  });

  xTensor.dispose();
  posTensor.dispose();

  // Average inverse-transformed policies
  const avgPolicy = new Float32Array(seqLen);
  for (let s = 0; s < numSymmetries; s++) {
    const start = s * seqLen;
    const symmPolicy = probsData.slice(start, start + seqLen);
    const restored = inverseTransformPolicy(symmPolicy, symmetryMaps[s].map);
    for (let i = 0; i < seqLen; i++) avgPolicy[i] += restored[i];
  }
  for (let i = 0; i < seqLen; i++) avgPolicy[i] /= numSymmetries;

  return avgPolicy;
}

/**
 * Build mixed training batch from fresh, replay, and tactical data.
 */
function buildTrainingBatch(freshData, replayBuffer, tacticalData) {
  const maxSize = GOMOKU_TRAIN_CFG.replayBufferMax;
  const batch = [];

  // 40% fresh
  const freshCount = Math.min(freshData.length, Math.floor(maxSize * 0.4));
  const freshSampled = shuffleArray([...freshData]).slice(0, freshCount);
  batch.push(...freshSampled);

  // 40% replay
  const replayCount = Math.min(replayBuffer.length, Math.floor(maxSize * 0.4));
  const replaySampled = shuffleArray([...replayBuffer]).slice(0, replayCount);
  batch.push(...replaySampled);

  // 20% tactical
  const tacticalCount = Math.min(tacticalData.length, Math.floor(maxSize * 0.2));
  const tacticalSampled = shuffleArray([...tacticalData]).slice(0, tacticalCount);
  batch.push(...tacticalSampled);

  return shuffleArray(batch);
}

/**
 * Train model on a batch of positions.
 */
async function trainOnBatch(model, data, seqLen, { epochs, batchSize, lr, progressCb, phaseLabel }) {
  if (data.length === 0) return;

  const augmented = augmentWithSymmetries(data, seqLen);
  const N = augmented.length;
  const planesPerCell = 3;

  progressCb?.({ type: 'train.status', payload: {
    message: `[${phaseLabel}] Training on ${N} positions (${data.length} raw + symmetry)...`
  }});

  function policyLossFromLogits(yTrue, yPred) {
    const logProbs = tf.logSoftmax(yPred, -1);
    return tf.neg(tf.sum(tf.mul(yTrue, logProbs), -1)).mean();
  }

  model.compile({
    optimizer: tf.train.adam(lr),
    loss: [policyLossFromLogits, 'meanSquaredError'],
    lossWeights: [1.0, GOMOKU_TRAIN_CFG.weightValue],
  });

  const planesArr = new Float32Array(N * seqLen * planesPerCell);
  const posArr = new Int32Array(N * seqLen);
  const policyArr = new Float32Array(N * seqLen);
  const valueArr = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    planesArr.set(augmented[i].planes, i * seqLen * planesPerCell);
    policyArr.set(augmented[i].policy, i * seqLen);
    valueArr[i] = augmented[i].value;
    for (let j = 0; j < seqLen; j++) posArr[i * seqLen + j] = j;
  }

  const xTensor = tf.tensor3d(planesArr, [N, seqLen, planesPerCell]);
  const posTensor = tf.tensor2d(posArr, [N, seqLen], 'int32');
  const yPolicyTensor = tf.tensor2d(policyArr, [N, seqLen]);
  const yValueTensor = tf.tensor2d(valueArr, [N, 1]);

  const effectiveBatch = Math.min(batchSize, N);

  await model.fit([xTensor, posTensor], [yPolicyTensor, yValueTensor], {
    epochs,
    batchSize: effectiveBatch,
    shuffle: true,
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        progressCb?.({ type: 'train.progress', payload: {
          phase: phaseLabel,
          epoch: epoch + 1,
          epochs,
          loss: Number(logs.loss ?? 0).toFixed(4),
          percent: Math.round(((epoch + 1) / epochs) * 100),
        }});
      },
    },
  });

  xTensor.dispose();
  posTensor.dispose();
  yPolicyTensor.dispose();
  yValueTensor.dispose();
}

/**
 * Augment data with 8 D4 symmetries.
 */
function augmentWithSymmetries(data, seqLen) {
  const N = Math.round(Math.sqrt(seqLen));
  const symmetryMaps = getSymmetryMaps(N);
  const augmented = [];

  for (const sample of data) {
    for (const { map } of symmetryMaps) {
      augmented.push({
        planes: transformPlanes(sample.planes, map, 3),
        policy: transformPolicy(sample.policy, map),
        value: sample.value,
      });
    }
  }

  return augmented;
}

/**
 * Load existing model or create new one.
 */
async function loadOrCreateModel(modelDir, seqLen, progressCb) {
  const modelPath = path.join(modelDir, 'model.json');
  const exists = await fs.stat(modelPath).then(() => true).catch(() => false);

  if (exists) {
    try {
      const model = await tf.loadLayersModel(`file://${modelPath}`);
      progressCb?.({ type: 'train.status', payload: { message: 'Loaded existing model' } });
      return model;
    } catch (e) {
      progressCb?.({ type: 'train.status', payload: { message: `Failed to load model: ${e.message}. Creating new.` } });
    }
  }

  // Create new model
  const cfg = GOMOKU_TRANSFORMER_CFG;
  progressCb?.({ type: 'train.status', payload: { message: `Creating new transformer: seqLen=${seqLen}, d=${cfg.dModel}, layers=${cfg.numLayers}` } });

  const model = buildPVTransformerSeq({
    seqLen,
    dModel: cfg.dModel,
    numLayers: cfg.numLayers,
    heads: cfg.heads,
    dropout: cfg.dropout,
  });

  return model;
}

export default trainGomokuWithProgress;
