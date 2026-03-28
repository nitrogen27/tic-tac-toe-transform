// Test script: evaluate TTT5 model strength
// Compares: Model vs Random, Model vs Heuristic, Heuristic vs Random
import tfpkg from './src/tf.mjs';
const tf = tfpkg;
import { buildPVTransformerSeq, maskLogits } from './src/model_pv_transformer_seq.mjs';
import { createGameAdapter } from './src/game_adapter.mjs';
import { TTT5_BOARD_N, TTT5_WIN_LEN, TTT5_TRANSFORMER_CFG } from './src/config.mjs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const SEQ_LEN = 25;
const adapter = createGameAdapter({ variant: 'ttt5', winLen: TTT5_WIN_LEN });
const MODEL_DIR = path.join(__dirname, 'saved', 'ttt5_transformer');

// ===== Players =====

function randomMove(board) {
  const moves = adapter.legalMoves(board);
  return moves[Math.floor(Math.random() * moves.length)];
}

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
  return scored[0]?.mv ?? randomMove(board);
}

function modelMove(model, board, player) {
  return tf.tidy(() => {
    const planes = adapter.encodePlanes(board, player);
    const x = tf.tensor3d(Array.from(planes), [1, SEQ_LEN, 3]);
    const posIndices = tf.tensor2d(
      Array.from({ length: SEQ_LEN }, (_, i) => i), [1, SEQ_LEN], 'int32'
    );
    const [logits, valueTensor] = model.predict([x, posIndices]);
    const mask = adapter.maskLegalMoves(board);
    const maskTensor = tf.tensor2d(Array.from(mask), [1, SEQ_LEN]);
    const masked = maskLogits(logits, maskTensor);
    const probs = tf.softmax(masked, -1).dataSync();
    const value = valueTensor.dataSync()[0];

    // Safety: win/block first
    const winMove = adapter.findImmediateWin(board, player);
    if (winMove >= 0) return { move: winMove, value, confidence: 1.0 };
    const blockMove = adapter.findImmediateBlock(board, player);
    if (blockMove >= 0) return { move: blockMove, value, confidence: probs[blockMove] };

    // Argmax
    const legalM = adapter.legalMoves(board);
    let bestMove = legalM[0], bestProb = -1;
    for (const m of legalM) {
      if (probs[m] > bestProb) { bestProb = probs[m]; bestMove = m; }
    }
    return { move: bestMove, value, confidence: bestProb };
  });
}

// ===== Play a game =====
function playGame(playerX, playerO) {
  let board = adapter.emptyBoard();
  let player = 1; // X starts
  let moveCount = 0;

  while (!adapter.isTerminal(board)) {
    const move = player === 1 ? playerX(board, player) : playerO(board, player);
    const mv = typeof move === 'object' ? move.move : move;
    board = adapter.applyMove(board, mv, player);
    player = -player;
    moveCount++;
  }

  const w = adapter.winner(board);
  return { winner: w, moveCount };
}

// ===== Run tournament =====
async function runTournament(name, playerA, playerB, numGames) {
  let winsA = 0, winsB = 0, draws = 0;
  let totalMoves = 0;

  for (let i = 0; i < numGames; i++) {
    // Alternate colors for fairness
    let result;
    if (i % 2 === 0) {
      result = playGame(playerA, playerB);
      if (result.winner === 1) winsA++;
      else if (result.winner === -1) winsB++;
      else draws++;
    } else {
      result = playGame(playerB, playerA);
      if (result.winner === 1) winsB++;
      else if (result.winner === -1) winsA++;
      else draws++;
    }
    totalMoves += result.moveCount;
  }

  const avgMoves = (totalMoves / numGames).toFixed(1);
  console.log(`  ${name}: A=${winsA}W B=${winsB}W D=${draws} (${numGames} games, avg ${avgMoves} moves)`);
  return { winsA, winsB, draws, avgMoves };
}

// ===== Main =====
async function main() {
  const NUM_GAMES = 40; // 40 games per matchup (20 as X, 20 as O)

  console.log('='.repeat(60));
  console.log('TTT5 Model Strength Evaluation (5x5, 4-in-a-row)');
  console.log('='.repeat(60));

  // 1. Baseline: Heuristic vs Random
  console.log('\n--- Baseline: Heuristic vs Random ---');
  const heuristicVsRandom = await runTournament(
    'Heuristic(A) vs Random(B)',
    (b, p) => heuristicMove(b, p),
    (b, p) => randomMove(b),
    NUM_GAMES
  );

  // 2. Try to load model
  const modelPath = path.join(MODEL_DIR, 'model.json');
  if (!fs.existsSync(modelPath)) {
    console.log('\n❌ Model not found at', modelPath);
    console.log('Train a model first with "Обучить с нуля"');
    process.exit(1);
  }

  console.log('\nLoading model...');
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  console.log('✓ Model loaded');

  // 3. Model vs Random
  console.log('\n--- Model vs Random ---');
  const modelVsRandom = await runTournament(
    'Model(A) vs Random(B)',
    (b, p) => modelMove(model, b, p),
    (b, p) => randomMove(b),
    NUM_GAMES
  );

  // 4. Model vs Heuristic
  console.log('\n--- Model vs Heuristic ---');
  const modelVsHeuristic = await runTournament(
    'Model(A) vs Heuristic(B)',
    (b, p) => modelMove(model, b, p),
    (b, p) => heuristicMove(b, p),
    NUM_GAMES
  );

  // 5. Summary
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  const hWinRate = ((heuristicVsRandom.winsA / NUM_GAMES) * 100).toFixed(1);
  const mVsRWinRate = ((modelVsRandom.winsA / NUM_GAMES) * 100).toFixed(1);
  const mVsRLossRate = ((modelVsRandom.winsB / NUM_GAMES) * 100).toFixed(1);
  const mVsHWinRate = ((modelVsHeuristic.winsA / NUM_GAMES) * 100).toFixed(1);
  const mVsHLossRate = ((modelVsHeuristic.winsB / NUM_GAMES) * 100).toFixed(1);
  const mVsHDrawRate = ((modelVsHeuristic.draws / NUM_GAMES) * 100).toFixed(1);

  console.log(`
┌──────────────────────────────────────┐
│ Heuristic vs Random:  ${hWinRate}% win rate │
│ Model vs Random:      ${mVsRWinRate}% win rate │
│ Model vs Heuristic:   ${mVsHWinRate}% win / ${mVsHLossRate}% loss / ${mVsHDrawRate}% draw │
└──────────────────────────────────────┘`);

  if (parseFloat(mVsRWinRate) > parseFloat(hWinRate)) {
    console.log('\n✅ Model is STRONGER than heuristic against random opponents!');
  } else if (parseFloat(mVsRWinRate) > 50) {
    console.log('\n⚠️  Model beats random but not as well as heuristic');
  } else {
    console.log('\n❌ Model is weaker than random baseline — needs more training');
  }

  if (parseFloat(mVsHWinRate) > 40) {
    console.log('✅ Model competes well against heuristic');
  } else if (parseFloat(mVsHWinRate) > 20) {
    console.log('⚠️  Model has some strength vs heuristic but needs improvement');
  } else {
    console.log('❌ Model cannot beat heuristic — needs more MCTS iterations');
  }

  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
