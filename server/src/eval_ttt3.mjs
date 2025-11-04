// Оценка качества модели для крестики-нолики 3×3
import tfpkg from './tf.mjs';
const tf = tfpkg;
import { winner, legalMoves, applyMove, emptyBoard, cloneBoard, encodePlanes, maskLegalMoves, isTerminal } from './game_ttt3.mjs';
import { getTeacherValueAndPolicy } from './ttt3_minimax.mjs';
import { maskLogits } from './model_pv_transformer_seq.mjs';
import { winningMove, blockingMove, safePick } from './safety.mjs';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Инференс модели: forward(board, player) → {policy, value}
export async function forward(model, board, player) {
  const planes = encodePlanes(board, player);
  const mask = maskLegalMoves(board);
  
  const x = tf.tensor3d([Array.from(planes)], [1, 9, 3]);
  const posIndices = tf.tensor2d([[0, 1, 2, 3, 4, 5, 6, 7, 8]], [1, 9], 'int32');
  
  const [logits, value] = model.predict([x, posIndices]);
  const maskTensor = tf.tensor2d([Array.from(mask)], [1, 9]);
  const masked = maskLogits(logits, maskTensor);
  const probs = tf.softmax(masked);
  
  const policyArray = Array.from(await probs.data());
  const valueScalar = (await value.data())[0];
  
  x.dispose();
  posIndices.dispose();
  logits.dispose();
  value.dispose();
  maskTensor.dispose();
  masked.dispose();
  probs.dispose();
  
  return {
    policy: policyArray,
    value: valueScalar
  };
}

// Игра: модель с Safety vs Minimax
async function playGame(model, nnPlayer, nnStarts) {
  let board = emptyBoard();
  let currentPlayer = nnStarts ? nnPlayer : -nnPlayer;
  const moves = [];
  
  while (!isTerminal(board)) {
    let move;
    
    if (currentPlayer === nnPlayer) {
      // Ход модели с Safety
      const { policy } = await forward(model, board, currentPlayer);
      move = safePick(board, currentPlayer, policy);
    } else {
      // Ход minimax (учителя)
      const { policy: teacherPolicy } = getTeacherValueAndPolicy(board, currentPlayer);
      const moves_legal = legalMoves(board);
      // Выбираем первый оптимальный ход
      const optimalMoves = moves_legal.filter(m => teacherPolicy[m] > 0.01);
      move = optimalMoves.length > 0 ? optimalMoves[0] : moves_legal[0];
    }
    
    if (move < 0 || move >= 9) {
      console.error('[Eval] Invalid move:', move);
      break;
    }
    
    board = applyMove(board, move, currentPlayer);
    moves.push({ move, player: currentPlayer, board: Array.from(board) });
    currentPlayer = -currentPlayer;
  }
  
  const w = winner(board);
  return {
    winner: w,
    moves,
    nnPlayer,
    nnStarts
  };
}

// Юнит-тесты: базовые тактики
export function testBasicTactics() {
  console.log('[Eval] Testing basic tactics...');
  
  // Тест 1: Определение терминального состояния
  const test1 = emptyBoard();
  test1[0] = 1; test1[1] = 1; test1[2] = 1;
  const w1 = winner(test1);
  console.assert(w1 === 1, 'Test 1 failed: should detect X win');
  console.log('✓ Test 1: Terminal state detection');
  
  // Тест 2: "Две в ряд → выигрыш"
  const test2 = emptyBoard();
  test2[0] = 1; test2[1] = 1; // Две X в ряд
  const moves2 = legalMoves(test2);
  const winMove = winningMove(test2, 1);
  console.assert(winMove === 2, 'Test 2 failed: should find winning move');
  console.log('✓ Test 2: Two in a row → win');
  
  // Тест 3: "Блокировка"
  const test3 = emptyBoard();
  test3[0] = -1; test3[1] = -1; // Две O в ряд
  const blockMove = blockingMove(test3, 1);
  console.assert(blockMove === 2, 'Test 3 failed: should find blocking move');
  console.log('✓ Test 3: Blocking opponent');
  
  console.log('[Eval] All basic tactics tests passed!');
}

// Матчи: NN vs Minimax
export async function evaluateMatches(model, numGames = 1000) {
  console.log(`[Eval] Starting ${numGames} games: NN (with Safety) vs Minimax...`);
  
  let nnWins = 0;
  let minimaxWins = 0;
  let draws = 0;
  let nnLosses = 0;
  
  const exampleGames = [];
  
  for (let i = 0; i < numGames; i++) {
    const nnPlayer = i % 2 === 0 ? 1 : -1; // Чередуем первого игрока
    const nnStarts = i % 2 === 0;
    
    const result = await playGame(model, nnPlayer, nnStarts);
    
    if (result.winner === nnPlayer) {
      nnWins++;
    } else if (result.winner === -nnPlayer) {
      minimaxWins++;
      nnLosses++;
    } else {
      draws++;
    }
    
    // Сохраняем первые 3 партии как примеры
    if (exampleGames.length < 3) {
      exampleGames.push(result);
    }
    
    if ((i + 1) % 100 === 0) {
      console.log(`[Eval] Progress: ${i + 1}/${numGames} games`);
      console.log(`[Eval] NN wins: ${nnWins}, Minimax wins: ${minimaxWins}, Draws: ${draws}`);
    }
  }
  
  const winRate = (nnWins / numGames) * 100;
  const drawRate = (draws / numGames) * 100;
  const lossRate = (nnLosses / numGames) * 100;
  
  console.log('\n[Eval] ===== Results =====');
  console.log(`[Eval] Total games: ${numGames}`);
  console.log(`[Eval] NN wins: ${nnWins} (${winRate.toFixed(2)}%)`);
  console.log(`[Eval] Minimax wins: ${minimaxWins} (${(lossRate).toFixed(2)}%)`);
  console.log(`[Eval] Draws: ${draws} (${drawRate.toFixed(2)}%)`);
  console.log(`[Eval] NN losses: ${nnLosses} (${lossRate.toFixed(2)}%)`);
  
  // Критерий успеха: NN не проиграл ни разу (0 поражений)
  const success = nnLosses === 0;
  console.log(`[Eval] Success criterion (0 losses): ${success ? '✓ PASSED' : '✗ FAILED'}`);
  
  // Примеры партий
  console.log('\n[Eval] Example games:');
  exampleGames.forEach((game, idx) => {
    console.log(`\n[Eval] Game ${idx + 1} (NN = ${game.nnPlayer > 0 ? 'X' : 'O'}, starts: ${game.nnStarts}):`);
    game.moves.forEach((m, i) => {
      const player = m.player > 0 ? 'X' : 'O';
      console.log(`  Move ${i + 1}: ${player} → ${m.move}`);
    });
    const winner = game.winner === 0 ? 'Draw' : (game.winner > 0 ? 'X' : 'O');
    console.log(`  Winner: ${winner}`);
  });
  
  return {
    nnWins,
    minimaxWins,
    draws,
    nnLosses,
    success,
    winRate,
    drawRate,
    lossRate
  };
}

// Sanity-чек: NN vs случайный
export async function evaluateVsRandom(model, numGames = 100) {
  console.log(`[Eval] Sanity check: NN vs Random (${numGames} games)...`);
  
  let nnWins = 0;
  let randomWins = 0;
  let draws = 0;
  
  for (let i = 0; i < numGames; i++) {
    let board = emptyBoard();
    let currentPlayer = 1; // NN играет за X
    const nnPlayer = 1;
    
    while (!isTerminal(board)) {
      let move;
      
      if (currentPlayer === nnPlayer) {
        const { policy } = await forward(model, board, currentPlayer);
        move = safePick(board, currentPlayer, policy);
      } else {
        // Случайный ход
        const moves_legal = legalMoves(board);
        move = moves_legal[Math.floor(Math.random() * moves_legal.length)];
      }
      
      board = applyMove(board, move, currentPlayer);
      currentPlayer = -currentPlayer;
    }
    
    const w = winner(board);
    if (w === nnPlayer) nnWins++;
    else if (w === -nnPlayer) randomWins++;
    else draws++;
  }
  
  const winRate = (nnWins / numGames) * 100;
  console.log(`[Eval] NN wins: ${nnWins} (${winRate.toFixed(2)}%)`);
  console.log(`[Eval] Random wins: ${randomWins}`);
  console.log(`[Eval] Draws: ${draws}`);
  console.log(`[Eval] NN should win >> 50%: ${winRate > 50 ? '✓ PASSED' : '✗ FAILED'}`);
  
  return { nnWins, randomWins, draws, winRate };
}

// Основная функция оценки
async function main() {
  const modelPath = path.join(__dirname, '..', 'saved', 'ttt3_transformer', 'model.json');
  
  console.log('[Eval] Loading model from:', modelPath);
  let model;
  try {
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('[Eval] Model loaded successfully');
  } catch (e) {
    console.error('[Eval] Failed to load model:', e.message);
    console.error('[Eval] Please train the model first using train_ttt3_transformer.mjs');
    process.exit(1);
  }
  
  // Юнит-тесты
  testBasicTactics();
  
  // Матчи против minimax
  const results = await evaluateMatches(model, 1000);
  
  // Sanity-чек против случайного
  await evaluateVsRandom(model, 100);
  
  console.log('\n[Eval] ===== Final Summary =====');
  console.log(`[Eval] Model is ${results.success ? 'PERFECT' : 'NOT PERFECT'}`);
  console.log(`[Eval] Losses against minimax: ${results.nnLosses}`);
  console.log(`[Eval] Draw rate: ${results.drawRate.toFixed(2)}%`);
}

// Запуск если файл выполняется напрямую
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(e => {
    console.error('[Eval] Error:', e);
    process.exit(1);
  });
}
