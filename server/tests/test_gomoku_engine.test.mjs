// Test suite for Gomoku Engine V2
import { GomokuBoard } from '../src/engine/board.mjs';
import { scanAllPatterns, P, findWinningMoves, evaluateMovePatterns } from '../src/engine/patterns.mjs';
import { evaluate } from '../src/engine/evaluate.mjs';
import { generateCandidates } from '../src/engine/candidates.mjs';
import { vcfSearch } from '../src/engine/threats.mjs';
import { iterativeDeepeningSearch } from '../src/engine/search.mjs';
import { createGomokuEngine } from '../src/engine/index.mjs';
import { buildSymmetryMaps, transformBoard, inverseTransformPolicy } from '../src/engine/symmetry_nxn.mjs';

const assert = (cond, msg) => {
  if (!cond) throw new Error(`ASSERTION FAILED: ${msg}`);
};

// ===== Board Tests =====
function testBoard() {
  console.log('--- Board Tests ---');

  const b = new GomokuBoard(9, 5);
  assert(b.N === 9, 'board size');
  assert(b.size === 81, 'board total cells');
  assert(b.winLen === 5, 'win length');

  // Make and undo move
  b.makeMove(40, 1); // center
  assert(b.cells[40] === 1, 'stone placed');
  assert(b.moveCount === 1, 'move count');
  assert(b.lastMove === 40, 'last move');

  const h1 = b.hashKey();
  b.undoMove();
  assert(b.cells[40] === 0, 'stone removed');
  assert(b.moveCount === 0, 'move count after undo');
  const h0 = b.hashKey();
  assert(h0 !== h1, 'hash changed');

  // fromArray
  const arr = new Int8Array(81);
  arr[40] = 1;
  arr[41] = -1;
  const b2 = GomokuBoard.fromArray(arr, 9, 5);
  assert(b2.cells[40] === 1, 'fromArray stone 1');
  assert(b2.cells[41] === -1, 'fromArray stone -1');
  assert(b2.moveCount === 2, 'fromArray move count');

  // isWinningMove
  const b3 = new GomokuBoard(9, 5);
  // Place 4 in a row horizontally (row 0, cols 0-3)
  for (let c = 0; c < 4; c++) b3.makeMove(c, 1);
  // Place stone to complete 5
  b3.cells[4] = 1;
  assert(b3.isWinningMove(4, 1), 'five in a row detected');
  b3.cells[4] = 0;

  console.log('  Board tests passed!');
}

// ===== Pattern Tests =====
function testPatterns() {
  console.log('--- Pattern Tests ---');

  const N = 9;
  const winLen = 5;

  // Open three: _XXX_ on row 0
  const cells1 = new Int8Array(81);
  cells1[1] = 1; cells1[2] = 1; cells1[3] = 1;
  const counts1 = scanAllPatterns(cells1, N, winLen, 1);
  assert(counts1[P.OPEN_THREE] >= 1, `open three detected (got ${counts1[P.OPEN_THREE]})`);

  // Half four: XXXX_ on row 0
  const cells2 = new Int8Array(81);
  cells2[0] = 1; cells2[1] = 1; cells2[2] = 1; cells2[3] = 1;
  const counts2 = scanAllPatterns(cells2, N, winLen, 1);
  assert(counts2[P.HALF_FOUR] >= 1 || counts2[P.OPEN_FOUR] >= 1, `four detected`);

  // Five in a row
  const cells3 = new Int8Array(81);
  for (let i = 0; i < 5; i++) cells3[i] = 1;
  const counts3 = scanAllPatterns(cells3, N, winLen, 1);
  assert(counts3[P.FIVE] >= 1, 'five detected');

  // Winning moves detection
  const b4 = new GomokuBoard(9, 5);
  // 4 in a row with open end
  for (let i = 0; i < 4; i++) b4.cells[i] = 1;
  const winMoves = findWinningMoves(b4, 1);
  assert(winMoves.includes(4), `winning move at 4 (got ${winMoves})`);

  console.log('  Pattern tests passed!');
}

// ===== Evaluation Tests =====
function testEvaluation() {
  console.log('--- Evaluation Tests ---');

  const b = new GomokuBoard(9, 5);

  // Empty board: should be roughly 0
  const emptyScore = evaluate(b, 1);
  assert(Math.abs(emptyScore) < 100, `empty board score near 0 (got ${emptyScore})`);

  // Board with threat: should be positive for attacker
  b.cells[0] = 1; b.cells[1] = 1; b.cells[2] = 1; b.cells[3] = 1;
  const threatScore = evaluate(b, 1);
  assert(threatScore > 5000, `threat score positive (got ${threatScore})`);

  console.log('  Evaluation tests passed!');
}

// ===== Candidates Tests =====
function testCandidates() {
  console.log('--- Candidate Tests ---');

  // Empty board: should suggest center
  const b1 = new GomokuBoard(9, 5);
  const c1 = generateCandidates(b1);
  assert(c1.length === 1, `empty board: 1 candidate (got ${c1.length})`);
  assert(c1[0] === 40, `center move (got ${c1[0]})`);

  // Board with some stones: candidates near stones
  const b2 = new GomokuBoard(15, 5);
  b2.makeMove(112, 1); // center of 15x15
  const c2 = generateCandidates(b2);
  assert(c2.length > 0, 'has candidates');
  assert(c2.length <= 30, `limited candidates (got ${c2.length})`);

  // All candidates should be within radius 2
  for (const m of c2) {
    const r = (m / 15) | 0, c = m % 15;
    const dist = Math.max(Math.abs(r - 7), Math.abs(c - 7));
    assert(dist <= 2, `candidate within radius (dist=${dist})`);
  }

  console.log('  Candidate tests passed!');
}

// ===== VCF Tests =====
function testVCF() {
  console.log('--- VCF Tests ---');

  // Position with forced win: XXXX_ (one move to win)
  const b1 = new GomokuBoard(9, 5);
  for (let i = 0; i < 4; i++) b1.cells[i] = 1;
  const vcf1 = vcfSearch(b1, 1, 4);
  assert(vcf1 !== null, 'VCF found for XXXX_');
  assert(vcf1.move === 4, `VCF move at 4 (got ${vcf1.move})`);

  // No VCF: XX___ with opponent blocking
  const b2 = new GomokuBoard(9, 5);
  b2.cells[0] = 1; b2.cells[1] = 1;
  b2.cells[4] = -1; // opponent blocks extension
  const vcf2 = vcfSearch(b2, 1, 6);
  assert(vcf2 === null, 'no VCF for blocked XX');

  console.log('  VCF tests passed!');
}

// ===== Search Tests =====
function testSearch() {
  console.log('--- Search Tests ---');

  // Winning position: should find the win
  const b1 = new GomokuBoard(9, 5);
  for (let i = 0; i < 4; i++) b1.cells[i] = 1;
  const r1 = iterativeDeepeningSearch(b1, 1, { maxDepth: 4, timeLimitMs: 1000 });
  assert(r1.move === 4, `search finds winning move (got ${r1.move})`);
  assert(r1.score >= 500000, `winning score (got ${r1.score})`);

  // Blocking position: opponent has XXXX_, must block
  const b2 = new GomokuBoard(9, 5);
  for (let i = 0; i < 4; i++) b2.cells[i] = -1; // opponent four
  b2.cells[36] = 1; // our stone elsewhere
  const r2 = iterativeDeepeningSearch(b2, 1, { maxDepth: 4, timeLimitMs: 1000 });
  assert(r2.move === 4, `search blocks opponent (got ${r2.move})`);

  console.log('  Search tests passed!');
}

// ===== Symmetry Tests =====
function testSymmetry() {
  console.log('--- Symmetry Tests ---');

  const maps = buildSymmetryMaps(9);
  assert(maps.length === 8, '8 symmetries');

  // Identity should map to itself
  const id = maps[0].map;
  for (let i = 0; i < 81; i++) {
    assert(id[i] === i, `identity map[${i}] === ${i}`);
  }

  // Transform and inverse should compose to identity
  const policy = new Float32Array(81);
  for (let i = 0; i < 81; i++) policy[i] = Math.random();

  for (const { name, map } of maps) {
    const transformed = new Float32Array(81);
    for (let i = 0; i < 81; i++) transformed[i] = policy[map[i]];
    const restored = inverseTransformPolicy(transformed, map);
    for (let i = 0; i < 81; i++) {
      assert(Math.abs(restored[i] - policy[i]) < 1e-6, `symmetry ${name} roundtrip at ${i}`);
    }
  }

  console.log('  Symmetry tests passed!');
}

// ===== Full Engine Integration Tests =====
function testEngine() {
  console.log('--- Engine Integration Tests ---');

  const engine = createGomokuEngine({ N: 9, winLen: 5 });

  // Empty board: should play center
  const empty = new Int8Array(81);
  const r1 = engine.bestMove(empty, 1, { timeLimitMs: 500 });
  assert(r1.move >= 0, 'engine returns a move');
  assert(r1.source, 'result has source');
  assert(r1.value !== undefined, 'result has value');

  // Winning position
  const winning = new Int8Array(81);
  for (let i = 0; i < 4; i++) winning[i] = 1;
  const r2 = engine.bestMove(winning, 1, { timeLimitMs: 100 });
  assert(r2.move === 4, `engine finds win (got ${r2.move})`);
  assert(r2.source === 'safety_win', `source=safety_win (got ${r2.source})`);

  // Blocking position
  const blocking = new Int8Array(81);
  for (let i = 0; i < 4; i++) blocking[i] = -1;
  blocking[40] = 1;
  const r3 = engine.bestMove(blocking, 1, { timeLimitMs: 100 });
  assert(r3.move === 4, `engine blocks (got ${r3.move})`);

  // 15x15 board
  const engine15 = createGomokuEngine({ N: 15, winLen: 5 });
  const empty15 = new Int8Array(225);
  const r4 = engine15.bestMove(empty15, 1, { timeLimitMs: 1000 });
  assert(r4.move >= 0, 'engine15 returns a move');
  console.log(`  15x15 first move: ${r4.move} (${(r4.move / 15) | 0},${r4.move % 15}), source=${r4.source}`);

  // 16x16 board
  const engine16 = createGomokuEngine({ N: 16, winLen: 5 });
  const empty16 = new Int8Array(256);
  const r5 = engine16.bestMove(empty16, 1, { timeLimitMs: 1000 });
  assert(r5.move >= 0, 'engine16 returns a move');
  console.log(`  16x16 first move: ${r5.move} (${(r5.move / 16) | 0},${r5.move % 16}), source=${r5.source}`);

  console.log('  Engine integration tests passed!');
}

// ===== Self-Play Test (engine vs engine) =====
function testSelfPlay() {
  console.log('--- Self-Play Test ---');

  const engine = createGomokuEngine({ N: 9, winLen: 5 });
  const board = new Int8Array(81);
  const players = [1, -1];
  let moveCount = 0;
  let winner = null;

  while (moveCount < 81) {
    const player = players[moveCount % 2];
    const result = engine.bestMove(board, player, { timeLimitMs: 200, maxDepth: 6 });

    if (result.move < 0) break;
    board[result.move] = player;
    moveCount++;

    // Check for winner (use GomokuBoard)
    const b = GomokuBoard.fromArray(board, 9, 5);
    winner = b.winner();
    if (winner !== null) break;
  }

  console.log(`  Game ended after ${moveCount} moves. Winner: ${winner === 0 ? 'draw' : (winner === 1 ? 'X' : 'O')}`);
  assert(winner !== undefined, 'game completed');

  // Play a 15x15 game (short)
  engine.clearCache();
  const engine15 = createGomokuEngine({ N: 15, winLen: 5 });
  const board15 = new Int8Array(225);
  let moveCount15 = 0;

  for (let i = 0; i < 20; i++) { // Just 20 moves for speed
    const player = players[i % 2];
    const result = engine15.bestMove(board15, player, { timeLimitMs: 500, maxDepth: 6 });
    if (result.move < 0) break;
    board15[result.move] = player;
    moveCount15++;
  }

  console.log(`  15x15: Played ${moveCount15} moves successfully`);
  assert(moveCount15 === 20, '15x15 played 20 moves');

  console.log('  Self-play tests passed!');
}

// ===== Run All Tests =====
console.log('=== Gomoku Engine V2 Test Suite ===\n');

try {
  testBoard();
  testPatterns();
  testEvaluation();
  testCandidates();
  testVCF();
  testSearch();
  testSymmetry();
  testEngine();
  testSelfPlay();

  console.log('\n=== ALL TESTS PASSED ===');
} catch (e) {
  console.error('\n!!! TEST FAILED !!!');
  console.error(e.message);
  console.error(e.stack);
  process.exit(1);
}
