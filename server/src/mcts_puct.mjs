// PUCT search using an external NN callback that returns { policy, value }.
// Value is always interpreted from the current player's perspective.
// Supports leaf-parallel batched evaluation via virtual loss.
import { legalMoves, getWinner } from './game_nxn.mjs';

// ===== Dirichlet noise sampling =====
function randn() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

function gammaRandom(alpha) {
  let a = alpha;
  let boost = 1;
  if (a < 1) {
    boost = Math.pow(Math.random(), 1 / a);
    a += 1;
  }
  const d = a - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  let x, v;
  for (;;) {
    do {
      x = randn();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * boost;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * boost;
  }
}

function sampleDirichlet(alpha, n) {
  const samples = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    samples[i] = gammaRandom(alpha);
    sum += samples[i];
  }
  if (sum > 1e-10) {
    for (let i = 0; i < n; i++) samples[i] /= sum;
  } else {
    const uniform = 1 / n;
    for (let i = 0; i < n; i++) samples[i] = uniform;
  }
  return samples;
}

export function mctsPUCT({
  N,
  nnEval,
  C_puct = 1.5,
  sims = 400,
  temperature = 1.0,
  dirichletAlpha = 0,
  dirichletEpsilon = 0,
  winnerFn,
  legalMovesFn,
  candidateMovesFn,
  keyFn,
  virtualLoss = 3,       // Virtual loss for leaf parallelism
  batchParallel = 8,      // Number of parallel simulations per batch
}){
  const getWin = winnerFn || ((board) => getWinner(board, N));
  const getMoves = legalMovesFn || ((board) => legalMoves(board));
  const getCandidates = candidateMovesFn || ((board, _player, legal) => legal);
  const getKey = keyFn || ((board, player) => `${player}|${Array.from(board).join('')}`);
  const table = new Map();

  // Pending expansions: deduplicate identical leaves reached in the same batched rollout.
  const pendingExpand = new Map();

  function buildNode(board, player, policy, legal, movePool) {
    const legalMask = new Uint8Array(board.length);
    for (const mv of legal) legalMask[mv] = 1;

    const moves = [];
    const seen = new Uint8Array(board.length);
    for (const mv of movePool.length ? movePool : legal){
      if (!legalMask[mv] || seen[mv]) continue;
      seen[mv] = 1;
      moves.push(mv);
    }

    const priors = new Float32Array(board.length);
    let sum = 0;
    for (const mv of moves){
      const prior = Number.isFinite(policy[mv]) ? Math.max(0, policy[mv]) : 0;
      priors[mv] = prior;
      sum += prior;
    }

    if (sum <= 1e-8){
      const uniform = moves.length > 0 ? 1 / moves.length : 0;
      for (const mv of moves) priors[mv] = uniform;
    } else {
      for (const mv of moves) priors[mv] /= sum;
    }

    return {
      moves,
      P: priors,
      Ns: 0,
      Nsa: new Uint32Array(board.length),
      Wsa: new Float32Array(board.length),
      Qsa: new Float32Array(board.length),
      VL: new Float32Array(board.length),  // Virtual losses per action
    };
  }

  // Select best move using PUCT with virtual loss
  function selectMove(node) {
    let bestMove = -1;
    let bestScore = -Infinity;
    const totalVisits = node.Ns + node.VL.reduce((s, v) => s + v, 0);
    const sqrtNs = Math.sqrt(totalVisits + 1e-8);

    for (const mv of node.moves){
      const effectiveN = node.Nsa[mv] + node.VL[mv];
      const q = effectiveN > 0
        ? (node.Wsa[mv] - node.VL[mv] * 1.0) / effectiveN  // VL biases Q down
        : 0;
      const u = C_puct * node.P[mv] * sqrtNs / (1 + effectiveN);
      const score = q + u;
      if (score > bestScore){
        bestScore = score;
        bestMove = mv;
      }
    }
    return bestMove;
  }

  // Single simulation traversal — returns { value, path } or { needsExpand, ... }
  function simulateSync(board, player, path) {
    const terminal = getWin(board);
    if (terminal !== null){
      if (terminal === player) return { value: 1 };
      if (terminal === -player) return { value: -1 };
      return { value: 0 };
    }

    const stateKey = getKey(board, player);
    let node = table.get(stateKey);
    if (!node){
      // Leaf node — needs NN evaluation
      const legal = getMoves(board, player);
      return { needsExpand: true, board: [...board], player, legal, stateKey };
    }

    const bestMove = selectMove(node);
    if (bestMove < 0) return { value: 0 };

    // Apply virtual loss
    node.VL[bestMove] += virtualLoss;
    path.push({ node, move: bestMove });

    board[bestMove] = player;
    const result = simulateSync(board, -player, path);
    board[bestMove] = 0; // Undo move

    return result;
  }

  // Backup path with a value and remove virtual losses
  function backup(path, leafValue) {
    let value = leafValue;
    for (let i = path.length - 1; i >= 0; i--) {
      value = -value;
      const { node, move } = path[i];
      node.VL[move] -= virtualLoss;
      node.Ns += 1;
      node.Nsa[move] += 1;
      node.Wsa[move] += value;
      node.Qsa[move] = node.Wsa[move] / node.Nsa[move];
    }
  }

  function rootPolicy(node, boardLength){
    const pi = new Float32Array(boardLength);
    if (!node || node.moves.length === 0) return pi;

    if (temperature <= 1e-6){
      let bestMove = node.moves[0];
      for (const mv of node.moves){
        if (node.Nsa[mv] > node.Nsa[bestMove]) bestMove = mv;
      }
      pi[bestMove] = 1;
      return pi;
    }

    let total = 0;
    const invTemp = 1 / temperature;
    for (const mv of node.moves){
      const visits = Math.max(1e-8, node.Nsa[mv]);
      pi[mv] = Math.pow(visits, invTemp);
      total += pi[mv];
    }

    if (total <= 1e-8){
      const uniform = 1 / node.moves.length;
      for (const mv of node.moves) pi[mv] = uniform;
      return pi;
    }

    for (const mv of node.moves) pi[mv] /= total;
    return pi;
  }

  async function run(board, player){
    let simsCompleted = 0;

    while (simsCompleted < sims) {
      // Launch a batch of parallel simulations
      const batchSize = Math.min(batchParallel, sims - simsCompleted);
      const simResults = [];

      for (let b = 0; b < batchSize; b++) {
        const path = [];
        const boardCopy = [...board];
        const result = simulateSync(boardCopy, player, path);
        simResults.push({ result, path });
      }

      // Collect all leaf nodes that need NN evaluation
      pendingExpand.clear();
      for (let i = 0; i < simResults.length; i++) {
        if (simResults[i].result.needsExpand) {
          const leaf = simResults[i];
          const key = leaf.result.stateKey;
          const existing = pendingExpand.get(key);
          if (existing) {
            existing.paths.push(leaf.path);
          } else {
            pendingExpand.set(key, {
              result: leaf.result,
              paths: [leaf.path],
            });
          }
        }
      }

      // Batch evaluate all leaves
      if (pendingExpand.size > 0) {
        const uniqueLeaves = Array.from(pendingExpand.values());
        const evalPromises = uniqueLeaves.map((leaf) =>
          nnEval(leaf.result.board, leaf.result.player)
        );
        const evalResults = await Promise.all(evalPromises);

        for (let j = 0; j < uniqueLeaves.length; j++) {
          const { result, paths } = uniqueLeaves[j];
          const { policy, value } = evalResults[j];
          const legal = result.legal;
          const movePool = getCandidates(result.board, result.player, legal, policy) || legal;
          const node = buildNode(result.board, result.player, policy, legal, movePool);
          node.value = value;
          table.set(result.stateKey, node);
          for (const path of paths) {
            backup(path, value);
          }
        }
      }

      // Backup terminal/already-expanded results
      for (let i = 0; i < simResults.length; i++) {
        if (!simResults[i].result.needsExpand) {
          backup(simResults[i].path, simResults[i].result.value);
        }
      }

      simsCompleted += batchSize;

      // After first batch, apply Dirichlet noise at root
      if (simsCompleted === batchSize && dirichletAlpha > 0 && dirichletEpsilon > 0) {
        const rootKey = getKey(board, player);
        const rootNode = table.get(rootKey);
        if (rootNode && rootNode.moves.length > 0) {
          const noise = sampleDirichlet(dirichletAlpha, rootNode.moves.length);
          for (let j = 0; j < rootNode.moves.length; j++) {
            const mv = rootNode.moves[j];
            rootNode.P[mv] = (1 - dirichletEpsilon) * rootNode.P[mv] + dirichletEpsilon * noise[j];
          }
        }
      }
    }

    const stateKey = getKey(board, player);
    const node = table.get(stateKey);
    const pi = rootPolicy(node, board.length);
    const value = node
      ? node.moves.reduce((best, mv) => node.Nsa[mv] > best.visits ? { visits: node.Nsa[mv], value: node.Qsa[mv] } : best, { visits: -1, value: 0 }).value
      : 0;
    return { pi, value, table };
  }

  return { run };
}
