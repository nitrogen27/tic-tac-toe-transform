// Minimal MCTS (PUCT) using external NN callback that returns {policy,value} with legal-mask baked.
// Designed to be batched by a queue upstream.
import { legalMoves, getWinner, cloneBoard } from './game_nxn.mjs';

export function mctsPUCT({ N, nnEval, C_puct=1.5, sims=400, temperature=1.0 }){
  // Node structure: N(s), W(s,a), Q(s,a), P(s,a)
  const table = new Map(); // key -> {N, P:Array, W:Array, Q:Array, player}
  function key(board, player){ return player+'|'+Array.from(board).join(''); }

  async function evalLeaf(board, player){
    // Call NN: returns {policy[L], value}
    const { policy, value } = await nnEval(board, player);
    const moves = legalMoves(board);
    // Normalize policy on legal
    let sum=0; for (const m of moves) sum += policy[m];
    const P = policy.map((p,i)=> (moves.includes(i) && sum>0)? p/sum : 0);
    return { P, value };
  }

  async function simulate(board, player){
    const term = getWinner(board, N);
    if (term!==null){
      if (term===player) return +1;
      if (term===-player) return -1;
      return 0;
    }
    const k = key(board, player);
    let node = table.get(k);
    if (!node){
      const { P, value } = await evalLeaf(board, player);
      table.set(k, { N:0, P, W: new Float32Array(P.length), Q: new Float32Array(P.length), player });
      return value;
    }
    // Select
    const s = table.get(k);
    const moves = legalMoves(board);
    let best=-1e9, bestMove=-1;
    const Nsum = s.N + 1e-9;
    for (const a of moves){
      const U = C_puct * s.P[a] * Math.sqrt(Nsum) / (1 + s.W[a]); // approximate visit count by W to avoid second array
      const score = s.Q[a] + U;
      if (score>best){ best=score; bestMove=a; }
    }
    const mv = bestMove;
    const v0 = board[mv]; board[mv]=player;
    const v = await simulate(board, -player);
    board[mv]=v0;
    // Backup
    s.N += 1;
    s.W[mv] += (player * v); // perspective trick
    s.Q[mv] = s.W[mv] / (Math.max(1, s.W[mv]!==0 ? Math.abs(s.W[mv]) : 1));
    return v;
  }

  async function run(board, player){
    for (let i=0;i<sims;i++) await simulate(board, player);
    const k = key(board, player);
    const s = table.get(k);
    const moves = legalMoves(board);
    const pi = new Float32Array(board.length);
    // Use "visits" proxy by |W| for simplicity; in production keep N(s,a)
    let total = 0;
    for (const a of moves){ const w = Math.max(1e-6, Math.abs(s.W[a])); pi[a]=Math.pow(w, 1/temperature); total += pi[a]; }
    for (const a of moves){ pi[a] /= total; }
    return { pi, table };
  }

  return { run };
}
