// Generic N×N board helpers (tic-tac-toe/gomoku-like)
export function emptyBoard(N){ return new Int8Array(N*N).fill(0); }
export function cloneBoard(b){ const c = new Int8Array(b.length); c.set(b); return c; }
export function legalMoves(board){ const m=[]; for (let i=0;i<board.length;i++) if (board[i]===0) m.push(i); return m; }
export const DIRS = [[1,0],[0,1],[1,1],[1,-1]];
export const inb = (N,r,c)=> r>=0 && c>=0 && r<N && c<N;
export function countOccupied(board){
  let count = 0;
  for (let i = 0; i < board.length; i++) if (board[i] !== 0) count++;
  return count;
}
export function centerIndex(N){
  const mid = Math.floor(N / 2);
  return mid * N + mid;
}

// Gomoku/TicTacToe winner check (five-in-row if N>=5, else 3-in-row for N=3)
export function getWinner(board, N){
  const need = (N>=5?5:3);
  for (let r=0;r<N;r++){
    for (let c=0;c<N;c++){
      const who = board[r*N+c]; if (!who) continue;
      for (const [dr,dc] of DIRS){
        let k=1, rr=r+dr, cc=c+dc;
        while(inb(N,rr,cc) && board[rr*N+cc]===who){ k++; rr+=dr; cc+=dc; if (k>=need) return who; }
      }
    }
  }
  // draw detection: no empty
  for (let i=0;i<board.length;i++) if (board[i]===0) return null;
  return 0;
}

// Encode to 3 planes: my/opponent/empty  -> Float32Array length L*3
export function encodePlanes(board, player){
  const L = board.length;
  const out = new Float32Array(L*3);
  for (let i=0;i<L;i++){
    const base = i*3, v = board[i];
    out[base+0] = (v===player)?1:0;
    out[base+1] = (v!==0 && v!==player)?1:0;
    out[base+2] = (v===0)?1:0;
  }
  return out;
}

// Mask of legal moves (Float32Array length L with 1 for legal)
export function maskFromBoard(board){
  const L = board.length, m = new Float32Array(L);
  for (let i=0;i<L;i++) m[i] = (board[i]===0)?1:0;
  return m;
}

// Safety: immediate win or block one-ply
export function findImmediateWin(board, N, player){
  const moves = legalMoves(board);
  for (const mv of moves){
    const v = board[mv]; board[mv] = player;
    const w = getWinner(board, N);
    board[mv] = v;
    if (w===player) return mv;
  }
  return -1;
}
export function findImmediateBlock(board, N, player){
  const moves = legalMoves(board);
  for (const mv of moves){
    const v = board[mv]; board[mv] = -player;
    const w = getWinner(board, N);
    board[mv] = v;
    if (w===-player) return mv;
  }
  return -1;
}

// ===== Configurable win-length variants =====

// Winner detection with configurable win length (e.g. 4-in-a-row on 5x5)
export function getWinnerWithLen(board, N, winLen){
  for (let r=0;r<N;r++){
    for (let c=0;c<N;c++){
      const who = board[r*N+c]; if (!who) continue;
      for (const [dr,dc] of DIRS){
        let k=1, rr=r+dr, cc=c+dc;
        while(inb(N,rr,cc) && board[rr*N+cc]===who){ k++; rr+=dr; cc+=dc; if (k>=winLen) return who; }
      }
    }
  }
  for (let i=0;i<board.length;i++) if (board[i]===0) return null;
  return 0;
}

export function findImmediateWinWithLen(board, N, winLen, player){
  const moves = legalMoves(board);
  for (const mv of moves){
    const v = board[mv]; board[mv] = player;
    const w = getWinnerWithLen(board, N, winLen);
    board[mv] = v;
    if (w===player) return mv;
  }
  return -1;
}

export function findImmediateBlockWithLen(board, N, winLen, player){
  const moves = legalMoves(board);
  for (const mv of moves){
    const v = board[mv]; board[mv] = -player;
    const w = getWinnerWithLen(board, N, winLen);
    board[mv] = v;
    if (w===-player) return mv;
  }
  return -1;
}

export function collectImmediateWinsWithLen(board, N, winLen, player, moves = legalMoves(board)){
  const wins = [];
  for (const mv of moves){
    const v = board[mv];
    board[mv] = player;
    const w = getWinnerWithLen(board, N, winLen);
    board[mv] = v;
    if (w === player) wins.push(mv);
  }
  return wins;
}

export function countImmediateWinsWithLen(board, N, winLen, player, moves = legalMoves(board)){
  return collectImmediateWinsWithLen(board, N, winLen, player, moves).length;
}

function neighborhoodScore(board, N, move, player, radius = 2){
  const row = Math.floor(move / N);
  const col = move % N;
  const mid = (N - 1) / 2;
  let score = Math.max(0, N - (Math.abs(row - mid) + Math.abs(col - mid)));

  for (let dr = -radius; dr <= radius; dr++){
    for (let dc = -radius; dc <= radius; dc++){
      if (dr === 0 && dc === 0) continue;
      const rr = row + dr;
      const cc = col + dc;
      if (!inb(N, rr, cc)) continue;
      const v = board[rr * N + cc];
      if (v === player) score += 3;
      else if (v === -player) score += 2;
    }
  }

  return score;
}

// Candidate pruning for larger boards: prefer empty cells near existing stones.
export function candidateMovesWithLen(board, N, winLen, {
  radius = 2,
  maxMoves = 18,
  policyProbs = null,
  includePolicyTopK = 6,
} = {}){
  const moves = legalMoves(board);
  if (moves.length === 0) return [];
  if (countOccupied(board) === 0) return [centerIndex(N)];

  const marked = new Uint8Array(board.length);
  for (let i = 0; i < board.length; i++){
    if (board[i] === 0) continue;
    const row = Math.floor(i / N);
    const col = i % N;
    for (let dr = -radius; dr <= radius; dr++){
      for (let dc = -radius; dc <= radius; dc++){
        const rr = row + dr;
        const cc = col + dc;
        if (!inb(N, rr, cc)) continue;
        const idx = rr * N + cc;
        if (board[idx] === 0) marked[idx] = 1;
      }
    }
  }

  const candidates = [];
  for (const mv of moves){
    if (marked[mv]) candidates.push(mv);
  }

  const pool = candidates.length ? candidates : moves.slice();

  if (policyProbs){
    const topPolicy = moves
      .slice()
      .sort((a, b) => (policyProbs[b] ?? 0) - (policyProbs[a] ?? 0))
      .slice(0, includePolicyTopK);
    for (const mv of topPolicy){
      if (!pool.includes(mv)) pool.push(mv);
    }
  }

  if (pool.length <= maxMoves) return pool;

  const scored = pool.map((mv) => {
    let tactical = 0;
    const prev = board[mv];
    board[mv] = 1;
    const spanForX = countImmediateWinsWithLen(board, N, winLen, 1);
    board[mv] = -1;
    const spanForO = countImmediateWinsWithLen(board, N, winLen, -1);
    board[mv] = prev;

    tactical += Math.max(spanForX, spanForO) * 10;
    const local = Math.max(
      neighborhoodScore(board, N, mv, 1, 1),
      neighborhoodScore(board, N, mv, -1, 1)
    );
    const prior = policyProbs ? (policyProbs[mv] ?? 0) * 8 : 0;
    return { mv, score: tactical + local + prior };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, maxMoves).map(({ mv }) => mv);
}

export function findDoubleThreatMoveWithLen(board, N, winLen, player, moves = legalMoves(board)){
  for (const mv of moves){
    const prev = board[mv];
    board[mv] = player;
    const immediateWins = countImmediateWinsWithLen(board, N, winLen, player);
    board[mv] = prev;
    if (immediateWins >= 2) return mv;
  }
  return -1;
}

// Best-effort defense for positions where plain one-ply block is not enough.
export function findBestDefensiveMoveWithLen(board, N, winLen, player, moves = legalMoves(board), policyProbs = null){
  let bestMove = -1;
  let bestOppWins = Infinity;
  let bestOwnWins = -1;
  let bestScore = -Infinity;

  for (const mv of moves){
    const prev = board[mv];
    board[mv] = player;
    const oppWins = countImmediateWinsWithLen(board, N, winLen, -player);
    const ownWins = countImmediateWinsWithLen(board, N, winLen, player);
    board[mv] = prev;

    const heuristic = neighborhoodScore(board, N, mv, player, 1) + (policyProbs ? (policyProbs[mv] ?? 0) : 0);

    const isBetter =
      oppWins < bestOppWins ||
      (oppWins === bestOppWins && ownWins > bestOwnWins) ||
      (oppWins === bestOppWins && ownWins === bestOwnWins && heuristic > bestScore);

    if (isBetter){
      bestMove = mv;
      bestOppWins = oppWins;
      bestOwnWins = ownWins;
      bestScore = heuristic;
    }
  }

  return bestMove;
}
