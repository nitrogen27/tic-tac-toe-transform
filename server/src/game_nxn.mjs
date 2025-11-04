// Generic N×N board helpers (tic-tac-toe/gomoku-like)
export function emptyBoard(N){ return new Int8Array(N*N).fill(0); }
export function cloneBoard(b){ const c = new Int8Array(b.length); c.set(b); return c; }
export function legalMoves(board){ const m=[]; for (let i=0;i<board.length;i++) if (board[i]===0) m.push(i); return m; }
export const DIRS = [[1,0],[0,1],[1,1],[1,-1]];
export const inb = (N,r,c)=> r>=0 && c>=0 && r<N && c<N;

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
