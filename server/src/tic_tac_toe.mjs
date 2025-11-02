export function emptyBoard() { return Array(9).fill(0); }

export function getWinner(board) {
  const lines = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6],
  ];
  for (const [a,b,c] of lines) {
    if (board[a] && board[a] === board[b] && board[b] === board[c]) return board[a];
  }
  if (board.every(v => v !== 0)) return 0;
  return null;
}

export function legalMoves(board) {
  const arr = [];
  for (let i=0;i<9;i++) if (board[i]===0) arr.push(i);
  return arr;
}

export function relativeCells(board, current=1) {
  return board.map(v => v===0?0:(v===current?1:2));
}

// Оптимизированный minimax с alpha-beta pruning
function minimax(board, player, me, alpha = -Infinity, beta = Infinity) {
  const w = getWinner(board);
  if (w !== null) {
    if (w === me) return {score: 1};
    if (w === 0) return {score: 0};
    return {score: -1};
  }
  const moves = legalMoves(board);
  
  // Быстрая проверка: если есть один ход, возвращаем его сразу
  if (moves.length === 1) {
    return {move: moves[0], score: 0};
  }
  
  let best = null;
  
  if (player === me) {
    // Максимизируем для текущего игрока
    for (const m of moves) {
      board[m] = player;
      const r = minimax(board, player===1?2:1, me, alpha, beta);
      board[m] = 0;
      const score = r.score;
      
      if (best === null || score > best.score) {
        best = {move: m, score};
        alpha = Math.max(alpha, score);
      }
      
      // Alpha-beta pruning: можем прервать если нашли оптимальный ход
      if (score === 1 || beta <= alpha) {
        break;
      }
    }
  } else {
    // Минимизируем для оппонента
    for (const m of moves) {
      board[m] = player;
      const r = minimax(board, player===1?2:1, me, alpha, beta);
      board[m] = 0;
      const score = r.score;
      
      if (best === null || score < best.score) {
        best = {move: m, score};
        beta = Math.min(beta, score);
      }
      
      // Alpha-beta pruning
      if (score === -1 || beta <= alpha) {
        break;
      }
    }
  }
  
  return best ?? {move: moves[0], score: 0};
}

export function teacherBestMove(board, current=1) {
  return minimax([...board], current, current).move;
}
