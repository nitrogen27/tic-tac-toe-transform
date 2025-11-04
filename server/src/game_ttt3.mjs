// Игровая логика для крестики-нолики 3×3
// Представление доски: Int8Array(9), значения {-1, 0, +1}

export function emptyBoard() {
  return new Int8Array(9).fill(0);
}

export function cloneBoard(board) {
  return new Int8Array(board);
}

// Определение победителя: возвращает -1 (O), +1 (X), 0 (ничья), null (игра не окончена)
export function winner(board) {
  const lines = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // горизонтальные
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // вертикальные
    [0, 4, 8], [2, 4, 6]              // диагональные
  ];
  
  for (const [a, b, c] of lines) {
    const val = board[a];
    if (val !== 0 && val === board[b] && val === board[c]) {
      return val; // +1 для X, -1 для O
    }
  }
  
  // Проверка на ничью (все клетки заполнены)
  for (let i = 0; i < 9; i++) {
    if (board[i] === 0) {
      return null; // Игра не окончена
    }
  }
  
  return 0; // Ничья
}

// Легальные ходы (индексы пустых клеток)
export function legalMoves(board) {
  const moves = [];
  for (let i = 0; i < 9; i++) {
    if (board[i] === 0) {
      moves.push(i);
    }
  }
  return moves;
}

// Применение хода (не изменяет исходную доску)
export function applyMove(board, move, player) {
  const newBoard = cloneBoard(board);
  if (newBoard[move] !== 0) {
    throw new Error(`Invalid move: cell ${move} is already occupied`);
  }
  newBoard[move] = player;
  return newBoard;
}

// Проверка терминального состояния
export function isTerminal(board) {
  const w = winner(board);
  return w !== null; // null означает, что игра продолжается
}

// Кодирование доски в 3 плоскости (my/opponent/empty) → Float32Array(27)
// player: +1 для X, -1 для O
export function encodePlanes(board, player) {
  const out = new Float32Array(27); // 9 * 3
  for (let i = 0; i < 9; i++) {
    const base = i * 3;
    const val = board[i];
    if (val === player) {
      out[base + 0] = 1.0; // my
      out[base + 1] = 0.0; // opponent
      out[base + 2] = 0.0; // empty
    } else if (val === -player) {
      out[base + 0] = 0.0; // my
      out[base + 1] = 1.0; // opponent
      out[base + 2] = 0.0; // empty
    } else {
      out[base + 0] = 0.0; // my
      out[base + 1] = 0.0; // opponent
      out[base + 2] = 1.0; // empty
    }
  }
  return out;
}

// Маска легальных ходов → Float32Array(9)
export function maskLegalMoves(board) {
  const mask = new Float32Array(9);
  for (let i = 0; i < 9; i++) {
    mask[i] = board[i] === 0 ? 1.0 : 0.0;
  }
  return mask;
}
