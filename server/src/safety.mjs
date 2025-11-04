// Safety-правила для игры: мгновенный выигрыш и блокировка
import { winner, legalMoves, applyMove, cloneBoard } from './game_ttt3.mjs';

// Проверка мгновенного выигрыша для игрока
export function winningMove(board, player) {
  const moves = legalMoves(board);
  for (const move of moves) {
    const newBoard = applyMove(board, move, player);
    const w = winner(newBoard);
    if (w === player) {
      return move; // Найден выигрышный ход
    }
  }
  return -1; // Нет выигрышного хода
}

// Проверка необходимости блокировки (оппонент имеет выигрышный ход)
export function blockingMove(board, player) {
  const opponent = -player;
  const moves = legalMoves(board);
  
  // Проверяем каждую доступную позицию: если оппонент сделает ход в неё, выиграет ли он?
  for (const move of moves) {
    // Симулируем ход оппонента в эту позицию
    const testBoard = applyMove(board, move, opponent);
    const w = winner(testBoard);
    if (w === opponent) {
      // Оппонент выиграет, если сделает ход в эту позицию
      // Нужно заблокировать эту позицию
      return move;
    }
  }
  return -1; // Нет необходимости блокировать
}

// Безопасный выбор хода с применением safety-правил
export function safePick(board, player, policyProbs) {
  // 1. Проверяем мгновенный выигрыш
  const winMove = winningMove(board, player);
  if (winMove >= 0) {
    console.log('[SafePick] Winning move found:', winMove);
    return winMove;
  }
  
  // 2. Проверяем необходимость блокировки
  const blockMove = blockingMove(board, player);
  if (blockMove >= 0) {
    console.log('[SafePick] Blocking move found:', blockMove);
    return blockMove;
  }
  
  // 3. Используем argmax от policy среди легальных ходов
  // Если policy указывает на оптимальные ходы (вероятность > 0), выбираем среди них
  const moves = legalMoves(board);
  if (moves.length === 0) {
    return -1;
  }
  
  // Находим максимальную вероятность среди легальных ходов
  let bestMove = moves[0];
  let bestProb = policyProbs[moves[0]];
  
  for (const move of moves) {
    if (policyProbs[move] > bestProb) {
      bestProb = policyProbs[move];
      bestMove = move;
    }
  }
  
  console.log('[SafePick] Using policy argmax:', bestMove, 'prob:', bestProb.toFixed(4));
  return bestMove;
}
