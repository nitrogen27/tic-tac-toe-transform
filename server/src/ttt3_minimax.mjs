// Полный minimax с мемоизацией симметричных позиций для крестики-нолики 3×3
import { winner, legalMoves, applyMove, isTerminal, cloneBoard, encodePlanes, emptyBoard } from './game_ttt3.mjs';
import { winningMove, blockingMove } from './safety.mjs';

// Мемоизация позиций (нормализованных через симметрии)
const cache = new Map();

// Генерация всех симметрий позиции (вращения и отражения)
function symmetries(board) {
  const syms = [];
  const b = Array.isArray(board) ? board : Array.from(board);
  
  // Исходная позиция
  syms.push(b);
  
  // Вращения на 90°, 180°, 270°
  syms.push([b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]);
  syms.push([b[8], b[7], b[6], b[5], b[4], b[3], b[2], b[1], b[0]]);
  syms.push([b[2], b[5], b[8], b[1], b[4], b[7], b[0], b[3], b[6]]);
  
  // Отражения (вертикальное и горизонтальное)
  syms.push([b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6]]);
  syms.push([b[6], b[7], b[8], b[3], b[4], b[5], b[0], b[1], b[2]]);
  
  // Диагональные отражения
  syms.push([b[0], b[3], b[6], b[1], b[4], b[7], b[2], b[5], b[8]]);
  syms.push([b[8], b[5], b[2], b[7], b[4], b[1], b[6], b[3], b[0]]);
  
  return syms;
}

// Нормализация позиции (выбираем каноническую форму)
function normalize(board) {
  // Преобразуем Int8Array в обычный массив для работы с симметриями
  const boardArray = Array.from(board);
  const syms = symmetries(boardArray);
  // Выбираем лексикографически минимальную
  syms.sort((a, b) => {
    for (let i = 0; i < 9; i++) {
      if (a[i] !== b[i]) return a[i] - b[i];
    }
    return 0;
  });
  return syms[0].join(',');
}

// Преобразование хода из канонической формы обратно к исходной
function transformMove(move, fromCanonical, toOriginal) {
  const origSyms = symmetries(Array.from(toOriginal));
  const canonSyms = symmetries(Array.from(fromCanonical));
  
  // Находим индекс канонической формы в списке симметрий исходной
  const canonKey = canonSyms[0].join(',');
  for (let idx = 0; idx < origSyms.length; idx++) {
    if (origSyms[idx].join(',') === canonKey) {
      // Находим соответствующее преобразование
      if (idx === 0) return move;
      // Применяем обратное преобразование (упрощенная версия)
      // Для полной реализации нужно хранить матрицы преобразований
      return move; // Упрощение: возвращаем исходный ход
    }
  }
  return move;
}

// Minimax с мемоизацией
function minimax(board, player, depth = 0) {
  const canonKey = normalize(board);
  
  if (cache.has(canonKey)) {
    return cache.get(canonKey);
  }
  
  const w = winner(board);
  if (w !== null) {
    // Терминальное состояние: +1 если выиграл текущий игрок, -1 если проиграл, 0 ничья
    const result = {
      value: w === player ? 1 : (w === -player ? -1 : 0),
      optimalMoves: []
    };
    cache.set(canonKey, result);
    return result;
  }
  
  const moves = legalMoves(board);
  if (moves.length === 0) {
    const result = { value: 0, optimalMoves: [] };
    cache.set(canonKey, result);
    return result;
  }
  
  let bestValue = -Infinity;
  const optimalMoves = [];
  
  for (const move of moves) {
    const newBoard = applyMove(board, move, player);
    const result = minimax(newBoard, -player, depth + 1);
    const moveValue = -result.value; // Инвертируем для оппонента
    
    if (moveValue > bestValue) {
      bestValue = moveValue;
      optimalMoves.length = 0;
      optimalMoves.push(move);
    } else if (Math.abs(moveValue - bestValue) < 1e-6) {
      optimalMoves.push(move);
    }
  }
  
  const result = { value: bestValue, optimalMoves };
  cache.set(canonKey, result);
  return result;
}

// Получить value и policy для позиции
export function getTeacherValueAndPolicy(board, player) {
  const moves = legalMoves(board);
  const policy = new Float32Array(9);
  
  // Проверяем критические ходы (выигрыш и блокировка) ПЕРЕД использованием minimax
  // Это гарантирует, что модель учится правильно блокировать угрозы
  const winMove = winningMove(board, player);
  
  if (winMove >= 0) {
    // Есть выигрышный ход - он единственный оптимальный
    policy[winMove] = 1.0;
    return {
      value: 1, // Гарантированный выигрыш
      policy
    };
  }
  
  // Проверяем необходимость блокировки
  const blockMove = blockingMove(board, player);
  if (blockMove >= 0) {
    // Есть необходимость блокировки - это критический ход
    policy[blockMove] = 1.0;
    // Проверяем, что после блокировки игра продолжается
    const testBoard = applyMove(board, blockMove, player);
    const testResult = minimax(testBoard, -player);
    return {
      value: -testResult.value, // Инвертируем для оппонента
      policy
    };
  }
  
  // Если нет критических ходов, используем minimax
  const result = minimax(board, player);
  
  if (result.optimalMoves.length > 0) {
    // Равномерное распределение по оптимальным ходам
    const prob = 1.0 / result.optimalMoves.length;
    for (const move of result.optimalMoves) {
      policy[move] = prob;
    }
  } else if (moves.length > 0) {
    // Если нет оптимальных ходов (не должно быть, но на всякий случай)
    const prob = 1.0 / moves.length;
    for (const move of moves) {
      policy[move] = prob;
    }
  }
  
  return {
    value: result.value, // +1, 0, или -1
    policy
  };
}

// Генератор батчей для обучения
export function* teacherBatches({ batchSize = 512 }) {
  const cache = new Map();
  const visited = new Set();
  
  // Генерируем все достижимые позиции
  function* generatePositions(board, player, depth = 0) {
    const canonKey = normalize(board);
    if (visited.has(canonKey)) return;
    visited.add(canonKey);
    
    yield { board: cloneBoard(board), player };
    
    if (!isTerminal(board)) {
      const moves = legalMoves(board);
      for (const move of moves) {
        const newBoard = applyMove(board, move, player);
        yield* generatePositions(newBoard, -player, depth + 1);
      }
    }
  }
  
  // Собираем все позиции
  const allPositions = [];
  for (const pos of generatePositions(emptyBoard(), 1)) {
    allPositions.push(pos);
  }
  
  console.log(`[TeacherBatches] Generated ${allPositions.length} unique positions`);
  
  // Генерируем батчи
  let batch = [];
  for (const { board, player } of allPositions) {
    const { value, policy } = getTeacherValueAndPolicy(board, player);
    const planes = encodePlanes(board, player);
    
    batch.push({ planes, policy, value });
    
    if (batch.length >= batchSize) {
      yield {
        x: new Float32Array(batch.flatMap(b => Array.from(b.planes))),
        yPolicy: new Float32Array(batch.flatMap(b => Array.from(b.policy))),
        yValue: new Float32Array(batch.map(b => b.value)),
        count: batch.length
      };
      batch = [];
    }
  }
  
  // Последний батч
  if (batch.length > 0) {
    yield {
      x: new Float32Array(batch.flatMap(b => Array.from(b.planes))),
      yPolicy: new Float32Array(batch.flatMap(b => Array.from(b.policy))),
      yValue: new Float32Array(batch.map(b => b.value)),
      count: batch.length
    };
  }
}

