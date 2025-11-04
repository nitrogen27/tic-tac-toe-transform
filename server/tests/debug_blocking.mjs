// Отладка blockingMove
import { emptyBoard, applyMove, winner, legalMoves } from '../src/game_ttt3.mjs';
import { blockingMove } from '../src/safety.mjs';

const board = emptyBoard();
applyMove(board, 0, 1); // X в 0
applyMove(board, 3, 1); // X в 3

console.log('Board:', Array.from(board));
console.log('Legal moves:', legalMoves(board));

// Проверяем, может ли X выиграть, сделав ход в 6
const testBoard = applyMove(board, 6, 1);
console.log('Test board (X in 6):', Array.from(testBoard));
console.log('Winner:', winner(testBoard));

// Проверяем blockingMove
const blockMove = blockingMove(board, -1);
console.log('Blocking move:', blockMove);

