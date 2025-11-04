import { describe, it, expect } from 'vitest'

// Игровая логика (скопирована из App.vue)
function getWinner(board) {
  const lines = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6],
  ];
  for (const [a,b,c] of lines) {
    if (board[a] && board[a] === board[b] && board[b] === board[c]) return board[a];
  }
  if (board.every(v => v !== 0)) return 0; // Ничья
  return null; // Игра продолжается
}

function legalMoves(board) {
  const moves = [];
  for (let i = 0; i < board.length; i++) {
    if (board[i] === 0) moves.push(i);
  }
  return moves;
}

describe('Game Logic', () => {
  describe('getWinner', () => {
    it('should return null for empty board', () => {
      const board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      expect(getWinner(board)).toBe(null);
    });

    it('should detect horizontal win for player 1', () => {
      const board = [1, 1, 1, 0, 0, 0, 0, 0, 0];
      expect(getWinner(board)).toBe(1);
    });

    it('should detect horizontal win for player 2', () => {
      const board = [0, 0, 0, 2, 2, 2, 0, 0, 0];
      expect(getWinner(board)).toBe(2);
    });

    it('should detect vertical win', () => {
      const board = [1, 0, 0, 1, 0, 0, 1, 0, 0];
      expect(getWinner(board)).toBe(1);
    });

    it('should detect diagonal win (top-left to bottom-right)', () => {
      const board = [1, 0, 0, 0, 1, 0, 0, 0, 1];
      expect(getWinner(board)).toBe(1);
    });

    it('should detect diagonal win (top-right to bottom-left)', () => {
      const board = [0, 0, 2, 0, 2, 0, 2, 0, 0];
      expect(getWinner(board)).toBe(2);
    });

    it('should detect draw', () => {
      const board = [1, 2, 1, 2, 1, 2, 2, 1, 2];
      expect(getWinner(board)).toBe(0);
    });

    it('should return null for game in progress', () => {
      const board = [1, 2, 0, 0, 1, 0, 0, 0, 0];
      expect(getWinner(board)).toBe(null);
    });
  });

  describe('legalMoves', () => {
    it('should return all positions for empty board', () => {
      const board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      expect(legalMoves(board)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('should return only empty positions', () => {
      const board = [1, 0, 2, 0, 1, 0, 0, 0, 2];
      expect(legalMoves(board)).toEqual([1, 3, 5, 6, 7]);
    });

    it('should return empty array for full board', () => {
      const board = [1, 2, 1, 2, 1, 2, 2, 1, 2];
      expect(legalMoves(board)).toEqual([]);
    });
  });

  describe('Game state transitions', () => {
    it('should handle valid move', () => {
      const board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      const newBoard = [...board];
      newBoard[4] = 1;
      expect(newBoard[4]).toBe(1);
      expect(board[4]).toBe(0); // Original unchanged
    });

    it('should prevent invalid move (occupied cell)', () => {
      const board = [1, 0, 0, 0, 0, 0, 0, 0, 0];
      const legal = legalMoves(board);
      expect(legal).not.toContain(0);
      expect(legal).toContain(1);
    });
  });
});
