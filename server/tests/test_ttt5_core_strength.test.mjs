import { beforeEach, describe, expect, it } from 'vitest';
import { createGameAdapter } from '../src/game_adapter.mjs';
import { mctsPUCT } from '../src/mcts_puct.mjs';
import { inferencePolicy } from '../src/inference_policy.mjs';
import { clearGameHistory, getGameHistoryStats, saveGameMove } from '../service.mjs';

describe('Current core strength improvements', () => {
  beforeEach(() => {
    clearGameHistory();
  });

  it('MCTS batches identical leaf evaluations instead of recomputing them', async () => {
    const adapter = createGameAdapter({ variant: 'ttt3' });
    let evalCalls = 0;

    const nnEval = async (b) => {
      evalCalls += 1;
      const policy = new Float32Array(b.length);
      const legal = adapter.legalMoves(b);
      const p = legal.length > 0 ? 1 / legal.length : 0;
      for (const mv of legal) policy[mv] = p;
      return { policy, value: 0 };
    };

    const mcts = mctsPUCT({
      N: adapter.N,
      nnEval,
      sims: 8,
      batchParallel: 8,
      temperature: 1.0,
      winnerFn: (b) => adapter.winner(b),
      legalMovesFn: (b) => adapter.legalMoves(b),
      candidateMovesFn: (b) => adapter.candidateMoves(b, { maxMoves: 9 }),
    });

    await mcts.run(adapter.emptyBoard(), 1);
    expect(evalCalls).toBe(1);
  });

  it('PUCT prefers a forced winning move on 3x3', async () => {
    const adapter = createGameAdapter({ variant: 'ttt3' });
    const board = new Int8Array([
      1, 1, 0,
      -1, -1, 0,
      0, 0, 0,
    ]);

    const nnEval = async (b) => {
      const policy = new Float32Array(b.length);
      const legal = adapter.legalMoves(b);
      const p = legal.length > 0 ? 1 / legal.length : 0;
      for (const mv of legal) policy[mv] = p;
      return { policy, value: 0 };
    };

    const mcts = mctsPUCT({
      N: adapter.N,
      nnEval,
      sims: 48,
      temperature: 1e-6,
      winnerFn: (b) => adapter.winner(b),
      legalMovesFn: (b) => adapter.legalMoves(b),
      candidateMovesFn: (b) => adapter.candidateMoves(b, { maxMoves: 9 }),
    });

    const { pi } = await mcts.run(adapter.cloneBoard(board), 1);
    const bestMove = pi.indexOf(Math.max(...pi));
    expect(bestMove).toBe(2);
  });

  it('Inference policy chooses a double-threat move on 5x5', async () => {
    const adapter = createGameAdapter({ variant: 'ttt5', winLen: 4 });
    const board = adapter.emptyBoard();
    board[11] = 1;
    board[12] = 1;
    board[8] = 1;
    board[18] = 1;
    board[6] = -1;
    board[16] = -1;

    const nnEval = async (b) => {
      const policy = new Float32Array(b.length);
      const legal = adapter.legalMoves(b);
      const p = legal.length > 0 ? 1 / legal.length : 0;
      for (const mv of legal) policy[mv] = p;
      policy[0] = 0.95;
      return { policy, value: 0.25 };
    };

    const result = await inferencePolicy({
      adapter,
      nnEval,
      board,
      player: 1,
      useMCTS: false,
    });

    expect(result.move).toBe(13);
    expect(result.source).toBe('safety_double_threat');
  });

  it('TTT5 moves do not pollute TTT3 incremental replay', async () => {
    await saveGameMove({
      board: Array(25).fill(0),
      move: 12,
      current: 1,
      variant: 'ttt5',
      gameId: 'ignore-me',
    });

    expect(getGameHistoryStats().count).toBe(0);
  });
});
