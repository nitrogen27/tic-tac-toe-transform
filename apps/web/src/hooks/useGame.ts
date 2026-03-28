/**
 * Convenience hook that bundles game state + common dispatchers.
 */

import { useCallback } from "react";
import { useGameState, useGameDispatch, type GameMode } from "../store/gameStore";
import { WIN_LENGTH } from "../utils/constants";
import type { Position } from "../api/types";

export function useGame() {
  const state = useGameState();
  const dispatch = useGameDispatch();

  const placeStone = useCallback(
    (index: number) => dispatch({ type: "PLACE_STONE", index }),
    [dispatch]
  );

  const undo = useCallback(() => dispatch({ type: "UNDO" }), [dispatch]);

  const newGame = useCallback(
    (boardSize?: number) => dispatch({ type: "NEW_GAME", boardSize }),
    [dispatch]
  );

  const setBoardSize = useCallback(
    (boardSize: number) => dispatch({ type: "SET_BOARD_SIZE", boardSize }),
    [dispatch]
  );

  const setMode = useCallback(
    (mode: GameMode) => dispatch({ type: "SET_MODE", mode }),
    [dispatch]
  );

  /** Build a Position payload for the API. */
  const toPosition = useCallback((): Position => {
    return {
      boardSize: state.boardSize,
      winLength: WIN_LENGTH,
      currentPlayer: state.currentPlayer,
      cells: state.cells,
      lastMove: state.lastMove,
    };
  }, [state.boardSize, state.currentPlayer, state.cells, state.lastMove]);

  return {
    ...state,
    placeStone,
    undo,
    newGame,
    setBoardSize,
    setMode,
    toPosition,
  };
}
