/**
 * Hook for triggering analysis via the API.
 */

import { useCallback } from "react";
import { useGameState, useGameDispatch } from "../store/gameStore";
import { analyze as apiAnalyze } from "../api/client";
import { WIN_LENGTH } from "../utils/constants";

export function useAnalysis() {
  const state = useGameState();
  const dispatch = useGameDispatch();

  const runAnalysis = useCallback(async () => {
    dispatch({ type: "SET_ANALYZING", isAnalyzing: true });
    try {
      const result = await apiAnalyze({
        position: {
          boardSize: state.boardSize,
          winLength: WIN_LENGTH,
          currentPlayer: state.currentPlayer,
          cells: state.cells,
          lastMove: state.lastMove,
        },
        topK: 5,
        timeLimitMs: 3000,
        includePv: true,
      });
      dispatch({ type: "SET_ANALYSIS", analysis: result });
    } catch (err) {
      console.error("Analysis failed:", err);
      dispatch({ type: "SET_ANALYZING", isAnalyzing: false });
    }
  }, [state.boardSize, state.currentPlayer, state.cells, state.lastMove, dispatch]);

  return {
    analysis: state.analysis,
    isAnalyzing: state.isAnalyzing,
    runAnalysis,
  };
}
