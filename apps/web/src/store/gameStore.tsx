/**
 * Game state management via React Context + useReducer.
 */

import React, { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import type { CellValue, Player, AnalyzeResponse } from "../api/types";
import { DEFAULT_BOARD_SIZE, WIN_LENGTH } from "../utils/constants";
import { emptyCells } from "../utils/boardUtils";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export type GameMode = "play" | "analyze" | "review";

export interface GameState {
  boardSize: number;
  cells: CellValue[];
  currentPlayer: Player;
  moveHistory: number[];
  lastMove: number;
  mode: GameMode;
  analysis: AnalyzeResponse | null;
  isAnalyzing: boolean;
}

const initialState: GameState = {
  boardSize: DEFAULT_BOARD_SIZE,
  cells: emptyCells(DEFAULT_BOARD_SIZE),
  currentPlayer: 1,
  moveHistory: [],
  lastMove: -1,
  mode: "play",
  analysis: null,
  isAnalyzing: false,
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export type GameAction =
  | { type: "PLACE_STONE"; index: number }
  | { type: "UNDO" }
  | { type: "NEW_GAME"; boardSize?: number }
  | { type: "SET_BOARD_SIZE"; boardSize: number }
  | { type: "SET_MODE"; mode: GameMode }
  | { type: "SET_ANALYSIS"; analysis: AnalyzeResponse | null }
  | { type: "SET_ANALYZING"; isAnalyzing: boolean };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "PLACE_STONE": {
      if (state.cells[action.index] !== 0) return state;
      const newCells = [...state.cells];
      newCells[action.index] = state.currentPlayer;
      return {
        ...state,
        cells: newCells as CellValue[],
        currentPlayer: (state.currentPlayer === 1 ? -1 : 1) as Player,
        moveHistory: [...state.moveHistory, action.index],
        lastMove: action.index,
        analysis: null,
      };
    }

    case "UNDO": {
      if (state.moveHistory.length === 0) return state;
      const history = [...state.moveHistory];
      const lastIdx = history.pop()!;
      const newCells = [...state.cells];
      newCells[lastIdx] = 0;
      return {
        ...state,
        cells: newCells as CellValue[],
        currentPlayer: (state.currentPlayer === 1 ? -1 : 1) as Player,
        moveHistory: history,
        lastMove: history.length > 0 ? history[history.length - 1] : -1,
        analysis: null,
      };
    }

    case "NEW_GAME": {
      const bs = action.boardSize ?? state.boardSize;
      return {
        ...initialState,
        boardSize: bs,
        cells: emptyCells(bs),
      };
    }

    case "SET_BOARD_SIZE":
      return {
        ...initialState,
        boardSize: action.boardSize,
        cells: emptyCells(action.boardSize),
      };

    case "SET_MODE":
      return { ...state, mode: action.mode };

    case "SET_ANALYSIS":
      return { ...state, analysis: action.analysis, isAnalyzing: false };

    case "SET_ANALYZING":
      return { ...state, isAnalyzing: action.isAnalyzing };

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const GameStateCtx = createContext<GameState>(initialState);
const GameDispatchCtx = createContext<Dispatch<GameAction>>(() => {});

export function GameProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(gameReducer, initialState);
  return (
    <GameStateCtx.Provider value={state}>
      <GameDispatchCtx.Provider value={dispatch}>{children}</GameDispatchCtx.Provider>
    </GameStateCtx.Provider>
  );
}

export function useGameState(): GameState {
  return useContext(GameStateCtx);
}

export function useGameDispatch(): Dispatch<GameAction> {
  return useContext(GameDispatchCtx);
}
