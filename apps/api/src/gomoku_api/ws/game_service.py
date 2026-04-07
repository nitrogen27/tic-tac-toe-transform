"""In-memory game history and session management."""

from __future__ import annotations

import uuid
from collections import deque


class GameService:
    """Tracks game sessions and move history for the legacy Vue UI."""

    def __init__(self, max_history: int = 10_000) -> None:
        self._games: dict[str, dict] = {}
        self._history: deque[dict] = deque(maxlen=max_history)
        self._total_moves: int = 0

    def start_game(self, player_role: int = 1, variant: str = "ttt3") -> str:
        game_id = str(uuid.uuid4())
        self._games[game_id] = {
            "gameId": game_id,
            "playerRole": player_role,
            "variant": variant,
            "moves": [],
            "finished": False,
            "analyzed": False,
        }
        return game_id

    def save_move(self, board: list, move: int, current: int, game_id: str | None = None, variant: str = "ttt3") -> dict:
        entry = {"board": board, "move": move, "current": current, "variant": variant}
        self._history.append(entry)
        self._total_moves += 1
        if game_id and game_id in self._games:
            self._games[game_id]["moves"].append(entry)
        return self.get_stats()

    def finish_game(self, game_id: str, winner: int = 0, **kwargs) -> dict:
        game = self._games.get(game_id)
        if game:
            game["winner"] = winner
            game["finished"] = True
            game["analyzed"] = False
        return self.get_stats()

    def get_stats(self) -> dict:
        return {
            "totalGames": len(self._games),
            "totalMoves": self._total_moves,
            "count": len(self._history),
        }

    def get_history(self) -> list[dict]:
        return list(self._history)

    def get_finished_game(self, game_id: str) -> dict | None:
        game = self._games.get(game_id)
        if not game or not game.get("finished"):
            return None
        return dict(game)

    def get_finished_games(self, variant: str | None = None, *, unanalyzed_only: bool = False) -> list[dict]:
        games: list[dict] = []
        for game in self._games.values():
            if not game.get("finished"):
                continue
            if variant and game.get("variant") != variant:
                continue
            if unanalyzed_only and bool(game.get("analyzed")):
                continue
            games.append(dict(game))
        return games

    def mark_game_analyzed(self, game_id: str, *, positions: int = 0) -> dict | None:
        game = self._games.get(game_id)
        if not game:
            return None
        game["analyzed"] = True
        game["analyzedPositions"] = int(max(0, positions))
        return dict(game)

    def clear_history(self) -> dict:
        self._history.clear()
        self._games.clear()
        count = self._total_moves
        self._total_moves = 0
        return {"cleared": True, "count": count}
